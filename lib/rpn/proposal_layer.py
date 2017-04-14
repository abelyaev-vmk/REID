# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

import caffe
import numpy as np
import json
from core.config import cfg
from .generate_anchors import generate_anchors
from core.bbox_transform import bbox_transform_inv, clip_boxes
from core.nms_wrapper import nms
from utils.timer import Timer

t = {'total': Timer(), 'nms': Timer(), 'sort': Timer()}
DEBUG = False


class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid JSON
        try:
            layer_params = json.loads(self.param_str_)
        except:
            layer_params = json.loads(self.param_str)

        self._feat_stride = layer_params['feat_stride']
        self._anchors = generate_anchors(base_size=self._feat_stride,
                                         ratios=layer_params['anchor_ratios'],
                                         scales=layer_params['anchor_scales'],
                                         shift_num_xy=layer_params['anchor_shift_num_xy'])
        self._min_size = layer_params.get('min_size', cfg.TRAIN.RPN_MIN_SIZE)
        self._pre_nms_topN = layer_params.get('pre_nms_topN', cfg.TRAIN.RPN_PRE_NMS_TOP_N)
        self._post_nms_topN = layer_params.get('post_nms_topN', cfg.TRAIN.RPN_POST_NMS_TOP_N)
        self._nms_thresh = layer_params.get('nms_thresh', cfg.TRAIN.RPN_NMS_THRESH)
        self._num_classes = layer_params['num_classes']
        self._clip_proposals = layer_params.get('clip_proposals', True)

        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print('feat_stride: {}'.format(self._feat_stride))
            print('anchors:')
            print(self._anchors)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        t['total'].tic()

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, :, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print('scale: {}'.format(im_info[2]))

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print('score map size: {}'.format(scores.shape))

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, self._num_classes + 1, self._num_anchors))
        scores = scores.transpose((0, 2, 1)).reshape((-1, self._num_classes + 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        if self._clip_proposals:
            proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, self._min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep, :]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        t['sort'].tic()
        order = np.sum(scores[:, 1:], axis=1).ravel().argsort()[::-1]
        t['sort'].toc()
        if self._pre_nms_topN > 0:
            order = order[:self._pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order, :]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        t['nms'].tic()
        fg_scores = np.sum(scores[:, 1:], axis=1).reshape((-1, 1))
        keep = nms(np.hstack((proposals, fg_scores)),
                   self._nms_thresh)
        t['nms'].toc()
        if self._post_nms_topN > 0:
            keep = keep[:self._post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep, :]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        t['total'].toc()

        # print('ptl total: {:.4f}, nms: {:.4f}, sort: {:.4f}'.format(t['total'].average_time,
        #                                                             t['nms'].average_time,
        #                                                             t['sort'].average_time))

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
