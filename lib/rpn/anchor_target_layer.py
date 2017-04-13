# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

import os
import caffe
import json
from core.config import cfg
import numpy as np
import numpy.random as npr
from .generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from core.bbox_transform import bbox_transform
from utils.timer import Timer


DEBUG = True
t = Timer()


class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        layer_params = json.loads(self.param_str)
        self._feat_stride = layer_params['feat_stride']
        self._iters = 0
        self._periodic_tn_enabled = True
        self._square_targets = layer_params['square_targets']
        self._square_targets_ky = layer_params.get('square_targets_ky', 0.5)

        self._positive_overlap = layer_params['positive_overlap']
        self._negative_overlap = layer_params['negative_overlap']
        self._batchsize = layer_params['batchsize']
        self._max_tn_fraction = layer_params['tn_fraction']
        self._fg_fraction = layer_params['fg_fraction']
        self._name = layer_params['name']
        self._num_classes = layer_params['num_classes']
        self._ratios = layer_params['anchor_ratios']


        assert self._square_targets and len(self._ratios) == 1

        self._anchors = generate_anchors(base_size=self._feat_stride,
                                         ratios=layer_params['anchor_ratios'],
                                         scales=layer_params['anchor_scales'],
                                         shift_num_xy=layer_params['anchor_shift_num_xy'])

        self._num_anchors = self._anchors.shape[0]
        t.tic()

        if DEBUG:
            print('anchors:')
            print(self._anchors)
            print('anchor shapes:')
            print(np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            )))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)
        self._top_neg_fraction = cfg.TRAIN.RPN_LINEAR_START_TNF

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print('AnchorTargetLayer: height', height, 'width', width)

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap
        t.toc()
        self._iters += 1

        assert bottom[0].shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].shape[-2:]

        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        gt_ignored_boxes = bottom[2].data

        # im_info
        im_info = bottom[3].data[0, :]

        if DEBUG:
            print('')
            print('class_distrib', gt_boxes[:, 4])
            print('class_pid', gt_boxes[:, 5])
            print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print('scale: {}'.format(im_info[2]))
            print('height, width: ({}, {})'.format(height, width))
            #print 'rpn: gt_boxes.shape', gt_boxes.shape
            #print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride

        if 0 and DEBUG:
            for i in range(6):
                print('bottom[{}].shape = {}'.format(i, bottom[i].data.shape))
            print(shift_x.shape, shift_y.shape, width, height)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        if DEBUG:
            print(self._anchors.shape, shifts.shape)
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]


        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))
            print('sq_ky:', self._square_targets_ky)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print('anchors.shape', anchors.shape)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)

        if gt_boxes.shape[0]:
            boxes = _square_boxes(gt_boxes, self._ratios[0], self._square_targets_ky) \
                if self._square_targets else gt_boxes
            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(boxes, dtype=np.float))

            argmax_overlaps = overlaps.argmax(axis=1)

            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < self._negative_overlap] = 0

            # gt_argmax_overlaps = overlaps.argmax(axis=0)
            # gt_max_overlaps = overlaps[gt_argmax_overlaps,
            #                            np.arange(overlaps.shape[1])]
            # gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            # fg label: for each gt, anchor with highest overlap
            # if np.max(max_overlaps) > 0.5:
            #     labels[gt_argmax_overlaps] = 1

            # fg label: above threshold IOU

            obj_classes = gt_boxes[argmax_overlaps, 4]
            overlap_mask = max_overlaps >= self._positive_overlap
            labels[overlap_mask] = obj_classes[overlap_mask]

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < self._negative_overlap] = 0
        else:
            labels.fill(0)

        # ignored label
        if len(gt_ignored_boxes):
            boxes = _square_boxes(gt_ignored_boxes, self._ratios[0], self._square_targets_ky) \
                if self._square_targets else gt_ignored_boxes
            ignored_overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(boxes, dtype=np.float))

            iargmax_overlaps = ignored_overlaps.argmax(axis=1)
            imax_overlaps = ignored_overlaps[np.arange(len(inds_inside)), iargmax_overlaps]
            labels[imax_overlaps > 0.3] = -1

        # subsample positive labels if we have too many
        # num_fg = len(fg_inds)
        num_fg = int(self._fg_fraction * self._batchsize)
        fg_inds = np.where(labels >= 1)[0]

        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        if cfg.TRAIN.RPN_LINEAR_TNF_K > 0:
            self._top_neg_fraction = min(self._top_neg_fraction + cfg.TRAIN.RPN_LINEAR_TNF_K, #0.00008
                                         self._max_tn_fraction)
        else:
            self._top_neg_fraction = self._max_tn_fraction

        if cfg.TRAIN.RPN_PERIODIC_TN > 0 and self._iters % cfg.TRAIN.RPN_PERIODIC_TN == 0:
            self._periodic_tn_enabled = not self._periodic_tn_enabled
            if self._periodic_tn_enabled:
                print('Switch on top negatives with fraction %.6f' % self._top_neg_fraction)
            else:
                print('Switch off top negatives')

        if not self._periodic_tn_enabled:
            self._top_neg_fraction = 0

        if self._iters % 250 == 0:
            print('Current TNF: %.6f' % self._top_neg_fraction)

        # subsample negative labels if we have too many
        # num_bg = max(num_fg, self._batchsize - num_fg)
        num_bg = self._batchsize - np.sum(labels >= 1)

        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            keep_first = int(np.floor(num_bg * self._top_neg_fraction))

            if keep_first > 0:
                # scores are (1, A, H, W) format
                # transpose to (1, H, W, A)
                # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)

                scores = np.zeros((self._num_anchors * bottom[5].shape[2] * bottom[5].shape[3], 1))

                for class_id in range(1, self._num_classes + 1):
                    indx_from = class_id * self._num_anchors
                    indx_to = indx_from + self._num_anchors
                    tmp = bottom[5].data[:, indx_from:indx_to, :, :]
                    tmp = tmp.transpose((0, 2, 3, 1)).reshape((-1, 1))
                    scores += tmp

                scores = scores[inds_inside]
                order = scores[bg_inds].ravel().argsort()[::-1]
                bg_inds = bg_inds[order]

            disable_inds = npr.choice(
                bg_inds[keep_first:], size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

        # print (np.round(t.diff, 4), np.round(t.average_time, 4),
        #        np.sum(labels == -1), np.sum(labels == 0), np.sum(labels == 1))
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if gt_boxes.shape[0]:
            bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels >= 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels >= 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels >= 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels >= 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels >= 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels >= 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print('means:')
            print(means)
            print('stdevs:')
            print(stds)

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print(self._name + ': max max_overlap', np.max(max_overlaps) if 'max_overlaps' in locals() else 0)
            print(self._name + ': num_positive', np.sum(labels >= 1))
            print(self._name + ': num_negative', np.sum(labels == 0))
            self._fg_sum += np.sum(labels >= 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print(self._name + ': num_positive avg', self._fg_sum / self._count)
            print(self._name + ': num_negative avg', self._bg_sum / self._count)

        # labels
        # print(np.unique(labels))
        # labels[labels >= 1] = 1
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

        t.tic()

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] >= 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def _square_boxes(boxes, ratio, ky=0.5):
    if boxes.shape[0] == 0:
        return boxes

    ret = boxes.copy()
    gt_sz = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]) / ratio)
    gt_cntr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    gt_cntr_y = ky * (boxes[:, 1] + boxes[:, 3])
    ret[:, 0] = gt_cntr_x - gt_sz * 0.5
    ret[:, 1] = gt_cntr_y - gt_sz * 0.5 * ratio
    ret[:, 2] = gt_cntr_x + gt_sz * 0.5
    ret[:, 3] = gt_cntr_y + gt_sz * 0.5 * ratio
    return ret
