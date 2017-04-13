# --------------------------------------------------------
# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

import os
import caffe
import json
from core.config import cfg
import numpy as np
import numpy.random as npr
from core.bbox_transform import width_height_transform
from utils.timer import Timer

from utils.cython_bbox_maps import (
    get_bbox_coverage,
    get_objects_size_regression_matrix,
    get_bbox_levels
)

DEBUG = False
t = Timer()


class BboxSegmentationLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        layer_params = json.loads(self.param_str)
        self._feat_stride = layer_params['feat_stride']
        self._iters = 0

        self._batchsize = layer_params['batchsize']
        self._fg_fraction = layer_params['fg_fraction']
        self._name = layer_params['name']

        t.tic()

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print('BboxSegmentationLayer: height', height, 'width', width)
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # labels
        top[0].reshape(1, 1, height, width)
        top[1].reshape(1, 1, height, width)
        top[2].reshape(1, 8 * 4, height, width)
        top[3].reshape(1, 8 * 4, height, width)
        top[4].reshape(1, 8 * 4, height, width)
        top[5].reshape(1, 1, height, width)


    def forward(self, bottom, top):
        t.toc()
        self._iters += 1

        assert bottom[0].shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].shape[-2:]

        # im_info
        im_info = bottom[3].data[0, :]

        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        gt_ignored_boxes = bottom[2].data
        gt_boxes_contiguous = np.ascontiguousarray(gt_boxes, dtype=np.float)

        ####################################################################
        gt_coverage = get_bbox_coverage(height, width, int(self._feat_stride),
                                        gt_boxes_contiguous,
                                        0.4, 0.75, 0.4)
        ignored_coverage = get_bbox_levels(height, width, int(self._feat_stride),
                                        np.ascontiguousarray(gt_ignored_boxes, dtype=np.float),
                                        0.4, 0.75, 0.4)
        gt_coverage[ignored_coverage > 0] = -1
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.array(gt_coverage.ravel(), dtype=np.float32)

        num_fg = int(self._fg_fraction * self._batchsize)
        fg_inds = np.where(labels >= 1)[0]

        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        ####################################################################
        objects_size_matrix = get_objects_size_regression_matrix(
            height, width, int(self._feat_stride),
            gt_boxes_contiguous,
            0.6, 0.75, 0.4
        )
        objects_size_matrix = objects_size_matrix.reshape((-1, 4))
        slabels, sreg = width_height_transform(objects_size_matrix)

        fg_inds = np.where(slabels != -1)[0]
        if len(fg_inds) > self._batchsize:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - self._batchsize), replace=False)
            slabels[disable_inds] = -1
            sreg[disable_inds, :] = 7777

        inside_weights = np.zeros((sreg.shape[0], sreg.shape[1]), dtype=np.float32)
        inside_weights[sreg < 7776] = 1.0

        wh_num_examples = np.sum(slabels != -1)
        outside_weights = np.zeros((sreg.shape[0], sreg.shape[1]), dtype=np.float32)
        outside_weights[sreg < 7776] = 1.0 / wh_num_examples if wh_num_examples else 0
        sreg[sreg >= 7776] = 0
        ####################################################################

        overlap_matrix = get_bbox_levels(
            height, width, int(self._feat_stride),
            gt_boxes_contiguous,
            0.9, 0.9, 0.5
        )
        overlap_matrix -= 1
        overlap_matrix = overlap_matrix.ravel()

        bc_samples_inds = np.where(overlap_matrix >= 0)[0]
        if len(bc_samples_inds) > num_fg:
            disable_inds = npr.choice(
                bc_samples_inds, size=(len(bc_samples_inds) - num_fg), replace=False)
            overlap_matrix[disable_inds] = -1
        ####################################################################

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

                scores = bottom[5].data[:, 1, :, :]
                scores = scores.reshape((-1, 1))

                order = scores[bg_inds].ravel().argsort()[::-1]
                bg_inds = bg_inds[order]

            disable_inds = npr.choice(
                bg_inds[keep_first:], size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

        # labels
        labels = labels.reshape((1, height, width, 1)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        slabels = slabels.reshape((1, height, width, 1)).transpose(0, 3, 1, 2)
        slabels = slabels.reshape((1, 1, height, width))
        top[1].reshape(*slabels.shape)
        top[1].data[...] = slabels

        # sreg
        sreg = sreg.reshape((1, height, width, sreg.shape[1])).transpose(0, 3, 1, 2)
        top[2].reshape(*sreg.shape)
        top[2].data[...] = sreg

        # sreg_inside_weights
        inside_weights = inside_weights \
            .reshape((1, height, width, inside_weights.shape[1])).transpose(0, 3, 1, 2)
        assert inside_weights.shape[2] == height
        assert inside_weights.shape[3] == width
        top[3].reshape(*inside_weights.shape)
        top[3].data[...] = inside_weights

        # sreg_outside_weights
        outside_weights = outside_weights \
            .reshape((1, height, width, outside_weights.shape[1])).transpose(0, 3, 1, 2)
        assert outside_weights.shape[2] == height
        assert outside_weights.shape[3] == width
        top[4].reshape(*outside_weights.shape)
        top[4].data[...] = outside_weights

        overlap_matrix = overlap_matrix.reshape((1, height, width, 1)).transpose(0, 3, 1, 2)
        overlap_matrix = overlap_matrix.reshape((1, 1, height, width))
        top[5].reshape(*overlap_matrix.shape)
        top[5].data[...] = overlap_matrix

        t.tic()

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
