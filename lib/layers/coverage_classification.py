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

from utils.cython_bbox_maps import get_bbox_coverage, get_bbox_levels

DEBUG = False
t = Timer()


class CoverageClassificationLayer(caffe.Layer):

    def setup(self, bottom, top):

        layer_params = json.loads(self.param_str)
        self._feat_stride = layer_params['feat_stride']
        self._batchsize = layer_params['batchsize']
        self._fg_fraction = layer_params['fg_fraction']
        self._proj_boundaries = layer_params['proj_boundaries']
        self._top_neg_fraction = layer_params['tn_fraction']

        self._name = layer_params['name']
        self._iters = 0

        t.tic()

        height, width = bottom[0].data.shape[-2:]
        # labels
        top[0].reshape(1, 1, height, width)

        if DEBUG:
            print('CoverageClassificationLayer: height', height, 'width', width)
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

    def forward(self, bottom, top):
        self._iters += 1

        t.toc()

        assert bottom[0].shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].shape[-2:]

        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        gt_ignored_boxes = bottom[2].data
        gt_boxes_contiguous = np.ascontiguousarray(gt_boxes, dtype=np.float)

        gt_coverage = get_bbox_coverage(height, width, int(self._feat_stride),
                                        gt_boxes_contiguous,
                                        self._proj_boundaries[0], self._proj_boundaries[1], self._proj_boundaries[2])
        ignored_coverage = get_bbox_levels(height, width, int(self._feat_stride),
                                        np.ascontiguousarray(gt_ignored_boxes, dtype=np.float),
                                        1.0, 1.0, 0.5)
        gt_coverage[ignored_coverage > 0] = -1

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.array(gt_coverage.ravel(), dtype=np.float32)

        fg_inds = np.where(labels >= 1)[0]
        if self._batchsize == -1:
            num_fg = len(fg_inds)
        else:
            num_fg = int(self._fg_fraction * self._batchsize)

        if DEBUG:
            self._fg_sum += len(fg_inds)
            print(self._name)
            print('num_fg: %4d; avg_num_fg: %4d' % (len(fg_inds), int(self._fg_sum / self._iters)))
            print('num ignored blocks:', np.sum(ignored_coverage > 0))

        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        if self._batchsize == -1:
            num_bg = num_fg
        else:
            num_bg = self._batchsize - np.sum(labels >= 1)

        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            keep_first = int(np.floor(num_bg * self._top_neg_fraction))

            if keep_first > 0:
                # scores are (1, A, H, W) format
                # transpose to (1, H, W, A)
                # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)

                scores = bottom[3].data[:, 1, :, :]
                scores = scores.reshape((-1, 1))

                order = scores[bg_inds].ravel().argsort()[::-1]
                bg_inds = bg_inds[order]

            disable_inds = npr.choice(
                bg_inds[keep_first:], size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        labels = labels.reshape((1, height, width, 1)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        t.tic()

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

