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

from utils.cython_bbox_maps import get_bbox_levels


DEBUG = False
t = Timer()


class BboxCountLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        layer_params = json.loads(self.param_str)
        self._feat_stride = layer_params['feat_stride']
        self._max_count = layer_params['max_count']
        self._proj_boundaries = layer_params['proj_boundaries']
        self._batchsize = layer_params['batchsize']
        self._name = layer_params['name']

        self._iters = 0

        t.tic()

        height, width = bottom[0].data.shape[-2:]

        # labels
        top[0].reshape(1, 1, height, width)

    def forward(self, bottom, top):
        self._iters += 1
        t.toc()

        assert bottom[0].shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].shape[-2:]

        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        gt_boxes_contiguous = np.ascontiguousarray(gt_boxes, dtype=np.float)

        overlap_matrix = get_bbox_levels(
            height, width, int(self._feat_stride),
            gt_boxes_contiguous,
            self._proj_boundaries[0], self._proj_boundaries[1], self._proj_boundaries[2]
        )
        overlap_matrix[overlap_matrix > self._max_count] = self._max_count

        overlap_matrix -= 1
        overlap_matrix = overlap_matrix.ravel()

        bc_samples_inds = np.where(overlap_matrix >= 0)[0]
        if len(bc_samples_inds) > self._batchsize:
            disable_inds = npr.choice(
                bc_samples_inds, size=(len(bc_samples_inds) - self._batchsize), replace=False)
            overlap_matrix[disable_inds] = -1

        overlap_matrix = overlap_matrix.reshape((1, height, width, 1)).transpose(0, 3, 1, 2)
        overlap_matrix = overlap_matrix.reshape((1, 1, height, width))
        top[0].reshape(*overlap_matrix.shape)
        top[0].data[...] = overlap_matrix

        t.tic()

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
