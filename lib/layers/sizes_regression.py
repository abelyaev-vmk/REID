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

from utils.cython_bbox_maps import get_objects_size_regression_matrix

DEBUG = False
t = Timer()


class SizesRegressionLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        layer_params = json.loads(self.param_str)
        self._iters = 0

        self._feat_stride = layer_params['feat_stride']
        self._batchsize = layer_params['batchsize']
        self._proj_boundaries = layer_params['proj_boundaries']

        self._num_scales = layer_params['num_scales']
        self._scale_base = layer_params['scale_base']
        self._scale_power = layer_params['scale_power']

        self._name = layer_params['name']

        t.tic()

        height, width = bottom[0].data.shape[-2:]

        # labels
        top[0].reshape(1, 1, height, width)

        # regression values
        top[1].reshape(1, self._num_scales * 4, height, width)
        top[1].reshape(1, self._num_scales * 4, height, width)
        top[2].reshape(1, self._num_scales * 4, height, width)

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

        objects_size_matrix = get_objects_size_regression_matrix(
            height, width, int(self._feat_stride),
            gt_boxes_contiguous,
            self._proj_boundaries[0], self._proj_boundaries[1], self._proj_boundaries[2]
        )

        objects_size_matrix = objects_size_matrix.reshape((-1, 4))

        labels, sreg = width_height_transform(objects_size_matrix,
                                              num_scales=self._num_scales,
                                              scale_base=self._scale_base,
                                              scale_power=self._scale_power)

        fg_inds = np.where(labels != -1)[0]
        if len(fg_inds) > self._batchsize:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - self._batchsize), replace=False)
            labels[disable_inds] = -1
            sreg[disable_inds, :] = 7777

        if DEBUG:
            print(self._name)
            print('labels.bincount:', np.bincount(labels.astype(np.int) + 1))

        inside_weights = np.zeros((sreg.shape[0], sreg.shape[1]), dtype=np.float32)
        inside_weights[sreg < 7776] = 1.0

        wh_num_examples = np.sum(labels != -1)
        outside_weights = np.zeros((sreg.shape[0], sreg.shape[1]), dtype=np.float32)
        outside_weights[sreg < 7776] = 1.0 / wh_num_examples if wh_num_examples else 0
        sreg[sreg >= 7776] = 0

        labels = labels.reshape((1, height, width, 1)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # sreg
        sreg = sreg.reshape((1, height, width, sreg.shape[1])).transpose(0, 3, 1, 2)
        top[1].reshape(*sreg.shape)
        top[1].data[...] = sreg

        # sreg_inside_weights
        inside_weights = inside_weights \
            .reshape((1, height, width, inside_weights.shape[1])).transpose(0, 3, 1, 2)
        assert inside_weights.shape[2] == height
        assert inside_weights.shape[3] == width
        top[2].reshape(*inside_weights.shape)
        top[2].data[...] = inside_weights

        # sreg_outside_weights
        outside_weights = outside_weights \
            .reshape((1, height, width, outside_weights.shape[1])).transpose(0, 3, 1, 2)
        assert outside_weights.shape[2] == height
        assert outside_weights.shape[3] == width
        top[3].reshape(*outside_weights.shape)
        top[3].data[...] = outside_weights

        t.tic()

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
