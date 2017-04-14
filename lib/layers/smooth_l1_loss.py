# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

import caffe
import numpy as np
import json


class SmoothL1LossLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = json.loads(self.param_str_)

        self._sigma2 = layer_params.get('sigma', 1) ** 2
        self._has_weights = len(bottom) >= 3

        if self._has_weights:
            assert len(bottom) == 4, "If weights are used, must specify both " \
                                     "inside and outside weights"

    def forward(self, bottom, top):
        self.diff = bottom[0].data - bottom[1].data
        if self._has_weights:
            self.diff *= bottom[2].data

        mask = np.abs(self.diff) < 1 / self._sigma2
        self.errors[mask] = 0.5 * (self.diff[mask] ** 2) * self._sigma2
        mask = np.invert(mask)
        self.errors[mask] = np.abs(self.diff[mask]) - 0.5 / self._sigma2

        if self._has_weights:
            self.errors *= bottom[3].data

        top[0].data[...] = np.sum(self.errors) / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        mask = np.abs(self.diff) < 1 / self._sigma2
        self.diff[mask] = self.diff[mask] * self._sigma2
        mask = np.invert(mask)
        self.diff[mask] = np.sign(self.diff[mask])

        if self._has_weights:
            self.diff *= bottom[2].data * bottom[3].data

        for i in range(2):
            if not propagate_down[i]:
                continue
            sign = 1 if i == 0 else -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num

    def reshape(self, bottom, top):

        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")

        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.errors = np.zeros_like(bottom[0].data, dtype=np.float32)

        top[0].reshape(1)