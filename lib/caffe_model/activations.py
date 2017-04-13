# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe_model.layers import BaseLayer

class ReLU(BaseLayer):
    def __init__(self, layer_name=None, negative_slope=0.0):
        super(ReLU, self).__init__(layer_name, "ReLU", 1, 1)
        self._params.relu_param.negative_slope = negative_slope
        self._inplace = True


class ELU(BaseLayer):
    def __init__(self, layer_name=None, alpha=1.0):
        super(ELU, self).__init__(layer_name, "ELU", 1, 1)
        self._params.elu_param.alpha = alpha
        self._inplace = True