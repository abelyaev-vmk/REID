# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe.proto import caffe_pb2
from caffe_model.layers import BaseLayer


class MaxPooling(BaseLayer):
    def __init__(self, name, kernel_size, stride):
        super(MaxPooling, self).__init__(name, "Pooling", 1, 1)
        self._params.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
        self._params.pooling_param.kernel_size = kernel_size
        self._params.pooling_param.stride = kernel_size
        self._inplace = True


class AveragePooling(BaseLayer):
    def __init__(self, name, kernel_size, stride):
        super(AveragePooling, self).__init__(name, "Pooling", 1, 1)
        self._params.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
        self._params.pooling_param.kernel_size = kernel_size
        self._params.pooling_param.stride = kernel_size
        self._inplace = True