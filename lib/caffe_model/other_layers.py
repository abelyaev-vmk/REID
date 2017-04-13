# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe.proto import caffe_pb2
from caffe_model.layers import BaseLayer
from caffe_model.layers import PythonLayer


class DropoutLayer(BaseLayer):
    def __init__(self, name, dropout_ratio):
        super(DropoutLayer, self).__init__(name, "Dropout", 1, 1)
        self._inplace = True

        self._params.dropout_param.dropout_ratio = dropout_ratio


class ReshapeLayer(BaseLayer):
    def __init__(self, name, reshape_dim):
        super(ReshapeLayer, self).__init__(name, "Reshape", 1, 1)
        self._inplace = False
        self._params.reshape_param.shape.dim.extend(reshape_dim)

    def slots_out_names(self):
        return ['']


class SoftmaxLayer(BaseLayer):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__(name, "Softmax", 1, 1)
        self._inplace = False

    def slots_out_names(self):
        return ['']


class SoftmaxWithLossLayer(BaseLayer):
    def __init__(self, name, loss_weight=1, ignore_label=-1):
        super(SoftmaxWithLossLayer, self).__init__(name, "SoftmaxWithLoss", 2, 1)
        self._inplace = False
        self._params.loss_param.normalization = caffe_pb2.LossParameter.VALID
        self._params.loss_param.ignore_label = ignore_label
        self._params.propagate_down.extend([True, False])
        self._params.loss_weight.extend([loss_weight])

    def slots_out_names(self):
        return ['']


class SmoothL1LossLayer(BaseLayer):
    def __init__(self, name, sigma=3, loss_weight=1):
        super(SmoothL1LossLayer, self).__init__(name, "SmoothL1Loss", 4, 1)
        self._inplace = False
        self._params.loss_weight.extend([loss_weight])
        self._params.smooth_l1_loss_param.sigma = sigma

    def slots_out_names(self):
        return ['']


class SmoothL1LossPyLayer(PythonLayer):
        def __init__(self, name, sigma=3, loss_weight=1):
            self._layer_params = {'sigma': sigma}

            super(SmoothL1LossPyLayer, self).__init__(name, 'layers.smooth_l1_loss.SmoothL1LossLayer',
                                                      self._layer_params, 4, 1)
            self._inplace = False
            self._params.loss_weight.extend([loss_weight])

        def slots_out_names(self):
            return ['']
