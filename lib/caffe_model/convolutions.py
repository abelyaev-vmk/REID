# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe.proto import caffe_pb2

from caffe_model.layers import BaseLayer
from caffe_model.fillers import GaussianFiller
from caffe_model.fillers import ConstantFiller
from caffe_model.activations import ReLU
from caffe_model.activations import ELU
from caffe_model.other_layers import DropoutLayer


class ConvolutionLayer(BaseLayer):
    def __init__(self, name, num_filters,
                 kernel_size, stride=1, pad=0,
                 weight_filler=GaussianFiller(std=0.01),
                 lr_mult=1):
        super(ConvolutionLayer, self).__init__(name, 'Convolution', 1, 1)
        self._inplace = False

        self._params.convolution_param.num_output = num_filters
        self._params.convolution_param.kernel_size.extend([kernel_size])
        self._params.convolution_param.pad.extend([pad])
        self._params.convolution_param.stride.extend([stride])

        self._params.convolution_param.weight_filler.MergeFrom(weight_filler.to_proto())
        self._params.convolution_param.bias_filler.MergeFrom(ConstantFiller().to_proto())

        weight_blob_param = caffe_pb2.ParamSpec(lr_mult=1 * lr_mult)
        bias_blob_param = caffe_pb2.ParamSpec(lr_mult=2 * lr_mult)

        self._params.param.extend([weight_blob_param, bias_blob_param])

    def slots_out_names(self):
        return ['']


class ConvWithActivation(BaseLayer):
    def __init__(self, name, num_filters,
                 kernel_size, stride=1, pad=1,
                 weight_filler=GaussianFiller(std=0.01),
                 lr_mult=1, activation='relu',
                 activation_params=None,
                 dropout=0.0):

        conv_layer = ConvolutionLayer(name, num_filters, kernel_size,
                                      stride, pad, weight_filler, lr_mult)

        if activation_params is None:
            activation_params = dict()

        if activation == 'relu':
            act_layer = ReLU(name + '_relu', **activation_params)
        elif activation == 'elu':
            act_layer = ELU(name + '_elu', **activation_params)
        else:
            raise ValueError("Unknown activation function '%s'" % activation)

        act_layer.connect_to(conv_layer)
        last_layer = act_layer

        if dropout > 0:
            dropout_layer = DropoutLayer(name + '_drop', dropout)
            dropout_layer.connect_to(act_layer)
            last_layer = dropout_layer

        super(ConvWithActivation, self).__init__(name, '', 1, 1)

        self._slots_in = conv_layer.slots_in
        self._slots_out = last_layer.slots_out
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        raise NotImplementedError()