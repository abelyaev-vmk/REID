# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe.proto import caffe_pb2


VARIANCE_NORM = {'AVERAGE': caffe_pb2.FillerParameter.AVERAGE,
                 'FAN_IN': caffe_pb2.FillerParameter.FAN_IN,
                 'FAN_OUT': caffe_pb2.FillerParameter.FAN_OUT}


class WeightFiller(object):
    def __init__(self):
        self._filler = caffe_pb2.FillerParameter()

    def to_proto(self):
        return self._filler


class GaussianFiller(WeightFiller):
    def __init__(self, std, mean=0.0):
        super(GaussianFiller, self).__init__()

        self._filler.type = 'gaussian'
        self._filler.std = std
        self._filler.mean = mean


class XavierFiller(WeightFiller):
    def __init__(self, variance_norm='AVERAGE'):
        super(XavierFiller, self).__init__()

        self._filler.type = 'xavier'
        self._filler.variance_norm = VARIANCE_NORM[variance_norm]


class MSRAFiller(WeightFiller):
    def __init__(self, variance_norm='AVERAGE'):
        super(MSRAFiller, self).__init__()

        self._filler.type = 'msra'
        self._filler.variance_norm = VARIANCE_NORM[variance_norm]


class ConstantFiller(WeightFiller):
    def __init__(self):
        super(ConstantFiller, self).__init__()
        self._filler.type = 'constant'
