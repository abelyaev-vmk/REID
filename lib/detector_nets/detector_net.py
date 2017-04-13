# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from abc import abstractmethod
from abc import ABCMeta


class DetectorNet(metaclass=ABCMeta):
    @property
    @abstractmethod
    def caffe_model(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def losses_names(self):
        pass

    @abstractmethod
    def extract_detections(self, net):
        pass