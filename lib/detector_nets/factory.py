# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------
from detector_nets.rpn import RPN
from detector_nets.seg_net import SegNet


def create_network(config):
    if config.TYPE == "RPN":
        return RPN(config)
    elif config.TYPE == "SegNet":
        return SegNet(config)
    else:
        raise NotImplementedError(config.TYPE)