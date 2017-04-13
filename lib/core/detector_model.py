# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------
import numpy as np
from caffe_model.model import CaffeNetworkModel
from caffe_model.data_layers import PythonDataLayer
from detector_nets.factory import create_network
from tempfile import mkstemp


class DetectorModel(object):
    def __init__(self, model_config):
        self._config = model_config

        self._model = CaffeNetworkModel(self._config.PRETRAINED_MODEL_CONFIG,
                                        input_slot_name='data')

        self._model.add_layer(PythonDataLayer('input', 2),
                              named_slots_out=[('data', 0), ('im_info', 1),
                                               ('gt_boxes', 2), ('ignored_boxes', 3)],
                              phase='train')

        self._dnets = []
        for dnet_config in self._config.ATTACHED_NETS:
            if dnet_config.get('DISABLED', False):
                continue
            dnet = create_network(dnet_config)
            self._model.merge(dnet.caffe_model, dnet_config.PARENT_LAYER)
            self._dnets.append(dnet)

    def net_params(self, phase):
        return self._model.get_net_params(phase=phase)

    def create_temp_test_prototxt(self):
        fd, path = mkstemp()

        with open(path, 'w') as f:
            f.write(str(self.net_params('test')))

        return fd, path

    def extract_detections(self, net):
        bboxes = []
        scores = []
        for dnet in self._dnets:
            tbboxes, tscores = dnet.extract_detections(net)
            bboxes.append(tbboxes)
            scores.append(tscores)

        return np.vstack(bboxes), np.vstack(scores)

    def get_loss_blobs_names(self):
        losses = []
        for dnet in self._dnets:
            losses += dnet.losses_names
        return losses