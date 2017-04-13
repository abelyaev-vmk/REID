# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------
from caffe.proto import caffe_pb2
from tempfile import mkstemp
import os


class SolverConfig(object):
    def __init__(self, net_name, cfg, net_param):
        self._cfg = cfg

        self._solver = caffe_pb2.SolverParameter()
        self._solver.net_param.MergeFrom(net_param)

        self._solver.iter_size = cfg.IMS_PER_BATCH

        self._solver.base_lr = cfg.BASE_LR
        self._solver.lr_policy = cfg.LR_POLICY.TYPE

        if cfg.LR_POLICY.TYPE == "step":
            self._solver.gamma = cfg.LR_POLICY.GAMMA
        elif cfg.LR_POLICY.TYPE == "multistep":
            self._solver.stepvalue.extend(cfg.LR_POLICY.STEPS)
            self._solver.gamma = cfg.LR_POLICY.GAMMA
        elif cfg.LR_POLICY == 'fixed':
            pass
        else:
            raise NotImplementedError(cfg.LR_POLICY.TYPE)

        self._solver.momentum = cfg.MOMENTUM
        self._solver.weight_decay = cfg.WEIGHT_DECAY
        self._solver.display = cfg.DISPLAY.PERIOD
        self._solver.average_loss = cfg.DISPLAY.AVERAGE_LOSS
        self._solver.snapshot = 0
        self._solver.snapshot_prefix = net_name

        self._solver_file = mkstemp()

        print('Created solver path:', self._solver_file[1])
        with open(self._solver_file[1], 'w') as f:
            f.write(str(self._solver))

    def get_path(self):
        return self._solver_file[1]

    def close(self):
        os.close(self._solver_file[0])
        os.remove(self._solver_file[1])
