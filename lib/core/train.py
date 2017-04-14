# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

import caffe
import caffe.draw
from core.config import cfg
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import pickle
import json

from datasets.collections import ImagesCollection
from core.detector_model import DetectorModel
from caffe_model.solver_config import SolverConfig
from layers.caffe_layer import CaffeLayer


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, images_dbs, output_dir,
                 pretrained_model=None, losses_names=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self._score_std = 0
        self._score_mean = 0
        self._so_force_iter = 0
        self._so_force_mode = False
        self._so_bad_samples = set()

        print('>>>>>>>' * 10000, solver_prototxt)

        CaffeLayer.reid_append_train_layers(solver_prototxt)

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print(('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model))
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self._data_layer = self.solver.net.layers[0]
        self._data_layer.set_images_dbs(images_dbs)
        self._data_layer.net = self.solver.net
        self._data_layer.set_losses(losses_names)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        iters_info = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            print('new loop')
            timer.tic()
            self.solver.net.layers[0].next_blob()
            print('blob ok')
            print(self.solver.net.layers[0])
            self.solver.step(1)
            print('step ok')
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

            iters_losses = self.solver.net.layers[0].get_losses()
            if not self._so_force_mode:
                iters_info += iters_losses

            if self.solver.iter % 100 == 0:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                save_path = os.path.join(self.output_dir, 'iters_info.pickle')
                with open(save_path, 'wb') as f:
                    pickle.dump(iters_info, f)

            if cfg.TRAIN.ENABLE_SMART_ORDER:
                tail_len = cfg.TRAIN.SO_TAIL_LEN
                scores = np.array([sum(x[1].values()) for x in iters_info])
                self._score_std = np.std(scores[-tail_len:])
                self._score_mean = np.mean(scores[-tail_len:])

                if not self._so_force_mode:
                    # if len(iters_losses) >= tail_len:
                    for sample, losses in iters_losses:
                        if sum(losses.values()) > self._score_mean + 1 * self._score_std:
                            self._so_bad_samples.add(sample)

                    self._data_layer.update_roidb_losses(iters_losses)
                    self._data_layer._score_mean = self._score_mean

                    if len(self._so_bad_samples) > cfg.TRAIN.SO_FORCE_BATCHSIZE:
                        self._so_force_mode = True
                        self._so_force_iter = 0
                        self._data_layer.enable_force_mode(self._so_bad_samples)

                    if self.solver.iter % 100 == 0:
                        print('bad sample cnt', len(self._so_bad_samples),
                              self._score_mean, self._score_std)
                else:
                    self._so_force_iter += len(iters_losses)

                    max_force_iter = len(self._so_bad_samples) * cfg.TRAIN.SO_FORCE_ROUNDS
                    if self._so_force_iter > max_force_iter:
                        self._so_force_mode = False
                        self._so_bad_samples = set()
                        self._data_layer.disable_force_mode()

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths


def train_net(output_dir):
    max_iters = cfg.TRAIN.SOLVER.TRAIN_ITERS
    pretrained_model = cfg.MODEL.WEIGHTS_PATH

    model = DetectorModel(cfg.MODEL)
    imgs_dbs = [ImagesCollection(x) for x in cfg.TRAIN.DATASETS]

    solver = SolverConfig(cfg.MODEL.NAME, cfg.TRAIN.SOLVER,
                          model.net_params('train'))

    """Train a network."""
    sw = SolverWrapper(solver.get_path(), imgs_dbs, output_dir,
                       pretrained_model=pretrained_model,
                       losses_names=model.get_loss_blobs_names())

    if cfg.DRAW_NET:
        caffe.draw.draw_net_to_file(model.net_params('train'),
                                    os.path.join(output_dir, 'net.png'),
                                    'LR')

    print('Solving...')
    model_paths = sw.train_model(max_iters)
    print('done solving')

    solver.close()

    return model_paths
