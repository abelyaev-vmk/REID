# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from detector_nets.detector_net import DetectorNet
from caffe_model.model import CaffeNetworkModel
from caffe_model.convolutions import ConvWithActivation
from caffe_model.convolutions import ConvolutionLayer
from caffe_model.detector_layers import AnchorTargetLayer
from caffe_model.detector_layers import ProposalLayer
from caffe_model.other_layers import ReshapeLayer
from caffe_model.other_layers import SoftmaxLayer
from caffe_model.other_layers import SoftmaxWithLossLayer
from caffe_model.other_layers import SmoothL1LossPyLayer
import numpy as np


class RPN(DetectorNet):
    def __init__(self, config):
        self._cfg = config

        self._anchors_params = {
            'anchor_ratios': config.ANCHOR_RATIOS,
            'anchor_scales': config.ANCHOR_SCALES,
            'anchor_shift_num_xy': config.ANCHOR_SHIFT_NUM_XY
        }

        assert len(config.ANCHOR_SCALES) == len(config.ANCHOR_SHIFT_NUM_XY) or len(config.ANCHOR_SHIFT_NUM_XY) == 1
        if len(config.ANCHOR_SHIFT_NUM_XY) == 1:
            self._num_anchors = len(config.ANCHOR_SCALES)
        else:
            self._num_anchors = sum([x[0] * x[1] for x in config.ANCHOR_SHIFT_NUM_XY])
        self._num_anchors *= len(config.ANCHOR_RATIOS)
        self._num_classes = self._cfg.get('NUM_CLASSES', 1)

        self._losses_names = []

        self._scores_bottom = None
        self._rois_top = None

        self._model = CaffeNetworkModel()
        self._init_model()

    def _init_model(self):
        m = self._model

        def p(x):
            return self.name + '_' + x

        exec(self._cfg.ARCHITECTURE)
        # m.add_layer(ConvWithActivation(p('conv1'), 512, 3))
        # m.add_layer(ConvWithActivation(p('conv2'), 512, 3, dropout=0.3),
        #             parent_layer=-1)
        #
        # m.add_layer(ConvWithActivation(p('output_cls'), 384, 3),
        #             parent_layer=p('conv2'))
        #
        # m.add_layer(ConvWithActivation(p('output_box'), 384, 3),
        #             parent_layer=p('conv2'))


        m.add_layer(ConvolutionLayer(p('cls_score'), (self._num_classes + 1) * self._num_anchors, 3, pad=0),
                    parent_layer=p('output_cls'))

        m.add_layer(ReshapeLayer(p('cls_score_reshape'),
                                 reshape_dim=(0,self._num_classes + 1,-1,0)),
                    parent_layer=p('cls_score'))

        m.add_layer(SoftmaxLayer(p('cls_prob')),
                    parent_layer=p('cls_score_reshape'))
        m.add_layer(ReshapeLayer(p('cls_prob_reshape'),
                                 reshape_dim=(0,(self._num_classes + 1) * self._num_anchors,-1,0)),
                    parent_layer=p('cls_prob'))

        m.add_layer(ConvolutionLayer(p('bbox_pred'), 4 * self._num_anchors, 3, pad=0),
                    parent_layer=p('output_box'))

        self._scores_bottom = p('proposal_rois_scores')
        self._rois_top = p('proposal_rois')
        m.add_layer(ProposalLayer(p('proposal'), self._anchors_params,
                                  pre_nms_topN=self._cfg.PRE_NMS_TOP_N,
                                  post_nms_topN=self._cfg.POST_NMS_TOP_N,
                                  nms_thresh=self._cfg.NMS_THRESH,
                                  min_size=self._cfg.get('MIN_SIZE', 16),
                                  clip_proposals=self._cfg.get('CLIP_PROPOSALS', True),
                                  num_classes=self._num_classes),
                    slots_list=[(p('cls_prob_reshape'), 0),
                                (p('bbox_pred'), 0),
                                (None, 'im_info')],
                    phase='test')

        anchors_target_params = {'batchsize': self._cfg.BATCHSIZE,
                                 'square_targets': self._cfg.SQUARE_TARGETS,
                                 # 'square_targets_ky': self._cfg.SQUARE_TARGETS_KY,
                                 'positive_overlap': self._cfg.POSITIVE_OVERLAP,
                                 'negative_overlap': self._cfg.NEGATIVE_OVERLAP,
                                 'tn_fraction': self._cfg.TOP_NEGATIVE_FRACTION,
                                 'fg_fraction': self._cfg.FG_FRACTION,
                                 'num_classes': self._num_classes,
                                 'name': self.name}

        m.add_layer(AnchorTargetLayer(p('anchor_target'), self._anchors_params,
                                      anchors_target_params),
                    slots_list=[(p('cls_score'), 0), (None, 'gt_boxes'),
                                (None, 'ignored_boxes'), (None, 'im_info'),
                                (None, 'data'), (p('cls_prob_reshape'), 0)],
                     phase='train')

        loss_weight = self._cfg.get('LOSS_WEIGHT', 1)
        m.add_layer(SoftmaxWithLossLayer(p('loss_cls'), loss_weight=loss_weight),
                    slots_list=[(p('cls_score_reshape'), 0), (p('anchor_target'), 0)],
                    phase='train')
        
        m.add_layer(SmoothL1LossPyLayer(p('loss_bbox'), loss_weight=loss_weight),
                    slots_list=[(p('bbox_pred'), 0), (p('anchor_target'), 1),
                                (p('anchor_target'), 2), (p('anchor_target'), 3)],
                    phase='train')

        self._losses_names = [p('loss_cls'), p('loss_bbox')]

    def extract_detections(self, net):
        scores = net.blobs[self._scores_bottom].data.copy().reshape(-1, self._num_classes + 1)
        # scores = np.hstack((1 - scores, scores))

        rois = net.blobs[self._rois_top].data.copy()

        return rois, scores

    @property
    def name(self):
        return self._cfg.NAME

    @property
    def losses_names(self):
        return self._losses_names

    @property
    def caffe_model(self):
        return self._model