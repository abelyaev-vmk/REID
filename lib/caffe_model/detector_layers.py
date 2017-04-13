# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe_model.layers import PythonLayer


class AnchorTargetLayer(PythonLayer):
    def __init__(self, name, anchors_params, layer_params, feat_stride=None):

        self._layer_params = {'feat_stride': feat_stride}
        self._layer_params.update(anchors_params)
        self._layer_params.update(layer_params)
        super(AnchorTargetLayer, self).__init__(name, 'rpn.anchor_target_layer.AnchorTargetLayer',
                                                self._layer_params, 6, 4)
        self._dynamic_params = ['stride']

    def set_dynamic_param(self, param, value):
        if param not in self._dynamic_params:
            raise ValueError()

        self._layer_params['feat_stride'] = value
        self.update_layer_params(self._layer_params)

    def slots_out_names(self):
        return ['labels', 'bbox_targets', 'inside_weights', 'outside_weights']


class ProposalLayer(PythonLayer):
    def __init__(self, name, anchors_params, min_size=16,
                 pre_nms_topN=12000, post_nms_topN=2000, nms_thresh=0.7,
                 num_classes=1,clip_proposals=True, feat_stride=None):

        self._layer_params = {'feat_stride': feat_stride, 'min_size': min_size,
                              'pre_nms_topN': pre_nms_topN,
                              'post_nms_topN': post_nms_topN,
                              'nms_thresh': nms_thresh,
                              'num_classes': num_classes,
                              'clip_proposals': clip_proposals}

        self._layer_params.update(anchors_params)
        super(ProposalLayer, self).__init__(name, 'rpn.proposal_layer.ProposalLayer',
                                            self._layer_params, 3, 2)
        self._dynamic_params = ['stride']

    def set_dynamic_param(self, param, value):
        if param not in self._dynamic_params:
            raise ValueError()

        self._layer_params['feat_stride'] = value
        self.update_layer_params(self._layer_params)

    def slots_out_names(self):
        return ['rois', 'rois_scores']


class SegmentationTargetLayer(PythonLayer):
    def __init__(self, name, layer_params, feat_stride=None):

        self._layer_params = {'feat_stride': feat_stride}
        self._layer_params.update(layer_params)
        super(SegmentationTargetLayer, self).__init__(name, 'layers.bbox_segmentation_layer.BboxSegmentationLayer',
                                                self._layer_params, 6, 6)
        self._dynamic_params = ['stride']

    def set_dynamic_param(self, param, value):
        if param not in self._dynamic_params:
            raise ValueError()

        self._layer_params['feat_stride'] = value
        self.update_layer_params(self._layer_params)

    def slots_out_names(self):
        return ['labels', 'slabels', 'sreg', 'sreg_inside', 'sreg_outside', 'bc_labels']


class CoverageClassificationLayer(PythonLayer):
        def __init__(self, name, layer_params, feat_stride=None):
            self._layer_params = {'feat_stride': feat_stride}
            self._layer_params.update(layer_params)
            super(CoverageClassificationLayer, self).__init__(name, 'layers.coverage_classification.CoverageClassificationLayer',
                                                              self._layer_params, 5, 1)
            self._dynamic_params = ['stride']

        def set_dynamic_param(self, param, value):
            if param not in self._dynamic_params:
                raise ValueError()

            self._layer_params['feat_stride'] = value
            self.update_layer_params(self._layer_params)

        def slots_out_names(self):
            return ['labels']


class SizesRegressionLayer(PythonLayer):
        def __init__(self, name, layer_params, feat_stride=None):
            self._layer_params = {'feat_stride': feat_stride}
            self._layer_params.update(layer_params)
            super(SizesRegressionLayer, self).__init__(name, 'layers.sizes_regression.SizesRegressionLayer',
                                                              self._layer_params, 2, 4)
            self._dynamic_params = ['stride']

        def set_dynamic_param(self, param, value):
            if param not in self._dynamic_params:
                raise ValueError()

            self._layer_params['feat_stride'] = value
            self.update_layer_params(self._layer_params)

        def slots_out_names(self):
            return ['labels', 'regression', 'weights_inside', 'weights_outside']


class BboxCountLayer(PythonLayer):
    def __init__(self, name, layer_params, feat_stride=None):
        self._layer_params = {'feat_stride': feat_stride}
        self._layer_params.update(layer_params)
        super(BboxCountLayer, self).__init__(name, 'layers.bbox_count.BboxCountLayer',
                                                   self._layer_params, 2, 1)
        self._dynamic_params = ['stride']

    def set_dynamic_param(self, param, value):
        if param not in self._dynamic_params:
            raise ValueError()

        self._layer_params['feat_stride'] = value
        self.update_layer_params(self._layer_params)

    def slots_out_names(self):
        return ['count_labels']