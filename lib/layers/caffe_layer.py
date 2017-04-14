# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2017 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Andrey Belyaev
# --------------------------------------------------------


class CaffeLayer:
    def __init__(self, name='MyLayer', type='MyType', bottoms=('MyBottom1', 'MyBottom2'), tops=None, params=None):
        self.name, self.type, self.bottoms, self.tops, self.params = name, type, bottoms, tops, params
        if self.tops is None:
            self.tops = (self.name,)

    def to_string(self, level=0):
        def params_to_str(param, level):
            ans, offset = '', ' ' * level
            if type(param) != dict:
                for k, v in param:
                    ans += params_to_str({k: v}, level)
                return ans
            for p, v in param.items():
                if type(v) == str:
                    ans += offset + '%s: %s\n' % (p, v[1:] if v[0] == "@" else '\'%s\'' % v)
                elif type(v) == dict:
                    ans += offset + '%s {\n' % p + \
                           params_to_str(v, level + 2) + \
                           offset + '}\n'
                else:
                    ans += offset + '%s: %s\n' % (p, str(v) if type(v) != bool else 'true' if v else 'false')
            return ans

        offset = ' ' * level
        offset2, offset3 = offset * 2, offset * 3
        ans = offset + 'layer {\n' + \
              offset2 + 'name: \'%s\'\n' % self.name + \
              offset2 + 'type: \'%s\'\n' % self.type
        for top in self.tops:
            ans += offset2 + 'top: \'%s\'\n' % top
        for bottom in self.bottoms:
            ans += offset2 + 'bottom: \'%s\'\n' % bottom
        ans += params_to_str(self.params, level + 2) if self.params is not None else ''
        return ans + offset + "}\n"

    def append_to_solver(self, solver_prototxt):
        lines = []
        break_line = 0
        for i, line in enumerate(open(solver_prototxt, 'r')):
            if line[:5] == 'name:':
                break_line = None
            lines.append(line)
            if line[0] == '}' and break_line is not None:
                break_line = i
        with open(solver_prototxt, 'w') as f:
            f.writelines(lines[:break_line])
            f.write(self.to_string(level=2))
            if break_line is not None:
                f.writelines(lines[break_line:])

    @staticmethod
    def get_prev_param_str(txt):
        for line in open(txt, 'r').readlines():
            if 'param_str' in line:
                return line.split('param_str')[-1]

    @staticmethod
    def reid_append_train_layers(solver_prototxt):

        # ===== ROI
        proposal_param_str = "{\"anchor_shift_num_xy\": [[2, 2], [2, 2], [2, 2], [2, 2], [1, 1], [1, 1], [1, 1], " \
                             "[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], " \
                             "\"feat_stride\": 16, " \
                             "\"num_classes\": 1, " \
                             "\"anchor_scales\": [2.5, 3.025, 3.66025, 4.4289024999999995, 5.358972025, " \
                             "6.484356150249999, 7.846070941802498, 9.493745839581022, " \
                             "11.487432465893036, 13.899793283730574, 16.818749873313998, " \
                             "20.350687346709933, 24.62433168951902, 29.795441344318014], " \
                             "\"square_targets\": true, " \
                             "\"name\": \"rpn_big\", " \
                             "\"batchsize\": 256, " \
                             "\"anchor_ratios\": [1.0]}"
        proposal_layer = CaffeLayer(name='proposal', type='Python',
                                    bottoms=('rpn_big_cls_prob_reshape', 'rpn_big_bbox_pred', 'input_im_info'),
                                    tops=('rpn_rois',),
                                    params=[['python_param', {'module': 'rpn.proposal_layer',
                                                              'layer': 'ProposalLayer',
                                                              'param_str': proposal_param_str}]])
        roi_data_layer = CaffeLayer(name='roi-data', type='Python', bottoms=('rpn_rois', 'input_gt_boxes'),
                                    tops=('rois', 'labels', 'bbox_targets', 'bbox_inside_weights',
                                          'bbox_outside_weights', 'pid_label'),
                                    params=[['python_param', {'module': 'rpn.proposal_target_layer',
                                                              'layer': 'ProposalTargetLayer',
                                                              'param_str': "{\"num_classes\": 2, "
                                                                           "\"bg_aux_label\": 5532}"}]])
        roi_pooling_layer = CaffeLayer(name='roi-pool', type='ROIPooling', bottoms=('conv5_3', 'rois'),
                                       params=[['roi_pooling_param', {'pooled_h': 7,
                                                                      'pooled_w': 7,
                                                                      'spatial_scale': 0.0625}]])

        # ===== Fully Connected for bbox regression
        fc6_layer = CaffeLayer(name='fc6', type='InnerProduct', bottoms=('roi-pool',),
                               params=[['param', {'lr_mult': 1.0}],
                                       ['param', {'lr_mult': 2.0}],
                                       ['inner_product_param', {'num_output': 4096}]])
        relu6_layer = CaffeLayer(name='relu6', type='ReLU', bottoms=('fc6',), tops=('fc6',))
        drop6_layer = CaffeLayer(name='drop6', type='Dropout', bottoms=('fc6',), tops=('fc6',),
                                 params=[['dropout_param', {'dropout_ratio': 0.5}]])

        fc7_layer = CaffeLayer(name='fc7', type='InnerProduct', bottoms=('fc6',),
                               params=[['param', {'lr_mult': 1.0}],
                                       ['param', {'lr_mult': 2.0}],
                                       ['inner_product_param', {'num_output': 4096}]])
        relu7_layer = CaffeLayer(name='relu7', type='ReLU', bottoms=('fc7',), tops=('fc7',))
        drop7_layer = CaffeLayer(name='drop7', type='Dropout', bottoms=('fc7',), tops=('fc7',),
                                 params=[['dropout_param', {'dropout_ratio': 0.5}]])

        bbox_pred_layer = CaffeLayer(name='bbox_pred', type='InnerProduct', bottoms=('fc7',),
                                     params=[['param', {'lr_mult': 1.0}],
                                             ['param', {'lr_mult': 2.0}],
                                             ['inner_product_param', {'num_output': 8,
                                                                      'weight_filler': {'type': 'gaussian',
                                                                                        'std': 0.001},
                                                                      'bias_filler': {'type': 'constant',
                                                                                      'value': 0}}]
                                             ])
        loss_bbox_layer = CaffeLayer(name='loss_bbox', type='SmoothL1Loss',
                                     bottoms=('bbox_pred', 'bbox_targets',
                                              'bbox_inside_weights', 'bbox_outside_weights'),
                                     params=[['loss_weight', 1]])

        # ===== Fully Connected for pid
        feat_layer = CaffeLayer(name='feat', type='InnerProduct', bottoms=('fc7',),
                                params=[['param', {'lr_mult': 1, 'decay_mult': 1}],
                                        ['param', {'lr_mult': 2, 'decay_mult': 0}],
                                        ['inner_product_param', {'num_output': 256,
                                                                 'weight_filler': {'type': 'gaussian',
                                                                                   'std': 0.01},
                                                                 'bias_filler': {'type': 'constant',
                                                                                 'value': 0}}]])
        relu8_layer = CaffeLayer(name='relu8', type='ReLU', bottoms=('feat',), tops=('feat',))
        drop8_layer = CaffeLayer(name='drop8', type='Dropout', bottoms=('feat',), tops=('feat',),
                                 params=[['dropout_param', {'dropout_ratio': 0.5}]])

        pid_score_layer = CaffeLayer(name='pid_score', type='InnerProduct', bottoms=('feat',),
                                     params=[['param', {'lr_mult': 1, 'decay_mult': 1}],
                                             ['param', {'lr_mult': 2, 'decay_mult': 0}],
                                             ['inner_product_param', {'num_output': 5533,
                                                                      'weight_filler': {'type': 'gaussian',
                                                                                        'std': 0.001},
                                                                      'bias_filler': {'type': 'constant',
                                                                                      'value': 0}}]])
        pid_loss_layer = CaffeLayer(name='pid_loss', type='SoftmaxWithLoss',
                                    bottoms=('pid_score', 'pid_label'),
                                    params=[['propagate_down', 1],
                                            ['propagate_down', 0],
                                            ['loss_weight', 1],
                                            ['loss_param', {'ignore_label': -1,
                                                            'normalize': True}]])
        pid_accuracy_layer = CaffeLayer(name='pid_accuracy', type='Accuracy',
                                        bottoms=('pid_score', 'pid_label'),
                                        params=[['accuracy_param', {'ignore_label': -1}]])

        for layer in (proposal_layer, roi_data_layer, roi_pooling_layer,
                      fc6_layer, relu6_layer, drop6_layer,
                      fc7_layer, relu7_layer, drop7_layer,
                      bbox_pred_layer, loss_bbox_layer,
                      feat_layer, relu8_layer, drop8_layer,
                      pid_score_layer, pid_loss_layer, pid_accuracy_layer):
            layer.append_to_solver(solver_prototxt)

    @staticmethod
    def reid_append_test_layers(solver_prototxt):

        # ===== ROI
        roi_pooling_layer = CaffeLayer(name='roi-pool', type='ROIPooling', bottoms=('conv5_3', 'rpn_big_proposal_rois'),
                                       params=[['roi_pooling_param', {'pooled_h': 7,
                                                                      'pooled_w': 7,
                                                                      'spatial_scale': 0.0625}]])

        # ===== Fully Connected for bbox regression
        fc6_layer = CaffeLayer(name='fc6', type='InnerProduct', bottoms=('roi-pool',),
                               params=[['param', {'lr_mult': 1.0}],
                                       ['param', {'lr_mult': 2.0}],
                                       ['inner_product_param', {'num_output': 4096}]])
        relu6_layer = CaffeLayer(name='relu6', type='ReLU', bottoms=('fc6',), tops=('fc6',))
        drop6_layer = CaffeLayer(name='drop6', type='Dropout', bottoms=('fc6',), tops=('fc6',),
                                 params=[['dropout_param', {'dropout_ratio': 0.5}]])

        fc7_layer = CaffeLayer(name='fc7', type='InnerProduct', bottoms=('fc6',),
                               params=[['param', {'lr_mult': 1.0}],
                                       ['param', {'lr_mult': 2.0}],
                                       ['inner_product_param', {'num_output': 4096}]])
        relu7_layer = CaffeLayer(name='relu7', type='ReLU', bottoms=('fc7',), tops=('fc7',))
        drop7_layer = CaffeLayer(name='drop7', type='Dropout', bottoms=('fc7',), tops=('fc7',),
                                 params=[['dropout_param', {'dropout_ratio': 0.5}]])

        bbox_pred_layer = CaffeLayer(name='bbox_pred', type='InnerProduct', bottoms=('fc7',),
                                     params=[['param', {'lr_mult': 1.0}],
                                             ['param', {'lr_mult': 2.0}],
                                             ['inner_product_param', {'num_output': 8,
                                                                      'weight_filler': {'type': 'gaussian',
                                                                                        'std': 0.001},
                                                                      'bias_filler': {'type': 'constant',
                                                                                      'value': 0}}]])

        # ===== Fully Connected for pid
        feat_layer = CaffeLayer(name='feat', type='InnerProduct', bottoms=('fc7',),
                                params=[['param', {'lr_mult': 1, 'decay_mult': 1}],
                                        ['param', {'lr_mult': 2, 'decay_mult': 0}],
                                        ['inner_product_param', {'num_output': 256,
                                                                 'weight_filler': {'type': 'gaussian',
                                                                                   'std': 0.01},
                                                                 'bias_filler': {'type': 'constant',
                                                                                 'value': 0}}]])
        relu8_layer = CaffeLayer(name='relu8', type='ReLU', bottoms=('feat',), tops=('feat',))
        drop8_layer = CaffeLayer(name='drop8', type='Dropout', bottoms=('feat',), tops=('feat',),
                                 params=[['dropout_param', {'dropout_ratio': 0.5}]])

        pid_score_layer = CaffeLayer(name='pid_score', type='InnerProduct', bottoms=('feat',),
                                     params=[['param', {'lr_mult': 1, 'decay_mult': 1}],
                                             ['param', {'lr_mult': 2, 'decay_mult': 0}],
                                             ['inner_product_param', {'num_output': 5533,
                                                                      'weight_filler': {'type': 'gaussian',
                                                                                        'std': 0.001},
                                                                      'bias_filler': {'type': 'constant',
                                                                                      'value': 0}}]])
        pid_prob_layer = CaffeLayer(name='pid_prob', type='Softmax', bottoms=('pid_score',))
        for layer in (roi_pooling_layer,
                      fc6_layer, relu6_layer, drop6_layer,
                      fc7_layer, relu7_layer, drop7_layer,
                      bbox_pred_layer,
                      feat_layer, relu8_layer, drop8_layer,
                      pid_score_layer, pid_prob_layer):
            layer.append_to_solver(solver_prototxt)
