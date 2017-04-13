# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe_model.layers import PythonLayer


class PythonDataLayer(PythonLayer):
    def __init__(self, name, num_classes):

        layer_params = {'num_classes': num_classes}
        super(PythonDataLayer, self).__init__(name, 'layers.roi_data_layer.RoIDataLayer',
                                              layer_params, 0, 4)

        for slot, dim in zip(self.slots_out, [[1, 3, 224, 224], [1, 5], [1,5], [1,5]]):
            slot.dim = dim

    def slots_out_names(self):
        return ['data', 'im_info', 'gt_boxes', 'ignored_boxes']