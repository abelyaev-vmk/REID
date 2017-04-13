# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from abc import abstractmethod
from abc import ABCMeta

from caffe_model.slots import LayerSlotIn
from caffe_model.slots import LayerSlotOut
from caffe.proto import caffe_pb2
import json


class CaffeAbstractLayer(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def slots_in(self):
        pass

    @property
    @abstractmethod
    def slots_out(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def is_inplace_layer(self):
        pass

    @abstractmethod
    def slots_out_names(self):
        pass

    @property
    @abstractmethod
    def dynamic_params(self):
        pass

    @abstractmethod
    def set_dynamic_param(self, param, value):
        pass


class BaseLayer(CaffeAbstractLayer):
    def __init__(self, name, layer_type,
                 num_slots_in, num_slots_out):
        self._params = caffe_pb2.LayerParameter()
        self._params.name = name
        self._params.type = layer_type
        self._inplace = False
        self._dynamic_params = []

        self._slots_in = [LayerSlotIn(self) for i in range(num_slots_in)]
        self._slots_out = [LayerSlotOut(self) for i in range(num_slots_out)]

    def connect_to(self, layer):
        assert len(self.slots_in) == len(layer.slots_out)
        for input_slot, output_slot in zip(self.slots_in, layer.slots_out):
            input_slot.connect(output_slot)

    @property
    def params(self):
        return self._params

    @property
    def slots_in(self):
        return self._slots_in

    @property
    def slots_out(self):
        return self._slots_out

    @property
    def name(self):
        return self._params.name

    def is_inplace_layer(self):
        return self._inplace

    def slots_out_names(self):
        return ['top' + str(i + 1)
                for i in range(len(self.slots_out))]

    @property
    def dynamic_params(self):
        return self._dynamic_params

    def set_dynamic_param(self, param, value):
        if param not in self._dynamic_params:
            raise ValueError()


class CaffeProtoLayer(BaseLayer):
    def __init__(self, params):
        super(CaffeProtoLayer, self).__init__(name="",
                                              layer_type="",
                                              num_slots_in=len(params.bottom),
                                              num_slots_out=len(params.top))
        self._params = params

        self._slots_out = [LayerSlotOut(self, bottom_name)
                           for bottom_name in params.top]


class PythonLayer(BaseLayer):
    def __init__(self, name, python_class_name, layer_params, num_slots_in, num_slots_out):
        super(PythonLayer, self).__init__(name, 'Python',
                                          num_slots_in, num_slots_out)

        splitted = python_class_name.split('.')

        self._params.python_param.module = '.'.join(splitted[:-1])
        self._params.python_param.layer = splitted[-1]
        self.update_layer_params(layer_params)

    def update_layer_params(self, layer_params):
        self._params.python_param.param_str = json.dumps(layer_params)