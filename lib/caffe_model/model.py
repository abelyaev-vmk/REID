# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe.proto import caffe_pb2
from google.protobuf import text_format
from copy import deepcopy

from caffe_model.layers import CaffeProtoLayer
import datetime


class CaffeNetworkModel(object):
    def __init__(self, prototxt_path=None, name=None, input_slot_name='data'):
        self._named_slots = dict()
        self._unconnected_slots_in = []
        self._layers = []
        self._layers_phases = dict()

        if name is None:
            name = "GeneratedModel_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
        self._name = name

        if prototxt_path is not None:
            self._layers = parse_model_prototxt(prototxt_path)
            input_slots, output_slots = self.get_free_slots()

            if len(input_slots):
                assert len(input_slots) == 1
                self._unconnected_slots_in.append((input_slot_name, input_slots[0]))

    def add_layer(self, layer, parent_layer=None,
                  slots_list=None, named_slots_out=None, phase='any'):

        if parent_layer is not None:
            if type(parent_layer) is str:
                parent_layer = self.find_layer(parent_layer)
            elif parent_layer == -1:
                parent_layer = self._layers[-1]
            else:
                assert False

            assert parent_layer is not None

            for slot_in, slot_out in zip(layer.slots_in, parent_layer.slots_out):
                slot_in.connect(slot_out)
        elif slots_list is not None:
            for (layer_name, slot_id), slot_in in zip(slots_list, layer.slots_in):
                if layer_name is not None:
                    parent_layer = self.find_layer(layer_name)
                    assert parent_layer is not None
                    slot_in.connect(parent_layer.slots_out[slot_id])
                else:
                    self._unconnected_slots_in.append((slot_id, slot_in))

        if named_slots_out is not None:
            for slot_name, slot_id in named_slots_out:
                self._named_slots[slot_name] = layer.slots_out[slot_id]

        self._layers_phases[layer.name] = phase
        self._layers.append(layer)
        self._try_connect_slots()

    def merge(self, model, parent_layer):
        input_slots, _ = model.get_free_slots()
        assert len(input_slots) > 0

        parent_layer = self.find_layer(parent_layer)
        assert parent_layer is not None
        assert len(parent_layer.slots_out) == 1
        assert not set(self._layers_phases.keys()) & set(model._layers_phases.keys())

        for input_slot in input_slots:
            if len(input_slot.layer.slots_in) == 1:
                input_slot.connect(parent_layer.slots_out[0])

        self._named_slots.update(model._named_slots)
        self._layers_phases.update(model._layers_phases)
        self._unconnected_slots_in += model._unconnected_slots_in
        self._layers += model._layers

        self._try_connect_slots()

    def find_layer(self, layer_name):
        for layer in self._layers:
            if layer.name == layer_name:
                return layer
        return None

    def get_free_slots(self):
        input_slots = []
        output_slots = []
        for layer in self._layers:
            for slot in layer.slots_in:
                if not slot.is_connected():
                    input_slots.append(slot)
            for slot in layer.slots_out:
                if not slot.is_connected():
                    output_slots.append(slot)

        return input_slots, output_slots

    def get_topsorted_layers(self, phase):
        used = set()
        topological_order = []

        def dfs(layer):
            used.add(layer.name)
            layer_phase = self._layers_phases.get(layer.name, 'any')

            if phase != 'any' and layer_phase != phase and layer_phase != 'any':
                return

            for slot in layer.slots_out:
                for connected_slot in slot.connected:
                    next_layer = connected_slot.layer
                    if next_layer.name not in used:
                        dfs(next_layer)

            topological_order.append(layer)

        for layer in self._layers:
            if layer.name not in used:
                dfs(layer)

        return list(reversed(topological_order))

    def get_layers_strides(self, phase='any'):
        strides = dict()
        layer_receptive_field = dict()

        for layer in self.get_topsorted_layers(phase):
            stride = 1
            receptive_field = 1

            for slots_in in layer.slots_in:
                for slot in slots_in.connected:
                    stride = max(stride, strides.get(slot.layer.name, 1))
                    receptive_field = max(receptive_field,
                                          layer_receptive_field.get(slot.layer.name, 1))

            multipler = 1
            kernel_size = 1
            if layer.params.type == 'Convolution' or layer.params.type == 'Deconvolution':
                conv_strides = layer.params.convolution_param.stride
                conv_stride_h = layer.params.convolution_param.stride_h
                conv_stride_w = layer.params.convolution_param.stride_w

                if conv_strides:
                    multipler = conv_strides[0]
                if conv_stride_h or conv_stride_w:
                    assert conv_stride_h == conv_stride_w
                    multipler = conv_stride_h

                kernel_sizes = layer.params.convolution_param.kernel_size
                kernel_size_h = layer.params.convolution_param.kernel_h
                kernel_size_w = layer.params.convolution_param.kernel_w
                if kernel_sizes:
                    kernel_size = kernel_sizes[0]
                if kernel_size_h or kernel_size_w:
                    assert kernel_size_h == kernel_size_w
                    kernel_size = kernel_size_h

            elif layer.params.type == 'Pooling':
                multipler = layer.params.pooling_param.stride
                pool_stride_h = layer.params.pooling_param.stride_h
                pool_stride_w = layer.params.pooling_param.stride_w
                if pool_stride_h or pool_stride_w:
                    assert pool_stride_w == pool_stride_h
                    multipler = pool_stride_w

                kernel_size = layer.params.pooling_param.kernel_size
                kernel_size_h = layer.params.pooling_param.kernel_h
                kernel_size_w = layer.params.pooling_param.kernel_w
                if kernel_size_h or kernel_size_w:
                    assert kernel_size_h == kernel_size_w
                    kernel_size = kernel_size_h

            if layer.params.type == 'Deconvolution':
                assert stride % multipler == 0
                stride //= multipler
            else:
                receptive_field += 2 * ((kernel_size - 1) // 2) * stride
                stride *= multipler

            strides[layer.name] = stride
            layer_receptive_field[layer.name] = receptive_field
            # print(layer.name, receptive_field)

        return strides

    def get_net_params(self, phase='any'):
        self._assign_names_to_slots(phase)

        proto_layers = []

        ts_layers = self.get_topsorted_layers(phase)
        used_layers = {layer.name for layer in ts_layers}

        strides = self.get_layers_strides(phase)

        params = caffe_pb2.NetParameter()

        for slot_name, slot in self._named_slots.items():
            if slot.proto_name is None:
                for next_slot in slot.connected:
                    if next_slot.layer.name in used_layers:
                        break
                else:
                    continue
                slot.proto_name = slot_name
                params.input.extend([slot_name])
                params.input_shape.extend([caffe_pb2.BlobShape(dim=slot.dim)])

        for layer in ts_layers:

            for dynamic_param in layer.dynamic_params:
                if dynamic_param == 'stride':
                    layer.set_dynamic_param(dynamic_param, strides[layer.name])
                else:
                    raise NotImplementedError("%s" % dynamic_param)

            tops = [slot.proto_name for slot in layer.slots_out]
            bottoms = [slot.proto_name for slot in layer.slots_in]

            proto_layer = deepcopy(layer.params)
            del proto_layer.bottom[:]
            proto_layer.bottom.extend(bottoms)

            del proto_layer.top[:]
            proto_layer.top.extend(tops)

            proto_layers.append(proto_layer)

        params.layer.extend(proto_layers)
        params.name = self._name

        return params

    def _assign_names_to_slots(self, phase):
        for layer in self.get_topsorted_layers('any'):
            for slot in layer.slots_out:
                slot.proto_name = None

        ts_layers = self.get_topsorted_layers(phase)

        for layer in ts_layers:
            for i, slot_out in enumerate(layer.slots_out):
                if slot_out.name is not None:
                    slot_out.proto_name = slot_out.name
                else:
                    if layer.is_inplace_layer():
                        slot_out.proto_name = layer.slots_in[0].proto_name
                    else:
                        slot_suffix = layer.slots_out_names()[i]
                        slot_out.proto_name = layer.name
                        if slot_suffix:
                            slot_out.proto_name += '_' + slot_suffix

    def _try_connect_slots(self):
        unconnected_slots = []
        for slot_name, slot in self._unconnected_slots_in:
            if slot_name in self._named_slots:
                slot.connect(self._named_slots[slot_name])
            else:
                unconnected_slots.append((slot_name, slot))
        self._unconnected_slots_in = unconnected_slots


def parse_model_prototxt(prototxt_path):
    net_params = load_prototxt(prototxt_path)

    layers = []
    top2slot = dict()

    for layer_params in net_params.layer:
        layer = CaffeProtoLayer(layer_params)

        for bottom_name, slot in zip(layer_params.bottom, layer.slots_in):
            if bottom_name in top2slot:
                slot.connect(top2slot[bottom_name])

        for top_name, slot in zip(layer_params.top, layer.slots_out):
            top2slot[top_name] = slot

        layers.append(layer)

    return layers


def load_prototxt(prototxt_path):
    params = caffe_pb2.NetParameter()

    with open(prototxt_path, "r") as f:
        text_format.Merge(str(f.read()), params)

    return params
