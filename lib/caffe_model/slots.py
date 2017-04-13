# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

class LayerSlot(object):
    last_free_id = 0

    def __init__(self, layer, name=None, dim=None):
        self.layer = layer
        self.name = name
        self._proto_name = None
        self._dim = dim
        self._id = LayerSlot.last_free_id
        LayerSlot.last_free_id += 1
        self._connected = set()

    @property
    def proto_name(self):
        return self._proto_name

    @proto_name.setter
    def proto_name(self, value):
        self._proto_name = value

    @property
    def connected(self):
        return list(self._connected)

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim):
        self._dim = dim

    def connect(self, slot):
        if slot not in self._connected:
            self._connected.add(slot)
            slot.connect(self)

    def disconnect(self):
        for slot in self._connected:
            slot.remove(self)
        self._connected.clear()

    def is_connected(self):
        return len(self._connected)

    def remove(self, slot):
        self._connected.remove(slot)

    def __hash__(self):
        return hash(self._id)


class LayerSlotIn(LayerSlot):
    def __init__(self, layer, name=None):
        super(LayerSlotIn, self).__init__(layer, name)

    def connect(self, slot):
        super(LayerSlotIn, self).connect(slot)
        assert len(self._connected) == 1

    @property
    def proto_name(self):
        return self.connected[0].proto_name

    @proto_name.setter
    def proto_name(self, value):
        raise NotImplementedError()


class LayerSlotOut(LayerSlot):
    def __init__(self, layer, name=None, dim=None):
        super(LayerSlotOut, self).__init__(layer, name, dim)
