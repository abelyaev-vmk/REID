# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from abc import abstractmethod
from abc import ABCMeta
from copy import deepcopy
import cv2


class ImageSample(metaclass=ABCMeta):
    """Интерфейс класса для доступа к изображению"""

    @property
    @abstractmethod
    def bgr_data(self):
        pass

    @property
    @abstractmethod
    def marking(self):
        pass

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def max_size(self):
        pass

    @property
    @abstractmethod
    def scales(self):
        pass

    def __hash__(self):
        return hash(self.id)


class ImageFileSampleCV(ImageSample):
    """Изображение изначально не хранится в оперативной памяти,
       при необходимости каждый раз загружается с жёсткого диска
    """
    _prev_id=None
    _prev_bgr_data=None

    def __init__(self, image_path, marking, max_size, scales):
        self._image_path = image_path
        self._marking = marking
        self._max_size = max_size
        self._scales = scales

    @property
    def bgr_data(self):
        """Загрузка изображения с жёсткого диска, если предыдущее обращение
        было к этому же классу, то изображение не загружается повторно

        Returns:
            numpy array с dtype=np.uint8, содержащий пиксели изображения
        """
        if ImageFileSampleCV._prev_id == self.id:
            data = ImageFileSampleCV._prev_bgr_data
        else:
            data = cv2.imread(self._image_path)
            ImageFileSampleCV._prev_bgr_data = data
            ImageFileSampleCV._prev_id = self.id

        return data.copy()

    @property
    def marking(self):
        return self._marking

    @property
    def id(self):
        return self._image_path

    @property
    def max_size(self):
        return self._max_size

    @property
    def scales(self):
        return self._scales


class FlippedImageSample(ImageSample):
    """Отражает по горизонтали исходное изображение
    """
    def __init__(self, image_sample: ImageSample):
        self._sample = image_sample
        self._id = self._sample.id + '_flipped'
        self._marking = None
        self._max_size = image_sample.max_size
        self._scales = image_sample.scales

    def _load(self):
        self._width = self._sample.bgr_data.shape[1]
        m = deepcopy(self._sample.marking)
        for obj in m:
            obj['x'] = self._width - obj['x'] - obj['w']
        self._marking = m

    @property
    def bgr_data(self):
        return self._sample.bgr_data[:, ::-1, :]

    @property
    def marking(self):
        if self._marking is None:
            self._load()
        return self._marking

    @property
    def id(self):
        return self._id

    @property
    def max_size(self):
        return self._max_size

    @property
    def scales(self):
        return self._scales