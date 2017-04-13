# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from abc import abstractmethod
from abc import ABCMeta
import random

from datasets.collections import ImagesCollection


class AbstractImagesIterator(metaclass=ABCMeta):
    """Интерфейс итератора коллекции изображений.
    Данный класс позволяет итерировать по загруженной коллекции изображений.
    """

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> tuple:
        """
        Получить следующий элемент из коллекции

        Returns:
            ImageSample
        """
        pass

    @abstractmethod
    def _init_iter(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class DirectIterator(AbstractImagesIterator):
    """Итератор коллекции в прямом порядке"""

    def __init__(self, images_collection: ImagesCollection):
        """
        Args:
            images_collection: Коллекция изображений
        Returns:
        """
        super().__init__()

        self._images_collection = images_collection

        self._init_iter()

    def _init_iter(self):
        self._indx = 0

    def __next__(self) -> tuple:
        if self._indx >= len(self._images_collection):
            raise StopIteration()
        else:
            sample = self._images_collection[self._indx]
            self._indx += 1

            return sample

    def __len__(self):
        return len(self._images_collection)


class RandomOrderIterator(AbstractImagesIterator):
    """Итератор коллекции в случайном порядке"""

    def __init__(self, images_collection: ImagesCollection):
        """
        Args:
            images_collection: Коллекция изображений
        Returns:
        """
        super().__init__()

        self._images_collection = images_collection

        self._order = list(range(len(images_collection)))
        self._init_iter()

    def _init_iter(self):
        self._indx = 0
        random.shuffle(self._order)

    def __next__(self) -> tuple:
        if self._indx >= len(self._images_collection):
            raise StopIteration()
        else:
            sample = self._images_collection[self._order[self._indx]]
            self._indx += 1

            return sample

    def __len__(self):
        return len(self._images_collection)


class MultiRandomOrderIterator(AbstractImagesIterator):
    """Итератор нескольких коллекций в случайном порядке как единого целого"""

    def __init__(self, images_collections: list):
        """

        Args:
            images_collections: список коллекций ImagesCollection

        Returns:

        """
        super().__init__()

        self._iters = [RandomOrderIterator(x) for x in images_collections]
        self._len = sum(len(x) for x in images_collections)
        self._order = []
        for i, x in enumerate(images_collections):
            self._order += [i] * len(x)

        self._init_iter()

    def _init_iter(self):
        for iter in self._iters:
            iter._init_iter()

        random.shuffle(self._order)
        self._indx = 0

    def __next__(self) -> tuple:
        if self._indx >= len(self._order):
            raise StopIteration()

        sample = next(self._iters[self._order[self._indx]])
        self._indx += 1
        return sample

    def __len__(self):
        return self._len


class MultiDirectIterator(AbstractImagesIterator):
    """Итератор нескольких коллекций в прямом порядке как единого целого"""

    def __init__(self, images_collections: list):
        """
        Args:
            images_collections: список коллекций ImagesCollection

        Returns:

        """
        super().__init__()

        self._iters = [DirectIterator(x) for x in images_collections]
        self._len = sum(len(x) for x in images_collections)
        self._order = []
        for i, x in enumerate(images_collections):
            self._order += [i] * len(x)

        self._init_iter()

    def _init_iter(self):
        for iter in self._iters:
            iter._init_iter()

        self._indx = 0

    def __next__(self) -> tuple:
        if self._indx >= len(self._order):
            raise StopIteration()
        sample = next(self._iters[self._order[self._indx]])
        self._indx += 1
        return sample

    def __len__(self):
        return self._len


class InfinityLoopIterator(AbstractImagesIterator):
    """ Обёртка над итератором, такая что при исчерапании базового итератора,
        он инициализируется заново.
    """

    def __init__(self, images_iterator):
        """

        Args:
            images_iterator: Итератор AbstractImagesIterator

        Returns:

        """
        super().__init__()

        self._iter = images_iterator

    def __next__(self) -> tuple:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter._init_iter()
            return next(self._iter)

    def _init_iter(self):
        self._iter._init_iter()

    def __len__(self):
        return len(self._iter)

