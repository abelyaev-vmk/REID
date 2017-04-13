# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

import os.path as osp
from datasets.image_sample import ImageSample
from datasets.loaders import load_bboxes_dataset_with_json_marking
from datasets.loaders import load_images_from_directory_without_marking


class ImagesCollection(object):
    """Коллекция изображений, поддерживает различные форматы:
        BBOX_JSON_MARKING - изображения с разметкой в json
                (см. loaders.load_bboxes_dataset_with_json_marking)
        IMAGES_DIR - директория с изображениями в форматах
                jpg, png, jpeg без разметки

    """

    def __init__(self, params: dict):
        """
        Args:
            params: Параметры коллекции, имеют формат:
                {
                    NAME: имя датасета (default None)
                    TYPE: тип датасета
                    PATH: путь к датасету
                    MARKING_NAME: (опционально) только для 'BBOX_JSON_MARKING'
                    SCALES: [500, 600]
                    MAX_SIZE: 1000
                }

        Returns:

        """
        assert params['TYPE'] in ['BBOX_JSON_MARKING', 'IMAGES_DIR', 'GML_FACES_MARKING']

        self._params = params
        self._samples = None
        self._max_size = params['MAX_SIZE']
        self._scales = params['SCALES']
        self._num_backgrounds = None
        self.imgs_path = None
        self._name = params.get('NAME', None)
        self.extract_clusters = params.get('EXTRACT_CLUSTERS', False)

        if params['TYPE'] in ['BBOX_JSON_MARKING', 'GML_FACES_MARKING']:
            json_format = {'BBOX_JSON_MARKING': 'default',
                           'GML_FACES_MARKING': 'gml_faces'}[params['TYPE']]
            self.imgs_path = osp.join(params['PATH'], 'imgs')
            if 'MARKING_NAME' in params:
                self._samples = \
                    load_bboxes_dataset_with_json_marking(
                        params['PATH'], params['MARKING_NAME'],
                        self._max_size, self._scales, json_format)
            else:
                self._samples = \
                    load_bboxes_dataset_with_json_marking(
                        params['PATH'], 'marking.json',
                        self._max_size, self._scales, json_format)

        elif params['TYPE'] == 'IMAGES_DIR':
            self.imgs_path = params['PATH']

            scan_recurse = params.get('RECURSE', False)
            self._samples = \
                load_images_from_directory_without_marking(
                    params['PATH'], self._max_size, self._scales, scan_recurse)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def scales(self) -> list:
        return self._scales

    def __len__(self):
        return len(self._samples)

    @property
    def num_backgrounds(self):
        """ Количество изображений, не содержащих объеков переднего плана """

        if self._num_backgrounds is not None:
            return self._num_backgrounds

        self._num_backgrounds = 0
        for sample in self._samples:
            obj_count = 0
            for object in sample.marking:
                if object['class'] > 0:
                    obj_count += 1
            if not obj_count:
                self._num_backgrounds += 1

        return self._num_backgrounds

    @property
    def num_classes(self):
        """ Количество используемых классов в датасете
         (для многоклассовой классификации > 2).

        Это значение вычисляется как максимальный по модулю
        номер класса + 1.
        """

        max_class = 0
        for sample in self._samples:
            for object in sample.marking:
                max_class = max(object['class'], max_class)

        return max_class + 1

    def __getitem__(self, key: int) -> ImageSample:
        return self._samples[key]
