# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------
import json
import os
from pathlib import Path
from datasets.image_sample import ImageFileSampleCV


def load_bboxes_dataset_with_json_marking(dataset_path: str,
                                          marking_filename: str,
                                          max_size: int, scales: list,
                                          json_format: str) -> list:
    """Данная функция загружает набор данных, которые представлены совокупностью
    изображений и размеченными объектами.

    Структура директории с разметкой:
        dataset_path/marking_filename: разметка всех изображений
        dataset_path/imgs: директория, содержащая файлы изображений

    Структура файла разметки 'default':
        {
            "image01.jpg": [
                {
                    x: координата левого верхнего угла по горизонтали,
                    y: координата левого верхнего угла по вертикали,
                    w: ширина объекта,
                    h: высота объекта,
                    class: номер класса объекта, может отсутствовать (default: 1),
                        0 - класс фона
                    ignore: игнорировать или нет объект (Boolean),
                        этот параметр может отсутствовать (default: False)
                }
            ], ...
        }
    "image01.jpg" - относительный путь к изображению по отношению к dataset_path/imgs.

    Структура файла разметки 'gml_faces':
        {
            "image01.jpg": {
                "faces": [
                    {
                        x: координата левого верхнего угла по горизонтали,
                        y: координата левого верхнего угла по вертикали,
                        w: ширина объекта,
                        h: высота объекта,
                        class: номер класса объекта, может отсутствовать (default: 1),
                            0 - класс фона
                        ignore: игнорировать или нет объект (Boolean),
                            этот параметр может отсутствовать (default: False)

                        * В данной разметке могут присутствовать дополнительные поля, но они
                        никак не будут использованы
                    }
                ], ...
            }
        }
    "image01.jpg" - относительный путь к изображению по отношению к dataset_path/imgs.

    Args:
        dataset_path (str): путь к директории с разметкой и изображениями
        marking_filename (str): имя файла разметки
        max_size (int): при прогоне сети по этому изображению, размер наибольшей стороны
                        не может превосходить этого значения
        scales (list):  на этом изображении алгоритм будет применён на нескольких масштабах,
                        которые соответствуют изображениям с длиной наименьшей стороны из scales
        json_format (str): формат файла с разметкой: "default", "gml_faces"

    Returns:
        list: Список объектов типа ImageFileSampleCV
    """
    if json_format not in {"default", "gml_faces"}:
        raise NotImplementedError(json_format)

    marking_path = os.path.join(dataset_path, marking_filename)

    with open(marking_path, 'r') as f:
        marking = json.load(f)

    samples = []
    for image_name, image_marking in sorted(marking.items()):
        image_path = os.path.join(dataset_path, 'imgs', image_name)

        if json_format == 'gml_faces':
            image_marking = image_marking['faces']

        for obj in image_marking:
            if 'ignore' not in obj:
                obj['ignore'] = False
            if 'class' not in obj:
                obj['class'] = 1
            # else:
            #     if obj['class'] == 'triangle':
            #         obj['class'] = 1
            #     else:
            #         obj['class'] = 2

        image_sample = ImageFileSampleCV(image_path, image_marking,
                                         max_size, scales)
        samples.append(image_sample)

    return samples


def load_images_from_directory_without_marking(
        images_path: str, max_size: int, scales: list,
        recurse: bool) -> list:
    """Загружает все изображения в форматах *.jpg,*.jpeg,*.png
    из указанной директории без разметки.

    Данная функция полезна для подготовки данных для тестирования на них детектора.

    Args:
        images_path: путь к папке, содержащей изображения
        max_size: максимальный размер стороны изображения для данного датасета
        scales: список масштабов
        recurse: искать изображения рекурсивно во всех поддиректориях
    Returns:
        list: Список объектов типа ImageFileSampleCV
    """

    images_dir = Path(images_path)
    images_files = []
    for format in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        if recurse:
            images_files += list(images_dir.rglob(format.lower()))
            images_files += list(images_dir.rglob(format.upper()))
        else:
            images_files += list(images_dir.glob(format.lower()))
            images_files += list(images_dir.glob(format.upper()))

    images_files = sorted(list(set(images_files)))

    return [ImageFileSampleCV(str(image_name), [], max_size, scales)
            for image_name in images_files]