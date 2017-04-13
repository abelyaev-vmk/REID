# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from copy import deepcopy
import math
import cv2


def extract_objects_from_image(image, marking, params=None):
    """ Из изображения с json-разметкой (прямоугольники с параметрами x, y, w, h)
    извлекаются содержащиеся в нём объекты. Каждый выходной объект представлен
    фрагментом изображения и разметкой данного объекта, в координатах исходного
    изображения.

    Args:
        image: BGR-изображение в формате np.uint8
        marking: список словарей, описывающих содержащиеся на изображении объекты
        params: параметры вырезания исходных фрагментов в виде словаря:
            crop_beyond_borders(str) - способ обработки фрагментов, которые выходят
            за границы изображения. Допустимые значения параметра:
                'skip' - фрагменты, выходящие за пределы границ будут проигнорированы
                'constant_fill'(default) - открытые области будут продолжены от границы
                    константными значениями
                'reflect_fill' - зеркальное отражение открытых областей

            expansion_strategy(dict) - стратегия расширения области вокруг объектов,
                                 задаётся в виде словаря
                {'type':<тип стратегии>, <имя_дополнительного_параметра>:<его_значение>,...}
                1) None(default) - расширение не производится, объекты вырезаются, как есть
                2) 'type':'linear_side' - линейное увеличение длин сторон области с коэффициентом 'k',
                                     задаваемым доп. параметром
                3) 'type':'linear_area' - линейное увелечение размера фрагмента относительно его
                    площади с сохранением исходных пропорций
                4) 'type':'inscribed' - вписывание исходной области в прямоугольник, который задается
                    параметрами 'area_k' - его размер относительно площади исходной области,
                    'ratio' - отношение сторон w/h этого прямоугольника. В случае, если 'area_k' не будет
                    обеспечивать минимальный размер области, которая может содержать исходную фигуру, то
                    будет выбран минимальный по площаи прямоугольник, включающий исходную область

            min_neighbour_overlap(float) - минимальный порог overlap расширенной области с другими объектами,
                    начиная с которого, они их разметка будет включена в исходный объект с пометкой
                    ignored. (default: 0.2)
            min_true_neighbour_overlap(float) - минимальный порог overlap расширенной области с другими объектами,
                    начиная с которого, они их разметка будет включена в исходный объект на равне с целевым
                    объектом области. (default: 0.8)

    Returns:
        list: список пар (image, marking) - фрагментов изображений
    """

    # Default params
    p = {
        'crop_beyond_borders': 'constant_fill',
        'expansion_strategy': None,
        'min_neighbour_overlap': 0.2,
        'min_true_neighbour_overlap': 0.8
    }

    if params is not None:
        p.update(params)

    image_h, image_w = image.shape[0:2]

    min_x, min_y = 0, 0
    max_x, max_y = image_w - 1, image_h - 1

    samples = []
    for i, obj in enumerate(marking):
        if obj['ignore']:
            continue

        obj_box = [int(obj['x']), int(obj['y']), int(obj['w']), int(obj['h'])]
        obj_field = expand_field(obj_box, p['expansion_strategy'])
        field_marking = [deepcopy(obj)]

        for j, other_obj in enumerate(marking):
            if i == j:
                continue

            other_obj_box = [other_obj['x'], other_obj['y'],
                             other_obj['w'], other_obj['h']]

            other_obj_area = other_obj['w'] * other_obj['h']
            overlap = common_area(obj_field, other_obj_box) / other_obj_area

            if overlap < p['min_neighbour_overlap']:
                continue

            m = deepcopy(other_obj)
            m['ignore'] = other_obj['ignore'] or \
                overlap < p['min_true_neighbour_overlap']

            field_marking.append(m)

        local_marking = deepcopy(field_marking)
        for tobj in local_marking:
            tobj['x'] -= obj_field[0]
            tobj['y'] -= obj_field[1]

        min_x = min(min_x, obj_field[0])
        min_y = min(min_y, obj_field[1])
        max_x = max(max_x, obj_field[0] + obj_field[2] - 1)
        max_y = max(max_y, obj_field[1] + obj_field[3] - 1)

        if p['crop_beyond_borders'] == 'skip' and \
                (obj_field[0] < 0 or obj_field[1] < 0 or
                 obj_field[0] + obj_field[2] - 1 >= image_w or
                 obj_field[1] + obj_field[3] - 1 >= image_h):
            continue

        samples.append((obj_field, local_marking))

    left_border = -min_x
    top_border = -min_y
    right_border = max_x - image_w + 1
    bottom_border = max_y - image_h + 1

    if p['crop_beyond_borders'] == 'skip':
        bordered_image = image
    else:
        if p['crop_beyond_borders'] == 'constant_fill':
            border_type = cv2.BORDER_CONSTANT
        elif p['crop_beyond_borders'] == 'reflect_fill':
            border_type = cv2.BORDER_REFLECT
        else:
            raise NotImplementedError(p['crop_beyond_borders'])

        bordered_image = cv2.copyMakeBorder(image, top_border, bottom_border,
                                            left_border, right_border,
                                            border_type)

        for obj_field, _ in samples:
            obj_field[0] += left_border
            obj_field[1] += top_border

    output = []
    for obj_field, field_marking in samples:
        crop_image = bordered_image[obj_field[1]:obj_field[1]+obj_field[3],
                                    obj_field[0]:obj_field[0]+obj_field[2]]

        output.append((crop_image, field_marking))

    return output


def expand_field(obj_box, exp_strategy):
    """ Вспомогательная функция для расширения области объекта

    Args:
        obj_box(list): исходная область в формате [x, y, w, h]
        exp_strategy(dict): стратегия расширения (см. описание параметра expansion_strategy
            функции extract_objects_from_image)

    Returns:
        list: расширенная область в формате [x, y, w, h]
    """

    if exp_strategy is None:
        return obj_box
    elif exp_strategy['type'] == 'linear_side':
        return linear_side_expand(obj_box, exp_strategy['k'])
    elif exp_strategy['type'] == 'linear_area':
        return inscribed_expand(obj_box, exp_strategy['k'],
                                obj_box[2]/obj_box[3])
    elif exp_strategy['type'] == 'inscribed':
        return inscribed_expand(obj_box,
                                exp_strategy['area_k'],
                                exp_strategy['ratio'])
    else:
        raise NotImplementedError(exp_strategy['type'])


def linear_side_expand(obj_box, k):
    """ Реализация стратегии расширения 'linear_side'
        (см. описание параметров extract_objects_from_image)

    Args:
        obj_box(list): исходная область в формате [x, y, w, h]
        k(float): линейный коэффициент расширения сторон
    Returns:
        list: расширенная область в формате [x, y, w, h]
    """
    assert k >= 1

    nw = obj_box[2] * k
    nh = obj_box[3] * k

    delta_w = nw - obj_box[2]
    delta_h = nh - obj_box[3]

    return [int(obj_box[0] - 0.5 * delta_w),
            int(obj_box[1] - 0.5 * delta_h),
            int(round(nw)), int(round(nh))]


def inscribed_expand(box, area_k, ratio):
    """ Реализация стратегии расширения 'inscribed'
        (см. описание параметров extract_objects_from_image)

    Args:
        box(list): исходная область в формате [x, y, w, h]
        area_k(float): площадь новой области относительно площади старой
            (линейный коэффициент), должен быть >= 1.0
        ratio(float): отношение сторон нового прямоугольника
    Returns:
        list: расширенная область в формате [x, y, w, h]
    """
    assert area_k >= 1

    new_area = area_k * box[2] * box[3]
    cx = box[0] + 0.5 * box[2]
    cy = box[1] + 0.5 * box[3]

    nh = round(max(box[3], math.sqrt(new_area / ratio)))
    nw = round(max(box[2], nh * ratio))

    nx = int(cx - 0.5 * nw)
    ny = int(cy - 0.5 * nh)

    return [nx, ny, int(nw), int(nh)]


def get_boxes_iou(a, b):
    """Вычисление метрики IoU для двух прямоугольников

    IoU = IntersectionArea(a, b) / UnionArea(a, b)

    Args:
        a: прямоугольник в формате [x, y, w, h]
        b: прямоугольник в формате [x, y, w, h]
    Returns:
        float: значение IoU(a, b)
    """
    common = common_area(a, b)
    union = a[2] * a[3] + b[2] * b[3] - common
    return common / union


def common_area(a, b):
    """Вычисление общей площади двух прямоугольников

    Args:
        a: прямоугольник в формате [x, y, w, h]
        b: прямоугольник в формате [x, y, w, h]
    Returns:
        float: общая площадь прямоугольников a и b
    """
    x_overlap = segments_overlap((a[0], a[2]), (b[0], b[2]))
    y_overlap = segments_overlap((a[1], a[3]), (b[1], b[3]))
    common = 1. * x_overlap * y_overlap
    return common


def segments_overlap(l, r):
    """Длина общего пересечения двух одномерных отрезков

    Args:
        l: отрезок в формате [start_x, segment_length]
        r: отрезок в формате [start_x, segment_length]
    Returns:
        float: длина общего пересечения отрезкой l и r
    """

    if l[0] > r[0]:
        l, r = r, l
    far_r, far_l = map(sum, (r, l))
    if r[0] > far_l:
        return 0
    if far_l > far_r:
        return r[1]
    return far_l - r[0]


def resize_sample(sample, resize_type, target_size):
    """Масштабирование объекта, представленного фрагментом изображения

    Args:
        sample(tuple): пара (<фрагмент изображения в формате BGR np.uint8>,
                             <разметка объектов, содержащихся в этом фрагменте>)
                       Разметка объектов должна быть представлена в виде списка, первым элеметом
                       которого является ключевой объект фрагмента (расположенный в его центре)
        resize_type(str): Целевой параметр масштабирования, именно его значение будет приведено
            к требуемому размеру. Возможны следующие значения:
                1) 'width' - относительно ширины фрагмента
                2) 'height' - относительно высоты фрагмента
                3) 'object_width' - относительно ширины ключевого объекта внутри фрагмента
                4) 'object_height' - относительно высоты ключевого объекта внутри фрагмента
        target_size(int): новое значение целевого параметра масштабирования

    Returns:
        tuple: отмасштабированный сэмпл в формате аналогичном входному аргументу sample
    """

    if resize_type == 'width':
        scale_value = target_size / sample[0].size[1]
    elif resize_type == 'height':
        scale_value = target_size / sample[0].size[0]
    elif resize_type == 'object_width':
        scale_value = target_size / sample[1][0]['w']
    elif resize_type == 'object_height':
        scale_value = target_size / sample[1][0]['h']
    else:
        raise NotImplementedError(resize_type)

    nmarking = deepcopy(sample[1])
    scaled_image = cv2.resize(sample[0], (0, 0), fx=scale_value, fy=scale_value)

    for obj in nmarking:
        obj['x'] *= scale_value
        obj['y'] *= scale_value
        obj['w'] *= scale_value
        obj['h'] *= scale_value

    return (scaled_image, nmarking)