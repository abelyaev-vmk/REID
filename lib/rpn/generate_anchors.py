# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6),
                     shift_num_xy=[(1,1)]):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    assert len(shift_num_xy) == len(scales) or \
           len(shift_num_xy) == 1

    total_anchors = []
    for i, (shift_num_x, shift_num_y) in enumerate(shift_num_xy):
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        anchors = base_anchor[np.newaxis, :]

        anchors = _shift_x_enum(anchors, shift_num_x)
        anchors = _shift_y_enum(anchors, shift_num_y)
        anchors = _ratio_enum(anchors, ratios)

        if len(shift_num_xy) == 1:
            anchors = _scale_enum(anchors, scales)
        else:
            anchors = _scale_enum(anchors, [scales[i]])

        total_anchors.append(anchors)

    total_anchors = np.vstack(total_anchors)

    return total_anchors


def _whctrs(anchors):
    """
    Return widths, heights, x centers, and y centers for anchors (windows).
    """

    w = anchors[:, 2] - anchors[:, 0] + 1
    h = anchors[:, 3] - anchors[:, 1] + 1
    x_ctr = anchors[:, 0] + 0.5 * (w - 1)
    y_ctr = anchors[:, 1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctrs, y_ctrs):
    """
    Given a vector of widths (ws) and heights (hs) around centers
    (x_ctrs, y_ctrs), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    x_ctrs = x_ctrs[:, np.newaxis]
    y_ctrs = y_ctrs[:, np.newaxis]
    anchors = np.hstack((x_ctrs - 0.5 * (ws - 1),
                         y_ctrs - 0.5 * (hs - 1),
                         x_ctrs + 0.5 * (ws - 1),
                         y_ctrs + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchors, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchors.
    """

    anchors_r = np.repeat(anchors, len(ratios), axis=0)
    ratios_t = np.tile(ratios, len(anchors))

    ws, hs, x_ctrs, y_ctrs = _whctrs(anchors_r)
    sizes = ws * hs
    sizes_ratios = sizes / ratios_t
    ws = np.round(np.sqrt(sizes_ratios))
    hs = np.round(ws * ratios_t)

    anchors = _mkanchors(ws, hs, x_ctrs, y_ctrs)
    return anchors


def _scale_enum(anchors, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    anchors_r = np.repeat(anchors, len(scales), axis=0)
    scales_t = np.tile(scales, len(anchors))

    ws, hs, x_ctrs, y_ctrs = _whctrs(anchors_r)
    ws = ws * scales_t
    hs = hs * scales_t

    anchors = _mkanchors(ws, hs, x_ctrs, y_ctrs)
    return anchors


def _shift_x_enum(anchors, num_x):
    assert num_x > 0
    k = (np.arange(1, num_x + 1) - 0.5) / num_x

    anchors_r = np.repeat(anchors, num_x, axis=0)
    k = np.tile(k, len(anchors))

    ws, hs, x_ctrs, y_ctrs = _whctrs(anchors_r)

    x_ctrs = anchors_r[:, 0] + k * (ws - 1)

    anchors = _mkanchors(ws, hs, x_ctrs, y_ctrs)
    return anchors


def _shift_y_enum(anchors, num_y):
    assert num_y > 0
    k = (np.arange(1, num_y + 1) - 0.5) / num_y

    anchors_r = np.repeat(anchors, num_y, axis=0)
    k = np.tile(k, len(anchors))

    ws, hs, x_ctrs, y_ctrs = _whctrs(anchors_r)

    y_ctrs = anchors_r[:, 1] + k * (hs - 1)

    anchors = _mkanchors(ws, hs, x_ctrs, y_ctrs)
    return anchors


if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
