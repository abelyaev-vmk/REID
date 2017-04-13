# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t


# kw, kh, ky, 0.4, 0.75, 0.4
def get_bbox_coverage(unsigned int H, unsigned int W,
                      unsigned int stride, np.ndarray[DTYPE_t, ndim=2] boxes,
                      double kw, double kh, double ky):
    cdef unsigned int N = boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] coverage = np.zeros((H, W), dtype=DTYPE)
    cdef unsigned int x, y, i
    cdef int x1, y1, x2, y2

    for i in range(N):
        w = boxes[i, 2] - boxes[i, 0] + 1
        h = boxes[i, 3] - boxes[i, 1] + 1

        x1, y1, x2, y2 = get_boundaries(boxes[i, :], stride, W, H,
                                        1.0, 1.0, 0.5)
        for y in range(y1, y2):
            for x in range(x1, x2):
                if coverage[y, x] == 0:
                    coverage[y, x] = -1

        if h < 30 or w < 16:
            continue

        x1, y1, x2, y2 = get_boundaries(boxes[i, :], stride, W, H,
                                        kw, kh, ky)

        for y in range(y1, y2):
            for x in range(x1, x2):
                if coverage[y, x] <= 0:
                    coverage[y, x] = 1

    return coverage


# kw kh ky 0.6, 0.75, 0.4
def get_objects_size_regression_matrix(unsigned int H, unsigned int W,
              unsigned int stride, np.ndarray[DTYPE_t, ndim=2] boxes,
              double kw, double kh, double ky):
    cdef unsigned int N = boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] out_map = np.zeros((H, W, 4), dtype=DTYPE)
    cdef unsigned int x, y, i
    cdef int x1, y1, x2, y2
    cdef w, h

    for i in range(N):
        w = boxes[i, 2] - boxes[i, 0] + 1
        h = boxes[i, 3] - boxes[i, 1] + 1
        if h < 30 or w < 16:
            continue
        x1, y1, x2, y2 = get_boundaries(boxes[i, :], stride, W, H,
                                        kw, kh, ky)

        for y in range(y1, y2):
            for x in range(x1, x2):
                if h < out_map[y, x, 1] or out_map[y, x, 1] == 0:
                    out_map[y, x, 0] = w
                    out_map[y, x, 1] = h
                    out_map[y, x, 2] = boxes[i, 0] + 0.5 * w - (x + 0.5) * stride
                    out_map[y, x, 3] = boxes[i, 1] + 0.5 * h - (y + 0.5) * stride

    return out_map


# kw kh ky 0.9 0.9 0.5
def get_bbox_levels(unsigned int H, unsigned int W,
              unsigned int stride, np.ndarray[DTYPE_t, ndim=2] boxes,
              double kw, double kh, double ky):
    cdef unsigned int N = boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] out_map = np.zeros((H, W), dtype=DTYPE)
    cdef unsigned int x, y, i
    cdef int x1, y1, x2, y2

    for i in range(N):
        w = boxes[i, 2] - boxes[i, 0] + 1
        h = boxes[i, 3] - boxes[i, 1] + 1
        if h < 30 or w < 16:
            continue

        x1, y1, x2, y2 = get_boundaries(boxes[i, :], stride, W, H,
                                kw, kh, ky)
        for y in range(y1, y2):
            for x in range(x1, x2):
                out_map[y, x] += 1

    return out_map


def get_boundaries(np.ndarray[DTYPE_t, ndim=1] box,
                   unsigned int stride,
                   unsigned int size_x, unsigned int size_y,
                   double kw, double kh, double ky):
    cdef DTYPE_t w, h, sw, sh, tx, ty
    cdef int start_w, end_w, start_h, end_h

    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1
    sw = max(kw * w, 20)
    sh = max(kh * h, 20)

    tx = box[0] + 0.5 * w - 0.5 * sw
    ty = box[1] + max(ky * h - 0.5 * sh, 0)

    start_w = int(tx + stride * 0.5) // stride
    end_w = int(tx + sw - 1 - stride * 0.5) // stride

    start_h = int(ty + stride * 0.5) // stride
    end_h = int(ty + sh - 1 - stride * 0.5) // stride

    start_h = min(max(start_h, 0), size_y - 1)
    start_w = min(max(start_w, 0), size_x - 1)
    end_h = min(max(end_h + 1, 0), size_y)
    end_w = min(max(end_w + 1, 0), size_x)

    if start_h >= end_h:
        start_h = end_h - 1
    if start_w >= end_w:
        start_w = end_w - 1

    return start_w, start_h, end_w, end_h