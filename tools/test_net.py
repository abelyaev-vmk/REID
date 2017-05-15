#!/usr/bin/env python3

# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from core.test import test_net, test_probe
from core.test_probe import test_net_on_probe_set
from core.config import cfg, cfg_from_file, cfg_from_list
from core.config import get_output_dir
from scipy.io import loadmat
import os.path as osp
import caffe
import argparse
import pprint
import time, os, sys
import datetime


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')

    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--exp_dir', dest='exp_dir', default=None, type=str)
    parser.add_argument('--datasets', nargs='*', default=[], required=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def load_probe(rois_dir, images_dir):
    import json
    import numpy as np
    protoc = json.load(open(osp.join(rois_dir, 'videoset.json'), 'r'))
    images, rois = [], []
    for im_name in protoc.keys():
        for item in protoc[im_name]:
            box = np.array([item['x'], item['y'], item['w'], item['h']])
            box[2:] += box[:2]
            images.append(osp.join(images_dir, im_name))
            rois.append(box)
    return protoc, images, rois


def evaluate(log_dir):
    import pickle
    import json
    import numpy as np
    from sklearn.metrics import average_precision_score

    def _compute_iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union

    gallery_feat = json.load(open('gallery_features.pkl', 'r'))
    gallery_det = json.load(open('videoset.json', 'r'))
    probe_feat = np.load(open('probe_features.npy', 'rb'))
    gt_marking = json.load(open(osp.join(cfg.TEST.DATASETS[0].PATH[:-5], 'marking.json'), 'r'))
    name_to_det_feat = {}
    for name, feat in gallery_feat:
        inds = np.where(np.array(list(map(lambda gd: gd['score'], gallery_det[name]))) >= 0.5)[0]
        if len(inds) > 0:
            det = np.array(gallery_det[name])[inds]
            det = np.array(list(map(lambda d: [d['x'], d['y'], d['w'], d['h']], det)))
            feat = np.array(feat)[inds]
            name_to_det_feat[name] = (det, feat)
    num_images = sum([len(gt) for _, gt in gallery_det.items()])
    num_probe = len(probe_feat)

    aps = []
    top1_acc = []
    for i in range(num_probe):
        y_true, y_score = [], []
        feat_p = probe_feat[i][1][np.newaxis, :]
        imname = probe_feat[i][0].split('/')[-1]
        count_gt = 0
        count_tp = 0
        for item in gt_marking[imname]:
            gt = np.array([item['x'], item['y'], item['w'], item['h']])
            count_gt += (gt[2] * gt[3] > 0)
            if imname not in name_to_det_feat:
                continue
            det, feat_g = name_to_det_feat[imname]
            dis = np.sum((feat_p - feat_g) ** 2, axis=1)
            label = np.zeros(len(dis), dtype=np.int32)
            if gt[2] * gt[3] > 0:
                w, h = gt[2], gt[3]
                gt[2:] += gt[:2]
                thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                inds = np.argsort(dis)
                dis = dis[inds]
                # set the label of the first box matched to gt to 1
                for j, roi in enumerate(det[inds, :4]):
                    roi[2:] += roi[:2]
                    if _compute_iou(roi, gt) >= thresh:
                        label[j] = 1
                        count_tp += 1
                        break
            y_true.extend(list(label))
            y_score.extend(list(-dis))

        assert count_tp <= count_gt
        recall_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * recall_rate
        if not np.isnan(ap):
            aps.append(ap)
        else:
            aps.append(0)
        maxind = np.argmax(y_score)
        top1_acc.append(y_true[maxind])

    print('mAP: {:.2%}'.format(np.mean(aps)))
    print('top-1: {:.2%}'.format(np.mean(top1_acc)))


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if args.exp_dir is not None:
        cfg.EXP_DIR = args.exp_dir

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    output_dir_name = 'test'
    if args.datasets:
        output_dir_name += '_' + '_'.join(args.datasets)
    output_dir_name += '_' + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    output_dir = get_output_dir(output_dir_name, None)
    test_net(args.caffemodel, output_dir, args.datasets)

    rois_dir = 'logs/last_run'
    images_dir = cfg.TEST.DATASETS[0].PATH

    _, probe_images, probe_rois = load_probe(
        rois_dir, images_dir)

    net = caffe.Net('models/vgg16/test_query_norm.prototxt', args.caffemodel, caffe.TEST)
    test_net_on_probe_set(net, probe_images, probe_rois, 'feat', rois_dir)

    evaluate(rois_dir)
