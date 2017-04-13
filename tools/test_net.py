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
from core.test import test_net
from core.config import cfg, cfg_from_file, cfg_from_list
from core.config import get_output_dir
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
