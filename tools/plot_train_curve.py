#!/usr/bin/env python3.4
# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

import argparse
import numpy as np
import re


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Plot train-loss curve')
    parser.add_argument('--log', dest='log_path',
                        help='path to train log file', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, required=True)

    args = parser.parse_args()
    return args


def parse_global_train_loss(log_path):
    iters = []
    losses = []

    with open(log_path, 'r') as f:
        for line in f.readlines():
            m = re.match(r".*Iteration ([-+]?\d+), loss = ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)", line)
            if m:
                iter, loss = m.group(1, 2)
                iters.append(int(iter))
                losses.append(float(loss))

    return np.array(iters[5:]), np.array(losses[5:])


def parse_bbox_train_loss(log_path):
    losses = []

    with open(log_path, 'r') as f:
        for line in f.readlines():
            m = re.match(r".*loss_bbox = ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)*", line)
            if m:
                losses.append(float(m.group(1)))

    return np.array(losses[5:])

def parse_pid_train_loss(log_path):
    losses = []

    with open(log_path, 'r') as f:
        for line in f.readlines():
            m = re.match(r".*pid_loss = ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)*", line)
            if m:
                losses.append(float(m.group(1)))

    return np.array(losses[5:])


def moving_average(a, n=3):
    try:
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret[:n - 1] /= np.arange(1, n)
        ret[n - 1:] /= n
        return ret
    except ValueError as e:
        return moving_average(a, n - 1) if n > 2 else 0


if __name__ == '__main__':
    args = parse_args()

    log_path = args.log_path

    iters, global_losses = parse_global_train_loss(log_path)
    bbox_losses = parse_bbox_train_loss(log_path)
    pid_losses = parse_pid_train_loss(log_path)

    if len(bbox_losses) != len(global_losses) != len(pid_losses):
        l = min(len(bbox_losses), len(global_losses), len(pid_losses))
        iters, global_losses, bbox_losses, pid_losses = \
            iters[:l], global_losses[:l], bbox_losses[:l], pid_losses[:l]

    from bokeh.plotting import figure
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    TOOLS = "pan,wheel_zoom,reset,save,box_select"

    p1 = figure(width=1600, height=800, title="Train loss from %s" % log_path,
                tools=TOOLS, webgl=True)
    p1.line(iters, global_losses, legend="Train loss", color='blue')
    p1.line(iters, bbox_losses, legend="BBox loss", color='green')
    p1.line(iters, pid_losses, legend="PID loss", color='yellow')
    p1.ygrid[0].ticker.desired_num_ticks = 30
    p1.xgrid[0].ticker.desired_num_ticks = 20

    p1.line(iters, moving_average(global_losses, 100),
            color='red', legend="MA100 Train loss")

    html = file_html(p1, CDN, title=log_path)

    with open(args.output, 'w') as f:
        f.write(html)
