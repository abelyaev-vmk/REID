import argparse
import numpy as np
import os.path as osp
import json
import pickle
from lib.core.config import cfg


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='/home/mopkobka/CourseWork/Reid-dataset/dataset/dataset/Test_data/marking.json')
    parser.add_argument('--predict', type=str, default='logs/last_run/probe_features.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    gt = json.load(open(args.gt))
    pred = np.load(args.predict)
    print(len(pred[0]))
