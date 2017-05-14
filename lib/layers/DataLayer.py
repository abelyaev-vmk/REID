import sys, os
sys.path.insert(0, "/home/sasha/caffe1/caffe/python/")
import caffe
from scipy.misc import imread, imresize
import numpy as np
import random
from time import time
import cv2 as cv
from random import shuffle
from itertools import groupby


class TestDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # load path to images and labels
        print('Param str', self.param_str_)
        with open('logs/last_run/videoset.json', 'r') as f:
            self.content = f.readlines()
        # self.content = [x[:-1] for x in self.content]
        # self.content = map(lambda x: (x.split(' ')[0], int(x.split(' ')[1])), self.content)
        # self.mean = np.load('mean.npy').transpose((1, 2, 0))
        # self.seed = 1337
        # self.idx = 0
        # random.seed(self.seed)

    # initialize layer's shape
    def reshape(self,bottom,top):
        if len(bottom)>0:
            raise Exception('cannot have bottoms for input layer')
        if len(top) != 2:
            raise Exception('Need to define two tops: data and label')

        img, label = self.load_image_and_label(0)
        top[0].reshape(50, *img.shape)
        top[1].reshape(50, 1)

    def forward(self,bottom,top):
        for i in range(50):
            self.idx += 1
            self.idx = self.idx % len(self.content)
            img, label = self.load_image_and_label(self.idx)
            top[0].data[i, ...] = img
            top[1].data[i] = float(label)


    def backward(self, top, propagate_down, bottom):
        pass

    def load_image_and_label(self, idx):
        img = cv.imread(self.content[idx][0])
        img = np.array(img, dtype=np.float32)
        img = img[:-255, :, :]
        img = cv.resize(img, (256, 256))
        img -= self.mean
        img = cv.resize(img, (227, 227))
        img = img.transpose((2, 0, 1))
        return img, self.content[idx][1]
