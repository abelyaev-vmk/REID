import os
import caffe
import numpy as np
import cv2 as cv
import json
from core.config import cfg


class TestDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.path_to_dataset = cfg.TEST.DATASETS[0].PATH
        self.rois = self.parse_json(json.load(open('logs/last_run/videoset.json', 'r')))
        self.idx = 0
        self.batch_size = 16

    # initialize layer's shape
    def reshape(self, bottom, top):
        if len(bottom) > 0:
            raise Exception('cannot have bottoms for input layer')
        if len(top) != 2:
            raise Exception('Need to define two tops: data and label')

        img, roi = self.load_image_and_rois(0)
        top[0].reshape(self.batch_size, *img.shape)
        top[1].reshape(self.batch_size, 4)

    def forward(self, bottom, top):
        for i in range(self.batch_size):
            img, roi = self.load_image_and_rois(self.idx)
            print(self.rois[self.idx][0])
            top[0].data[i, ...] = img
            top[1].data[i, ...] = [1, 1, 2, 2]
            self.idx += 1
            self.idx = self.idx % len(self.rois)

    def backward(self, top, propagate_down, bottom):
        pass

    def parse_json(self, js):
        #  return [[IMAGE_NAME, ROI], ..]
        rois = []
        for image, regions in js.items():
            for region in regions:
                if int(region['class']) == 1 and float(region['score']) > 0.7:
                    rois.append([image, region['x'], region['y'], region['w'], region['h']])
        return rois

    def load_image_and_rois(self, idx):
        img = cv.imread(os.path.join(self.path_to_dataset, self.rois[idx][0]))
        x, y, w, h = self.rois[idx][1:]
        W, H = img.shape[:2]
        x1, y1, x2, y2 = np.array([x / W, y / H, (x + w) / W, (y + h) / H], dtype=np.float32) * 244.
        img = np.array(img, dtype=np.float32)
        im_scale = 244
        img = cv.resize(img, (244, 244), interpolation=cv.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        print(x1, x2, y1, y2)
        return img, [x1, y1, x2, y2]
