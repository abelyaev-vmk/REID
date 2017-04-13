# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
import random
from core.config import cfg
import numpy as np
import numpy.random as npr
import yaml
import cv2

from utils.blob import prep_im_for_blob, im_list_to_blob
from datasets.iterators import MultiRandomOrderIterator
from datasets.iterators import InfinityLoopIterator
from datasets.image_sample import FlippedImageSample

IMS_PER_BATCH = 1

class RoIDataLayer(caffe.Layer):
    """ data layer used for training."""

    def _get_next_normal_minibatch(self):
        minibatch = []
        while self._imdbs_iter and len(minibatch) < IMS_PER_BATCH:
            sample = next(self._imdbs_iter)

            if cfg.TRAIN.USE_FLIPPED and np.random.randint(0, 2):
                sample = FlippedImageSample(sample)

            if cfg.TRAIN.ENABLE_SMART_ORDER:
                loss = self._roidb_losses.get(sample.id, 1e6)

                if loss < 0.5 * self._score_mean and \
                        np.random.uniform() < cfg.TRAIN.SO_GOOD_SKIP_PROB:
                    # print('Skipped', sample.id, 'last loss', loss, 'mean loss', self._score_mean)
                    continue

            minibatch.append(sample)

        return minibatch

    def _get_next_force_minibatch(self):
        minibatch = []
        while len(minibatch) < cfg.TRAIN.IMS_PER_BATCH:
            if self._force_cur >= len(self._forced_samples):
                self._force_cur = 0
                random.shuffle(self._forced_samples)

            minibatch.append(self._forced_samples[self._force_cur])
            self._force_cur += 1

        return minibatch

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        if self._force_mode:
            minibatch_db = self._get_next_force_minibatch()
        else:
            minibatch_db = self._get_next_normal_minibatch()

        return get_minibatch(minibatch_db, self._num_classes)

    def enable_force_mode(self, forced_samples):
        self._forced_samples = forced_samples

        self._force_cur = 0
        self._force_mode = True

        print('Force mode was enabled with %d samples' % len(self._forced_samples))

    def disable_force_mode(self):
        self._force_mode = False

    def set_images_dbs(self, images_dbs):
        """Set the roidb to be used by this layer during training."""
        self._imdbs = images_dbs
        self._imdbs_iter = InfinityLoopIterator(
            MultiRandomOrderIterator(self._imdbs))

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # random.seed(cfg.RNG_SEED)
        # np.random.seed(cfg.RNG_SEED)

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}
        self._forward_images = []
        self._losses = []
        self._blobs = None
        self._roidb_losses = {}
        self._score_mean = 0
        self._force_mode = False
        self._forced_samples = []
        self._force_cur = 0
        self._imdbs = None
        self._imdbs_iter = None
        self._losses_blobs = None

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0

        max_scale = max(max(x.SCALES) for x in cfg.TRAIN.DATASETS)
        max_size =  max(x.MAX_SIZE for x in cfg.TRAIN.DATASETS)

        top[idx].reshape(IMS_PER_BATCH, 3, 1000, 1500)
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(1, 3)
        self._name_to_top_map['im_info'] = idx
        idx += 1

        top[idx].reshape(1, 5)
        self._name_to_top_map['gt_boxes'] = idx
        idx += 1

        top[idx].reshape(1, 5)
        self._name_to_top_map['ignored_boxes'] = idx
        idx += 1

        print('RoiDataLayer: name_to_top:', self._name_to_top_map)
        assert len(top) == len(self._name_to_top_map)

    def next_blob(self):
        pass

    def set_losses(self, losses_blobs):
        self._losses_blobs = losses_blobs

    def get_losses(self):
        iters_count = len(self._forward_images)

        losses = self._losses[-iters_count+1:] + [self.get_last_loss()]
        ret = list(zip(self._forward_images, losses))

        self._forward_images = []
        self._losses = []
        return ret

    def get_last_loss(self):
        ret = dict()
        for loss_blob in self._losses_blobs:
            ret[loss_blob] = float(self.net.blobs[loss_blob].data.copy())

        return ret

    def update_roidb_losses(self, loss_data):
        for sample, losses in loss_data:
            self._roidb_losses[sample.id] = losses[0]

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        self._blobs, samples = self._get_next_minibatch()
        self._forward_images += samples

        for blob_name, blob in self._blobs.items():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

        if not samples:
            return
        # for sample in samples:
        #     print(sample.id, len(sample.marking), sample.max_size)

        self._losses.append(self.get_last_loss())




    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def get_minibatch(samples, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    assert len(samples) <= 1, "Single batch only"

    if not samples:
        blobs = {
            'data': np.zeros((1, 3, 128, 128), dtype=np.float32),
            'gt_boxes': np.empty(shape=(0,5), dtype=np.float32),
            'ignored_boxes': np.empty(shape=(0,5), dtype=np.float32),
            'im_info': np.array([[128, 128, 1]], dtype=np.float32)
        }
        return blobs, samples

    sample = samples[0]
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(sample.scales))

    # Get the input image blob, formatted for caffe
    im_blob, gt_boxes, ignored_boxes, im_scale = \
        _convert_sample(sample, random_scale_inds)

    blobs = {
        'data': im_blob,
        'gt_boxes': gt_boxes,
        'ignored_boxes': ignored_boxes,
        'im_info': np.array([[im_blob.shape[2], im_blob.shape[3], im_scale]],
                dtype=np.float32)
    }

    return blobs, samples

def illumination_jittering(im_sample, value):
    illumination = np.random.uniform(-1, 1) * value * 128
    light_map = np.full(im_sample.shape, illumination)
    return cv2.add(im_sample, light_map, dtype=cv2.CV_8U)

def contrast_jittering(im_sample, value):
    im_sample = im_sample.copy()
    contrast = 1 + np.random.uniform(-1, 1) * value
    for ch in range(3):
        im_sample[:, :, ch] = cv2.multiply(im_sample[:, :, ch], contrast, dtype=cv2.CV_8U)
    return im_sample

def gauss_noise_jittering(im_sample, value):
    noise_map = np.zeros(im_sample.shape)
    cv2.randn(noise_map, 0, np.random.uniform(0, value))
    # noise_map = np.random.normal(0.0, np.random.uniform(0, value), im_sample.shape)
    return cv2.add(im_sample, noise_map, dtype=cv2.CV_8U)


def _convert_sample(sample, scale_indx):
    target_size = sample.scales[scale_indx]

    if cfg.TRAIN.SCALE_JITTERING > 0:
        r = (2 * cfg.TRAIN.SCALE_JITTERING * random.random() - cfg.TRAIN.SCALE_JITTERING) / 100.0
        target_size = int(target_size * (1 + r))

    ar_mult = 1.0
    if cfg.TRAIN.RATIO_JITTERING > 0:
        r = (2 * cfg.TRAIN.RATIO_JITTERING * random.random() - cfg.TRAIN.RATIO_JITTERING) / 100.0
        ar_mult += r

    sample_image = sample.bgr_data.copy()

    if random.random() < 0.2:
        ksize = 2 * np.random.randint(1, 3) + 1
        sample_image = cv2.GaussianBlur(sample_image, (ksize,ksize), 0)

    sample_image = illumination_jittering(sample_image, 0.3)
    sample_image = contrast_jittering(sample_image, 0.3)
    sample_image = gauss_noise_jittering(sample_image, 10)


    im, im_scale = prep_im_for_blob(sample_image, cfg.PIXEL_MEANS,
                                    target_size, sample.max_size, ar_mult)
    old_ratio = sample.bgr_data.shape[1] / sample.bgr_data.shape[0]
    new_ratio = im.shape[1] / im.shape[0]

    ar_mult = new_ratio / old_ratio

    gt_boxes = []
    ignored_boxes = []
    for x in sample.marking:
        if x['class'] < 1:
            continue
        box = [x['x'] * ar_mult, x['y'],
               (x['x'] + x['w'] - 1) * ar_mult, x['y'] + x['h'] - 1,
               x['class']]
        if x['ignore']:
            ignored_boxes.append(box)
        else:
            gt_boxes.append(box)

    if gt_boxes:
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
    else:
        gt_boxes = np.empty(shape=(0,5), dtype=np.float32)
    if ignored_boxes:
        ignored_boxes = np.array(ignored_boxes, dtype=np.float32)
    else:
        ignored_boxes = np.empty(shape=(0, 5), dtype=np.float32)

    gt_boxes[:, 0:4] *= im_scale
    ignored_boxes[:, 0:4] *= im_scale

    blob = im_list_to_blob([im])

    return blob, gt_boxes, ignored_boxes, im_scale