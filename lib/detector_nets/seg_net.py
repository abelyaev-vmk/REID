# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from detector_nets.detector_net import DetectorNet
from caffe_model.model import CaffeNetworkModel
from caffe_model.convolutions import ConvWithActivation
from caffe_model.convolutions import ConvolutionLayer
from caffe_model.detector_layers import CoverageClassificationLayer
from caffe_model.detector_layers import SizesRegressionLayer
from caffe_model.detector_layers import BboxCountLayer

from caffe_model.other_layers import ReshapeLayer
from caffe_model.other_layers import SoftmaxLayer
from caffe_model.other_layers import SoftmaxWithLossLayer
from caffe_model.other_layers import SmoothL1LossPyLayer

from sklearn.cluster import DBSCAN, KMeans
from core.nms_wrapper import nms

import numpy as np
import xgboost as xgb
import math


class SegNet(DetectorNet):
    def __init__(self, config):
        self._cfg = config

        self._losses_names = []

        self._scores_bottom = None
        self._rois_top = None
        self._stride = 16

        self._model = CaffeNetworkModel()
        self._init_model()

        trained_gbm_model_path = config.TRAINED_GBM_MODEL
        self._gbm = None
        self._gbm = xgb.Booster({'nthread':4})
        self._gbm.load_model(trained_gbm_model_path)

    def _init_model(self):
        def p(x):
            return self.name + '_' + x

        m = self._model # variable uses in exec() call
        exec(self._cfg.ARCHITECTURE)

        self._add_coverage_classification(p('output_cls'))
        self._add_size_regression(p('sreg_output_cls'), p('sreg_output'))
        self._add_bbox_count(p('bc_output_cls'))

    def _add_coverage_classification(self, parent_layer):
        m = self._model
        p = lambda x: self.name + "_" + x

        if 'COVERAGE_CLASSIFICATION' not in self._cfg:
            return

        config = self._cfg.COVERAGE_CLASSIFICATION
        coverage_classification_params = {
            'batchsize': config.BATCHSIZE,
            'fg_fraction': config.FG_FRACTION,
            'tn_fraction': config.TOP_NEGATIVE_FRACTION,
            'proj_boundaries': config.PROJ_BOUNDARIES,
            'name': self.name
        }

        m.add_layer(ConvolutionLayer(p('cls_score'), 2, 3, pad=0),
                    parent_layer=parent_layer)

        m.add_layer(ReshapeLayer(p('cls_score_reshape'),
                                 reshape_dim=(0, 2, -1, 0)),
                    parent_layer=p('cls_score'))

        m.add_layer(SoftmaxLayer(p('cls_prob')),
                    parent_layer=p('cls_score_reshape'))
        m.add_layer(ReshapeLayer(p('cls_prob_reshape'),
                                 reshape_dim=(0, 2, -1,0)),
                    parent_layer=p('cls_prob'))

        m.add_layer(CoverageClassificationLayer(p('coverage_classification'), coverage_classification_params),
                    slots_list=[(p('cls_score'), 0), (None, 'gt_boxes'),
                                (None, 'ignored_boxes'), (p('cls_prob_reshape'), 0),
                                (None, 'im_info')],
                    phase='train')

        m.add_layer(SoftmaxWithLossLayer(p('loss_cls'), loss_weight=config.LOSS_WEIGHT),
                    slots_list=[(p('cls_score_reshape'), 0), (p('coverage_classification'), 0)],
                    phase='train')

        self._losses_names.append(p('loss_cls'))
        self.coverage_classification_output = p('cls_prob')

    def _add_size_regression(self, parent_cls, parent_reg):
        m = self._model
        p = lambda x: self.name + "_" + x

        if 'SIZES_REGRESSION' not in self._cfg:
            return

        config = self._cfg.SIZES_REGRESSION
        sizes_regression_params = {
            'name': self.name,
            'batchsize': config.BATCHSIZE,
            'proj_boundaries': config.PROJ_BOUNDARIES,
            'num_scales': config.NUM_SCALES,
            'scale_base': config.SCALE_BASE,
            'scale_power': config.SCALE_POWER,
        }

        m.add_layer(ConvolutionLayer(p('sreg_cls_score'), config.NUM_SCALES, 3, pad=0),
                    parent_layer=parent_cls)

        m.add_layer(ReshapeLayer(p('sreg_cls_score_reshape'),
                                 reshape_dim=(0, config.NUM_SCALES, -1, 0)),
                    parent_layer=p('sreg_cls_score'))

        m.add_layer(SoftmaxLayer(p('sreg_cls_prob')),
                    parent_layer=p('sreg_cls_score_reshape'),
                    phase='test')
        m.add_layer(ReshapeLayer(p('sreg_cls_prob_reshape'),
                                 reshape_dim=(0, config.NUM_SCALES, -1, 0)),
                    parent_layer=p('sreg_cls_prob'),
                    phase='test')

        m.add_layer(SizesRegressionLayer(p('sizes_regression'), sizes_regression_params),
                    slots_list=[(p('sreg_cls_score'), 0), (None, 'gt_boxes')],
                    phase='train')

        m.add_layer(SoftmaxWithLossLayer(p('sreg_loss_cls'), loss_weight=config.CLS_LOSS_WEIGHT),
                    slots_list=[(p('sreg_cls_score_reshape'), 0), (p('sizes_regression'), 0)],
                    phase='train')
        self._losses_names.append(p('sreg_loss_cls'))

        m.add_layer(ConvolutionLayer(p('sreg_pred'), config.NUM_SCALES * 4, 3, pad=0),
                    parent_layer=parent_reg)
        m.add_layer(SmoothL1LossPyLayer(p('sreg_loss'), loss_weight=config.REG_LOSS_WEIGHT),
                    slots_list=[(p('sreg_pred'), 0), (p('sizes_regression'), 1),
                                (p('sizes_regression'), 2), (p('sizes_regression'), 3)],
                    phase='train')
        self._losses_names.append(p('sreg_loss'))

        self.sizes_reg = p('sreg_pred')
        self.sizes_cls = p('sreg_cls_prob')

    def _add_bbox_count(self, parent_layer):
        m = self._model
        p = lambda x: self.name + "_" + x

        if 'BBOX_COUNT' not in self._cfg:
            return

        config = self._cfg.BBOX_COUNT
        bbox_count_params = {
            'name': self.name,
            'batchsize': config.BATCHSIZE,
            'proj_boundaries': config.PROJ_BOUNDARIES,
            'max_count': config.MAX_COUNT,
        }
        loss_weight = config.LOSS_WEIGHT

        num_classes = config.MAX_COUNT - 1
        m.add_layer(ConvolutionLayer(p('bc_cls_score'), num_classes, 3, pad=0),
                    parent_layer=parent_layer)

        m.add_layer(ReshapeLayer(p('bc_cls_score_reshape'),
                                 reshape_dim=(0, num_classes, -1, 0)),
                    parent_layer=p('bc_cls_score'))

        m.add_layer(SoftmaxLayer(p('bc_cls_prob')),
                    parent_layer=p('bc_cls_score_reshape'),
                    phase='test')
        m.add_layer(ReshapeLayer(p('bc_cls_prob_reshape'),
                                 reshape_dim=(0, num_classes, -1, 0)),
                    parent_layer=p('bc_cls_prob'),
                    phase='test')

        m.add_layer(BboxCountLayer(p('bbox_count'), bbox_count_params),
                    slots_list=[(p('bc_cls_score'), 0), (None, 'gt_boxes')],
                    phase='train')

        m.add_layer(SoftmaxWithLossLayer(p('bc_loss_cls'), loss_weight=loss_weight),
                    slots_list=[(p('bc_cls_score_reshape'), 0), (p('bbox_count'), 0)],
                    phase='train')

        self._losses_names.append(p('bc_loss_cls'))
        self.bbox_count = p('bc_cls_prob')

    def extract_bbox_count_prob(self, net):
        bc = net.blobs[self.bbox_count].data.copy().transpose((0, 2, 3, 1))
        return np.squeeze(bc, axis=0)

    def extract_sizes_regression(self, net):
        config = self._cfg.SIZES_REGRESSION

        scores = net.blobs[self.sizes_cls].data.copy().transpose((0, 2, 3, 1))
        pred = net.blobs[self.sizes_reg].data.copy().transpose((0, 2, 3, 1))
        scores = np.squeeze(scores, axis=0)
        pred = np.squeeze(pred, axis=0)
        pred = pred.reshape((pred.shape[0] * pred.shape[1], config.NUM_SCALES, 4))

        labels = np.argmax(scores, axis=2).ravel()
        sreg_base = config.SCALE_BASE * (config.SCALE_POWER ** labels)

        width = np.zeros_like(labels)
        height = np.zeros_like(labels)
        dx = np.zeros_like(labels)
        dy = np.zeros_like(labels)

        for i in range(config.NUM_SCALES):
            mask = labels == i
            width[mask] = np.exp(pred[mask, i, 0]) * sreg_base[mask]
            height[mask] = np.exp(pred[mask, i, 1]) * sreg_base[mask]
            dx[mask] = pred[mask, i, 2] * sreg_base[mask]
            dy[mask] = pred[mask, i, 3] * sreg_base[mask]

        width = width.reshape((scores.shape[0], scores.shape[1]))
        height = height.reshape((scores.shape[0], scores.shape[1]))
        dx = dx.reshape((scores.shape[0], scores.shape[1]))
        dy = dy.reshape((scores.shape[0], scores.shape[1]))
        sreg = np.dstack((width, height, dx, dy))
        return sreg

    def extract_coverage_scores(self, net):
        scores = net.blobs[self.coverage_classification_output].data.copy()
        return scores

    def extract_detections(self, net):
        clusters, noise, all_regions = self.extract_clusters(net)

        def separate_cluster(cluster):
            sclusters = kmean_clusterize(cluster, 2)
            sfeatures = [get_cluster_features(cluster) for cluster in sclusters]
            dtest = xgb.DMatrix(np.array(sfeatures))
            sp = self._gbm.predict(dtest)

            result = []
            for x, cp in zip(sclusters, sp):
                if cp > 1:
                    result += separate_cluster(x)
                elif cp == 1:
                    result.append(x)
            return result

        def get_cluster_density(cluster):
            area = sum(r.w * r.h for r in cluster) / len(cluster)
            area /= 16 ** 2
            return len(cluster) / area

        bad_clusters = []
        if clusters:
            features = [get_cluster_features(cluster) for cluster in clusters]
            dtest = xgb.DMatrix(np.array(features))
            p = self._gbm.predict(dtest)

            good_clusters = []
            for cluster, cp in zip(clusters, list(p)):
                if len(cluster) > 30 and get_cluster_density(cluster) < 0.15:
                    continue
                if cp == 0:
                    bad_clusters.append(cluster)
                elif cp == 1:
                    good_clusters.append(cluster)
                else:
                    good_clusters += separate_cluster(cluster)

            clusters = good_clusters

        def get_rect_from_cluster(cluster):
            rect = np.zeros(4)
            prob_sum = 0
            for region in cluster:
                rect += region.p * np.array([region.cx, region.cy,
                                             region.w, region.h])
                prob_sum += region.p
            rect /= prob_sum
            x, y, w, h = rect.tolist()

            rect = [0, x - 0.5 * w, y - 0.5 * h,
                    x + 0.5 * w - 1, y + 0.5 * h - 1]
            return rect, prob_sum

        detections_bboxes = []
        detections_scores = []
        for cluster in clusters:
            if not cluster:
                break

            rect, prob = get_rect_from_cluster(cluster)
            detections_bboxes.append(rect)
            detections_scores.append(prob)

        for cluster in bad_clusters:
            rect, prob = get_rect_from_cluster(cluster)
            max_iou = get_max_iou(rect, detections_bboxes)
            if max_iou < 0.3:
                detections_bboxes.append(rect)
                detections_scores.append(prob)

        for r in noise:
            rect, prob = get_rect_from_cluster([r])
            max_iou = get_max_iou(rect, detections_bboxes)
            if max_iou < 0.3:
                detections_bboxes.append(rect)
                detections_scores.append(prob)

        if not detections_bboxes:
            rois = np.empty((0, 5))
            scores = np.empty((0, 2))
        else:
            rois = np.array(detections_bboxes)
            scores = np.array(detections_scores).reshape((-1, 1))
            scores = np.hstack((1 - scores, scores))

        return rois, scores

    def extract_clusters(self, net):
        scores = self.extract_coverage_scores(net)
        sreg = self.extract_sizes_regression(net)

        scores = scores[0, 1, :, :].tolist()
        score_thresh = self._cfg.TEST_SCORE_THRESH

        num_y = len(scores)
        num_x = len(scores[0])

        regions = []
        for i in range(num_y):
            y = (i + 0.5) * self._stride
            for j in range(num_x):
                x = (j + 0.5) * self._stride
                prob = scores[i][j]
                if prob >= score_thresh:
                    regions.append(Region(x, y, prob, sreg[i, j, :].tolist()))

        clusters, noise = dbscan_clusterize(regions, 10, 3)
        return clusters, noise, regions

    def extract_nms_detections(self, regions):
        if not regions:
            return [], []

        boxes = []
        for region in regions:
            boxes.append([region.cx - region.w * 0.5, region.cy - region.h * 0.5,
                          region.cx + region.w * 0.5, region.cy + region.h * 0.5, region.p])
        boxes = np.array(boxes, dtype=np.float32)

        keep = nms(boxes, 0.6)
        boxes = boxes[keep, :]
        boxes = boxes[np.argsort(boxes[:, 4])[::-1], :].tolist()

        detections = []
        scores = []
        for box in boxes:
            detections.append([0, box[0], box[1], box[2], box[3]])
            scores.append(box[4])

        return detections, scores

    @property
    def name(self):
        return self._cfg.NAME

    @property
    def losses_names(self):
        return self._losses_names

    @property
    def caffe_model(self):
        return self._model


class Region(object):
    def __init__(self, x, y, prob, sreg):
        self.x = x
        self.y = y
        self.cx = x + sreg[2]
        self.cy = y + sreg[3]
        self.w = sreg[0]
        self.h = sreg[1]
        self.p = prob


def dbscan_clusterize(regions, eps, min_samples):
    if len(regions) < min_samples:
        return [], regions

    samples = np.array([[r.cx, r.cy] for r in regions])

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clusters_ids = clustering.fit_predict(samples)
    clusters, noise = convert_clusters(regions, clusters_ids)

    return clusters, noise


def kmean_clusterize(regions, n_clusters):
    samples = np.array([[r.cx, r.cy] for r in regions])

    clustering = KMeans(n_clusters=n_clusters)
    clusters_ids = clustering.fit_predict(samples)
    clusters, _ = convert_clusters(regions, clusters_ids)
    return clusters


def convert_clusters(regions, clusters_ids):
    num_clusters = np.bincount(clusters_ids + 1).shape[0] - 1
    noise = []
    clusters = [list() for i in range(num_clusters)]
    for region, cluster_id in zip(regions, clusters_ids):
        if cluster_id == -1:
            noise.append(region)
        else:
            clusters[cluster_id].append(region)

    return clusters, noise


def separate_cluster_old(cluster, stride):
    area = sum(r.w * r.h for r in cluster) / len(cluster)
    if len(cluster) < 40:
        width_p = cluster_diameter(cluster) / (sum(r.w for r in cluster) / len(cluster))
    else:
        width_p = 0
    area /= stride ** 2
    density = len(cluster) / area

    if len(cluster) >= 50:
        density_thresh = 0.59
    elif len(cluster) >= 35:
        density_thresh = 0.65
    elif len(cluster) >= 25:
        density_thresh = 0.75
    else:
        density_thresh = 2

    if width_p > 0.8:
        density_thresh = 0
    elif width_p > 0.5:
        density_thresh *= 0.75

    if density > density_thresh:
        tmp = kmean_clusterize(cluster, 2)
        # print(density, len(cluster), len(tmp[0]), len(tmp[1]))
        ret = []
        for x in tmp:
            ret += separate_cluster_old(x, stride)
        return ret
    else:
        return [cluster]


def cluster_diameter(cluster):
    def find_most_far(p):
        best_dist = 0
        best_p = p
        for r in cluster:
            dist = (r.cx - p.cx) ** 2 + (r.cy - p.cy) ** 2
            if dist > best_dist:
                best_dist = dist
                best_p = r
        return best_p

    last = cluster[0]
    b = last
    for i in range(2):
        last = b
        b = find_most_far(last)
    return math.sqrt((b.cx - last.cx) ** 2 + (b.cy - last.cy) ** 2)


def intersect_1d(x1, x2, y1, y2):
    return max(0, min(x2, y2) - max(x1, y1) + 1)


def rect_intersection(r1, r2):
    ix = intersect_1d(r1[1], r1[1] + r1[3] - 1,
                      r2[1], r2[1] + r2[3] - 1)
    if ix == 0:
        return 0
    iy = intersect_1d(r1[2], r1[2] + r1[4] - 1,
                      r2[2], r2[2] + r2[4] - 1)
    return ix * iy


def rect_union(r1, r2):
    return r1[3] * r1[4] + r2[3] * r2[4] - rect_intersection(r1, r2)


def rect_area(rect):
    return rect[3] * rect[4]


def rect_iou(r1, r2):
    return rect_intersection(r1, r2) / rect_union(r1, r2)


def get_max_iou(r, rects):
    best_iou = 0
    for x in rects:
        iou = rect_iou(r, x)
        best_iou = max(best_iou, iou)
    return best_iou


class Cluster(object):
    def __init__(self, regions, scale, stride=16):
        self.stride = stride
        self.regions = regions
        self.scale = scale
        prob_sum = sum(r.p for r in regions)
        self.prob = prob_sum / len(regions)
        w = np.array([r.w for r in regions])
        h = np.array([r.h for r in regions])
        cx = np.array([r.cx for r in regions])
        cy = np.array([r.cy for r in regions])

        self.cx_std = np.std(cx)
        self.cy_std = np.std(cy)
        self.w_nstd = np.std(w / np.mean(w))
        self.h_nstd = np.std(h / np.mean(h))
        self.w = sum(r.w * r.p for r in regions) / prob_sum
        self.h = sum(r.h * r.p for r in regions) / prob_sum
        self.cx = sum(r.cx * r.p for r in regions) / prob_sum
        self.cy = sum(r.cy * r.p for r in regions) / prob_sum
        self.density = len(regions) / ((self.w * self.h) / (stride ** 2))
        self._diameter = None
        self._elongation = None

    @property
    def diameter(self):
        if self._diameter is not None:
            return self._diameter

        def find_most_far(p):
            best_dist = 0
            best_p = p

            for r in self.regions:
                dist = (r.cx - p.cx) ** 2 + (r.cy - p.cy) ** 2
                if dist > best_dist:
                    best_dist = dist
                    best_p = r
            return best_p

        last = self.regions[0]
        b = last
        for i in range(3):
            last = b
            b = find_most_far(last)
        self._diameter = math.sqrt((b.cx - last.cx) ** 2 + (b.cy - last.cy) ** 2)
        return self._diameter

    @property
    def elongation(self):
        if self._elongation is not None:
            return self._elongation

        points = np.array([[r.cx, r.cy] for r in self.regions])

        m20 = ((points[:, 0] - self.cx) ** 2).sum()
        m02 = ((points[:, 1] - self.cy) ** 2).sum()
        m11 = ((points[:, 1] - self.cy) * (points[:, 0] - self.cx)).sum()
        nominator = m20 + m02 + math.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2)
        denominator = m20 + m02 - math.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2)
        if denominator > 0:
            self._elongation = nominator / denominator
        else:
            self._elongation = nominator
        return self._elongation

    @property
    def rect(self):
        return {'x': (self.cx - 0.5 * self.w) / self.scale,
                'y': (self.cy - 0.5 * self.h) / self.scale,
                'w': self.w / self.scale,
                'h': self.h / self.scale}

    def __len__(self):
        return len(self.regions)


def get_cluster_features(cluster):
    cluster = Cluster(cluster, 1)
    return [len(cluster), cluster.w, cluster.h, cluster.w * cluster.h, cluster.w / cluster.h,
            cluster.prob, cluster.density, cluster.diameter / min(cluster.w, cluster.h),
            cluster.elongation, cluster.w_nstd, cluster.h_nstd, cluster.cx_std, cluster.cy_std]
