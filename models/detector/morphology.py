#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : morphology
# @Date    : 2023/11/13 2:48
# @Author  : Jeerrzy


import copy
import cv2
import numpy as np
from skimage.morphology import skeletonize as skimage_skeletonize
from .unet import UnetDetector


IndividualLengthThresholdRange = [0.5, 1.5]
WormModelUpdateIouThreshold = 0.3


ECEWormModelInitTemplate = {
    'id': None,
    'body': None,
    'bbox': None,
    'length': None,
    'centroid': None,
    'endpoint': None,
    'crosspoint': None,
    'individual': None
}


class ECEWormModel(object):
    def __init__(self, _dict):
        for name, value in _dict.items():
            setattr(self, name, value)

    def update(self, _dict):
        if _dict['individual'] and compute_iou(_dict['bbox'], self.bbox, type='xywh') > WormModelUpdateIouThreshold:
            for name, value in _dict.items():
                if not name == 'id':
                    setattr(self, name, value)

    def fixModel(self):
        if self.individual:
            if len(self.endpoint) > 2:
                dist = list(map(lambda p: np.linalg.norm(p - np.array(self.centroid)), np.array(self.endpoint)))
                idx = np.argsort(dist)[-2:]
                self.endpoint = [self.endpoint[i] for i in idx]

    def obj2dict(self):
        return copy.deepcopy(self.__dict__)


class UnetMorphologyDetector(object):
    def __init__(self):
        self.unet = UnetDetector()

    def detect(self, ori_img):
        wormModelList = []
        binary_mask = self.unet.detect(ori_img)
        n, bodies, key_values, centroids, endpoints, crosspoints = morphologyPostProcessing(binary_mask)
        for i in range(n):
            init_template = ECEWormModelInitTemplate.copy()
            init_template['body'] = bodies[i]
            init_template['bbox'] = key_values[i][:4]
            init_template['length'] = key_values[i][-1]
            init_template['centroid'] = centroids[i]
            init_template['endpoint'] = endpoints[i]
            init_template['crosspoint'] = crosspoints[i]
            model = ECEWormModel(init_template)
            wormModelList.append(model)
        length_median = np.median([wormModel.length for wormModel in wormModelList])
        t1, t2 = IndividualLengthThresholdRange[0] * length_median, IndividualLengthThresholdRange[1] * length_median
        for wormModel in wormModelList:
            wormModel.individual = True if t1 <= wormModel.length <= t2 else False
            wormModel.fixModel()
        return wormModelList


def morphologyPostProcessing(binary_mask):
    # 提取骨架线
    skeleton_binary_mask = skeletonize(binary_mask)
    # 连通域分析
    num_labels, instance_mask, key_values, centroids = cv2.connectedComponentsWithStats(skeleton_binary_mask * 255)
    if not num_labels > 1:  # 默认跳过第一个
        return None
    bodies = []
    for mask_idx in range(1, num_labels):
        body = np.argwhere(instance_mask == mask_idx)
        body[:, [0, 1]] = body[:, [1, 0]]
        bodies.append(body)
    endpoints, crosspoints = [[] for i in range(num_labels)], [[] for i in range(num_labels)]
    # 计算邻接值以判断端点和交叉点
    kernelKPResponse = np.ones((3, 3), np.uint8)
    kernelKPResponse[1][1] = 0
    kp_response_mat = cv2.filter2D(skeleton_binary_mask, -1, kernelKPResponse)
    ep_response = (kp_response_mat == 1).astype(np.uint8)
    ep_response = cv2.bitwise_and(ep_response, skeleton_binary_mask)
    eps = np.argwhere(ep_response == 1)
    cp_response = (kp_response_mat >= 3).astype(np.uint8)
    cp_response = cv2.bitwise_and(cp_response, skeleton_binary_mask)
    cps = np.argwhere(cp_response == 1)
    if len(eps) > 0:
        for (x, y) in eps:
            mask_idx = instance_mask[x][y]
            endpoints[mask_idx].append([y, x])
    if len(cps) > 0:
        for (x, y) in cps:
            mask_idx = instance_mask[x][y]
            crosspoints[mask_idx].append([y, x])
    return num_labels-1, bodies, key_values[1:], centroids[1:], endpoints[1:], crosspoints[1:]


def skeletonize(binary_mask):
    return skimage_skeletonize(binary_mask).astype(np.uint8)


def compute_iou(bbox1, bbox2, type='xyxy'):
    """
    :param bbox1: [x1, y1, x2, y2]
    :param bbox2: [x1, y1, x2, y2]
    """
    if type == 'xywh':
        bbox1, bbox2 = xywh2xyxy(bbox1), xywh2xyxy(bbox2)
    xx1, yy1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    xx2, yy2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    intersec = (xx2 - xx1) * (yy2 - yy1)
    union = area1 + area2 - intersec
    iou = intersec / union
    return iou

def xywh2xyxy(bbox):
    (x, y, w, h) = bbox
    return [x, y, x+w, y+h]