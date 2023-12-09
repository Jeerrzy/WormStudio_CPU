#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : joint.py
# @Time     : 2023/11/16 19:47
# @Project  : WormStudio_CPU


import sys
from models.detector import *
from .sort import *


sys.path.append('..')


JointTrackerIouThreshold = 0.3


class WormMorphologyJointTracker(object):
    def __init__(self):
        self.wormModelList = []
        self.frame_count = 0
        self.iou_threshold = JointTrackerIouThreshold

    def update(self, yolo_trk_frame_data, unet_dets_frame_data):
        if len(self.wormModelList) > 0:
            self.wormModelList.clear()
        # 初始化
        yolo_trks = []
        for _id, x, y, w, h in yolo_trk_frame_data:
            yolo_trks.append([y, x, y+h, x+w])
            init_template = ECEWormModelInitTemplate.copy()
            init_template['id'] = _id
            init_template['bbox'] = [y, x, h, w]
            model = ECEWormModel(init_template)
            self.wormModelList.append(model)
        unet_dets = []
        for wormObjDict in unet_dets_frame_data:
            (x, y, w, h) = wormObjDict['bbox']
            unet_dets.append([x, y, x+w, y+h])
        # 匹配
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            unet_dets,
            yolo_trks,
            iou_threshold=self.iou_threshold
        )
        for m in matched:
            self.wormModelList[m[1]].update(unet_dets_frame_data[m[0]])

    def save_info(self):
        frame_result = {}
        for model in self.wormModelList:
            frame_result[int(model.id)] = model.obj2dict()
        return frame_result

