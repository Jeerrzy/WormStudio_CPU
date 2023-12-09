#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : yolo.py
# @Date    : 2023/11/14 4:19
# @Author  : Jeerrzy


import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from .nets.yolo import YoloBody
from .utils.utils import (cvtColor, get_classes, preprocess_input, resize_image)
from .utils.utils_bbox import decode_outputs, non_max_suppression


class YOLODetector(object):
    _defaults = {
        "model_path": 'models/detector/yolo/model_data/yolox_nano_worm_2023_1114.pth',
        "classes_path": 'models/detector/yolo/model_data/worm_classes.txt',
        "input_shape": [640, 640],
        "phi": 'nano',
        "confidence": 0.5,
        "nms_iou": 0.3,
        "letterbox_image": True,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()

    def generate(self):
        self.net = YoloBody(self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect(self, ori_img):
        image_shape = np.array(np.shape(ori_img)[0:2])
        image = Image.fromarray(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            results = non_max_suppression(
                outputs,
                self.num_classes,
                self.input_shape,
                image_shape,
                self.letterbox_image,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou
            )
            if results[0] is None: 
                return None
            else:
                return results[0][:, :4]

