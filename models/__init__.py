#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__
# @Date    : 2023/11/13 0:07
# @Author  : Jeerrzy


from .detector import *
from .tracker import *
from .optimizer import *
from .visualize import *
from .math import *


def yolo_mot_processing(src_video_path, result_txt_path):
    """
    :param src_video_path: 输入视频路径
    :param result_txt_path: txt格式的MOT结果路径
    """
    # 检测
    yolo_detect_video(src_video_path, result_txt_path)
    # 追踪
    yolo_track(result_txt_path, result_txt_path)
    # 优化
    yolo_optimize(result_txt_path, result_txt_path)


def yolo_unet_joint_mot_processing(src_video_path, cache_txt_path, result_json_path):
    """
    :param src_video_path: 输入视频路径
    :param cache_txt_path: txt格式的MOT缓存结果路径
    :param result_json_path: json格式的MOT缓存结果路径
    """
    unet_detect_video(src_video_path, result_json_path)
    yolo_mot_processing(src_video_path, cache_txt_path)
    joint_track(
        yolo_txt_path=cache_txt_path,
        unet_json_path=result_json_path,
        result_json_path=result_json_path
    )

