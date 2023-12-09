#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : math.py
# @Time     : 2023/11/18 13:25
# @Project  : WormStudio_CPU


import math
import json
import numpy as np


def compute_json_twist_number(json_path):
    idx2twist_number = {}
    with open(json_path, 'r') as f:
        rawData = json.load(f)
        id_num = rawData['id_num']
        frames = rawData['frames']
    for idx in range(1, id_num+1):
        angle_list = []
        for frame_idx in frames.keys():
            frame_data = frames[frame_idx]
            if str(idx) in frame_data.keys():
                idx_data = frame_data[str(idx)]
                if idx_data['centroid'] is not None and idx_data['endpoint'] is not None and len(idx_data['endpoint'])==2:
                    angle = get_angle(idx_data['endpoint'][0], idx_data['centroid'], idx_data['endpoint'][1])
                    angle_list.append(angle)
                else:
                    angle_list.append(None)
            else:
                angle_list.append(None)
        peaks = find_peaks(angle_list)
        idx2twist_number[idx] = len(peaks)
    return idx2twist_number


def get_angle(ep1, cp, ep2):
    ep1, cp, ep2 = np.array(ep1), np.array(cp), np.array(ep2)
    v1, v2 = ep1 - cp, ep2 - cp
    try:
        cos_value = (v1.dot(v2))/(np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_radian = math.acos(cos_value)
        angle_degree = math.degrees(angle_radian)
    except:
        angle_degree = 0
    return angle_degree


def find_peaks(angle_list):
    peaks = []
    noNoneAngleList = [[idx, angle] for idx, angle in enumerate(angle_list) if angle is not None]
    for idx in range(1, len(noNoneAngleList)-1):
        if noNoneAngleList[idx-1][1] < noNoneAngleList[idx][1] and noNoneAngleList[idx+1][1] < noNoneAngleList[idx][1]:
            peaks.append(noNoneAngleList[idx][0])
    return peaks

