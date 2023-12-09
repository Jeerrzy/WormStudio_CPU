#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : test2.py
# @Time     : 2023/11/16 0:59
# @Project  : WormStudio_CPU


import json
import math
import numpy as np
import matplotlib.pyplot as plt
from models import *


def get_angle(ep1, cp, ep2):
    ep1, cp, ep2 = np.array(ep1), np.array(cp), np.array(ep2)
    v1, v2 = ep1 - cp, ep2 - cp
    cos_value = (v1.dot(v2))/(np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_radian = math.acos(cos_value)
    angle_degree = math.degrees(angle_radian)
    return angle_degree


def find_peaks(angle_list):
    peaks = []
    noNoneAngleList = [[idx, angle] for idx, angle in enumerate(angle_list) if angle is not None]
    for idx in range(1, len(noNoneAngleList)-1):
        if noNoneAngleList[idx-1][1] < noNoneAngleList[idx][1] and noNoneAngleList[idx+1][1] < noNoneAngleList[idx][1]:
            peaks.append(noNoneAngleList[idx][0])
    return peaks


def compute_angle_sequence_test():
    idx = 7
    with open('./test.json', 'r') as f:
        rawData = json.load(f)
        frames = rawData['frames']
    idx_kp_list = []
    angle_list = []
    for frame_idx in frames.keys():
        frame_data = frames[frame_idx]
        if str(idx) in frame_data.keys():
            idx_data = frame_data[str(idx)]
            if idx_data['centroid'] is not None and idx_data['endpoint'] is not None and len(idx_data['endpoint'])==2:
                idx_kp_list.append([idx_data['endpoint'][0], idx_data['centroid'], idx_data['endpoint'][1]])
                angle = get_angle(idx_data['endpoint'][0], idx_data['centroid'], idx_data['endpoint'][1])
                angle_list.append(angle)
            else:
                idx_kp_list.append(None)
                angle_list.append(None)
        else:
            idx_kp_list.append(None)
            angle_list.append(None)
    return angle_list


# def find()

if __name__ == "__main__":
    angle_list = compute_angle_sequence_test()
    peaks = find_peaks(angle_list)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = [], []
    for idx, value in enumerate(angle_list):
        if value is not None:
            x.append(idx)
            y.append(value)
    print(len(peaks), peaks)
    ax.plot(np.array(x), np.array(y))
    ax.scatter(np.array(x), np.array(y))
    for x in peaks:
        ax.axvline(x=x, color='r')
    plt.show()

