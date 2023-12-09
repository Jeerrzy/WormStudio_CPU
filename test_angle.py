#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : test_angle.py
# @Time     : 2023/11/18 16:40
# @Project  : WormStudio_CPU


import json
import os
from models import *


def batch_angle_processing():
    root = 'E:/Worm_CPU_datasets/CPU_Datasets_2023_11_28_Results'
    angle_result = {}
    for jsonName in os.listdir(root):
        if jsonName.endswith('json'):
            jsonPath = os.path.join(root, jsonName)
            name, _ = os.path.splitext(jsonName)
            result = compute_json_twist_number(json_path=jsonPath)
            angle_result[name+'.mp4'] = result
    with open('./angle_result.json', 'w') as f:
        json.dump(angle_result, f, indent=2, cls=NpEncoder)


if __name__ == "__main__":
    batch_angle_processing()