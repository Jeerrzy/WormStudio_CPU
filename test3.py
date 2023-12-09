#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : test3.py
# @Time     : 2023/11/22 17:04
# @Project  : WormStudio_CPU


from models import *


if __name__ == "__main__":
    # yolo_unet_joint_mot_processing(
    #     src_video_path='./WeChat_20231122170312.mp4',
    #     result_json_path='./WeChat_result.json',
    #     cache_txt_path='./WeChat_result.txt'
    # )
    # visualizeVideo2json(
    #     src_video_path='./WeChat_20231122170312.mp4',
    #     out_video_path='./WeChat_result.mp4',
    #     json_path='./WeChat_result.json'
    # )
    result = compute_json_twist_number(json_path='./WeChat_result.json')
    print(result)