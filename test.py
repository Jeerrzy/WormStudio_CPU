#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test
# @Date    : 2023/11/13 0:17
# @Author  : Jeerrzy


import os
import datetime
import cv2
import numpy as np
from models import *


def unet_test():
    unet = UnetDetector()
    imageData = cv2.imread('./lz_test.png')
    binary_mask = unet.detect(imageData)
    n, bodies, key_values, centroids, endpoints, crosspoints = morphologyPostProcessing(binary_mask)
    for i in range(n):
        for (x, y) in bodies[i]:
            cv2.circle(imageData, (int(x), int(y)), 1, (0, 255, 0), -1)
        for (x, y) in endpoints[i]:
            cv2.circle(imageData, (int(x), int(y)), 2, (0, 0, 255), -1)
        (x, y) = centroids[i]
        cv2.circle(imageData, (int(x), int(y)), 2, (255, 0, 0), -1)
        for (x, y, w, h, l) in key_values:
            cv2.rectangle(imageData, (int(x), int(y), int(w), int(h)), (255, 0, 0), 2)
    cv2.imshow('demo', imageData)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./test_result.png', imageData)


def yolo_test0():
    yolo = YOLODetector()
    imageData = cv2.imread('./lz_test.png')
    bboxes = yolo.detect(imageData)
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(imageData, (int(y1), int(x1)), (int(y2), int(x2)), (0, 255, 0), 2)
    cv2.imshow('demo', imageData)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./test_result2.png', imageData)


def yolo_test():
    root = './images'
    yolo = YOLODetector()
    for imageName in os.listdir(root):
        imageData = cv2.imread(os.path.join(root, imageName))
        bboxes = yolo.detect(imageData)
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(imageData, (int(y1), int(x1)), (int(y2), int(x2)), (0, 255, 0), 2)
        cv2.imshow('demo', cv2.resize(imageData, (1280, 960)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def yolo_mot_test():
    for root, dirs, fileNames in os.walk('D:/BaiduNetdiskDownload/CPURawVideo'):
        for fileName in fileNames:
            print(fileName)
            name, _ = os.path.splitext(fileName)
            videoPath = os.path.join(root, fileName)
            txtPath = os.path.join('D:/BaiduNetdiskDownload/WormStudioCPUVideoRun/Yolo', name+'_Yolo.txt')
            outvideoPath = os.path.join('D:/BaiduNetdiskDownload/WormStudioCPUVideoRun/Yolo', name+'_Yolo.mp4')
            yolo_mot_processing(src_video_path=videoPath, result_txt_path=txtPath)
            visualizeVideo2txt(
                src_video_path=videoPath,
                out_video_path=outvideoPath,
                txt_path=txtPath,
                scale=0.25
            )


def unet_mot_test():
    for root, dirs, fileNames in os.walk('D:/BaiduNetdiskDownload/CPURawVideo'):
        for fileName in fileNames:
            print(fileName)
            name, _ = os.path.splitext(fileName)
            videoPath = os.path.join(root, fileName)
            outvideoPath = os.path.join('D:/BaiduNetdiskDownload/WormStudioCPUVideoRun/Unet', name + '_Unet.mp4')
            jsonPath = os.path.join('D:/BaiduNetdiskDownload/WormStudioCPUVideoRun/Unet', name+'_Unet.json')
            unet_detect_video(src_video_path=videoPath, result_json_path=jsonPath)
            visualizeVideo2json(
                src_video_path=videoPath,
                out_video_path=outvideoPath,
                json_path=jsonPath,
                scale=0.25
            )

def unet2yolo_test():
    for root, dirs, fileNames in os.walk('D:/BaiduNetdiskDownload/CPURawVideo'):
        for fileName in fileNames:
            print(fileName)
            srcVideoPath = os.path.join(root, fileName)
            name, _ = os.path.splitext(fileName)
            jsonPath = os.path.join('D:/BaiduNetdiskDownload/WormStudioCPUVideoRun/Unet', name+'_Unet.json')
            txtPath = os.path.join('D:/BaiduNetdiskDownload/WormStudioCPUVideoRun/Unet2yolo', name+'_Unet2yolo.txt')
            outVideoPath = os.path.join('D:/BaiduNetdiskDownload/WormStudioCPUVideoRun/Unet2yolo', name+'_Unet2yolo.mp4')
            unetJsonDict2yoloTxt(jsonPath, txtPath)
            yolo_track(txtPath, txtPath)
            yolo_optimize(txtPath, txtPath)
            visualizeVideo2txt(
                src_video_path=srcVideoPath,
                out_video_path=outVideoPath,
                txt_path=txtPath,
                scale=0.25
            )


def batch_joint_processing(dataset_root, result_root):
    for fileName in os.listdir(dataset_root):
        name, _ = os.path.splitext(fileName)
        filePath = os.path.join(dataset_root, fileName)
        outPath = os.path.join(result_root, fileName)
        jsonPath = os.path.join(result_root, name+'.json')
        txtPath = os.path.join(result_root, name+'.txt')
        yolo_unet_joint_mot_processing(
            src_video_path=filePath,
            result_json_path=jsonPath,
            cache_txt_path=txtPath
        )
        visualizeVideo2json(
            src_video_path=filePath,
            out_video_path=outPath,
            json_path=jsonPath,
            scale=0.25
        )


if __name__ == "__main__":
    yolo_test0()
    # batch_joint_processing(
    #     dataset_root='E:/Worm_CPU_datasets/CPU_Datasets_2023_11_28',
    #     result_root='E:/Worm_CPU_datasets/CPU_Datasets_2023_11_28_Results'
    # )


