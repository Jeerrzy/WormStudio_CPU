#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : utils.py
# @Time     : 2023/11/14 23:52
# @Project  : WormStudio_CPU


import json
import cv2
import numpy as np
from tqdm import tqdm


# FrameIDDrawConfig = {'pos': (150, 225), 'size': 10, 'color': (255, 0, 0), 'thickness': 3}
FrameIDDrawConfig = {'pos': (100, 150), 'size': 5, 'color': (255, 0, 0), 'thickness': 3}
BodyDrawConfig = {'size': 2, 'color': (0, 255, 0)}
EPDrawConfig = {'size': 8, 'color': (0, 0, 255)}
CPDrawConfig = {'size': 10, 'color': (255, 0, 0)}
BBoxDrawConfig = {'size': 2, 'color': (128, 0, 128)}
IDDrawConfig = {'size': 2, 'color': (130, 0, 75), 'thickness': 4}


def visualizeVideo2json(src_video_path, out_video_path, json_path, fps=None, scale=None):
    """
    :param src_video_path: 输入视频路径
    :param out_video_path: 输出视频路径
    :param json_path: json格式的MOT追踪序列文件
    :param fps: 控制输出视频的帧率
    :param scale: 控制输出视频的缩放尺度
    """
    print(f'visualizing {src_video_path} ...')
    with open(json_path) as f:
        rawData = json.load(f)
        videoInfo = rawData['videoInfo']
        frames = rawData['frames']
    src_video = cv2.VideoCapture(src_video_path)
    out_fps = videoInfo['fps'] if fps is None else fps
    out_w = videoInfo['width'] if scale is None else int(videoInfo['width'] * scale)
    out_h = videoInfo['height'] if scale is None else int(videoInfo['height'] * scale)
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'), out_fps, (out_w, out_h))
    frame_idx = 1
    with tqdm(total=videoInfo['frameNum'] - 1) as pbar:
        pbar.set_description('Processing')
        while src_video.isOpened():
            ret, frame = src_video.read()
            if not ret:
                break
            frame_result = frames[str(frame_idx)]
            cv2.putText(frame, str(frame_idx), FrameIDDrawConfig['pos'], cv2.FONT_HERSHEY_SIMPLEX,
                        FrameIDDrawConfig['size'], FrameIDDrawConfig['color'], FrameIDDrawConfig['thickness'])
            if type(frame_result) == dict:
                frame_result = list(frame_result.values())
            for wormObjDict in frame_result:
                drawObjDict2image(wormObjDict, frame)
            out_video.write(cv2.resize(frame, (out_w, out_h)))
            frame_idx += 1
            pbar.update(1)
    src_video.release()
    out_video.release()
    print(f'visualizing {src_video_path} down.')


def drawObjDict2image(wormObjDict, frame):
    """
    :param wormObjDict: 模型对象字典
    :param frame: 绘制图片
    """
    _id = wormObjDict['id']
    body = wormObjDict['body']
    bbox = wormObjDict['bbox']
    centroid = wormObjDict['centroid']
    endpoint = wormObjDict['endpoint']
    individual = wormObjDict['individual']
    if body is not None:
        for (x, y) in body:
            cv2.circle(frame, (int(x), int(y)), BodyDrawConfig['size'], BodyDrawConfig['color'], -1)
    if endpoint is not None:
        for (x, y) in endpoint:
            cv2.circle(frame, (int(x), int(y)), EPDrawConfig['size'], EPDrawConfig['color'], -1)
    if centroid is not None:
        (x, y) = centroid
        cv2.circle(frame, (int(x), int(y)), CPDrawConfig['size'], CPDrawConfig['color'], -1)
    if bbox is not None:
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (int(x), int(y), int(w), int(h)), BBoxDrawConfig['color'], BBoxDrawConfig['size'])
    if _id is not None:
        cv2.putText(frame, str(wormObjDict['id']), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    IDDrawConfig['size'], IDDrawConfig['color'], IDDrawConfig['thickness'])


def visualizeVideo2txt(src_video_path, out_video_path, txt_path, fps=None, scale=None):
    """
    :param src_video_path: 输入视频路径
    :param out_video_path: 输出视频路径
    :param txt_path: txt格式的MOT追踪序列文件
    :param fps: 控制输出视频的帧率
    :param scale: 控制输出视频的缩放尺度
    """
    frames = np.loadtxt(txt_path, dtype=int, delimiter=',')
    src_video = cv2.VideoCapture(src_video_path)
    videoInfo = {
        'frameNum': int(src_video.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(src_video.get(cv2.CAP_PROP_FPS))
    }
    out_fps = videoInfo['fps'] if fps is None else fps
    out_w = videoInfo['width'] if scale is None else int(videoInfo['width'] * scale)
    out_h = videoInfo['height'] if scale is None else int(videoInfo['height'] * scale)
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'), out_fps, (out_w, out_h))
    frame_idx = 1
    with tqdm(total=videoInfo['frameNum'] - 1) as pbar:
        pbar.set_description('Processing')
        while src_video.isOpened():
            ret, frame = src_video.read()
            if not ret:
                break
            frame_result = frames[frames[:, 0] == frame_idx]
            cv2.putText(frame, str(frame_idx), FrameIDDrawConfig['pos'], cv2.FONT_HERSHEY_SIMPLEX,
                        FrameIDDrawConfig['size'], FrameIDDrawConfig['color'], FrameIDDrawConfig['thickness'])
            for _, worm_id, x, y, w, h, _, _, _, _ in frame_result:
                cv2.rectangle(frame, (y, x), (y + h, x + w), BBoxDrawConfig['color'], BBoxDrawConfig['size'])
                cv2.putText(frame, str(worm_id), (y, x), cv2.FONT_HERSHEY_SIMPLEX,
                            IDDrawConfig['size'], IDDrawConfig['color'], IDDrawConfig['thickness'])
            out_video.write(cv2.resize(frame, (out_w, out_h)))
            frame_idx += 1
            pbar.update(1)
    src_video.release()
    out_video.release()
    print(f'visualizing {src_video_path} down.')


def visualizeVideo2txt_json(src_video_path, out_video_path, txt_path, json_path, fps=None, scale=None):
    """
    :param src_video_path: 输入视频路径
    :param out_video_path: 输出视频路径
    :param txt_path: txt格式的MOT追踪序列文件
    :param json_path: json格式的MOT追踪序列文件
    :param fps: 控制输出视频的帧率
    :param scale: 控制输出视频的缩放尺度
    """
    print(f'visualizing {src_video_path} ...')
    framesTxt = np.loadtxt(txt_path, dtype=int, delimiter=',')
    with open(json_path) as f:
        rawData = json.load(f)
        videoInfo = rawData['videoInfo']
        framesJson = rawData['frames']
    src_video = cv2.VideoCapture(src_video_path)
    out_fps = videoInfo['fps'] if fps is None else fps
    out_w = videoInfo['width'] if scale is None else int(videoInfo['width'] * scale)
    out_h = videoInfo['height'] if scale is None else int(videoInfo['height'] * scale)
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'), out_fps, (out_w, out_h))
    frame_idx = 1
    with tqdm(total=videoInfo['frameNum'] - 1) as pbar:
        pbar.set_description('Processing')
        while src_video.isOpened():
            ret, frame = src_video.read()
            if not ret:
                break
            # 绘制unet检测的json文件结果
            frame_result_json = framesJson[str(frame_idx)]
            cv2.putText(frame, str(frame_idx), FrameIDDrawConfig['pos'], cv2.FONT_HERSHEY_SIMPLEX,
                        FrameIDDrawConfig['size'], FrameIDDrawConfig['color'], FrameIDDrawConfig['thickness'])
            if type(frame_result_json) == dict:
                frame_result_json = list(frame_result_json.values())
            for wormObjDict in frame_result_json:
                drawObjDict2image(wormObjDict, frame)
            # 绘制yolo检测的txt文件结果
            frame_result_txt = framesTxt[framesTxt[:, 0] == frame_idx]
            cv2.putText(frame, str(frame_idx), FrameIDDrawConfig['pos'], cv2.FONT_HERSHEY_SIMPLEX,
                        FrameIDDrawConfig['size'], FrameIDDrawConfig['color'], FrameIDDrawConfig['thickness'])
            for _, worm_id, x, y, w, h, _, _, _, _ in frame_result_txt:
                cv2.rectangle(frame, (y, x), (y + h, x + w), BBoxDrawConfig['color'], BBoxDrawConfig['size'])
                cv2.putText(frame, str(worm_id), (y, x), cv2.FONT_HERSHEY_SIMPLEX,
                            IDDrawConfig['size'], IDDrawConfig['color'], IDDrawConfig['thickness'])
            out_video.write(cv2.resize(frame, (out_w, out_h)))
            frame_idx += 1
            pbar.update(1)
    src_video.release()
    out_video.release()
    print(f'visualizing {src_video_path} down.')