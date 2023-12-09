#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__
# @Date    : 2023/11/15 2:11
# @Author  : Jeerrzy


import json
import datetime
from tqdm import tqdm
from .morphology import *
from .yolo import YOLODetector


def yolo_detect_video(src_video_path, result_txt_path):
    """
    Params:
    src_video_path - 输入视频路径
    result_txt_path - txt格式缓存结果保存路径
    """
    print(f'use Yolo to detect {src_video_path} ...')
    detector = YOLODetector()
    src_video = cv2.VideoCapture(src_video_path)
    frame_id = 1
    with open(result_txt_path, 'w') as f:
        pbar = tqdm(total=int(src_video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        while True:
            ret, frame = src_video.read()
            if not ret:
                break
            bboxes = detector.detect(frame)
            if bboxes is not None:
                for (x1, y1, x2, y2) in bboxes:
                    """
                    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    """
                    f.write(f'%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame_id, -1, x1, y1, x2-x1, y2-y1)+'\n')
            frame_id += 1
            pbar.update(1)
    f.close()
    cv2.destroyAllWindows()
    src_video.release()
    print(f'Yolo detect {src_video_path} down.')


def unet_detect_video(src_video_path, result_json_path):
    """
    Params:
    src_video_path - 输入视频路径
    result_json_path - json格式缓存结果保存路径
    """
    print(f'use Unet to detect {src_video_path} ...')
    start_time = get_current_time_str()
    src_video = cv2.VideoCapture(src_video_path)
    detector = UnetMorphologyDetector()
    videoInfo = {
        'frameNum': int(src_video.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(src_video.get(cv2.CAP_PROP_FPS))
    }
    frames = {}
    frame_idx = 1
    with tqdm(total=videoInfo['frameNum']) as pbar:
        pbar.set_description('Processing')
        while src_video.isOpened():
            ret, frame = src_video.read()
            if not ret:
                break
            wormModelList = detector.detect(frame)
            frames[str(frame_idx)] = [wormModel.obj2dict() for wormModel in wormModelList]
            frame_idx += 1
            pbar.update(1)
    src_video.release()
    end_time = get_current_time_str()
    result = {
        'time': start_time + ' ~~~~ ' + end_time,
        'videoInfo': videoInfo,
        'frames': frames
    }
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    print(f'Unet detect {src_video_path} down.')


def unetJsonDict2yoloTxt(json_path, txt_path):
    with open(json_path, 'r') as jf:
        rawData = json.load(jf)
        frames = rawData['frames']
    with open(txt_path, 'w') as tf:
        for frame_idx in frames.keys():
            frame_result = frames[frame_idx]
            for wormObjDict in frame_result:
                if wormObjDict['individual']:
                    _id = wormObjDict['id'] if wormObjDict['id'] is not None else -1
                    x, y, w, h = wormObjDict['bbox']
                    """
                    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    """
                    tf.write(f'%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (int(frame_idx), _id, y, x, h, w) + '\n')
    tf.close()


def get_current_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)