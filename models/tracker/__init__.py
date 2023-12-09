#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @folder: tracker
# @author: jerrzzy
# @date: 2023/8/26


import json
import numpy as np
from .sort import Sort
from .joint import WormMorphologyJointTracker


def yolo_track(dets_seq_path, trk_result_path):
    """
    Params:
    dets_seq_path - MOT格式的检测结果文件路径
    trk_result_path - MOT格式的追踪结果文件路径
    """
    sort_tracker = Sort()
    det_data = np.loadtxt(dets_seq_path, delimiter=',')
    det_data[:, 4:6] += det_data[:, 2:4]  # convert [x,y,w,h] to [x1,y1,x2,y2]
    with open(trk_result_path, 'w') as f:
        for frame_id in range(1, int(det_data[:, 0].max())+1):
            frame_data = det_data[det_data[:, 0] == frame_id, 2:7]
            trackers = sort_tracker.update(frame_data)
            for d in trackers:
                """
                MOT: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                """
                f.write(f'%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame_id, d[4], d[0], d[1], d[2]-d[0], d[3]-d[1]) + '\n')
    f.close()


def joint_track(yolo_txt_path, unet_json_path, result_json_path):
    """
    :param yolo_txt_path: txt格式的经过yolo检测和MOT处理过程的监督信息文件
    :param unet_json_path: json格式的unet形态学检测结果
    :param result_json_path: 经过联合追踪优化后的形态学追踪结果
    """
    joint_tracker = WormMorphologyJointTracker()
    yolo_trks = np.loadtxt(yolo_txt_path, dtype=int, delimiter=',')
    trks_frames = {}
    with open(unet_json_path, 'r') as f:
        unet_dets_raw_data = json.load(f)
        unet_dets = unet_dets_raw_data['frames']
        unet_dets_raw_data['id_num'] = int(yolo_trks[:, 1].max())
    for frame_idx in range(1, int(yolo_trks[:, 0].max()) + 1):
        yolo_trk_frame_data = yolo_trks[yolo_trks[:, 0] == frame_idx, 1:6]
        unet_dets_frame_data = unet_dets[str(frame_idx)]
        joint_tracker.update(yolo_trk_frame_data, unet_dets_frame_data)
        trks_frames[str(frame_idx)] = joint_tracker.save_info()
    unet_dets_raw_data['frames'] = trks_frames
    with open(result_json_path, 'w') as f:
        json.dump(unet_dets_raw_data, f, indent=2, cls=NpEncoder)


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


