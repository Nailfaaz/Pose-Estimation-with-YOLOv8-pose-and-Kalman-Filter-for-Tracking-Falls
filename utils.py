# utils.py

import cv2
import json
import numpy as np
from ultralytics import YOLO

def load_pose_model(model_path):
    return YOLO(model_path)

def estimate_poses(model, frame):
    results = model(frame)
    detections = []
    for result in results:
        keypoints = result.keypoints.data
        boxes = result.boxes.data
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max, conf, _ = box.tolist()
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            detections.append((cx, cy, keypoints[i], box))
    return detections

def init_video_io(video_path, output_path, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video.")
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return cap, out

def draw_detection(frame, track_id, box, keypoints, is_fall):
    x_min, y_min, x_max, y_max, *_ = box.tolist()
    color = (0, 0, 255) if is_fall else (255, 0, 0)
    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
    cv2.putText(frame, f'ID: {track_id}', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    keypoints_subset = keypoints[[0, 9, 10, 15, 16]]  # Head, hands, legs
    for kp in keypoints_subset:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def save_keypoints_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
