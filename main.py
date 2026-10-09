# main.py

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import *
from models import KalmanFilterManual, SpeedBasedFallDetector
from utils import (
    load_pose_model, estimate_poses, init_video_io,
    draw_detection, save_keypoints_data
)

def main():
    # Load model and init video
    model = load_pose_model(MODEL_PATH)
    cap, writer = init_video_io(VIDEO_PATH, OUTPUT_VIDEO_PATH, VIDEO_FPS)

    frame_data = {'fall': [], 'non_fall': []}
    trackers, tracks, next_id = {}, [], 0
    fall_detector = SpeedBasedFallDetector(FALL_SPEED_THRESHOLD)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = estimate_poses(model, frame)

        # Predict all tracker positions
        for tracker in trackers.values():
            tracker.predict()

        if tracks and detections:
            track_states = np.array([trackers[tid].get_state() for tid in tracks])
            detection_points = np.array([[cx, cy] for cx, cy, _, _ in detections])
            cost = np.linalg.norm(track_states[:, None] - detection_points[None], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)

            unmatched_tracks = set(tracks)
            unmatched_detections = set(range(len(detections)))

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < ASSOCIATION_DISTANCE:
                    trackers[tracks[r]].update(np.array(detection_points[c]))
                    unmatched_tracks.discard(tracks[r])
                    unmatched_detections.discard(c)
        else:
            unmatched_tracks = set(tracks)
            unmatched_detections = set(range(len(detections)))

        # Create new tracks
        for i in unmatched_detections:
            tid = next_id
            next_id += 1
            trackers[tid] = KalmanFilterManual(detections[i][0], detections[i][1])
            tracks.append(tid)

        # Remove outdated tracks (optional logic here)

        # Draw and process detections
        for tid in list(tracks):
            tracker = trackers[tid]
            track_pos = tracker.get_state()
            for det in detections:
                cx, cy, keypoints, box = det
                if np.linalg.norm(track_pos - np.array([cx, cy])) < ASSOCIATION_DISTANCE:
                    fall = fall_detector.detect(keypoints, tid)
                    draw_detection(frame, tid, box, keypoints, fall)
                    flat_kp = keypoints[:, :2].flatten().tolist()
                    tag = 'fall' if fall else 'non_fall'
                    frame_data[tag].append({'frame': frame_count, 'keypoints': flat_kp})
                    break

        writer.write(frame)
        frame_count += 1
        cv2.imshow('Fall Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    save_keypoints_data(KEYPOINTS_JSON_PATH, frame_data)

if __name__ == "__main__":
    main()
