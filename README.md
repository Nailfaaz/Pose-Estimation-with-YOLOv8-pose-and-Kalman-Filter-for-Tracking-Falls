# Fall Detection with YOLOv8-Pose and a Kalman Filter

Detects people falling in a video. YOLOv8-pose finds each person and their body keypoints, a
hand-written Kalman filter tracks every person across frames, and a person is flagged as **falling**
when their head moves downward faster than a threshold.

## How it works

1. **Pose estimation:** `yolov8n-pose` returns a box and 17 keypoints per person.
2. **Tracking:** each person gets a constant-velocity Kalman filter (state `x, y, vx, vy`, written in
   NumPy). Every frame, predicted track positions are matched to detections with the Hungarian algorithm
   (`scipy.optimize.linear_sum_assignment`) on centre distance; matches farther than
   `ASSOCIATION_DISTANCE` (50 px) start a new track.
3. **Fall rule:** the head keypoint's displacement between frames is compared with
   `FALL_SPEED_THRESHOLD` (20 px/frame); a fall is reported only when the movement points downward
   (angle between 45° and 135°).
4. **Output:** an annotated video (red box = fall, blue = normal) and a JSON file with the keypoints of
   every frame, split into `fall` and `non_fall`, ready to train a classifier later.

## Run

```bash
pip install -r requirements.txt      # ultralytics, opencv-python, numpy, scipy
# put a video at src/input/dummy.mp4 (or change VIDEO_PATH in config.py)
python main.py                       # press q to stop
```

Ultralytics downloads `yolov8n-pose.pt` on first use. Results go to `src/output/`.

| Setting (`config.py`) | Default |
|---|---|
| `FALL_SPEED_THRESHOLD` | 20 px per frame |
| `ASSOCIATION_DISTANCE` | 50 px |
| `VIDEO_FPS` | 20 |

## Files

```
main.py     detection → tracking → fall check loop
models.py   KalmanFilterManual, SpeedBasedFallDetector
utils.py    YOLO pose wrapper, video I/O, drawing, JSON export
config.py   paths and thresholds
LesgoFall_multiple_analysis_speed.ipynb   original notebook version
```

## Limitations

The fall rule is a speed heuristic, so a fast bend or a camera shake can trigger it, and a slow fall can be
missed. The keypoint JSON is meant as training data for a learned fall classifier.
