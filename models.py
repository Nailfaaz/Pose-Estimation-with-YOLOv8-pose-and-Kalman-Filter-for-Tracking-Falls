# models.py

import numpy as np

class KalmanFilterManual:
    def __init__(self, x, y):
        self.state = np.array([x, y, 0, 0], dtype='float64')
        self.P = np.eye(4) * 1000
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 10
        self.I = np.eye(4)
        self.Q = np.eye(4)

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def get_state(self):
        return self.state[:2]


class SpeedBasedFallDetector:
    def __init__(self, threshold):
        self.prev_positions = {}
        self.threshold = threshold

    def detect(self, keypoints, track_id):
        try:
            head = keypoints[0, :2]
            if track_id in self.prev_positions:
                movement = head - self.prev_positions[track_id]
                speed = np.linalg.norm(movement)
                angle = np.arctan2(movement[1], movement[0])
                self.prev_positions[track_id] = head
                if speed > self.threshold and (np.pi/4 <= angle <= 3*np.pi/4):
                    return True
            else:
                self.prev_positions[track_id] = head
        except IndexError:
            return False
        return False
