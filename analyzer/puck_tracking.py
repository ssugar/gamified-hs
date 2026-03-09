"""Lightweight puck tracking and kinematics."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class PuckTracker:
    """Centroid tracker with Kalman smoothing for short shot clips."""

    def __init__(self, fps: float):
        self.fps = max(1.0, float(fps))
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def _init_state(self, cx: float, cy: float) -> None:
        self.kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self.initialized = True

    def step(self, measurement: tuple[float, float] | None) -> tuple[float, float, bool]:
        """Track one frame; returns (x, y, observed)."""
        if not self.initialized:
            if measurement is None:
                return 0.0, 0.0, False
            self._init_state(*measurement)
            return measurement[0], measurement[1], True

        pred = self.kf.predict()
        px, py = float(pred[0]), float(pred[1])
        if measurement is not None:
            mx, my = measurement
            self.kf.correct(np.array([[mx], [my]], dtype=np.float32))
            return float(mx), float(my), True
        return px, py, False


def extract_center(det: dict[str, Any] | None) -> tuple[float, float] | None:
    if not det:
        return None
    center = det.get("center")
    if not center or len(center) != 2:
        return None
    return float(center[0]), float(center[1])


def track_puck_records(
    detection_records: list[dict[str, Any]],
    fps: float,
) -> dict[str, Any]:
    """Track puck through records and compute velocity/acceleration per frame."""
    tracker = PuckTracker(fps=fps)
    dt = 1.0 / max(1.0, float(fps))

    rows: list[dict[str, Any]] = []
    prev_pt: np.ndarray | None = None
    prev_vel = 0.0

    for rec in detection_records:
        frame_index = int(rec["frame_index"])
        raw_frame_index = int(rec["raw_frame_index"])
        measurement = extract_center(rec.get("puck"))
        x, y, observed = tracker.step(measurement)

        pt = np.array([x, y], dtype=float)
        if prev_pt is None:
            vel = 0.0
            acc = 0.0
        else:
            vel = _distance(pt, prev_pt) / dt
            acc = (vel - prev_vel) / dt
        prev_pt = pt
        prev_vel = vel

        rows.append(
            {
                "frame_index": frame_index,
                "raw_frame_index": raw_frame_index,
                "puck_x": float(x),
                "puck_y": float(y),
                "velocity": float(vel),
                "acceleration": float(acc),
                "observed": bool(observed),
            }
        )

    peak_vel = max((r["velocity"] for r in rows), default=0.0)
    return {
        "per_frame": rows,
        "trajectory": [[r["puck_x"], r["puck_y"]] for r in rows],
        "peak_velocity_pixels": float(peak_vel),
    }

