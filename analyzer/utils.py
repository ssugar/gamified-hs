"""Utility functions for hockey shot analysis."""

import numpy as np
from typing import Optional

# MoveNet keypoint indices
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Keypoint index lookup
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def angle_between_points(
    p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray
) -> float:
    """Calculate angle at vertex formed by p1-vertex-p2 in degrees."""
    v1 = p1 - vertex
    v2 = p2 - vertex
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def line_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle of line p1->p2 relative to horizontal, in degrees."""
    delta = p2 - p1
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def angular_difference(a: float, b: float) -> float:
    """Signed shortest angular difference from a to b, handling ±180° wrapping."""
    diff = (b - a + 180.0) % 360.0 - 180.0
    return diff


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Midpoint between two points."""
    return (p1 + p2) / 2.0


def velocity(positions: list[np.ndarray], fps: float) -> list[float]:
    """Compute velocity (pixels/sec) from a list of positions."""
    if len(positions) < 2:
        return [0.0]
    dt = 1.0 / fps
    velocities = []
    for i in range(1, len(positions)):
        d = distance(positions[i], positions[i - 1])
        velocities.append(d / dt)
    return velocities


def acceleration(velocities: list[float], fps: float) -> list[float]:
    """Compute acceleration (change in velocity per frame)."""
    if len(velocities) < 2:
        return [0.0]
    accel = [0.0]
    for i in range(1, len(velocities)):
        accel.append(velocities[i] - velocities[i - 1])
    return accel


def smooth(values: list[float], window: int = 3) -> list[float]:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def get_keypoint(
    keypoints: np.ndarray, name: str, confidence_threshold: float = 0.2
) -> Optional[np.ndarray]:
    """Get keypoint position [y, x] if confidence is above threshold.

    MoveNet returns keypoints as [y, x, confidence].
    Returns as [x, y] for easier use with OpenCV.
    """
    idx = KP[name]
    kp = keypoints[idx]
    if kp[2] < confidence_threshold:
        return None
    return np.array([kp[1], kp[0]])  # Return as [x, y]


def get_keypoint_raw(
    keypoints: np.ndarray, name: str, confidence_threshold: float = 0.2
) -> Optional[np.ndarray]:
    """Get raw keypoint [y, x, confidence]."""
    idx = KP[name]
    kp = keypoints[idx]
    if kp[2] < confidence_threshold:
        return None
    return kp
