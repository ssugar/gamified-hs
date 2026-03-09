"""Release refinement and puck trajectory/lift analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from utils import acceleration


def _euclidean(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _get_blade_center(rec: dict[str, Any]) -> list[float] | None:
    blade = rec.get("stick_blade")
    if not blade:
        return None
    c = blade.get("center")
    if not c or len(c) != 2:
        return None
    return [float(c[0]), float(c[1])]


def refine_release_frame(
    shot_info: dict[str, Any],
    tracking_result: dict[str, Any],
    detection_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fuse wrist, puck-separation, and puck-velocity signals."""
    base_release = int(shot_info.get("release_frame", 0))
    shot_start = int(shot_info.get("shot_start_frame", 0))
    wrist_vel = shot_info.get("wrist_velocities", []) or [0.0]
    wrist_accel = acceleration(wrist_vel, fps=1.0)  # frame-domain spike score
    wrist_spike_idx = int(np.argmax(wrist_accel)) if wrist_accel else base_release

    per_frame = tracking_result.get("per_frame", [])
    n = min(len(per_frame), len(detection_records))
    if n < 3:
        return {
            "release_frame": base_release,
            "release_source": "wrist_only_fallback",
            "wrist_spike_frame": wrist_spike_idx,
        }

    distances: list[float] = []
    puck_vel: list[float] = []
    for i in range(n):
        puck_xy = [per_frame[i]["puck_x"], per_frame[i]["puck_y"]]
        blade_xy = _get_blade_center(detection_records[i])
        if blade_xy is None:
            distances.append(distances[-1] if distances else 0.0)
        else:
            distances.append(_euclidean(puck_xy, blade_xy))
        puck_vel.append(float(per_frame[i]["velocity"]))

    dist_delta = [0.0]
    for i in range(1, n):
        dist_delta.append(distances[i] - distances[i - 1])

    positive_delta = [d for d in dist_delta if d > 0]
    sep_threshold = max(3.0, float(np.percentile(positive_delta, 75))) if positive_delta else 5.0
    vel_threshold = max(5.0, float(np.percentile(puck_vel, 75)))

    # Candidate release frames satisfy: separation increase AND velocity spike.
    candidates: list[int] = []
    search_start = max(1, shot_start)
    for i in range(search_start, n):
        sep_ok = dist_delta[i] >= sep_threshold
        vel_ok = puck_vel[i] >= vel_threshold
        if sep_ok and vel_ok:
            candidates.append(i)

    final_release = base_release
    source = "wrist_only"
    if candidates:
        # Choose candidate nearest wrist spike to preserve V2 behavior bias.
        final_release = min(candidates, key=lambda c: abs(c - wrist_spike_idx))
        source = "fused_wrist_separation_velocity"

    return {
        "release_frame": int(final_release),
        "release_source": source,
        "wrist_spike_frame": int(wrist_spike_idx),
        "separation_threshold": float(sep_threshold),
        "velocity_threshold": float(vel_threshold),
    }


def classify_puck_lift(angle_deg: float) -> str:
    if angle_deg < 5.0:
        return "low_shot"
    if angle_deg <= 15.0:
        return "medium_lift"
    return "high_lift"


def analyze_post_release_trajectory(
    tracking_result: dict[str, Any],
    release_frame: int,
    window_frames: int = 12,
) -> dict[str, Any]:
    """Compute trajectory slope, launch angle, and lift class."""
    rows = tracking_result.get("per_frame", [])
    if len(rows) < 3:
        return {
            "puck_velocity_pixels": 0.0,
            "trajectory_slope": 0.0,
            "launch_angle_deg": 0.0,
            "puck_lift_classification": "unknown",
            "trajectory_points_post_release": [],
        }

    start = max(0, int(release_frame))
    end = min(len(rows), start + max(3, window_frames))
    seg = rows[start:end]
    if len(seg) < 2:
        return {
            "puck_velocity_pixels": 0.0,
            "trajectory_slope": 0.0,
            "launch_angle_deg": 0.0,
            "puck_lift_classification": "unknown",
            "trajectory_points_post_release": [],
        }

    p0 = np.array([seg[0]["puck_x"], seg[0]["puck_y"]], dtype=float)
    p1 = np.array([seg[-1]["puck_x"], seg[-1]["puck_y"]], dtype=float)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])

    slope = float(dy / dx) if abs(dx) > 1e-8 else 0.0
    # Image Y grows downward, so physical lift is -dy.
    launch_angle = math.degrees(math.atan2(-dy, abs(dx) + 1e-8))
    launch_angle = abs(float(launch_angle))
    lift_class = classify_puck_lift(launch_angle)

    post_vel = [float(r["velocity"]) for r in seg]
    avg_vel = float(np.mean(post_vel)) if post_vel else 0.0

    return {
        "puck_velocity_pixels": avg_vel,
        "trajectory_slope": slope,
        "launch_angle_deg": launch_angle,
        "puck_lift_classification": lift_class,
        "trajectory_points_post_release": [[r["puck_x"], r["puck_y"]] for r in seg],
    }

