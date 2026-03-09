"""Puck speed and shot power estimation."""

from __future__ import annotations

from typing import Any

import numpy as np


MPS_TO_MPH = 2.23693629


def _max_or_zero(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def estimate_shot_power(
    tracking_result: dict[str, Any],
    mechanics: dict[str, Any],
    shot_info: dict[str, Any],
    release_frame: int,
    calibration_m_per_pixel: float,
    fps: float,
) -> dict[str, Any]:
    """Estimate speed in mph and a composite 0-100 shot power score."""
    rows = tracking_result.get("per_frame", [])
    if not rows:
        return {
            "estimated_shot_speed_mph": 0.0,
            "peak_puck_velocity_mph": 0.0,
            "peak_puck_velocity_mps": 0.0,
            "shot_power_score": 0.0,
        }

    release = max(0, min(int(release_frame), len(rows) - 1))
    post_rows = rows[release:]
    pixel_vel = [float(r.get("velocity", 0.0)) for r in post_rows]
    if not pixel_vel:
        pixel_vel = [float(r.get("velocity", 0.0)) for r in rows]

    mps_vel = [v * calibration_m_per_pixel for v in pixel_vel]
    mph_vel = [v * MPS_TO_MPH for v in mps_vel]

    estimated_mph = float(np.mean(mph_vel[:5])) if mph_vel else 0.0
    peak_mps = _max_or_zero(mps_vel)
    peak_mph = peak_mps * MPS_TO_MPH

    # Rotation speeds are stored as deg/frame; convert to deg/s.
    timing = mechanics.get("timing_sequence", {})
    hip_deg_per_s = _max_or_zero(timing.get("hip_angular_velocities", [])) * max(1.0, float(fps))
    shoulder_deg_per_s = _max_or_zero(timing.get("shoulder_angular_velocities", [])) * max(
        1.0, float(fps)
    )
    wrist_vel = _max_or_zero(shot_info.get("wrist_velocities", []))

    # Normalized sub-scores for power model.
    speed_n = min(1.0, estimated_mph / 90.0)
    hip_n = min(1.0, hip_deg_per_s / 500.0)
    shoulder_n = min(1.0, shoulder_deg_per_s / 650.0)
    wrist_n = min(1.0, wrist_vel / 2.0)
    score = 100.0 * (0.4 * speed_n + 0.2 * hip_n + 0.2 * shoulder_n + 0.2 * wrist_n)

    return {
        "estimated_shot_speed_mph": float(round(estimated_mph, 2)),
        "peak_puck_velocity_mph": float(round(peak_mph, 2)),
        "peak_puck_velocity_mps": float(round(peak_mps, 2)),
        "shot_power_score": float(round(max(0.0, min(100.0, score)), 1)),
        "calibration_m_per_pixel": float(calibration_m_per_pixel),
    }

