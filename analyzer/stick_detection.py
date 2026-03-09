"""Stick geometry and blade orientation analysis."""

from __future__ import annotations

import math
from typing import Any


def _center(bbox: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _line_angle_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(math.degrees(math.atan2(dy, dx)))


def _normalize_signed_180(angle_deg: float) -> float:
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return float(a)


def classify_blade_state(blade_angle_relative_to_ice: float) -> str:
    """Classify blade openness based on angle to ice (horizontal)."""
    if blade_angle_relative_to_ice <= -10.0:
        return "closed_blade"
    if blade_angle_relative_to_ice >= 10.0:
        return "open_blade"
    return "neutral_blade"


def analyze_stick_frame(
    stick_shaft_det: dict[str, Any] | None,
    stick_blade_det: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute stick orientation and blade state for one frame."""
    if not stick_shaft_det or not stick_blade_det:
        return {
            "stick_orientation_deg": None,
            "blade_angle_deg": None,
            "blade_state": "unknown",
            "confidence": 0.0,
        }

    shaft_center = _center(stick_shaft_det["bbox"])
    blade_center = _center(stick_blade_det["bbox"])

    # Stick orientation: shaft->blade axis.
    orientation = _line_angle_deg(shaft_center, blade_center)
    orientation = _normalize_signed_180(orientation)

    # Relative to horizontal "ice" line.
    blade_angle = _normalize_signed_180(orientation)
    blade_state = classify_blade_state(blade_angle)

    confidence = min(
        float(stick_shaft_det.get("confidence", 0.0)),
        float(stick_blade_det.get("confidence", 0.0)),
    )

    return {
        "stick_orientation_deg": float(orientation),
        "blade_angle_deg": float(blade_angle),
        "blade_state": blade_state,
        "confidence": confidence,
        "shaft_center": [float(shaft_center[0]), float(shaft_center[1])],
        "blade_center": [float(blade_center[0]), float(blade_center[1])],
    }


def summarize_stick_analysis(per_frame: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stick analysis across video."""
    angles = [
        f["blade_angle_deg"]
        for f in per_frame
        if f.get("blade_angle_deg") is not None
    ]
    states = [f["blade_state"] for f in per_frame if f.get("blade_state") != "unknown"]

    if not angles:
        return {
            "median_blade_angle_deg": None,
            "blade_state": "unknown",
        }

    angles_sorted = sorted(angles)
    median_angle = angles_sorted[len(angles_sorted) // 2]

    state_counts: dict[str, int] = {}
    for s in states:
        state_counts[s] = state_counts.get(s, 0) + 1
    dominant_state = (
        max(state_counts.items(), key=lambda kv: kv[1])[0] if state_counts else "unknown"
    )

    return {
        "median_blade_angle_deg": float(median_angle),
        "blade_state": dominant_state,
    }

