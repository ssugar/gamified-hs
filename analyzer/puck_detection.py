"""YOLOv8-based puck and stick detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

CLASS_PUCK = 0
CLASS_STICK_SHAFT = 1
CLASS_STICK_BLADE = 2


@dataclass
class Detection:
    """Single detection record in pixel coordinates."""

    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> dict:
        return {
            "class_id": int(self.class_id),
            "confidence": float(self.confidence),
            "bbox": [int(v) for v in self.bbox],
            "center": [float(v) for v in self.center],
        }


def _clip_bbox(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


class YoloPuckStickDetector:
    """Thin wrapper around Ultralytics YOLO for puck/stick classes."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.2,
        imgsz: int = 640,
        device: str | None = None,
    ):
        if not model_path:
            raise ValueError("YOLO model path is required for V3 puck/stick analysis.")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Ultralytics is not installed. Install with `pip install ultralytics`."
            ) from exc

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

    def _predict(
        self, frame: np.ndarray, classes: list[int] | None = None
    ) -> list[Detection]:
        preds = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not preds:
            return []

        boxes = preds[0].boxes
        if boxes is None:
            return []

        detections: list[Detection] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            detections.append(
                Detection(
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def detect_frame(
        self,
        frame: np.ndarray,
        classes: list[int] | None = None,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[Detection]:
        """Detect objects in full frame or an ROI."""
        if roi is None:
            return self._predict(frame, classes=classes)

        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        if roi_frame.size == 0:
            return []

        local = self._predict(roi_frame, classes=classes)
        lifted: list[Detection] = []
        for det in local:
            bx1, by1, bx2, by2 = det.bbox
            lifted.append(
                Detection(
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                )
            )
        return lifted


def best_detection(
    detections: list[Detection], class_id: int
) -> Detection | None:
    """Get highest-confidence detection for a given class."""
    candidates = [d for d in detections if d.class_id == class_id]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)


def roi_around_detection(
    detection: Detection,
    frame_shape: tuple[int, int, int],
    radius_px: int = 200,
) -> tuple[int, int, int, int]:
    """Square ROI centered on detection center."""
    h, w = frame_shape[:2]
    cx, cy = detection.center
    x1 = int(round(cx - radius_px))
    y1 = int(round(cy - radius_px))
    x2 = int(round(cx + radius_px))
    y2 = int(round(cy + radius_px))
    return _clip_bbox(x1, y1, x2, y2, w, h)


def detect_objects_with_blade_roi(
    detector: YoloPuckStickDetector,
    frame: np.ndarray,
    roi_radius_px: int = 200,
) -> dict[str, dict[str, Any] | None]:
    """Detect stick first, then detect puck inside blade-centered ROI."""
    all_stick = detector.detect_frame(
        frame, classes=[CLASS_STICK_SHAFT, CLASS_STICK_BLADE]
    )
    blade = best_detection(all_stick, CLASS_STICK_BLADE)
    shaft = best_detection(all_stick, CLASS_STICK_SHAFT)

    puck: Detection | None = None
    roi_used: list[int] | None = None
    roi_anchor = blade if blade is not None else shaft
    if roi_anchor is not None:
        roi = roi_around_detection(roi_anchor, frame.shape, radius_px=roi_radius_px)
        roi_used = [int(v) for v in roi]
        roi_detections = detector.detect_frame(frame, classes=[CLASS_PUCK], roi=roi)
        puck = best_detection(roi_detections, CLASS_PUCK)
    else:
        fallback = detector.detect_frame(frame, classes=[CLASS_PUCK])
        puck = best_detection(fallback, CLASS_PUCK)

    return {
        "puck": puck.to_dict() if puck else None,
        "stick_blade": blade.to_dict() if blade else None,
        "stick_shaft": shaft.to_dict() if shaft else None,
        "blade_roi": roi_used,
    }


def run_detection_on_video(
    video_path: str,
    detector: YoloPuckStickDetector,
    frame_stride: int = 1,
    roi_radius_px: int = 200,
    max_frames: int = 0,
) -> list[dict[str, Any]]:
    """Run frame-wise detection and return records aligned to analysis frames."""
    if frame_stride < 1:
        frame_stride = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    records: list[dict[str, Any]] = []
    raw_idx = -1
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_idx += 1

        if raw_idx % frame_stride != 0:
            continue

        det = detect_objects_with_blade_roi(
            detector=detector,
            frame=frame,
            roi_radius_px=roi_radius_px,
        )
        records.append(
            {
                "frame_index": processed,
                "raw_frame_index": raw_idx,
                **det,
            }
        )
        processed += 1
        if max_frames > 0 and processed >= max_frames:
            break

    cap.release()
    return records
