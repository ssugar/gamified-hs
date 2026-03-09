"""Pose detection using Google MoveNet (TFLite)."""

import os
import urllib.request
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


import subprocess

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
THUNDER_URL = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
THUNDER_PATH = os.path.join(MODEL_DIR, "movenet_thunder.tflite")


def download_model() -> str:
    """Download MoveNet Thunder TFLite model if not already cached."""
    if os.path.exists(THUNDER_PATH):
        return THUNDER_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading MoveNet Thunder TFLite model...")
    urllib.request.urlretrieve(THUNDER_URL, THUNDER_PATH)
    print(f"Model saved to {THUNDER_PATH}")
    return THUNDER_PATH


def _get_video_rotation(video_path: str) -> int:
    """Get rotation metadata from video using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream_side_data=rotation',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=5
        )
        rot = result.stdout.strip()
        return int(float(rot)) if rot else 0
    except Exception:
        return 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation to frame based on metadata."""
    if rotation == -90 or rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 90 or rotation == -270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


class PoseDetector:
    """Detect body pose keypoints using MoveNet TFLite."""

    def __init__(self, model_path: str = None):
        """Initialize pose detector.

        Args:
            model_path: Path to TFLite model file. If None, downloads Thunder model.
        """
        path = model_path or THUNDER_PATH
        if not os.path.exists(path):
            path = download_model()

        self.interpreter = tflite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]
        print(f"MoveNet loaded: input size {self.input_size}x{self.input_size}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect pose keypoints in a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Array of shape (17, 3) with [y, x, confidence] for each keypoint.
            Coordinates are normalized to [0, 1].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        expected_dtype = self.input_details[0]['dtype']
        input_image = np.expand_dims(resized, axis=0).astype(expected_dtype)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Shape: (1, 1, 17, 3) -> (17, 3)
        return keypoints[0, 0, :, :]

    def detect_video(
        self, video_path: str, max_frames: int = 0, skip_frames: int = 0
    ) -> tuple:
        """Detect poses across all frames of a video.

        Args:
            video_path: Path to video file
            max_frames: Max frames to process (0 = all)
            skip_frames: Process every Nth frame (0 = every frame).
                         The effective fps is adjusted accordingly.

        Returns:
            Tuple of (list of keypoint arrays, effective_fps, frame_size)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Apply rotation metadata so pose detection sees upright frames
        rotation = _get_video_rotation(video_path)
        if rotation in (90, -90, 270, -270):
            width, height = height, width

        step = max(1, skip_frames)
        effective_fps = fps / step

        print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        if rotation:
            print(f"  Rotation metadata: {rotation}°")
        if step > 1:
            print(f"  Skipping every {step} frames -> effective {effective_fps:.1f}fps")

        all_keypoints = []
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if rotation:
                frame = _apply_rotation(frame, rotation)

            if frame_idx % step == 0:
                keypoints = self.detect(frame)
                all_keypoints.append(keypoints)
                processed += 1

                if processed % 30 == 0:
                    print(f"  Processed {processed} poses ({frame_idx}/{total_frames} frames)...")

                if max_frames > 0 and processed >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        print(f"  Done: {processed} poses from {frame_idx} frames")

        return all_keypoints, effective_fps, (width, height)
