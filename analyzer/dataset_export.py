"""Dataset helper: export frame candidates around release."""

from __future__ import annotations

from pathlib import Path

import cv2


def export_candidate_frames(
    video_path: str,
    release_frame: int,
    output_dir: str = "dataset_candidates",
    offsets: list[int] | None = None,
    frame_stride: int = 1,
) -> dict:
    """Export release-adjacent frames for fast YOLO labeling.

    Default exports: release-5, release-3, release, release+2.
    """
    if offsets is None:
        offsets = [-5, -3, 0, 2]

    if frame_stride < 1:
        frame_stride = 1

    release_raw = int(release_frame) * frame_stride
    target_raw_frames = sorted(set(max(0, release_raw + off * frame_stride) for off in offsets))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    exported: list[dict] = []
    wanted = set(target_raw_frames)
    raw_idx = -1

    stem = Path(video_path).stem
    while wanted:
        ret, frame = cap.read()
        if not ret:
            break
        raw_idx += 1
        if raw_idx not in wanted:
            continue

        rel = (raw_idx - release_raw) // frame_stride if frame_stride else (raw_idx - release_raw)
        rel_tag = f"r{rel:+d}"
        out_file = out_path / f"{stem}_{rel_tag}_raw{raw_idx}.jpg"
        cv2.imwrite(str(out_file), frame)
        exported.append(
            {
                "raw_frame_index": raw_idx,
                "relative_to_release": int(rel),
                "path": str(out_file),
            }
        )
        wanted.remove(raw_idx)

    cap.release()

    return {
        "output_dir": str(out_path),
        "release_frame": int(release_frame),
        "release_raw_frame": int(release_raw),
        "exported_frames": exported,
    }
