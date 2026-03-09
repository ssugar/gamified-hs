"""Visualization - annotated video with pose overlays."""

import cv2
import numpy as np
from utils import get_keypoint, KEYPOINT_NAMES, midpoint


# Skeleton connections for drawing
SKELETON_EDGES = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]

# Colors (BGR)
COLOR_SKELETON = (0, 255, 200)  # Cyan-green
COLOR_JOINT = (0, 200, 255)     # Orange
COLOR_HIP_LINE = (0, 0, 255)    # Red
COLOR_SHOULDER_LINE = (255, 100, 0)  # Blue
COLOR_RELEASE = (0, 255, 0)     # Green
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)


def to_pixel(normalized_point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    """Convert normalized [x, y] to pixel coordinates."""
    return (int(normalized_point[0] * width), int(normalized_point[1] * height))


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    confidence_threshold: float = 0.2
) -> np.ndarray:
    """Draw pose skeleton on frame."""
    h, w = frame.shape[:2]

    # Draw edges
    for start_name, end_name in SKELETON_EDGES:
        start = get_keypoint(keypoints, start_name, confidence_threshold)
        end = get_keypoint(keypoints, end_name, confidence_threshold)
        if start is not None and end is not None:
            p1 = to_pixel(start, w, h)
            p2 = to_pixel(end, w, h)
            cv2.line(frame, p1, p2, COLOR_SKELETON, 2, cv2.LINE_AA)

    # Draw joints
    for name in KEYPOINT_NAMES:
        pt = get_keypoint(keypoints, name, confidence_threshold)
        if pt is not None:
            px = to_pixel(pt, w, h)
            cv2.circle(frame, px, 4, COLOR_JOINT, -1, cv2.LINE_AA)

    return frame


def draw_hip_shoulder_lines(
    frame: np.ndarray,
    keypoints: np.ndarray
) -> np.ndarray:
    """Draw hip and shoulder lines with angle labels."""
    h, w = frame.shape[:2]

    lh = get_keypoint(keypoints, 'left_hip')
    rh = get_keypoint(keypoints, 'right_hip')
    ls = get_keypoint(keypoints, 'left_shoulder')
    rs = get_keypoint(keypoints, 'right_shoulder')

    if lh is not None and rh is not None:
        p1 = to_pixel(lh, w, h)
        p2 = to_pixel(rh, w, h)
        cv2.line(frame, p1, p2, COLOR_HIP_LINE, 3, cv2.LINE_AA)

    if ls is not None and rs is not None:
        p1 = to_pixel(ls, w, h)
        p2 = to_pixel(rs, w, h)
        cv2.line(frame, p1, p2, COLOR_SHOULDER_LINE, 3, cv2.LINE_AA)

    return frame


def draw_hip_ghost(
    frame: np.ndarray,
    keypoints: np.ndarray,
    start_keypoints: np.ndarray,
    frame_idx: int,
    shot_start: int,
    release_frame: int
) -> np.ndarray:
    """Draw ghost marker showing hip position at shot start + movement arrow."""
    if frame_idx < shot_start or frame_idx > release_frame + int(release_frame * 0.3):
        return frame

    h, w = frame.shape[:2]

    # Ghost hip position from shot start
    lh_start = get_keypoint(start_keypoints, 'left_hip')
    rh_start = get_keypoint(start_keypoints, 'right_hip')

    if lh_start is not None and rh_start is not None:
        ghost_mid = midpoint(lh_start, rh_start)
        ghost_px = to_pixel(ghost_mid, w, h)

        # Draw ghost marker (hollow circle)
        cv2.circle(frame, ghost_px, 10, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "START", (ghost_px[0] - 20, ghost_px[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # Current hip position
        lh = get_keypoint(keypoints, 'left_hip')
        rh = get_keypoint(keypoints, 'right_hip')
        if lh is not None and rh is not None:
            current_mid = midpoint(lh, rh)
            current_px = to_pixel(current_mid, w, h)

            # Arrow from ghost to current
            cv2.arrowedLine(frame, ghost_px, current_px, (0, 100, 255), 2,
                           cv2.LINE_AA, tipLength=0.3)

    return frame


def draw_metrics_overlay(
    frame: np.ndarray,
    frame_idx: int,
    shot_info: dict,
    score_result: dict | None = None,
    mechanics: dict | None = None,
    shooting_side: str = 'right'
) -> np.ndarray:
    """Draw metrics text overlay on frame.

    Phase banner goes top-left. Score overlay goes in the top corner
    opposite the shooting side so it doesn't obscure the shot action.
    """
    h, w = frame.shape[:2]
    line_height = 22

    # Frame info
    release = shot_info.get('raw_release_frame', shot_info.get('release_frame', -1))
    start = shot_info.get('raw_shot_start_frame', shot_info.get('shot_start_frame', -1))

    # Phase indicator
    if frame_idx < start:
        phase = "PRE-SHOT"
        phase_color = (150, 150, 150)
    elif frame_idx < release:
        phase = "LOADING"
        phase_color = (0, 200, 255)
    elif frame_idx == release:
        phase = "RELEASE"
        phase_color = (0, 255, 0)
    else:
        phase = "FOLLOW-THROUGH"
        phase_color = (200, 200, 0)

    # Draw phase banner (top-left)
    cv2.rectangle(frame, (0, 0), (280, 35), COLOR_TEXT_BG, -1)
    cv2.putText(frame, f"{phase}  [frame {frame_idx}]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2, cv2.LINE_AA)

    # Draw release frame marker
    if frame_idx == release:
        cv2.rectangle(frame, (0, 0), (w, h), COLOR_RELEASE, 4)
        cv2.putText(frame, "RELEASE!", (w // 2 - 60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_RELEASE, 3, cv2.LINE_AA)

    # Draw score overlay in top corner opposite shooting side
    if score_result and frame_idx >= release:
        score = score_result.get('total_score', 0)
        breakdown = score_result.get('breakdown', {})

        box_w = 280
        box_h = 25 + (len(breakdown) + 1) * line_height
        # Place opposite the shooting side so it doesn't cover the action
        if shooting_side == 'right':
            x_start = 0
        else:
            x_start = w - box_w
        y_start = 40  # below the phase banner

        cv2.rectangle(frame, (x_start, y_start), (x_start + box_w, y_start + box_h),
                      (0, 0, 0, 180), -1)

        x_text = x_start + 10
        y = y_start + 5

        cv2.putText(frame, f"Shot Score: {score:.0f}/100", (x_text, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y += line_height + 5

        for mechanic, data in breakdown.items():
            name = mechanic.replace('_', ' ').title()[:15]
            rating = data.get('rating', '?')
            pts = data.get('points', 0)
            max_pts = data.get('max_points', 0)

            # Color by rating
            if rating == 'excellent':
                color = (0, 255, 0)
            elif rating == 'good':
                color = (0, 200, 100)
            elif rating == 'moderate':
                color = (0, 200, 255)
            else:
                color = (0, 100, 255)

            text = f"{name}: {pts:.0f}/{max_pts}"
            cv2.putText(frame, text, (x_text, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += line_height

    return frame


def draw_v3_overlays(
    frame: np.ndarray,
    frame_idx: int,
    v3_data: dict | None = None,
) -> np.ndarray:
    """Draw V3 object detections, trajectory, and speed/launch overlays."""
    if not v3_data or not v3_data.get("enabled"):
        return frame

    detections = v3_data.get("detections", [])
    by_raw = {int(d.get("raw_frame_index", -1)): d for d in detections}
    current = by_raw.get(frame_idx)

    # Draw puck/stick bounding boxes.
    if current:
        puck = current.get("puck")
        shaft = current.get("stick_shaft")
        blade = current.get("stick_blade")
        for det, label, color in [
            (puck, "puck", (0, 255, 255)),
            (shaft, "stick_shaft", (255, 180, 0)),
            (blade, "stick_blade", (0, 120, 255)),
        ]:
            if not det:
                continue
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {det.get('confidence', 0.0):.2f}",
                (x1, max(12, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    # Draw trajectory path from tracked puck centers.
    tracking = v3_data.get("tracking", {})
    pts = []
    for row in tracking.get("per_frame", []):
        rfi = int(row.get("raw_frame_index", -1))
        if rfi <= frame_idx:
            pts.append((int(round(row["puck_x"])), int(round(row["puck_y"]))))
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), 2, cv2.LINE_AA)

    # Overlay speed/lift info.
    power = v3_data.get("power", {})
    traj = v3_data.get("trajectory", {})
    release = v3_data.get("release", {})
    rel_idx = int(release.get("release_frame", -1))
    rel_raw = rel_idx
    if detections and 0 <= rel_idx < len(detections):
        rel_raw = int(detections[rel_idx].get("raw_frame_index", rel_idx))

    h, w = frame.shape[:2]
    panel_w = 290
    panel_h = 110
    x0 = w - panel_w - 10
    y0 = 10
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    lines = [
        f"Shot Speed: {power.get('estimated_shot_speed_mph', 0.0):.1f} mph",
        f"Launch Angle: {traj.get('launch_angle_deg', 0.0):.1f} deg",
        f"Puck Lift: {traj.get('puck_lift_classification', 'unknown')}",
        f"Release Frame: {rel_idx}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(
            frame,
            txt,
            (x0 + 8, y0 + 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if frame_idx == rel_raw:
        cv2.circle(frame, (w // 2, h // 2), 40, (0, 255, 0), 3, cv2.LINE_AA)

    return frame


def _get_video_rotation(video_path: str) -> int:
    """Get rotation metadata from video using ffprobe."""
    import subprocess
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


def generate_annotated_video(
    input_path: str,
    output_path: str,
    keypoints_sequence: list[np.ndarray],
    shot_info: dict,
    score_result: dict | None = None,
    mechanics: dict | None = None,
    shooting_side: str = 'right',
    v3_data: dict | None = None,
    frame_stride: int = 1,
) -> None:
    """Generate annotated output video with pose overlays.

    Args:
        input_path: Path to original video
        output_path: Path for output annotated video
        keypoints_sequence: Keypoints per frame
        shot_info: Shot detection results
        score_result: Scoring results (optional)
        mechanics: Mechanics analysis results (optional)
        shooting_side: 'left' or 'right' - score overlay goes opposite side
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Detect rotation metadata (phones often record landscape with rotation flag)
    rotation = _get_video_rotation(input_path)
    if rotation in (90, -90, 270, -270):
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    if frame_stride < 1:
        frame_stride = 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _apply_rotation(frame, rotation)

        # Pose-driven overlays are aligned to analysis-frame cadence.
        if frame_idx % frame_stride == 0:
            analysis_idx = frame_idx // frame_stride
            if analysis_idx < len(keypoints_sequence):
                kps = keypoints_sequence[analysis_idx]
                frame = draw_skeleton(frame, kps)
                frame = draw_hip_shoulder_lines(frame, kps)

                start_frame = shot_info.get('shot_start_frame', 0)
                release = shot_info.get('release_frame', 0)
                if start_frame < len(keypoints_sequence):
                    frame = draw_hip_ghost(
                        frame, kps, keypoints_sequence[start_frame],
                        analysis_idx, start_frame, release
                    )

        frame = draw_metrics_overlay(
            frame, frame_idx, shot_info, score_result, mechanics,
            shooting_side
        )
        frame = draw_v3_overlays(frame, frame_idx, v3_data=v3_data)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")
