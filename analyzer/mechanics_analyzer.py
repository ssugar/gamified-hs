"""Analyze shot mechanics from pose keypoint sequences."""

import numpy as np
from utils import (
    get_keypoint, angle_between_points, line_angle,
    angular_difference, distance, midpoint
)


def find_reliable_end_frame(
    keypoints_sequence: list[np.ndarray],
    start_frame: int,
    release_frame: int
) -> int:
    """Find the last frame with reliable keypoint tracking.

    MoveNet often degrades in the last few frames before release due to
    motion blur. We detect this by checking shoulder width — when it drops
    significantly from the median, tracking has failed.

    Returns the last reliable frame index.
    """
    shoulder_widths = []
    frame_indices = []

    for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
        kps = keypoints_sequence[i]
        ls = get_keypoint(kps, 'left_shoulder')
        rs = get_keypoint(kps, 'right_shoulder')
        if ls is not None and rs is not None:
            shoulder_widths.append(distance(ls, rs))
            frame_indices.append(i)

    if len(shoulder_widths) < 3:
        return release_frame

    median_sw = float(np.median(shoulder_widths))

    # Walk backwards from release to find last frame where shoulder width
    # is within 70% of median (i.e., tracking is still reasonable).
    # Walking backwards avoids a single noisy frame at the end sneaking through.
    reliable_end = frame_indices[0]  # fallback: first frame
    for i in range(len(frame_indices) - 1, -1, -1):
        if shoulder_widths[i] >= median_sw * 0.7:
            reliable_end = frame_indices[i]
            break

    return reliable_end


def median_shoulder_width(
    keypoints_sequence: list[np.ndarray],
    start_frame: int,
    end_frame: int
) -> float:
    """Compute median shoulder width over a frame range. Robust to outliers."""
    widths = []
    for i in range(start_frame, min(end_frame + 1, len(keypoints_sequence))):
        kps = keypoints_sequence[i]
        ls = get_keypoint(kps, 'left_shoulder')
        rs = get_keypoint(kps, 'right_shoulder')
        if ls is not None and rs is not None:
            widths.append(distance(ls, rs))
    return float(np.median(widths)) if widths else 0.0


def analyze_mechanics(
    keypoints_sequence: list[np.ndarray],
    shot_info: dict,
    fps: float,
    shooting_side: str = 'right'
) -> dict:
    """Analyze shot mechanics for key phases."""
    release = shot_info['release_frame']
    start = shot_info['shot_start_frame']

    # Ensure we have valid frame range
    if start >= release:
        start = max(0, release - int(fps * 0.5))

    # Find the last frame with reliable tracking
    reliable_end = find_reliable_end_frame(keypoints_sequence, start, release)

    results = {
        'weight_transfer': analyze_weight_transfer(
            keypoints_sequence, start, reliable_end
        ),
        'hip_rotation': analyze_hip_rotation(
            keypoints_sequence, start, reliable_end, shooting_side
        ),
        'shoulder_rotation': analyze_shoulder_rotation(
            keypoints_sequence, start, reliable_end, shooting_side
        ),
        'shot_loading': analyze_shot_loading(
            keypoints_sequence, start, release, shooting_side
        ),
        'timing_sequence': analyze_timing_sequence(
            keypoints_sequence, start, reliable_end, fps, shooting_side
        ),
        'hand_separation': analyze_hand_separation(
            keypoints_sequence, start, reliable_end, shooting_side
        ),
    }

    return results


def analyze_weight_transfer(
    keypoints_sequence: list[np.ndarray],
    start_frame: int,
    release_frame: int
) -> dict:
    """Measure forward hip movement normalized to shoulder width.

    Uses total displacement (both X and Y) to handle any camera angle.
    hip_shift_ratio = hip_displacement / shoulder_width
    """
    hip_positions = []

    for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
        kps = keypoints_sequence[i]
        lh = get_keypoint(kps, 'left_hip')
        rh = get_keypoint(kps, 'right_hip')
        if lh is not None and rh is not None:
            hip_positions.append(midpoint(lh, rh))

    sw = median_shoulder_width(keypoints_sequence, start_frame, release_frame)

    if len(hip_positions) < 2 or sw == 0:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Could not track hip positions',
            'hip_shift': 0.0,
            'hip_shift_ratio': 0.0,
        }

    # Total displacement using both X and Y
    hip_displacement = distance(hip_positions[-1], hip_positions[0])
    hip_shift_ratio = hip_displacement / (sw + 1e-8)

    if hip_shift_ratio >= 0.15:
        score = 1.0
        rating = 'excellent'
    elif hip_shift_ratio >= 0.10:
        score = 0.75
        rating = 'good'
    elif hip_shift_ratio >= 0.05:
        score = 0.5
        rating = 'moderate'
    else:
        score = 0.25
        rating = 'weak'

    detail = f"Hip shift ratio: {hip_shift_ratio:.2f} (displacement/shoulder width)"

    return {
        'score': score,
        'rating': rating,
        'detail': detail,
        'hip_shift': float(hip_displacement),
        'hip_shift_ratio': float(hip_shift_ratio),
        'shoulder_width': float(sw),
    }


def analyze_hip_rotation(
    keypoints_sequence: list[np.ndarray],
    start_frame: int,
    release_frame: int,
    shooting_side: str
) -> dict:
    """Measure hip rotation during the shot.

    Uses two complementary signals:
    1. line_angle tilt: change in the 2D angle of the hip line (captures
       rotation visible as one hip rising/dropping relative to the other)
    2. depth rotation via hip width: when hips rotate toward/away from the
       camera, the projected distance between them shrinks. The ratio
       min_width / max_width ≈ cos(rotation_angle), so we recover the angle.

    The larger of the two signals is used as the rotation estimate, since
    from a front-facing camera depth rotation dominates, while from a side
    camera tilt dominates.
    """
    sw = median_shoulder_width(keypoints_sequence, start_frame, release_frame)
    hip_angles = []
    hip_widths = []

    for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
        kps = keypoints_sequence[i]
        lh = get_keypoint(kps, 'left_hip')
        rh = get_keypoint(kps, 'right_hip')
        ls = get_keypoint(kps, 'left_shoulder')
        rs = get_keypoint(kps, 'right_shoulder')

        # Only include frame if shoulder width is reasonable (tracking OK)
        if lh is not None and rh is not None:
            frame_sw = distance(ls, rs) if ls is not None and rs is not None else sw
            if frame_sw >= sw * 0.7:
                hip_angles.append(line_angle(lh, rh))
                hip_widths.append(distance(lh, rh))

    if len(hip_angles) < 2:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Could not track hip rotation',
            'rotation_degrees': 0.0,
        }

    # Signal 1: 2D tilt rotation from line_angle (handles ±180° wrapping)
    tilt_rotation = abs(angular_difference(hip_angles[0], hip_angles[-1]))

    # Signal 2: depth rotation from hip width change
    # Use median of first/last few frames for robustness against single-frame noise
    n_avg = min(3, len(hip_widths) // 2)
    if n_avg >= 1:
        start_width = float(np.median(hip_widths[:n_avg]))
        end_width = float(np.median(hip_widths[-n_avg:]))
        max_width = max(start_width, end_width)
        min_width = min(start_width, end_width)
        if max_width > 0:
            ratio = np.clip(min_width / max_width, 0.0, 1.0)
            depth_rotation = float(np.degrees(np.arccos(ratio)))
        else:
            depth_rotation = 0.0
    else:
        depth_rotation = 0.0

    rotation = max(tilt_rotation, depth_rotation)

    if rotation >= 15:
        score = 1.0
        rating = 'excellent'
    elif rotation >= 10:
        score = 0.75
        rating = 'good'
    elif rotation >= 5:
        score = 0.5
        rating = 'moderate'
    else:
        score = 0.25
        rating = 'weak'

    detail = f"Hips rotated {rotation:.1f} degrees during shot"
    if score < 0.5:
        detail += " - focus on rotating hips toward the target"

    return {
        'score': score,
        'rating': rating,
        'detail': detail,
        'rotation_degrees': float(rotation),
    }


def analyze_shoulder_rotation(
    keypoints_sequence: list[np.ndarray],
    start_frame: int,
    release_frame: int,
    shooting_side: str
) -> dict:
    """Measure shoulder rotation during the shot.

    Uses both 2D tilt and depth rotation (via shoulder width change),
    same approach as hip rotation analysis.
    """
    sw = median_shoulder_width(keypoints_sequence, start_frame, release_frame)
    shoulder_angles = []
    shoulder_widths = []

    for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
        kps = keypoints_sequence[i]
        ls = get_keypoint(kps, 'left_shoulder')
        rs = get_keypoint(kps, 'right_shoulder')
        if ls is not None and rs is not None:
            frame_sw = distance(ls, rs)
            if frame_sw >= sw * 0.7:
                shoulder_angles.append(line_angle(ls, rs))
                shoulder_widths.append(frame_sw)

    if len(shoulder_angles) < 2:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Could not track shoulder rotation',
            'rotation_degrees': 0.0,
        }

    # Signal 1: 2D tilt rotation (handles ±180° wrapping)
    tilt_rotation = abs(angular_difference(shoulder_angles[0], shoulder_angles[-1]))

    # Signal 2: depth rotation from shoulder width change
    n_avg = min(3, len(shoulder_widths) // 2)
    if n_avg >= 1:
        start_width = float(np.median(shoulder_widths[:n_avg]))
        end_width = float(np.median(shoulder_widths[-n_avg:]))
        max_width = max(start_width, end_width)
        min_width = min(start_width, end_width)
        if max_width > 0:
            ratio = np.clip(min_width / max_width, 0.0, 1.0)
            depth_rotation = float(np.degrees(np.arccos(ratio)))
        else:
            depth_rotation = 0.0
    else:
        depth_rotation = 0.0

    rotation = max(tilt_rotation, depth_rotation)

    if rotation >= 20:
        score = 1.0
        rating = 'excellent'
    elif rotation >= 12:
        score = 0.75
        rating = 'good'
    elif rotation >= 6:
        score = 0.5
        rating = 'moderate'
    else:
        score = 0.25
        rating = 'weak'

    detail = f"Shoulders rotated {rotation:.1f} degrees during shot"
    if score < 0.5:
        detail += " - rotate shoulders to generate more power"

    return {
        'score': score,
        'rating': rating,
        'detail': detail,
        'rotation_degrees': float(rotation),
    }


def analyze_shot_loading(
    keypoints_sequence: list[np.ndarray],
    start_frame: int,
    release_frame: int,
    shooting_side: str
) -> dict:
    """Check if the shooting wrist loads behind the hip before release.

    Computes "behind" relative to the estimated shot path (2D projection),
    rather than raw image X. This is more robust across camera angles.
    """
    wrist_name = f'{shooting_side}_wrist'
    hip_name = f'{shooting_side}_hip'

    # Estimate shot direction from early->late shooting wrist trajectory.
    wrist_positions = []
    for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
        w = get_keypoint(keypoints_sequence[i], wrist_name)
        if w is not None:
            wrist_positions.append(w)

    sw = median_shoulder_width(keypoints_sequence, start_frame, release_frame)
    if len(wrist_positions) < 2 or sw == 0:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Could not track shot loading',
            'wrist_behind_frames': 0,
            'max_behind_distance': 0.0,
            'max_behind_ratio': 0.0,
        }

    n_avg = min(3, len(wrist_positions) // 2)
    start_wrist = np.median(np.array(wrist_positions[:n_avg]), axis=0)
    end_wrist = np.median(np.array(wrist_positions[-n_avg:]), axis=0)
    shot_vec = end_wrist - start_wrist
    shot_vec_norm = float(np.linalg.norm(shot_vec))
    if shot_vec_norm < 1e-8:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Could not estimate shot direction',
            'wrist_behind_frames': 0,
            'max_behind_distance': 0.0,
            'max_behind_ratio': 0.0,
        }
    shot_dir = shot_vec / shot_vec_norm

    wrist_behind_count = 0
    total_checked = 0
    max_behind_distance = 0.0
    max_behind_ratio = 0.0

    for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
        kps = keypoints_sequence[i]
        wrist = get_keypoint(kps, wrist_name)
        hip = get_keypoint(kps, hip_name)

        if wrist is not None and hip is not None:
            total_checked += 1
            behind_distance = float(np.dot(hip - wrist, shot_dir))
            behind_ratio = behind_distance / (sw + 1e-8)
            if behind_ratio > 0.02:  # ignore tiny noise-level sign flips
                wrist_behind_count += 1
                max_behind_distance = max(max_behind_distance, behind_distance)
                max_behind_ratio = max(max_behind_ratio, behind_ratio)

    if total_checked == 0:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Could not track shot loading',
            'wrist_behind_frames': 0,
            'max_behind_distance': 0.0,
            'max_behind_ratio': 0.0,
        }

    behind_ratio = wrist_behind_count / total_checked

    if behind_ratio >= 0.35 and max_behind_ratio >= 0.18:
        score = 1.0
        rating = 'excellent'
    elif behind_ratio >= 0.25 or max_behind_ratio >= 0.12:
        score = 0.75
        rating = 'good'
    elif behind_ratio >= 0.12 or max_behind_ratio >= 0.06:
        score = 0.5
        rating = 'moderate'
    else:
        score = 0.25
        rating = 'weak'

    detail = (
        f"Hands behind body in {wrist_behind_count}/{total_checked} frames "
        f"(max load: {max_behind_ratio:.2f}x shoulder width)"
    )
    if score < 0.5:
        detail += " - bring the puck further behind your body before shooting"

    return {
        'score': score,
        'rating': rating,
        'detail': detail,
        'wrist_behind_frames': wrist_behind_count,
        'max_behind_distance': float(max_behind_distance),
        'max_behind_ratio': float(max_behind_ratio),
    }


def _width_to_angular_vel(widths: list[float]) -> list[float]:
    """Convert a series of projected widths to angular velocity (deg/frame).

    Width change reflects depth rotation: width = W*cos(θ), so
    Δθ ≈ arccos(w2/w1) per frame. We combine this with tilt for a
    better total rotation rate estimate.
    """
    if len(widths) < 2:
        return [0.0]
    vels = []
    for i in range(1, len(widths)):
        if widths[i-1] > 0:
            ratio = np.clip(widths[i] / widths[i-1], 0.0, 1.0) if widths[i] <= widths[i-1] else np.clip(widths[i-1] / widths[i], 0.0, 1.0)
            vels.append(float(np.degrees(np.arccos(ratio))))
        else:
            vels.append(0.0)
    return vels


def analyze_timing_sequence(
    keypoints_sequence: list[np.ndarray],
    start_frame: int,
    release_frame: int,
    fps: float,
    shooting_side: str
) -> dict:
    """Evaluate kinetic chain timing: hips -> shoulders -> wrists.

    Tracks angular velocity of hips and shoulders using both 2D tilt
    and depth rotation (width change) signals.
    Filters out frames with unreliable tracking to avoid false spikes.
    """
    sw = median_shoulder_width(keypoints_sequence, start_frame, release_frame)
    hip_angles = []
    hip_widths = []
    shoulder_angles = []
    shoulder_widths = []

    for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
        kps = keypoints_sequence[i]
        lh = get_keypoint(kps, 'left_hip')
        rh = get_keypoint(kps, 'right_hip')
        ls = get_keypoint(kps, 'left_shoulder')
        rs = get_keypoint(kps, 'right_shoulder')

        # Check tracking quality via shoulder width
        frame_sw = distance(ls, rs) if ls is not None and rs is not None else 0
        reliable = frame_sw >= sw * 0.7

        if lh is not None and rh is not None and reliable:
            hip_angles.append(line_angle(lh, rh))
            hip_widths.append(distance(lh, rh))
        else:
            hip_angles.append(hip_angles[-1] if hip_angles else 0)
            hip_widths.append(hip_widths[-1] if hip_widths else 0)

        if ls is not None and rs is not None and reliable:
            shoulder_angles.append(line_angle(ls, rs))
            shoulder_widths.append(frame_sw)
        else:
            shoulder_angles.append(shoulder_angles[-1] if shoulder_angles else 0)
            shoulder_widths.append(shoulder_widths[-1] if shoulder_widths else 0)

    if len(hip_angles) < 3 or len(shoulder_angles) < 3:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Not enough data for timing analysis',
            'sequence_correct': False,
            'arms_only': False,
        }

    # Angular velocity: combine tilt velocity + depth rotation velocity
    hip_tilt_vel = [abs(angular_difference(hip_angles[i-1], hip_angles[i])) for i in range(1, len(hip_angles))]
    hip_depth_vel = _width_to_angular_vel(hip_widths)
    hip_angular_vel = [max(t, d) for t, d in zip(hip_tilt_vel, hip_depth_vel)]

    shoulder_tilt_vel = [abs(angular_difference(shoulder_angles[i-1], shoulder_angles[i])) for i in range(1, len(shoulder_angles))]
    shoulder_depth_vel = _width_to_angular_vel(shoulder_widths)
    shoulder_angular_vel = [max(t, d) for t, d in zip(shoulder_tilt_vel, shoulder_depth_vel)]

    hip_peak_frame = int(np.argmax(hip_angular_vel)) if hip_angular_vel else 0
    shoulder_peak_frame = int(np.argmax(shoulder_angular_vel)) if shoulder_angular_vel else 0

    # Wrist peak is at release (relative to start)
    wrist_peak_frame = release_frame - start_frame

    # Detect if hips even rotate meaningfully
    max_hip_speed = max(hip_angular_vel) if hip_angular_vel else 0
    hip_rotation_started = max_hip_speed > 0.5  # threshold: 0.5 deg/frame

    # Check kinetic chain order
    hip_first = hip_peak_frame <= shoulder_peak_frame
    shoulders_before_wrist = shoulder_peak_frame <= wrist_peak_frame

    # Arms-only detection: wrist fires before hip rotation begins
    hip_start_frame = wrist_peak_frame  # default: assume simultaneous
    for i, v in enumerate(hip_angular_vel):
        if v > 0.3:
            hip_start_frame = i
            break

    arms_only = (not hip_rotation_started) or (hip_start_frame > wrist_peak_frame * 0.8)

    if arms_only:
        score = 0.25
        rating = 'weak'
        detail = 'Arms-only shot detected - start with hips and shoulders'
    elif hip_first and shoulders_before_wrist:
        score = 1.0
        rating = 'excellent'
        detail = 'Great kinetic chain: hips lead, then shoulders, then release'
    elif shoulders_before_wrist:
        score = 0.75
        rating = 'good'
        detail = 'Shoulders fire before release, but hips could lead more'
    elif hip_first:
        score = 0.5
        rating = 'moderate'
        detail = 'Hips start first but arms release too early'
    else:
        score = 0.25
        rating = 'weak'
        detail = 'Arms fire before body rotation - let hips and shoulders lead'

    return {
        'score': score,
        'rating': rating,
        'detail': detail,
        'sequence_correct': hip_first and shoulders_before_wrist,
        'arms_only': arms_only,
        'hip_peak_frame': int(hip_peak_frame + start_frame),
        'shoulder_peak_frame': int(shoulder_peak_frame + start_frame),
        'hip_angular_velocities': hip_angular_vel,
        'shoulder_angular_velocities': shoulder_angular_vel,
    }


def analyze_hand_separation(
    keypoints_sequence: list[np.ndarray],
    start_frame: int,
    release_frame: int,
    shooting_side: str
) -> dict:
    """Measure hand separation with a camera-angle-robust composite metric.

    Uses a robust percentile (P90) instead of max and combines:
    - wrist-to-wrist separation (capped to reduce perspective inflation)
    - top-hand distance from shoulders midpoint (stick-loading proxy)
    """
    sep_ratios = []
    top_hand_ratios = []
    effective_ratios = []

    top_wrist_name = 'right_wrist' if shooting_side == 'left' else 'left_wrist'

    sw = median_shoulder_width(keypoints_sequence, start_frame, release_frame)
    if sw == 0:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Could not track hand positions',
            'max_separation': 0.0,
            'separation_ratio': 0.0,
            'top_hand_ratio': 0.0,
            'effective_ratio': 0.0,
        }

    for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
        kps = keypoints_sequence[i]
        lw = get_keypoint(kps, 'left_wrist')
        rw = get_keypoint(kps, 'right_wrist')
        ls = get_keypoint(kps, 'left_shoulder')
        rs = get_keypoint(kps, 'right_shoulder')
        top_wrist = get_keypoint(kps, top_wrist_name)
        if lw is None or rw is None or ls is None or rs is None or top_wrist is None:
            continue

        frame_sw = distance(ls, rs)
        if frame_sw < sw * 0.7:
            continue

        shoulder_mid = midpoint(ls, rs)
        sep_ratio = distance(lw, rw) / (sw + 1e-8)
        top_hand_ratio = distance(top_wrist, shoulder_mid) / (sw + 1e-8)

        # Cap raw separation so one perspective-heavy frame can't dominate.
        sep_capped = min(sep_ratio, 1.6)
        effective_ratio = 0.5 * sep_capped + 0.5 * top_hand_ratio

        sep_ratios.append(sep_ratio)
        top_hand_ratios.append(top_hand_ratio)
        effective_ratios.append(effective_ratio)

    if not effective_ratios:
        return {
            'score': 0.0,
            'rating': 'unknown',
            'detail': 'Could not track hand positions',
            'max_separation': 0.0,
            'separation_ratio': 0.0,
            'top_hand_ratio': 0.0,
            'effective_ratio': 0.0,
        }

    p90_sep_ratio = float(np.percentile(sep_ratios, 90))
    p90_top_hand_ratio = float(np.percentile(top_hand_ratios, 90))
    effective_ratio = float(np.percentile(effective_ratios, 90))

    if effective_ratio >= 1.1:
        score = 1.0
        rating = 'excellent'
    elif effective_ratio >= 0.9:
        score = 0.75
        rating = 'good'
    elif effective_ratio >= 0.7:
        score = 0.5
        rating = 'moderate'
    else:
        score = 0.25
        rating = 'weak'

    detail = (
        f"Hand separation index: {effective_ratio:.2f} "
        f"(P90 wrists: {p90_sep_ratio:.2f}x, top hand: {p90_top_hand_ratio:.2f}x)"
    )

    return {
        'score': score,
        'rating': rating,
        'detail': detail,
        'max_separation': float(max(sep_ratios) * sw),
        'separation_ratio': float(p90_sep_ratio),
        'top_hand_ratio': float(p90_top_hand_ratio),
        'effective_ratio': float(effective_ratio),
    }
