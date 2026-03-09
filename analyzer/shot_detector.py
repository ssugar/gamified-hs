"""Detect shot release frame from pose keypoint sequences."""

import numpy as np
from utils import get_keypoint, velocity, smooth


def detect_shot_release(
    keypoints_sequence: list[np.ndarray],
    fps: float,
    shooting_side: str = 'right'
) -> dict:
    """Detect the shot release frame using acceleration spike on combined signal.

    Uses wrist + 0.5*elbow velocity as release signal.
    Detects the first frame where this signal exceeds 65% of its max,
    which corresponds to the acceleration spike before peak velocity.
    """
    wrist_name = f'{shooting_side}_wrist'
    elbow_name = f'{shooting_side}_elbow'

    # Extract wrist and elbow positions across frames
    wrist_positions = []
    elbow_positions = []
    for kps in keypoints_sequence:
        wpos = get_keypoint(kps, wrist_name, confidence_threshold=0.1)
        epos = get_keypoint(kps, elbow_name, confidence_threshold=0.1)
        wrist_positions.append(
            wpos if wpos is not None else
            (wrist_positions[-1] if wrist_positions else np.array([0.0, 0.0]))
        )
        elbow_positions.append(
            epos if epos is not None else
            (elbow_positions[-1] if elbow_positions else np.array([0.0, 0.0]))
        )

    if len(wrist_positions) < 5:
        return {
            'release_frame': len(keypoints_sequence) // 2,
            'shot_start_frame': 0,
            'wrist_velocities': [0.0],
            'release_signal': [0.0],
            'confidence': 0.0,
        }

    # Compute velocities
    wrist_vel = velocity(wrist_positions, fps)
    elbow_vel = velocity(elbow_positions, fps)
    wrist_vel_smooth = smooth(wrist_vel, window=5)
    elbow_vel_smooth = smooth(elbow_vel, window=5)

    # Combined release signal: wrist + 0.5 * elbow
    min_len = min(len(wrist_vel_smooth), len(elbow_vel_smooth))
    release_signal = [
        wrist_vel_smooth[i] + 0.5 * elbow_vel_smooth[i]
        for i in range(min_len)
    ]
    release_signal_smooth = smooth(release_signal, window=3)

    # Find release: first prominent local peak in the signal.
    # The global max is often the follow-through arm swing, not the release.
    # A peak is "significant" if it meets BOTH conditions:
    #   1. Prominence: rises at least 3x above the minimum in the preceding 1s
    #   2. Absolute: signal is at least 25% of the global max
    # This avoids both: (a) small noise peaks with high prominence due to
    # quiet preceding windows, and (b) missing the release when the
    # follow-through spike dominates the global max.
    max_signal = max(release_signal_smooth) if release_signal_smooth else 0
    min_threshold = 0.25 * max_signal
    lookback = max(1, int(fps))  # 1 second lookback window

    release_frame = int(np.argmax(release_signal_smooth)) + 1  # fallback: global peak
    n = len(release_signal_smooth)
    for i in range(1, n - 1):
        is_local_peak = (release_signal_smooth[i] >= release_signal_smooth[i - 1] and
                         release_signal_smooth[i] >= release_signal_smooth[i + 1])
        if is_local_peak and release_signal_smooth[i] >= min_threshold:
            # Check prominence: peak must rise 3x above preceding valley
            window_start = max(0, i - lookback)
            preceding_min = min(release_signal_smooth[window_start:i + 1])
            prominence = release_signal_smooth[i] / (preceding_min + 1e-8)
            if prominence >= 3.0:
                # Puck leaves blade after peak wrist velocity due to stick
                # flex unloading (~0.1s delay). Scale offset by fps.
                release_offset = max(2, round(fps * 0.1))
                release_frame = min(i + release_offset, len(keypoints_sequence) - 1)
                break

    release_frame = min(release_frame, len(keypoints_sequence) - 1)

    # Shot start = local minimum before release
    shot_start_frame = 0
    min_val = float('inf')
    search_start = max(0, release_frame - int(fps * 1.5))
    for i in range(release_frame, search_start, -1):
        idx = i - 1
        if 0 <= idx < len(release_signal_smooth):
            if release_signal_smooth[idx] < min_val:
                min_val = release_signal_smooth[idx]
                shot_start_frame = i

    # Confidence
    mean_signal = np.mean(release_signal_smooth) if release_signal_smooth else 0
    confidence = min(1.0, (max_signal / (mean_signal + 1e-8)) / 5.0)

    return {
        'release_frame': release_frame,
        'shot_start_frame': shot_start_frame,
        'wrist_velocities': wrist_vel_smooth,
        'elbow_velocities': elbow_vel_smooth,
        'release_signal': release_signal_smooth,
        'wrist_positions': wrist_positions,
        'confidence': float(confidence),
    }


def detect_shooting_side(keypoints_sequence: list[np.ndarray]) -> str:
    """Auto-detect which side the player shoots from.

    Looks at which wrist has more lateral movement relative to the body.
    """
    left_range = 0.0
    right_range = 0.0

    left_xs = []
    right_xs = []

    for kps in keypoints_sequence:
        lw = get_keypoint(kps, 'left_wrist')
        rw = get_keypoint(kps, 'right_wrist')
        if lw is not None:
            left_xs.append(lw[0])
        if rw is not None:
            right_xs.append(rw[0])

    if left_xs:
        left_range = max(left_xs) - min(left_xs)
    if right_xs:
        right_range = max(right_xs) - min(right_xs)

    # The shooting hand (bottom hand) typically has more movement
    return 'left' if left_range > right_range else 'right'
