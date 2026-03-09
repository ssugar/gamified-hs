#!/usr/bin/env python3
"""Hockey Shot Analyzer - Analyze wrist shot mechanics from video.

Usage:
    python main.py --video input.mp4 --output analyzed.mp4
    python main.py --video input.mp4 --output analyzed.mp4 --json results.json
"""

import argparse
import json
import sys
import os
import cv2

# Add analyzer directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose_detector import PoseDetector
from shot_detector import detect_shot_release, detect_shooting_side
from mechanics_analyzer import analyze_mechanics
from scoring import compute_shot_score
from visualization import generate_annotated_video
from puck_detection import YoloPuckStickDetector, run_detection_on_video
from stick_detection import analyze_stick_frame, summarize_stick_analysis
from puck_tracking import track_puck_records
from trajectory_analysis import refine_release_frame, analyze_post_release_trajectory
from power_estimation import estimate_shot_power
from dataset_export import export_candidate_frames
from utils import get_keypoint


def analyze_video(
    video_path: str,
    output_path: str = None,
    json_path: str = None,
    model_path: str = None,
    yolo_model_path: str = None,
    shooting_side: str = None,
    skip_frames: int = 0,
    roi_radius: int = 200,
    calibration_m_per_pixel: float = 0.0025,
    export_dataset_candidates: bool = False,
    dataset_dir: str = "dataset_candidates",
) -> dict:
    """Run full analysis pipeline on a video.

    Args:
        video_path: Path to input video
        output_path: Path for annotated output video (optional)
        json_path: Path to save JSON results (optional)
        model_path: Path to MoveNet model (optional, auto-downloads)
        shooting_side: 'left' or 'right' (optional, auto-detects)

    Returns:
        Analysis results dict
    """
    print(f"\n{'='*50}")
    print(f"Hockey Shot Analyzer")
    print(f"{'='*50}")
    print(f"Input: {video_path}")
    print()

    # Step 1: Pose Detection
    print("[1/8] Detecting poses...")
    detector = PoseDetector(model_path)
    keypoints_seq, fps, frame_size = detector.detect_video(
        video_path, skip_frames=skip_frames
    )

    if not keypoints_seq:
        print("ERROR: No frames processed")
        return {'error': 'No frames processed'}

    frame_stride = infer_effective_frame_stride(
        video_path=video_path,
        processed_frames=len(keypoints_seq),
        requested_skip=skip_frames,
    )

    # Step 2: Detect shooting side
    if not shooting_side:
        print("[2/8] Detecting shooting side...")
        shooting_side = detect_shooting_side(keypoints_seq)
        print(f"  Detected: {shooting_side}-handed shot")
    else:
        print(f"[2/8] Using specified shooting side: {shooting_side}")

    # Step 3: Detect shot release
    print("[3/8] Detecting shot release...")
    shot_info = detect_shot_release(keypoints_seq, fps, shooting_side)
    shot_info["raw_release_frame"] = int(shot_info["release_frame"] * frame_stride)
    shot_info["raw_shot_start_frame"] = int(shot_info["shot_start_frame"] * frame_stride)
    print(f"  Shot start frame: {shot_info['shot_start_frame']}")
    print(f"  Release frame: {shot_info['release_frame']}")
    print(f"  Detection confidence: {shot_info['confidence']:.2f}")

    # Step 4: Analyze mechanics
    print("[4/8] Analyzing mechanics...")
    mechanics = analyze_mechanics(keypoints_seq, shot_info, fps, shooting_side)

    # Step 5: Score
    print("[5/8] Computing score...")
    score_result = compute_shot_score(mechanics)

    # Step 6: V3 puck/stick analysis
    print("[6/8] Running V3 puck/stick analysis...")
    v3 = run_v3_analysis(
        video_path=video_path,
        keypoints_seq=keypoints_seq,
        shot_info=shot_info,
        mechanics=mechanics,
        fps=fps,
        frame_stride=frame_stride,
        yolo_model_path=yolo_model_path,
        roi_radius=roi_radius,
        calibration_m_per_pixel=calibration_m_per_pixel,
        shooting_side=shooting_side,
    )
    if v3.get("enabled"):
        print(f"  Refined release frame: {v3['release']['release_frame']}")
        print(f"  Estimated shot speed: {v3['power']['estimated_shot_speed_mph']:.1f} mph")
        print(f"  Launch angle: {v3['trajectory']['launch_angle_deg']:.1f}°")
    else:
        print(f"  V3 skipped: {v3.get('reason', 'unknown reason')}")

    # Step 7: Dataset helper export
    dataset_export = None
    print("[7/8] Preparing dataset candidates...")
    if export_dataset_candidates:
        export_release = (
            v3.get("release", {}).get("release_frame", shot_info["release_frame"])
            if v3.get("enabled")
            else shot_info["release_frame"]
        )
        dataset_export = export_candidate_frames(
            video_path=video_path,
            release_frame=int(export_release),
            output_dir=dataset_dir,
            frame_stride=frame_stride,
        )
        print(
            f"  Exported {len(dataset_export.get('exported_frames', []))} frame candidates "
            f"to {dataset_export.get('output_dir')}"
        )
    else:
        print("  Skipped (enable with --export-dataset)")

    # Print report
    print_report(score_result, mechanics, shot_info, shooting_side, v3=v3)

    # Generate annotated video
    print("[8/8] Rendering annotated video...")
    if output_path:
        print(f"\nGenerating annotated video...")
        generate_annotated_video(
            video_path, output_path, keypoints_seq,
            shot_info, score_result, mechanics, shooting_side,
            v3_data=v3 if v3.get("enabled") else None,
            frame_stride=frame_stride,
        )

    # Build full results
    results = {
        'video': video_path,
        'fps': fps,
        'total_frames': len(keypoints_seq),
        'frame_size': list(frame_size),
        'shooting_side': shooting_side,
        'shot_info': {
            'release_frame': shot_info['release_frame'],
            'shot_start_frame': shot_info['shot_start_frame'],
            'confidence': shot_info['confidence'],
        },
        'score': score_result,
        'mechanics': {
            k: {key: val for key, val in v.items()
                 if key not in ('wrist_positions', 'hip_angular_velocities',
                                'shoulder_angular_velocities')}
            for k, v in mechanics.items()
        },
        'v3': v3,
    }
    if dataset_export is not None:
        results["dataset_export"] = dataset_export

    # Save JSON
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {json_path}")

    return results


def print_report(
    score_result: dict,
    mechanics: dict,
    shot_info: dict,
    shooting_side: str,
    v3: dict | None = None,
):
    """Print a text report of the analysis."""
    print(f"\n{'='*50}")
    print(f"  SHOT MECHANICS REPORT")
    print(f"{'='*50}")
    print(f"\n  Shot Score: {score_result['total_score']:.0f} / 100")
    print(f"  Shooting Side: {shooting_side}")
    print(f"  Confidence: {shot_info['confidence']:.0%}")
    print()

    breakdown = score_result['breakdown']
    for mechanic, data in breakdown.items():
        name = mechanic.replace('_', ' ').title()
        rating = data['rating']
        points = data['points']
        max_pts = data['max_points']

        # Rating emoji
        emoji = {'excellent': '+', 'good': '+', 'moderate': '~', 'weak': '-', 'unknown': '?'}
        indicator = emoji.get(rating, '?')

        print(f"  [{indicator}] {name:20s}  {points:5.1f}/{max_pts}  ({rating})")
        if data['detail']:
            print(f"      {data['detail']}")
    print()

    for line in score_result.get('feedback', []):
        print(f"  {line}")

    if v3 and v3.get("enabled"):
        print("\n  V3 Puck + Power")
        print("  " + "-" * 44)
        power = v3.get("power", {})
        traj = v3.get("trajectory", {})
        stick = v3.get("stick_summary", {})
        puck_start = v3.get("puck_start_position", {})
        release = v3.get("release", {})
        print(f"  Estimated Shot Speed: {power.get('estimated_shot_speed_mph', 0.0):.1f} mph")
        print(f"  Peak Puck Velocity:  {power.get('peak_puck_velocity_mph', 0.0):.1f} mph")
        print(f"  Shot Power Score:    {power.get('shot_power_score', 0.0):.1f} / 100")
        print(f"  Launch Angle:        {traj.get('launch_angle_deg', 0.0):.1f}°")
        print(f"  Puck Lift:           {traj.get('puck_lift_classification', 'unknown')}")
        print(f"  Blade Orientation:   {stick.get('blade_state', 'unknown')}")
        print(f"  Release Frame (V3):  {release.get('release_frame', shot_info.get('release_frame', 0))}")
        print(f"  Puck Start Offset:   {puck_start.get('puck_start_offset_px', 0.0):.1f} px")
        if puck_start.get("feedback"):
            print(f"  Note: {puck_start.get('feedback')}")
    print(f"\n{'='*50}")


def _infer_front_foot_x(
    keypoints_sequence: list,
    frame_index: int,
    shooting_side: str,
) -> float | None:
    if not keypoints_sequence:
        return None
    idx = max(0, min(frame_index, len(keypoints_sequence) - 1))
    kps = keypoints_sequence[idx]
    if shooting_side == "right":
        ankle = get_keypoint(kps, "left_ankle")
        knee = get_keypoint(kps, "left_knee")
    else:
        ankle = get_keypoint(kps, "right_ankle")
        knee = get_keypoint(kps, "right_knee")
    pt = ankle if ankle is not None else knee
    if pt is None:
        return None
    return float(pt[0])


def _classify_puck_start_offset(offset_px: float) -> str:
    if offset_px < -25:
        return "too_far_back"
    if offset_px > 25:
        return "too_far_forward"
    return "ideal_shooting_pocket"


def infer_effective_frame_stride(
    video_path: str,
    processed_frames: int,
    requested_skip: int,
) -> int:
    """Infer effective analysis stride from real frame count and processed output."""
    requested_stride = max(1, requested_skip + 1)
    if processed_frames <= 0:
        return requested_stride

    cap = cv2.VideoCapture(video_path)
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    if total_raw <= 0:
        return requested_stride

    inferred = max(1, int(round(total_raw / max(1, processed_frames))))
    return inferred


def run_v3_analysis(
    video_path: str,
    keypoints_seq: list,
    shot_info: dict,
    mechanics: dict,
    fps: float,
    frame_stride: int,
    yolo_model_path: str | None,
    roi_radius: int,
    calibration_m_per_pixel: float,
    shooting_side: str,
) -> dict:
    """Run additive V3 pipeline; never breaks V2 flow if unavailable."""
    if not yolo_model_path:
        return {
            "enabled": False,
            "reason": "No YOLO model provided. Pass --yolo-model to enable V3.",
        }

    try:
        detector = YoloPuckStickDetector(model_path=yolo_model_path)
        detection_records = run_detection_on_video(
            video_path=video_path,
            detector=detector,
            frame_stride=frame_stride,
            roi_radius_px=roi_radius,
            max_frames=len(keypoints_seq),
        )
    except Exception as exc:
        return {
            "enabled": False,
            "reason": f"YOLO detection failed: {exc}",
        }

    tracking = track_puck_records(detection_records, fps=fps)
    release = refine_release_frame(shot_info, tracking, detection_records)
    trajectory = analyze_post_release_trajectory(
        tracking_result=tracking,
        release_frame=int(release["release_frame"]),
    )
    power = estimate_shot_power(
        tracking_result=tracking,
        mechanics=mechanics,
        shot_info=shot_info,
        release_frame=int(release["release_frame"]),
        calibration_m_per_pixel=calibration_m_per_pixel,
        fps=fps,
    )

    stick_per_frame = [
        analyze_stick_frame(rec.get("stick_shaft"), rec.get("stick_blade"))
        for rec in detection_records
    ]
    stick_summary = summarize_stick_analysis(stick_per_frame)

    start_frame = int(shot_info.get("shot_start_frame", 0))
    start_frame = max(0, min(start_frame, len(detection_records) - 1)) if detection_records else 0
    puck_start_x = None
    if detection_records and detection_records[start_frame].get("puck"):
        puck_start_x = float(detection_records[start_frame]["puck"]["center"][0])
    front_foot_x = _infer_front_foot_x(keypoints_seq, start_frame, shooting_side)

    if puck_start_x is None or front_foot_x is None:
        puck_start = {
            "puck_start_offset_px": 0.0,
            "classification": "unknown",
            "feedback": "",
        }
    else:
        offset = float(puck_start_x - front_foot_x)
        cls = _classify_puck_start_offset(offset)
        feedback = ""
        if cls != "ideal_shooting_pocket":
            feedback = "Puck slightly ahead of front foot helps lift the puck."
        puck_start = {
            "puck_start_offset_px": offset,
            "classification": cls,
            "feedback": feedback,
        }

    return {
        "enabled": True,
        "detections": detection_records,
        "stick_per_frame": stick_per_frame,
        "tracking": tracking,
        "release": release,
        "trajectory": trajectory,
        "power": power,
        "stick_summary": stick_summary,
        "puck_start_position": puck_start,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze hockey wrist shot mechanics from video'
    )
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', help='Output annotated video path')
    parser.add_argument('--json', help='Output JSON results path')
    parser.add_argument('--model', help='Path to MoveNet model (TFLite or SavedModel)')
    parser.add_argument('--yolo-model', help='Path to custom YOLOv8 model (.pt) for puck/stick')
    parser.add_argument(
        '--side', choices=['left', 'right'],
        help='Shooting side (auto-detected if not specified)'
    )
    parser.add_argument(
        '--skip-frames', type=int, default=0,
        help='Process every Nth frame (e.g. 4 for 240fps -> ~60fps analysis)'
    )
    parser.add_argument(
        '--roi-radius', type=int, default=200,
        help='ROI radius in pixels around detected stick blade for puck detection'
    )
    parser.add_argument(
        '--calibration', type=float, default=0.0025,
        help='Calibration factor (meters per pixel) for shot speed estimation'
    )
    parser.add_argument(
        '--export-dataset', action='store_true',
        help='Export candidate frames around release to dataset folder'
    )
    parser.add_argument(
        '--dataset-dir', default='dataset_candidates',
        help='Output directory for dataset helper frame exports'
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    analyze_video(
        video_path=args.video,
        output_path=args.output,
        json_path=args.json,
        model_path=args.model,
        yolo_model_path=args.yolo_model,
        shooting_side=args.side,
        skip_frames=args.skip_frames,
        roi_radius=args.roi_radius,
        calibration_m_per_pixel=args.calibration,
        export_dataset_candidates=args.export_dataset,
        dataset_dir=args.dataset_dir,
    )


if __name__ == '__main__':
    main()
