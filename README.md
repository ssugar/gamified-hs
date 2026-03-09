# Hockey Shot Analyzer

Video capture and AI-powered shot mechanics analysis for hockey players. Record wrist shots, get instant feedback on technique, and track improvement over time.

Uses MoveNet pose detection to evaluate weight transfer, hip/shoulder rotation, shot loading, and release timing — producing a 0-100 shot score with detailed coaching feedback.

## Features

- **Video Capture** — Record shots directly from browser with shooting side selection
- **Pose Detection** — MoveNet Thunder TFLite for 17-keypoint body tracking
- **Shot Detection** — Automatic release frame detection using wrist velocity with prominence-based filtering
- **Mechanics Analysis** — Weight transfer, hip rotation, shoulder rotation, shot loading, hand separation, release timing
- **Scoring** — Weighted 0-100 scoring with per-mechanic breakdown and coaching notes
- **Annotated Video** — Output video with skeleton overlay, phase indicators, hip ghost markers, and score display
- **V3 Puck/Stick Tracking** — Optional YOLO-based puck and stick detection for shot speed estimation, launch angle, and refined release frame

## Tech Stack

- **Frontend**: React 19 + Vite + TypeScript + Tailwind CSS 4
- **Backend**: Express.js + TypeScript + multer
- **Analyzer**: Python 3.11 + MoveNet TFLite + OpenCV + NumPy
- **Storage**: JSON files + video files on disk

## Quick Start

### Prerequisites

- Node.js 22+
- Python 3.11
- ffprobe (from ffmpeg, for video metadata)

### Setup

```bash
# Install frontend dependencies
npm install

# Install backend dependencies
cd server && npm install && cd ..

# Set up Python analyzer
cd analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

The MoveNet Thunder model (~12MB) downloads automatically on first analysis run.

### Running

```bash
# Frontend (port 5176)
npm run dev

# Backend (port 3003)
cd server && npm run dev

# Or run analyzer standalone
cd analyzer && source venv/bin/activate
python main.py --video input.mp4 --output analyzed.mp4 --json results.json
```

### Analyzer CLI Options

```
--video PATH        Input video path (required)
--output PATH       Output annotated video path
--json PATH         Output JSON results path
--side left|right   Shooting side (auto-detected if not specified)
--skip-frames N     Process every Nth frame (auto-set for high-fps video)
--yolo-model PATH   Path to YOLOv8 model for V3 puck/stick detection
--roi-radius N      ROI radius in pixels for puck detection (default: 200)
--calibration F     Meters per pixel for speed estimation (default: 0.0025)
--export-dataset    Export candidate frames around release for training data
```

## Project Structure

```
src/                  React frontend
  pages/              Record, Videos, Analysis pages
  components/         Layout, navigation
  api/                Backend API client
server/               Express backend
  index.ts            API endpoints (upload, analyze, stream)
analyzer/             Python ML pipeline
  main.py             CLI entry point
  pose_detector.py    MoveNet TFLite pose detection
  shot_detector.py    Shot release frame detection
  mechanics_analyzer.py  Mechanics analysis (rotation, weight transfer, etc.)
  scoring.py          0-100 scoring with weighted metrics
  visualization.py    Annotated video generation
  puck_detection.py   YOLO-based puck/stick detection (V3)
  puck_tracking.py    Puck tracking across frames
  trajectory_analysis.py  Post-release trajectory and launch angle
  power_estimation.py Shot speed estimation
```

## How It Works

1. **Upload or record** a wrist shot video
2. **Pose detection** extracts 17 body keypoints per frame using MoveNet
3. **Shot detection** finds the release frame via wrist velocity peaks
4. **Mechanics analysis** measures rotation (2D tilt + depth via width ratio), weight transfer, loading position, and timing sequence
5. **Scoring** rates each mechanic and produces coaching feedback
6. **Annotated video** renders skeleton overlay with phase indicators and score

## Scoring Breakdown

| Mechanic | Max Points | What It Measures |
|---|---|---|
| Weight Transfer | 25 | Hip movement toward target during shot |
| Hip Rotation | 20 | Hip turn through the shot (tilt + depth) |
| Shoulder Rotation | 20 | Shoulder turn and separation from hips |
| Shot Loading | 20 | Hands drawn back before release |
| Release Timing | 15 | Coordination of hip → shoulder → wrist sequence |

## License

MIT
