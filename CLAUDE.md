# Hockey Shot Analyzer

Video capture and AI-powered shot mechanics analysis for hockey players.
Uses MoveNet pose detection to evaluate weight transfer, hip/shoulder rotation, shot loading, and release timing.

## Access
- Frontend: http://100.87.167.25:5176 (Tailscale)
- Backend API: http://100.87.167.25:3003
- Local dev: http://localhost:5176 / http://localhost:3003

## Quick Start
```bash
# Frontend
npm run dev

# Backend
cd server && npm run dev

# Run analyzer standalone
cd analyzer && source venv/bin/activate
python main.py --video input.mp4 --output analyzed.mp4 --json results.json
```

## Tech Stack
- Frontend: React 19 + Vite + TypeScript + Tailwind CSS 4
- Backend: Express.js + TypeScript (tsx) + multer
- Analyzer: Python 3.11 + MoveNet TFLite + OpenCV + NumPy
- Storage: JSON files in data/ + video files in data/videos/ + analyses in data/analyses/

## Project Structure
- `src/` - React frontend (Record, Videos, Analysis pages)
- `server/` - Express backend with video upload, analysis triggering, streaming
- `analyzer/` - Python ML pipeline (pose detection, shot detection, mechanics analysis, scoring, visualization)
- `data/videos/` - Uploaded video files + metadata JSON
- `data/analyses/` - Analysis results JSON + annotated MP4 videos

## Key Files
- `server/index.ts` - All API endpoints (CRUD, video, analysis)
- `analyzer/main.py` - CLI entry point for shot analysis
- `analyzer/pose_detector.py` - MoveNet TFLite pose detection
- `analyzer/shot_detector.py` - Shot release frame detection
- `analyzer/mechanics_analyzer.py` - Weight transfer, rotation, loading, timing analysis
- `analyzer/scoring.py` - 0-100 scoring with weighted metrics
- `analyzer/visualization.py` - Annotated video generation with skeleton overlay

## Services (user-level systemd)
```bash
systemctl --user start/stop/restart gamified-hs-frontend
systemctl --user start/stop/restart gamified-hs-backend
journalctl --user -u gamified-hs-backend -f
```

## Ports
- Frontend: 5176
- Backend: 3003
