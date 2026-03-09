# V3 Public Dataset Workflow

This project now supports a clean path to bootstrap V3 (`puck`, `stick_shaft`, `stick_blade`) from public YOLO datasets and then fine-tune with your own shooting-lane clips.

## 1) Download Public YOLO Datasets

Put each downloaded dataset in its own folder under:

`data/public_datasets/`

Example:

`data/public_datasets/hockeyai/`  
`data/public_datasets/player_puck_stick/`  
`data/public_datasets/hockey_only_puck/`

Each source should include a `data.yaml` and YOLO labels/images.

## 2) Merge + Remap to V3 Classes

Run:

```bash
analyzer/venv/bin/python analyzer/prepare_public_datasets.py \
  --source-root data/public_datasets \
  --output-root data/puck_v3_public \
  --fresh
```

This script:
- Detects common YOLO folder layouts.
- Remaps source classes into:
  - `0 puck`
  - `1 stick_shaft`
  - `2 stick_blade`
- Keeps background/negative images by default.
- Writes merged `data.yaml` at `data/puck_v3_public/data.yaml`.

If you want to drop negatives:

```bash
analyzer/venv/bin/python analyzer/prepare_public_datasets.py \
  --source-root data/public_datasets \
  --output-root data/puck_v3_public \
  --fresh \
  --drop-negatives
```

## 3) Train Initial V3 Model

```bash
analyzer/venv/bin/python analyzer/train_yolo.py \
  --data data/puck_v3_public/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 16 \
  --project runs/puck_train \
  --name v3_public_base
```

Use the resulting best weights (example):

`runs/puck_train/v3_public_base/weights/best.pt`

## 4) Run V3 Analysis

```bash
analyzer/venv/bin/python analyzer/main.py \
  --video data/videos/panthers_wrist_shot.mp4 \
  --output /tmp/v3_annotated.mp4 \
  --json /tmp/v3_results.json \
  --yolo-model runs/puck_train/v3_public_base/weights/best.pt \
  --roi-radius 200 \
  --calibration 0.0025
```

## 5) Fine-Tune With Your Son's New Videos

As you add videos:

1. Run analyzer with `--export-dataset` to generate release-adjacent frame candidates.
2. Label `puck`, `stick_shaft`, `stick_blade`.
3. Add those labeled samples into your training source set.
4. Re-run merge script, then fine-tune:

```bash
analyzer/venv/bin/python analyzer/train_yolo.py \
  --data data/puck_v3_public/data.yaml \
  --model runs/puck_train/v3_public_base/weights/best.pt \
  --epochs 30 \
  --imgsz 960 \
  --batch 16 \
  --project runs/puck_train \
  --name v3_finetune_round1
```
