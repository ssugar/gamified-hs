#!/usr/bin/env python3
"""Train custom YOLOv8 puck/stick detector on YOLO-format datasets.

Example:
  python train_yolo.py --data data/hockey/data.yaml --model yolov8n.pt --epochs 100
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 puck/stick model")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Train image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or 0")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (0-1)")
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training")
    parser.add_argument("--project", default="runs/puck_train", help="Output project dir")
    parser.add_argument("--name", default="exp", help="Run name")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required. Install with `pip install ultralytics`."
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        val=not args.no_val,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
