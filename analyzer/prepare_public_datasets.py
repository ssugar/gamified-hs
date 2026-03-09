#!/usr/bin/env python3
"""Merge/remap public YOLO datasets into the V3 puck/stick schema.

Expected target classes:
  0 puck
  1 stick_shaft
  2 stick_blade

Usage:
  python prepare_public_datasets.py \
    --source-root data/public_datasets \
    --output-root data/puck_v3_public
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import re
import shutil
from pathlib import Path


TARGET_CLASSES = ["puck", "stick_shaft", "stick_blade"]

ALIASES = {
    "puck": {
        "puck",
        "hockey_puck",
        "ice_puck",
    },
    "stick_shaft": {
        "stick_shaft",
        "shaft",
        "stick",
        "hockey_stick",
    },
    "stick_blade": {
        "stick_blade",
        "blade",
        "hockey_stick_blade",
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_yolo_names(data_yaml_path: Path) -> list[str]:
    """Best-effort parser for YOLO names from data.yaml without extra deps."""
    text = data_yaml_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    names_line = ""
    collecting = False
    block: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            names_line = stripped[len("names:") :].strip()
            if names_line:
                break
            collecting = True
            continue
        if collecting:
            if not stripped:
                continue
            if re.match(r"^[a-zA-Z_]+\s*:", stripped):
                break
            block.append(stripped)

    if names_line:
        # formats:
        # names: [puck, stick]
        # names: {0: puck, 1: stick}
        try:
            parsed = ast.literal_eval(names_line)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            if isinstance(parsed, dict):
                return [str(v) for _, v in sorted(parsed.items(), key=lambda kv: int(kv[0]))]
        except Exception:
            pass

        # fallback bracket parsing
        if names_line.startswith("[") and names_line.endswith("]"):
            inner = names_line[1:-1]
            return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]

    if block:
        names: list[str] = []
        for line in block:
            # "- puck" or "0: puck"
            m_dash = re.match(r"^-\s*(.+)$", line)
            if m_dash:
                names.append(m_dash.group(1).strip().strip("'\""))
                continue
            m_kv = re.match(r"^\d+\s*:\s*(.+)$", line)
            if m_kv:
                names.append(m_kv.group(1).strip().strip("'\""))
        return names

    return []


def discover_pairs(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    """Discover (split, image, label) pairs across common YOLO layouts."""
    pairs: list[tuple[str, Path, Path]] = []
    split_patterns = [
        ("train", dataset_dir / "images" / "train", dataset_dir / "labels" / "train"),
        ("val", dataset_dir / "images" / "val", dataset_dir / "labels" / "val"),
        ("valid", dataset_dir / "images" / "valid", dataset_dir / "labels" / "valid"),
        ("test", dataset_dir / "images" / "test", dataset_dir / "labels" / "test"),
        ("train", dataset_dir / "train" / "images", dataset_dir / "train" / "labels"),
        ("val", dataset_dir / "val" / "images", dataset_dir / "val" / "labels"),
        ("valid", dataset_dir / "valid" / "images", dataset_dir / "valid" / "labels"),
        ("test", dataset_dir / "test" / "images", dataset_dir / "test" / "labels"),
    ]

    for split, img_dir, lbl_dir in split_patterns:
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = lbl_dir / (img_path.stem + ".txt")
            if label_path.exists():
                pairs.append((split, img_path, label_path))

    if pairs:
        return pairs

    # fallback flat layout
    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    if img_dir.exists() and lbl_dir.exists():
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = lbl_dir / (img_path.stem + ".txt")
            if label_path.exists():
                pairs.append(("unsplit", img_path, label_path))
    return pairs


def map_source_class(name: str) -> int | None:
    n = normalize_name(name)
    for target, aliases in ALIASES.items():
        if n in {normalize_name(a) for a in aliases}:
            return TARGET_CLASSES.index(target)
    return None


def ensure_layout(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def canonical_split(name: str) -> str:
    n = name.lower()
    if n in {"train"}:
        return "train"
    if n in {"val", "valid"}:
        return "val"
    if n in {"test"}:
        return "test"
    return "unsplit"


def write_data_yaml(out_root: Path) -> None:
    text = (
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(TARGET_CLASSES)}\n"
        f"names: {TARGET_CLASSES}\n"
    )
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def remap_label_file(
    src_label_path: Path,
    src_names: list[str],
) -> tuple[list[str], int]:
    mapped: list[str] = []
    unknown = 0
    lines = src_label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_cls = int(parts[0])
        except ValueError:
            continue
        if src_cls < 0 or src_cls >= len(src_names):
            unknown += 1
            continue
        tgt_cls = map_source_class(src_names[src_cls])
        if tgt_cls is None:
            continue
        mapped.append(" ".join([str(tgt_cls)] + parts[1:5]))
    return mapped, unknown


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged public YOLO datasets for V3")
    parser.add_argument(
        "--source-root",
        required=True,
        help="Folder containing one subfolder per downloaded YOLO dataset",
    )
    parser.add_argument(
        "--output-root",
        default="data/puck_v3_public",
        help="Merged output dataset root",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for unsplit dataset split allocation",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio for unsplit sources",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio for unsplit sources",
    )
    parser.add_argument(
        "--drop-negatives",
        action="store_true",
        help="Skip images with no mapped classes (default keeps them as negatives)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete and recreate output root before writing",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    random.seed(args.seed)

    if not source_root.exists():
        raise ValueError(f"Source root not found: {source_root}")
    if args.fresh and output_root.exists():
        shutil.rmtree(output_root)

    ensure_layout(output_root)
    write_data_yaml(output_root)

    dataset_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()])
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found under {source_root}")

    total_images = 0
    total_labels = 0
    total_boxes = 0
    total_unknown = 0

    for ds_dir in dataset_dirs:
        data_yaml = ds_dir / "data.yaml"
        if not data_yaml.exists():
            print(f"[skip] {ds_dir.name}: missing data.yaml")
            continue

        src_names = parse_yolo_names(data_yaml)
        if not src_names:
            print(f"[skip] {ds_dir.name}: could not parse class names from data.yaml")
            continue

        pairs = discover_pairs(ds_dir)
        if not pairs:
            print(f"[skip] {ds_dir.name}: no YOLO image/label pairs found")
            continue

        unsplit = [p for p in pairs if canonical_split(p[0]) == "unsplit"]
        if unsplit:
            pairs_split: list[tuple[str, Path, Path]] = []
            for split_name, img_p, lbl_p in pairs:
                s = canonical_split(split_name)
                if s != "unsplit":
                    pairs_split.append((s, img_p, lbl_p))
                    continue
                r = random.random()
                if r < args.test_ratio:
                    s = "test"
                elif r < args.test_ratio + args.val_ratio:
                    s = "val"
                else:
                    s = "train"
                pairs_split.append((s, img_p, lbl_p))
            pairs = pairs_split
        else:
            pairs = [(canonical_split(s), i, l) for s, i, l in pairs]

        mapped_for_ds = 0
        for split, img_path, label_path in pairs:
            mapped_lines, unknown = remap_label_file(label_path, src_names)
            total_unknown += unknown
            if not mapped_lines and args.drop_negatives:
                continue

            base = f"{normalize_name(ds_dir.name)}__{img_path.stem}"
            out_img = output_root / "images" / split / f"{base}{img_path.suffix.lower()}"
            out_lbl = output_root / "labels" / split / f"{base}.txt"

            shutil.copy2(img_path, out_img)
            out_lbl.write_text("\n".join(mapped_lines) + ("\n" if mapped_lines else ""), encoding="utf-8")

            total_images += 1
            total_labels += 1
            total_boxes += len(mapped_lines)
            mapped_for_ds += 1

        print(f"[ok] {ds_dir.name}: {mapped_for_ds} samples added")

    print("\nMerged dataset ready")
    print(f"  Output: {output_root}")
    print(f"  Images: {total_images}")
    print(f"  Labels: {total_labels}")
    print(f"  Boxes: {total_boxes}")
    print(f"  Unknown source-class lines skipped: {total_unknown}")
    print(f"  data.yaml: {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
