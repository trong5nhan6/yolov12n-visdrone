"""
scripts/prepare_visdrone.py
Convert VisDrone2019-DET annotations sang YOLO format.

VisDrone annotation format (CSV per image):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

YOLO format (txt per image):
    <class_id> <cx> <cy> <w> <h>   (normalized 0-1)

VisDrone categories → YOLO class IDs:
    0: ignored regions (BỎ QUA)
    1: pedestrian       → 0
    2: people           → 1
    3: bicycle          → 2
    4: car              → 3
    5: van              → 4
    6: truck            → 5
    7: tricycle         → 6
    8: awning-tricycle  → 7
    9: bus              → 8
   10: motor            → 9
   11: others (BỎ QUA)

Usage:
    python scripts/prepare_visdrone.py --visdrone-root /path/to/VisDrone2019-DET

    Sau khi chạy, cấu trúc output:
        datasets/VisDrone/
            images/
                train/  val/  test/
            labels/
                train/  val/  test/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# VisDrone category_id → YOLO class_id  (0, 11 bị bỏ qua)
CATEGORY_MAP = {
    1:  0,   # pedestrian
    2:  1,   # people
    3:  2,   # bicycle
    4:  3,   # car
    5:  4,   # van
    6:  5,   # truck
    7:  6,   # tricycle
    8:  7,   # awning-tricycle
    9:  8,   # bus
    10: 9,   # motor
}

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val":   "VisDrone2019-DET-val",
    "test":  "VisDrone2019-DET-test-dev",
}


def convert_annotation(
    ann_path: Path,
    img_path: Path,
    out_label_path: Path,
) -> int:
    """
    Convert một file annotation VisDrone → YOLO txt.

    Returns:
        Số objects đã convert (không tính ignored).
    """
    try:
        from PIL import Image
        img = Image.open(img_path)
        img_w, img_h = img.size
    except Exception as e:
        logger.warning(f"Không đọc được ảnh {img_path}: {e}")
        return 0

    lines = []
    n_converted = 0

    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            bbox_left   = int(parts[0])
            bbox_top    = int(parts[1])
            bbox_width  = int(parts[2])
            bbox_height = int(parts[3])
            # score      = parts[4]  # không dùng
            category    = int(parts[5])
            # truncation = parts[6]
            # occlusion  = parts[7]

            if category not in CATEGORY_MAP:
                continue  # ignored (0) hoặc others (11)

            class_id = CATEGORY_MAP[category]

            # Chuyển về YOLO format (center-normalized)
            cx = (bbox_left + bbox_width  / 2) / img_w
            cy = (bbox_top  + bbox_height / 2) / img_h
            w  = bbox_width  / img_w
            h  = bbox_height / img_h

            # Clip về [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))

            if w <= 0 or h <= 0:
                continue

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_converted += 1

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w") as f:
        f.write("\n".join(lines))

    return n_converted


def prepare_split(
    visdrone_root: Path,
    split_name: str,
    split_dir:  str,
    output_root: Path,
    copy_images: bool = True,
) -> None:
    """
    Convert một split (train/val/test).
    """
    src_dir = visdrone_root / split_dir
    if not src_dir.exists():
        logger.warning(f"Split directory not found: {src_dir} — skipping {split_name}")
        return

    src_images = src_dir / "images"
    src_annots = src_dir / "annotations"

    if not src_images.exists():
        logger.error(f"Images folder not found: {src_images}")
        return

    out_images = output_root / "images" / split_name
    out_labels = output_root / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_images.glob("*.jpg")) + sorted(src_images.glob("*.png"))
    logger.info(f"[{split_name}] {len(image_files)} images found")

    total_objects = 0
    skipped = 0

    for img_path in image_files:
        stem = img_path.stem

        # Copy ảnh
        if copy_images:
            dst_img = out_images / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

        # Convert annotation
        ann_path = src_annots / f"{stem}.txt"
        if not ann_path.exists():
            logger.debug(f"No annotation for {stem}, creating empty label")
            (out_labels / f"{stem}.txt").touch()
            skipped += 1
            continue

        out_label = out_labels / f"{stem}.txt"
        n = convert_annotation(ann_path, img_path, out_label)
        total_objects += n

    logger.info(
        f"[{split_name}] Done: "
        f"{len(image_files) - skipped} images converted, "
        f"{total_objects} objects, "
        f"{skipped} skipped (no annotation)"
    )


def verify_dataset(output_root: Path) -> None:
    """Kiểm tra dataset sau khi convert."""
    logger.info("\n── Dataset Verification ──")
    for split in ["train", "val", "test"]:
        img_dir = output_root / "images" / split
        lbl_dir = output_root / "labels" / split

        if not img_dir.exists():
            logger.info(f"  {split}: NOT FOUND")
            continue

        n_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        n_labels = len(list(lbl_dir.glob("*.txt")))

        logger.info(f"  {split:5s}: {n_images:5d} images, {n_labels:5d} labels")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert VisDrone2019-DET to YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--visdrone-root", default="datasets/VisDrone_raw",
                   help="Root directory của VisDrone2019-DET dataset")
    p.add_argument("--output-root", default="datasets/VisDrone",
                   help="Output directory cho YOLO format dataset")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"],
                   help="Splits để convert")
    p.add_argument("--no-copy-images", action="store_true",
                   help="Không copy ảnh (chỉ tạo labels) — dùng symlinks thay thế")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    visdrone_root = Path(args.visdrone_root)
    output_root   = Path(args.output_root)

    if not visdrone_root.exists():
        logger.error(
            f"VisDrone root not found: {visdrone_root}\n"
            f"Hãy download VisDrone2019-DET từ:\n"
            f"  https://github.com/VisDrone/VisDrone-Dataset\n"
            f"và giải nén vào: {visdrone_root}"
        )
        sys.exit(1)

    logger.info(f"VisDrone root : {visdrone_root}")
    logger.info(f"Output root   : {output_root}")
    logger.info(f"Splits        : {args.splits}")

    for split_name in args.splits:
        split_dir = SPLITS.get(split_name)
        if split_dir is None:
            logger.warning(f"Unknown split: {split_name}")
            continue
        prepare_split(
            visdrone_root = visdrone_root,
            split_name    = split_name,
            split_dir     = split_dir,
            output_root   = output_root,
            copy_images   = not args.no_copy_images,
        )

    verify_dataset(output_root)
    logger.info(f"\nConversion complete! Dataset ready at: {output_root}")
    logger.info("Tiếp theo: chỉnh đường dẫn trong configs/visdrone.yaml")


if __name__ == "__main__":
    main()
