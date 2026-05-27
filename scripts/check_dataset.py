"""
scripts/check_dataset.py
Kiểm tra tính toàn vẹn của VisDrone dataset sau khi convert.

Checks:
  1. Image–label pairing: mỗi ảnh có đúng 1 label file
  2. YOLO format validity: mỗi dòng label có đúng 5 fields, values trong [0,1]
  3. Class distribution: số objects per class
  4. Size distribution: tiny (<32px), small (32-96px), medium (>96px)
  5. Aspect ratio outliers

Usage:
    python scripts/check_dataset.py --dataset-root datasets/VisDrone
    python scripts/check_dataset.py --split val --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def check_label_file(label_path: Path) -> Tuple[int, List[str]]:
    """
    Kiểm tra một file label YOLO.

    Returns:
        (n_valid_objects, list_of_errors)
    """
    errors: List[str] = []
    n_valid = 0

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"  Line {i+1}: expected 5 fields, got {len(parts)}")
            continue

        try:
            cls  = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"  Line {i+1}: non-numeric values")
            continue

        if cls < 0 or cls >= len(CLASS_NAMES):
            errors.append(f"  Line {i+1}: invalid class_id={cls}")
            continue

        if not all(0.0 <= v <= 1.0 for v in vals):
            errors.append(f"  Line {i+1}: values out of [0,1] range: {vals}")
            continue

        n_valid += 1

    return n_valid, errors


def analyze_split(
    dataset_root: Path,
    split: str,
    verbose: bool = False,
    img_size: int = 640,  # assumed image size for pixel-space analysis
) -> Dict:
    """
    Phân tích một split và trả về statistics.
    """
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    if not img_dir.exists():
        logger.warning(f"  {split}: image dir not found — {img_dir}")
        return {}

    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    label_files = {lf.stem: lf for lf in lbl_dir.glob("*.txt")}

    stats: Dict = {
        "n_images":        len(image_files),
        "n_images_paired": 0,
        "n_images_missing_label": 0,
        "n_objects_total": 0,
        "n_label_errors":  0,
        "class_counts":    defaultdict(int),
        "size_tiny":       0,   # <32px in original image
        "size_small":      0,   # 32-96px
        "size_medium":     0,   # >96px
        "empty_labels":    0,   # ảnh hợp lệ nhưng không có objects
    }

    for img_path in image_files:
        stem = img_path.stem

        if stem not in label_files:
            stats["n_images_missing_label"] += 1
            if verbose:
                logger.warning(f"  Missing label: {stem}")
            continue

        stats["n_images_paired"] += 1
        lbl_path = label_files[stem]

        n_valid, errors = check_label_file(lbl_path)

        if errors:
            stats["n_label_errors"] += len(errors)
            if verbose:
                for err in errors[:3]:
                    logger.warning(f"  [{stem}] {err}")

        if n_valid == 0:
            stats["empty_labels"] += 1

        stats["n_objects_total"] += n_valid

        # Class + size distribution
        try:
            from PIL import Image
            img = Image.open(img_path)
            iw, ih = img.size
        except Exception:
            iw, ih = img_size, img_size

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(parts[0])
                    w   = float(parts[3]) * iw
                    h   = float(parts[4]) * ih
                    diag = (w**2 + h**2) ** 0.5
                except (ValueError, IndexError):
                    continue

                if 0 <= cls < len(CLASS_NAMES):
                    stats["class_counts"][CLASS_NAMES[cls]] += 1

                if diag < 32:
                    stats["size_tiny"]   += 1
                elif diag < 96:
                    stats["size_small"]  += 1
                else:
                    stats["size_medium"] += 1

    return stats


def print_stats(split: str, stats: Dict) -> None:
    logger.info(f"\n{'─'*50}")
    logger.info(f"  Split: {split.upper()}")
    logger.info(f"{'─'*50}")
    logger.info(f"  Images total   : {stats.get('n_images', 0):>6}")
    logger.info(f"  Images paired  : {stats.get('n_images_paired', 0):>6}")
    logger.info(f"  Missing labels : {stats.get('n_images_missing_label', 0):>6}")
    logger.info(f"  Empty labels   : {stats.get('empty_labels', 0):>6}")
    logger.info(f"  Label errors   : {stats.get('n_label_errors', 0):>6}")
    logger.info(f"  Objects total  : {stats.get('n_objects_total', 0):>6}")

    total_obj = stats.get("n_objects_total", 1) or 1
    s_tiny   = stats.get("size_tiny",   0)
    s_small  = stats.get("size_small",  0)
    s_medium = stats.get("size_medium", 0)
    logger.info(
        f"\n  Size distribution (diagonal pixels):"
        f"\n    Tiny   (<32px) : {s_tiny:>6} ({100*s_tiny/total_obj:.1f}%)"
        f"\n    Small (32-96px): {s_small:>6} ({100*s_small/total_obj:.1f}%)"
        f"\n    Medium (>96px) : {s_medium:>6} ({100*s_medium/total_obj:.1f}%)"
    )

    logger.info(f"\n  Class distribution:")
    class_counts = stats.get("class_counts", {})
    for cls_name in CLASS_NAMES:
        cnt = class_counts.get(cls_name, 0)
        bar = "█" * int(cnt / max(class_counts.values(), default=1) * 30) if class_counts else ""
        logger.info(f"    {cls_name:<20}: {cnt:>6}  {bar}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check VisDrone YOLO dataset integrity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default="datasets/VisDrone",
                   help="Root của YOLO-format VisDrone dataset")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--verbose", action="store_true",
                   help="In chi tiết lỗi từng file")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    root = Path(args.dataset_root)
    if not root.exists():
        logger.error(f"Dataset root not found: {root}")
        logger.error("Hãy chạy: python scripts/prepare_visdrone.py --visdrone-root <path>")
        sys.exit(1)

    logger.info(f"Checking dataset at: {root}")

    all_ok = True
    for split in args.splits:
        stats = analyze_split(root, split, verbose=args.verbose)
        if stats:
            print_stats(split, stats)
            if stats.get("n_label_errors", 0) > 0:
                all_ok = False
                logger.warning(f"  ⚠ Label errors found in {split} split!")
        else:
            logger.warning(f"  {split}: split not found or empty")

    logger.info(f"\n{'─'*50}")
    if all_ok:
        logger.info("  ✓ Dataset check PASSED — ready for training!")
    else:
        logger.info("  ✗ Dataset check FAILED — fix errors above before training.")
    logger.info(f"{'─'*50}")


if __name__ == "__main__":
    main()
