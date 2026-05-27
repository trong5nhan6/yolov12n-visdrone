"""
val.py — Standalone validation / test script.

Usage:
    # Val split (mặc định)
    python val.py --weights runs/train/yolov12n_baseline/weights/best.pt

    # Test split
    python val.py --weights runs/.../best.pt --split test

    # Custom data & thresholds
    python val.py --weights best.pt --data configs/visdrone.yaml --conf 0.25 --iou 0.5

    # Không đo efficiency (nhanh hơn)
    python val.py --weights best.pt --no-efficiency

    # Lưu kết quả vào CSV để so sánh experiments
    python val.py --weights best.pt --save-csv runs/results.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate YOLOv12n on VisDrone dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", required=True,
                   help="Path to model weights (.pt file)")
    p.add_argument("--data",    default="configs/visdrone.yaml",
                   help="Dataset config YAML")
    p.add_argument("--split",   default="val",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--conf",    type=float, default=0.001,
                   help="Confidence threshold for val")
    p.add_argument("--iou",     type=float, default=0.6,
                   help="IoU threshold for NMS during val")
    p.add_argument("--device",  default="",
                   help="Device: '', 'cpu', '0', 'cuda:0'")
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--save-json", action="store_true",
                   help="Save results to COCO-format JSON")
    p.add_argument("--plots",   action="store_true", default=True,
                   help="Save validation plots")
    p.add_argument("--project", default="runs/val")
    p.add_argument("--name",    default=None)

    # Efficiency options
    p.add_argument("--no-efficiency", action="store_true",
                   help="Bỏ qua đo efficiency metrics (nhanh hơn)")
    p.add_argument("--benchmark-runs", type=int, default=50,
                   help="Số lần lặp để đo latency/FPS")
    p.add_argument("--model-name", default="YOLOv12n",
                   help="Tên model để hiển thị trong report")
    p.add_argument("--idea", default="baseline",
                   help="Tên idea để hiển thị trong report")

    # CSV export
    p.add_argument("--save-csv", default=None,
                   help="Lưu metrics vào CSV file (append). VD: runs/results.csv")

    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    if not Path(args.data).exists():
        logger.error(f"Data config not found: {args.data}")
        sys.exit(1)

    logger.info("=" * 58)
    logger.info(f"  YOLOv12n VisDrone Validator")
    logger.info(f"  weights = {weights_path}")
    logger.info(f"  data    = {args.data}")
    logger.info(f"  split   = {args.split}")
    logger.info("=" * 58)

    from ultralytics import YOLO
    from utils.metrics import (
        compute_efficiency_metrics,
        print_full_report,
        save_metrics_csv,
    )

    model = YOLO(str(weights_path))
    name  = args.name or f"{weights_path.parent.parent.name}_{args.split}"

    # ── 1. Chạy validation ───────────────────────────────────────────────────
    logger.info(f"Running {args.split} evaluation…")
    val_results = model.val(
        data      = args.data,
        split     = args.split,
        imgsz     = args.imgsz,
        conf      = args.conf,
        iou       = args.iou,
        device    = args.device,
        batch     = args.batch,
        workers   = args.workers,
        save_json = args.save_json,
        plots     = args.plots,
        project   = args.project,
        name      = name,
    )

    # ── 2. Đo efficiency metrics ─────────────────────────────────────────────
    eff: dict = {}
    if not args.no_efficiency:
        logger.info("Measuring efficiency metrics (latency / FPS / GFLOPs)…")
        eff = compute_efficiency_metrics(
            weights_path   = weights_path,
            imgsz          = args.imgsz,
            device         = args.device,
            benchmark_runs = args.benchmark_runs,
        )
    else:
        logger.info("Efficiency metrics skipped (--no-efficiency)")

    # ── 3. In full report ────────────────────────────────────────────────────
    report = print_full_report(
        val_results  = val_results,
        eff          = eff,
        split        = args.split,
        model_name   = args.model_name,
        idea         = args.idea,
        weights_path = weights_path,
        logger_instance = logger,
    )

    # ── 4. Lưu CSV nếu được yêu cầu ─────────────────────────────────────────
    if args.save_csv:
        save_metrics_csv(report, args.save_csv)

    logger.info(f"Ultralytics results saved to: {args.project}/{name}")


if __name__ == "__main__":
    main()
