"""
predict.py — Inference on images / video / folder.

Usage:
    # Inference trên một ảnh
    python predict.py --weights runs/.../best.pt --source image.jpg

    # Inference trên thư mục ảnh
    python predict.py --weights best.pt --source datasets/VisDrone/VisDrone2019-DET-test-dev/images/

    # Inference với CGSR 2-pass (chỉ áp dụng cho idea=cgsr)
    python predict.py --weights best.pt --source img.jpg --idea cgsr

    # Không lưu ảnh, chỉ in kết quả
    python predict.py --weights best.pt --source img.jpg --no-save
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
        description="Run inference with YOLOv12n on images/video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",  required=True, help="Path to model weights")
    p.add_argument("--source",   required=True,
                   help="Source: image path, folder, video, or URL")
    p.add_argument("--idea",     default="baseline",
                   choices=["baseline", "cagi", "amsha", "rsfe", "cgsr", "iawr"],
                   help="Idea mode (affects inference pipeline for cgsr/iawr)")
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--iou",      type=float, default=0.45)
    p.add_argument("--device",   default="", help="Device: '', 'cpu', '0'")
    p.add_argument("--save",     action="store_true", default=True,
                   help="Save output images with detections drawn")
    p.add_argument("--no-save",  action="store_true",
                   help="Do not save output images")
    p.add_argument("--save-txt", action="store_true",
                   help="Save detection results as .txt labels")
    p.add_argument("--save-conf", action="store_true",
                   help="Include confidence scores in saved labels")
    p.add_argument("--show",     action="store_true",
                   help="Display results in a window (requires display)")
    p.add_argument("--project",  default="runs/predict")
    p.add_argument("--name",     default="exp")
    p.add_argument("--max-det",  type=int, default=300,
                   help="Maximum number of detections per image")
    p.add_argument("--stream",   action="store_true",
                   help="Use streaming mode for memory efficiency on large videos")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        sys.exit(1)

    save_flag = args.save and not args.no_save

    logger.info("=" * 60)
    logger.info(f"  YOLOv12n VisDrone Predictor")
    logger.info(f"  weights = {weights_path}")
    logger.info(f"  source  = {args.source}")
    logger.info(f"  idea    = {args.idea}")
    logger.info(f"  conf    = {args.conf}  |  iou = {args.iou}")
    logger.info("=" * 60)

    predict_kwargs = dict(
        source   = args.source,
        imgsz    = args.imgsz,
        conf     = args.conf,
        iou      = args.iou,
        device   = args.device,
        save     = save_flag,
        save_txt = args.save_txt,
        save_conf = args.save_conf,
        show     = args.show,
        project  = args.project,
        name     = args.name,
        max_det  = args.max_det,
        stream   = args.stream,
    )

    # ── CGSR 2-pass inference ────────────────────────────────────────────────
    if args.idea == "cgsr":
        logger.info("[CGSR] Using 2-pass selective re-detection inference")
        from omegaconf import OmegaConf
        cfg_path = "configs/idea_cgsr.yaml"
        cfg = OmegaConf.load(cfg_path) if Path(cfg_path).exists() else OmegaConf.create({})
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["pretrained"] = str(weights_path)
        cfg["model"] = "yolov12n"
        cfg["idea"] = "cgsr"

        from models.idea_cgsr import CGSRModel
        model = CGSRModel(cfg)
        results = model.predict_cgsr(**predict_kwargs)

    # ── IAWR adaptive-width inference ───────────────────────────────────────
    elif args.idea == "iawr":
        logger.info("[IAWR] Using adaptive width routing inference")
        from omegaconf import OmegaConf
        cfg_path = "configs/idea_iawr.yaml"
        cfg = OmegaConf.load(cfg_path) if Path(cfg_path).exists() else OmegaConf.create({})
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["pretrained"] = str(weights_path)
        cfg["model"] = "yolov12n"
        cfg["idea"] = "iawr"

        from models.idea_iawr import IAWRModel
        model = IAWRModel(cfg)
        results = model.predict_adaptive(**predict_kwargs)

    # ── Standard ultralytics inference ──────────────────────────────────────
    else:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        results = model.predict(**predict_kwargs)

    # ── Print summary ────────────────────────────────────────────────────────
    if not args.stream:
        total_boxes = 0
        for i, result in enumerate(results):
            n = len(result.boxes) if hasattr(result, "boxes") else 0
            total_boxes += n
            if i < 5:   # chỉ in 5 ảnh đầu
                logger.info(f"  Image {i+1}: {n} detections")

        logger.info("=" * 60)
        logger.info(f"  Total images   : {len(results)}")
        logger.info(f"  Total detections: {total_boxes}")
        if save_flag:
            logger.info(f"  Saved to       : {args.project}/{args.name}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
