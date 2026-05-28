"""
train.py — Main training entry point.

Usage:
    python train.py --model yolov12n --idea baseline
    python train.py --model yolov12n --idea cagi   --epochs 100 --batch 16
    python train.py --config configs/idea_rsfe.yaml --device cuda:0
    python train.py --model yolov12n --idea amsha  --test-every 5

After each epoch:  val runs automatically (ultralytics val=True).
Every N epochs:    test split runs via on_train_epoch_end callback.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

# ── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in handlers:
        h.setFormatter(fmt)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Redirect ultralytics LOGGER vào file để capture epoch/val output
    if log_file:
        ul_logger = logging.getLogger("ultralytics")
        ul_logger.propagate = False          # tránh double-print lên console
        for h in handlers:
            ul_logger.addHandler(h)
        ul_logger.setLevel(logging.INFO)


logger = logging.getLogger(__name__)


# ── Config helpers ───────────────────────────────────────────────────────────

def load_config(
    base_cfg: str   = "configs/base.yaml",
    exp_cfg:  str | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Merge base.yaml + experiment yaml + CLI overrides.
    Priority: CLI overrides > experiment yaml > base.yaml
    """
    cfg = OmegaConf.load(base_cfg)

    if exp_cfg and Path(exp_cfg).exists():
        exp = OmegaConf.load(exp_cfg)
        cfg = OmegaConf.merge(cfg, exp)
        logger.info(f"Merged experiment config: {exp_cfg}")

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def resolve_data_config(cfg: DictConfig, data_arg: str | None) -> str:
    """
    Quyết định data config path.
    Priority: --data CLI arg > cfg.data > configs/visdrone.yaml
    """
    if data_arg:
        return data_arg
    data = cfg.get("data", None)
    if data and Path(data).exists():
        return data
    default = "configs/visdrone.yaml"
    if Path(default).exists():
        return default
    raise FileNotFoundError(
        "Không tìm thấy data config. Hãy dùng --data <path> hoặc set 'data' trong config."
    )


# ── Test-every-N callback ────────────────────────────────────────────────────

def make_test_callback(model, data_cfg: str, cfg: DictConfig, weights_path: str = ""):
    """
    Tạo callback chạy test split sau mỗi N epochs với full metrics report.

    N = cfg.test_every_n_epochs (0 → không chạy test).
    Báo cáo đầy đủ: Accuracy metrics + Efficiency metrics + Efficiency ratios.
    """
    from utils.metrics import (
        compute_efficiency_metrics,
        print_full_report,
        save_metrics_csv,
    )

    test_every_n: int = int(cfg.get("test_every_n_epochs", 10))
    if test_every_n <= 0:
        return None

    model_name: str  = str(cfg.get("model", "YOLOv12n"))
    idea:       str  = str(cfg.get("idea",  "baseline"))
    save_csv:   str  = str(cfg.get("metrics_csv", "runs/metrics.csv"))
    measure_eff: bool = bool(cfg.get("test_measure_efficiency", True))

    def on_train_epoch_end(trainer) -> None:
        epoch = trainer.epoch + 1      # ultralytics: 0-indexed
        if epoch % test_every_n != 0:
            return

        logger.info(f"[TEST] Epoch {epoch} — running test split với full metrics…")
        try:
            # Resolve checkpoint path trước để dùng cho cả val lẫn efficiency
            try:
                best_pt = str(trainer.best) if hasattr(trainer, "best") else weights_path
                if not best_pt or not Path(best_pt).exists():
                    last_pt = str(trainer.last) if hasattr(trainer, "last") else ""
                    best_pt = last_pt if last_pt and Path(last_pt).exists() else ""
            except Exception:
                best_pt = weights_path

            # ── Accuracy: load fresh model từ checkpoint để tránh
            #    inference_mode contaminating GradScaler._scale trên PyTorch 2.12+
            from ultralytics import YOLO as _YOLO
            _val_src = best_pt if best_pt and Path(best_pt).exists() else model.pretrained
            _tmp = _YOLO(str(_val_src))

            # Val output → <experiment_dir>/val/  (không rải ra runs/detect/val*)
            _exp_dir = str(trainer.save_dir) if hasattr(trainer, "save_dir") else "runs"
            val_results = _tmp.val(
                data     = data_cfg,
                split    = str(cfg.get("test_split", "test")),
                imgsz    = int(cfg.get("imgsz", 640)),
                device   = str(cfg.get("device", "")),
                conf     = float(cfg.get("val_conf", 0.001)),
                iou      = float(cfg.get("val_iou",  0.6)),
                project  = _exp_dir,
                name     = "val",
                exist_ok = True,
            )
            del _tmp

            # Reset GradScaler sau val để đảm bảo _scale tensor không bị
            # nhiễm inference_mode metadata (PyTorch 2.12 regression)
            try:
                from ultralytics.utils.torch_utils import TORCH_2_4
                import torch as _torch
                trainer.scaler = (
                    _torch.amp.GradScaler("cuda", enabled=trainer.amp) if TORCH_2_4
                    else _torch.cuda.amp.GradScaler(enabled=trainer.amp)
                )
            except Exception:
                pass

            # ── Efficiency: đo từ best.pt hiện tại hoặc weights_path ────────
            eff: dict = {}
            if measure_eff:
                if best_pt:
                    logger.info(f"[TEST] Measuring efficiency from: {best_pt}")
                    eff = compute_efficiency_metrics(
                        weights_path   = best_pt,
                        imgsz          = int(cfg.get("imgsz", 640)),
                        device         = str(cfg.get("device", "")),
                        warmup_runs    = 5,    # ít hơn để không delay training
                        benchmark_runs = 20,   # đủ để có số ổn định
                    )

            # ── In full report ───────────────────────────────────────────────
            report = print_full_report(
                val_results     = val_results,
                eff             = eff,
                split           = str(cfg.get("test_split", "test")),
                model_name      = model_name,
                idea            = idea,
                logger_instance = logger,
            )
            # ── Lưu CSV để so sánh experiments ──────────────────────────────
            try:
                save_metrics_csv(report, save_csv, epoch=epoch)
            except Exception as csv_err:
                logger.debug(f"[TEST] CSV save failed: {csv_err}")

        except Exception as e:
            logger.warning(f"[TEST] Test run failed at epoch {epoch}: {e}")

    return on_train_epoch_end


# ── Per-epoch metrics logger ─────────────────────────────────────────────────

def make_epoch_log_callback(total_epochs: int):
    """
    Callback ghi metrics từng epoch vào logger:
        box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 | lr
    Gọi vào sự kiện on_fit_epoch_end (sau cả train lẫn val).
    """
    def on_fit_epoch_end(trainer) -> None:
        epoch = trainer.epoch + 1

        # ── Losses (trung bình toàn epoch) ──────────────────────────────────
        tloss = trainer.tloss
        if tloss is not None:
            try:
                loss_vals = tloss.tolist() if hasattr(tloss, "tolist") else list(tloss)
                box_l, cls_l, dfl_l = (loss_vals + [0, 0, 0])[:3]
                loss_str = f"box={box_l:.4f}  cls={cls_l:.4f}  dfl={dfl_l:.4f}"
            except Exception:
                loss_str = f"loss={tloss}"
        else:
            loss_str = "loss=N/A"

        # ── Val metrics ──────────────────────────────────────────────────────
        metrics = trainer.metrics or {}
        map50    = metrics.get("metrics/mAP50(B)",    metrics.get("mAP50",    None))
        map5095  = metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95", None))
        prec     = metrics.get("metrics/precision(B)", None)
        rec      = metrics.get("metrics/recall(B)",    None)

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A"

        metric_str = (
            f"P={_fmt(prec)}  R={_fmt(rec)}  "
            f"mAP50={_fmt(map50)}  mAP50-95={_fmt(map5095)}"
        )

        # ── Learning rate (pg0) ──────────────────────────────────────────────
        lr_val = list((trainer.lr or {}).values())
        lr_str = f"lr={lr_val[0]:.6f}" if lr_val else ""

        logger.info(
            f"[EPOCH {epoch:>3}/{total_epochs}]  {loss_str}  |  {metric_str}  |  {lr_str}"
        )

    return on_fit_epoch_end


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv12n on VisDrone with switchable ideas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / idea selection
    p.add_argument("--model",  default="yolov12n", help="Model name")
    p.add_argument("--idea",   default="baseline",
                   choices=["baseline", "cagi", "amsha", "rsfe", "cgsr", "iawr"],
                   help="Research idea to use")

    # Config
    p.add_argument("--config",     default=None,
                   help="Path to experiment YAML (auto-resolved if not set)")
    p.add_argument("--base-config", default="configs/base.yaml",
                   help="Base config YAML")
    p.add_argument("--data",       default=None,
                   help="Dataset config YAML (default: configs/visdrone.yaml)")

    # Training hyperparams (override config)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch",      type=int,   default=None)
    p.add_argument("--imgsz",      type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None,
                   help="Initial learning rate (override config lr0)")
    p.add_argument("--device",     default=None,
                   help="Device: '', 'cpu', '0', '0,1', 'cuda:0'")
    p.add_argument("--workers",    type=int,   default=None)
    p.add_argument("--pretrained", default=None,
                   help="Pretrained weights path (override config)")

    # Project / experiment name
    p.add_argument("--project",  default=None, help="Runs folder")
    p.add_argument("--name",     default=None, help="Experiment name")
    p.add_argument("--exist-ok", action="store_true",
                   help="Continue existing experiment")

    # Testing
    p.add_argument("--test-every", type=int, default=None,
                   dest="test_every",
                   help="Run test split every N epochs (0=disable)")

    # Logging
    p.add_argument("--log-level",  default="INFO")
    p.add_argument("--log-file",   default=None,
                   help="Path to log file (optional)")

    # Extra omegaconf overrides
    p.add_argument("overrides", nargs="*",
                   help="Extra config overrides in key=value format")

    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.log_file is None:
        args.log_file = f"logs/{args.model}_{args.idea}.log"
    setup_logging(args.log_level, args.log_file)

    logger.info("=" * 60)
    logger.info(f"  YOLOv12n VisDrone Trainer")
    logger.info(f"  model={args.model}  idea={args.idea}")
    logger.info("=" * 60)

    # ── 1. Resolve configs ───────────────────────────────────────────────────
    if args.config:
        exp_cfg_path = args.config
    else:
        # Auto-resolve: configs/baseline.yaml, configs/idea_cagi.yaml, ...
        if args.idea == "baseline":
            exp_cfg_path = "configs/baseline.yaml"
        else:
            exp_cfg_path = f"configs/idea_{args.idea}.yaml"

    cfg = load_config(
        base_cfg  = args.base_config,
        exp_cfg   = exp_cfg_path,
        overrides = args.overrides,
    )

    # CLI overrides (take priority over yaml)
    if args.model:       cfg.model       = args.model
    if args.idea:        cfg.idea        = args.idea
    if args.epochs:      cfg.epochs      = args.epochs
    if args.batch:       cfg.batch       = args.batch
    if args.imgsz:       cfg.imgsz       = args.imgsz
    if args.lr:          cfg.lr0         = args.lr
    if args.device is not None:   cfg.device  = args.device
    if args.workers:     cfg.workers     = args.workers
    if args.pretrained:  cfg.pretrained  = args.pretrained
    if args.project:     cfg.project     = args.project
    if args.name:        cfg.name        = args.name
    if args.test_every is not None:
        cfg.test_every_n_epochs = args.test_every

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── 2. Resolve data config ───────────────────────────────────────────────
    data_cfg = resolve_data_config(cfg, args.data)
    logger.info(f"Dataset config: {data_cfg}")

    # ── 3. Build model ───────────────────────────────────────────────────────
    from models import build_model
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model = build_model(cfg.model, cfg.idea, cfg_dict)

    # ── 4. Register test-every-N callback ────────────────────────────────────
    # Dự đoán đường dẫn best.pt để efficiency metrics có thể đọc sau training
    project_dir_early = str(cfg.get("project", "runs/train"))
    exp_name_early    = str(cfg.get("name", f"{cfg.model}_{cfg.idea}"))
    expected_best_pt  = f"{project_dir_early}/{exp_name_early}/weights/best.pt"

    # Lưu metrics_csv path vào config nếu chưa có — đặt tên theo idea
    if not cfg.get("metrics_csv"):
        cfg.metrics_csv = f"{project_dir_early}/metrics_{cfg.idea}.csv"

    test_cb = make_test_callback(model, data_cfg, cfg, weights_path=expected_best_pt)
    if test_cb is not None:
        model.yolo.add_callback("on_train_epoch_end", test_cb)
        logger.info(
            f"[TEST] Test callback registered: every {cfg.test_every_n_epochs} epochs"
        )
        logger.info(f"[TEST] Metrics CSV → {cfg.metrics_csv}")

    # Epoch metrics logger — ghi loss + mAP mỗi epoch vào log file
    epoch_log_cb = make_epoch_log_callback(total_epochs=int(cfg.get("epochs", 100)))
    model.yolo.add_callback("on_fit_epoch_end", epoch_log_cb)

    # ── 5. Train ─────────────────────────────────────────────────────────────
    experiment_name = cfg.get("name", f"{cfg.model}_{cfg.idea}")
    project_dir     = cfg.get("project", "runs/train")

    extra_kwargs: dict = {
        "workers":      int(cfg.get("workers", 4)),
        "exist_ok":     args.exist_ok,
        "cache":        cfg.get("cache", False),
        "patience":     int(cfg.get("patience", 50)),
        # ── Augmentation (tất cả forward để config có hiệu lực) ──────────────
        "hsv_h":        float(cfg.get("hsv_h",       0.015)),
        "hsv_s":        float(cfg.get("hsv_s",       0.7)),
        "hsv_v":        float(cfg.get("hsv_v",       0.4)),
        "degrees":      float(cfg.get("degrees",     0.0)),
        "translate":    float(cfg.get("translate",   0.1)),
        "scale":        float(cfg.get("scale",       0.5)),
        "shear":        float(cfg.get("shear",       0.0)),
        "perspective":  float(cfg.get("perspective", 0.0)),
        "flipud":       float(cfg.get("flipud",      0.0)),
        "fliplr":       float(cfg.get("fliplr",      0.5)),
        "mosaic":       float(cfg.get("mosaic",      1.0)),
        "mixup":        float(cfg.get("mixup",       0.0)),
        "copy_paste":   float(cfg.get("copy_paste",  0.0)),
        "close_mosaic": int(cfg.get("close_mosaic",  10)),
    }

    # Learning rate
    if cfg.get("lr0"):           extra_kwargs["lr0"]          = float(cfg.lr0)
    if cfg.get("lrf"):           extra_kwargs["lrf"]          = float(cfg.lrf)
    if cfg.get("momentum"):      extra_kwargs["momentum"]     = float(cfg.momentum)
    if cfg.get("weight_decay"):  extra_kwargs["weight_decay"] = float(cfg.weight_decay)
    if cfg.get("warmup_epochs"): extra_kwargs["warmup_epochs"] = float(cfg.warmup_epochs)
    if cfg.get("label_smoothing"): extra_kwargs["label_smoothing"] = float(cfg.label_smoothing)

    logger.info(f"Starting training → {project_dir}/{experiment_name}")

    model.train(
        data    = data_cfg,
        epochs  = int(cfg.get("epochs",  100)),
        batch   = int(cfg.get("batch",   16)),
        imgsz   = int(cfg.get("imgsz",   640)),
        device  = str(cfg.get("device",  "")),
        project = project_dir,
        name    = experiment_name,
        **extra_kwargs,
    )

    logger.info("Training complete!")
    logger.info(f"Results saved to: {project_dir}/{experiment_name}")
    logger.info(
        "Để chạy val thủ công:\n"
        f"  python val.py --model {cfg.model} --idea {cfg.idea} "
        f"  --weights {project_dir}/{experiment_name}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
