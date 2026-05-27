"""
utils/metrics.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive metrics module cho lightweight model evaluation.

Accuracy metrics:
    mAP50, mAP50-95, AP per-class, AR (Average Recall), Precision, Recall

Efficiency metrics:
    GFLOPs, Params (M), Model size (MB), FPS, Latency (ms/img),
    mAP50/GFLOPs, mAP50/Params (M), mAP50 × FPS

Usage:
    from utils.metrics import compute_efficiency_metrics, print_full_report

    metrics_val = model.val(...)
    eff = compute_efficiency_metrics(weights_path, imgsz=640, device="")
    print_full_report(metrics_val, eff, split="val")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# VisDrone 10 class names (theo thứ tự YOLO class_id)
VISDRONE_CLASS_NAMES = [
    "pedestrian",       # 0
    "people",           # 1
    "bicycle",          # 2
    "car",              # 3
    "van",              # 4
    "truck",            # 5
    "tricycle",         # 6
    "awning-tricycle",  # 7
    "bus",              # 8
    "motor",            # 9
]


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficiency_metrics(
    weights_path: str | Path,
    imgsz: int = 640,
    device: str = "",
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
    batch: int = 1,
) -> Dict[str, Any]:
    """
    Đo các efficiency metrics của model.

    Args:
        weights_path:   Đường dẫn tới .pt file
        imgsz:          Kích thước ảnh đầu vào
        device:         Device để benchmark ('', 'cpu', '0', 'cuda:0')
        warmup_runs:    Số lần warmup trước khi đo latency
        benchmark_runs: Số lần đo để lấy trung bình
        batch:          Batch size để đo throughput

    Returns:
        dict với keys: params_m, gflops, model_size_mb, latency_ms,
                       fps, fps_batch (throughput), device_used
    """
    weights_path = Path(weights_path)
    result: Dict[str, Any] = {
        "params_m":       None,
        "gflops":         None,
        "model_size_mb":  None,
        "latency_ms":     None,
        "fps":            None,
        "fps_batch":      None,
        "device_used":    None,
    }

    # ── Model size ────────────────────────────────────────────────────────────
    if weights_path.exists():
        result["model_size_mb"] = round(weights_path.stat().st_size / 1e6, 2)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        nn_model = model.model

        # Resolve device
        if device in ("", "cpu"):
            dev = torch.device("cpu")
        else:
            dev_str = f"cuda:{device}" if device.isdigit() else device
            dev = torch.device(dev_str if torch.cuda.is_available() else "cpu")

        result["device_used"] = str(dev)
        nn_model = nn_model.to(dev).eval()

    except Exception as e:
        logger.warning(f"[metrics] Cannot load model for efficiency metrics: {e}")
        return result

    # ── Params ────────────────────────────────────────────────────────────────
    try:
        total_params = sum(p.numel() for p in nn_model.parameters())
        result["params_m"] = round(total_params / 1e6, 3)
    except Exception as e:
        logger.debug(f"[metrics] Params count failed: {e}")

    # ── GFLOPs ───────────────────────────────────────────────────────────────
    try:
        from ultralytics.utils.torch_utils import get_flops
        result["gflops"] = round(get_flops(nn_model, imgsz), 2)
    except Exception:
        # Fallback: thop library
        try:
            from thop import profile
            dummy = torch.zeros(1, 3, imgsz, imgsz).to(dev)
            macs, _ = profile(nn_model, inputs=(dummy,), verbose=False)
            result["gflops"] = round(macs * 2 / 1e9, 2)  # MACs → GFLOPs
        except Exception as e2:
            logger.debug(f"[metrics] GFLOPs measurement failed: {e2}")

    # ── Latency & FPS (batch=1) ───────────────────────────────────────────────
    try:
        dummy_single = torch.zeros(1, 3, imgsz, imgsz).to(dev)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                nn_model(dummy_single)

        # Sync nếu dùng CUDA
        if dev.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        t_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(benchmark_runs):
                nn_model(dummy_single)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) / benchmark_runs * 1000
        result["latency_ms"] = round(latency_ms, 2)
        result["fps"]        = round(1000 / latency_ms, 1)

    except Exception as e:
        logger.debug(f"[metrics] Latency measurement failed: {e}")

    # ── Throughput FPS (batch > 1) ───────────────────────────────────────────
    if batch > 1:
        try:
            dummy_batch = torch.zeros(batch, 3, imgsz, imgsz).to(dev)
            with torch.no_grad():
                for _ in range(warmup_runs):
                    nn_model(dummy_batch)

            if dev.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(benchmark_runs):
                    nn_model(dummy_batch)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
            t1 = time.perf_counter()

            latency_batch = (t1 - t0) / benchmark_runs
            result["fps_batch"] = round(batch / latency_batch, 1)

        except Exception as e:
            logger.debug(f"[metrics] Batch throughput failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy metrics parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_accuracy_metrics(val_results: Any) -> Dict[str, Any]:
    """
    Trích xuất accuracy metrics từ ultralytics val() results.

    Returns dict với keys:
        map50, map50_95, precision, recall,
        ap50_per_class (dict {class_name: float}),
        ap_per_class (dict {class_name: float}),
        ar (float | None),
    """
    acc: Dict[str, Any] = {
        "map50":           None,
        "map50_95":        None,
        "precision":       None,
        "recall":          None,
        "ap50_per_class":  {},
        "ap_per_class":    {},
        "ar":              None,
    }

    if val_results is None:
        return acc

    try:
        box = val_results.box
        acc["map50"]     = round(float(box.map50), 4)
        acc["map50_95"]  = round(float(box.map),   4)
        acc["precision"] = round(float(box.mp),    4)
        acc["recall"]    = round(float(box.mr),    4)

        # Per-class AP50 và AP50-95
        if hasattr(box, "ap_class_index") and hasattr(box, "ap50") and hasattr(box, "ap"):
            for idx, ap50_val, ap_val in zip(
                box.ap_class_index,
                box.ap50,
                box.ap,
            ):
                idx = int(idx)
                name = (
                    VISDRONE_CLASS_NAMES[idx]
                    if idx < len(VISDRONE_CLASS_NAMES)
                    else f"class_{idx}"
                )
                acc["ap50_per_class"][name] = round(float(ap50_val), 4)
                acc["ap_per_class"][name]   = round(float(ap_val),   4)

        # Average Recall — ultralytics lưu trong box.mr (mean recall)
        # AR@100 có thể lấy từ stats nếu có
        acc["ar"] = acc["recall"]  # approximation

    except AttributeError as e:
        logger.debug(f"[metrics] Cannot parse accuracy metrics: {e}")

    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency ratios
# ─────────────────────────────────────────────────────────────────────────────

def compute_ratios(acc: Dict, eff: Dict) -> Dict[str, Any]:
    """
    Tính các efficiency ratio metrics.

    Returns dict với keys:
        map50_per_gflop, map50_per_param_m, map50_x_fps
    """
    ratios: Dict[str, Any] = {
        "map50_per_gflop":   None,
        "map50_per_param_m": None,
        "map50_x_fps":       None,
    }

    map50   = acc.get("map50")
    gflops  = eff.get("gflops")
    params  = eff.get("params_m")
    fps     = eff.get("fps")

    if map50 is not None and gflops and gflops > 0:
        ratios["map50_per_gflop"] = round(map50 / gflops, 4)

    if map50 is not None and params and params > 0:
        ratios["map50_per_param_m"] = round(map50 / params, 4)

    if map50 is not None and fps and fps > 0:
        ratios["map50_x_fps"] = round(map50 * fps, 2)

    return ratios


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val: Any, fmt: str = ".4f", unit: str = "") -> str:
    """Format giá trị, trả về 'N/A' nếu None."""
    if val is None:
        return "N/A"
    try:
        return f"{val:{fmt}}{unit}"
    except (ValueError, TypeError):
        return str(val)


def print_full_report(
    val_results: Any,
    eff: Optional[Dict] = None,
    split: str = "val",
    model_name: str = "YOLOv12n",
    idea: str = "baseline",
    weights_path: Optional[str | Path] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    In full metrics report và trả về dict tổng hợp.

    Args:
        val_results:    Kết quả từ model.val()
        eff:            Dict từ compute_efficiency_metrics() (None → bỏ qua)
        split:          'val' hoặc 'test'
        model_name:     Tên model để hiển thị
        idea:           Tên idea
        weights_path:   Đường dẫn weights (để tính model size nếu eff=None)
        logger_instance: Logger tùy chỉnh (None → dùng module logger)

    Returns:
        dict tổng hợp tất cả metrics
    """
    log = logger_instance or logger

    acc    = parse_accuracy_metrics(val_results)
    eff    = eff or {}
    ratios = compute_ratios(acc, eff)

    W = 58   # độ rộng bảng

    def line(text: str = "") -> None:
        log.info(text)

    def header(title: str) -> None:
        line("─" * W)
        log.info(f"  {title}")
        line("─" * W)

    def row(label: str, value: str) -> None:
        log.info(f"  {label:<32} {value}")

    # ── Header ────────────────────────────────────────────────────────────────
    line("=" * W)
    log.info(f"  EVALUATION REPORT  [{split.upper()}]  —  {model_name} / {idea}")
    line("=" * W)

    # ── 1. Accuracy metrics ───────────────────────────────────────────────────
    header("1. ACCURACY METRICS")
    row("mAP50",               _fmt(acc["map50"]))
    row("mAP50-95",            _fmt(acc["map50_95"]))
    row("Precision (mean)",    _fmt(acc["precision"]))
    row("Recall / AR (mean)",  _fmt(acc["recall"]))
    line()

    # Per-class AP50
    if acc["ap50_per_class"]:
        log.info("  Per-class AP50:")
        max_ap = max(acc["ap50_per_class"].values(), default=1.0) or 1.0
        for cls_name, ap50_val in acc["ap50_per_class"].items():
            bar_len = int(ap50_val / max_ap * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            log.info(f"    {cls_name:<20} {ap50_val:.4f}  {bar}")
        line()

    # Per-class AP50-95
    if acc["ap_per_class"]:
        log.info("  Per-class AP50-95:")
        for cls_name, ap_val in acc["ap_per_class"].items():
            log.info(f"    {cls_name:<20} {ap_val:.4f}")
    line()

    # ── 2. Efficiency metrics ─────────────────────────────────────────────────
    header("2. EFFICIENCY METRICS")
    row("Parameters",          _fmt(eff.get("params_m"),      ".3f", " M"))
    row("GFLOPs",              _fmt(eff.get("gflops"),        ".2f", " G"))
    row("Model size",          _fmt(eff.get("model_size_mb"), ".2f", " MB"))
    row("Latency (batch=1)",   _fmt(eff.get("latency_ms"),    ".2f", " ms/img"))
    row("FPS (batch=1)",       _fmt(eff.get("fps"),           ".1f", " fps"))
    if eff.get("fps_batch") is not None:
        row("Throughput (batch>1)", _fmt(eff.get("fps_batch"), ".1f", " fps"))
    row("Device",              str(eff.get("device_used") or "N/A"))
    line()

    # ── 3. Efficiency ratios ──────────────────────────────────────────────────
    header("3. EFFICIENCY RATIOS")
    row("mAP50 / GFLOPs",     _fmt(ratios["map50_per_gflop"],   ".4f"))
    row("mAP50 / Params (M)", _fmt(ratios["map50_per_param_m"], ".4f"))
    row("mAP50 × FPS",        _fmt(ratios["map50_x_fps"],       ".2f"))
    line("=" * W)

    # ── Return consolidated dict ──────────────────────────────────────────────
    return {
        "split":   split,
        "model":   model_name,
        "idea":    idea,
        **{f"acc_{k}": v for k, v in acc.items()},
        **{f"eff_{k}": v for k, v in eff.items()},
        **{f"ratio_{k}": v for k, v in ratios.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(
    report: Dict[str, Any],
    csv_path: str | Path,
) -> None:
    """
    Ghi kết quả metrics vào CSV để so sánh nhiều experiments.

    Mỗi lần gọi append 1 dòng. Tạo file mới + header nếu chưa có.
    """
    import csv
    import datetime

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    flat = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **report,
    }

    # Bỏ các key chứa dict (per-class) — không phù hợp với CSV flat
    flat_clean = {
        k: v for k, v in flat.items()
        if not isinstance(v, dict)
    }

    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_clean.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat_clean)

    logger.info(f"[metrics] Results appended to: {csv_path}")
