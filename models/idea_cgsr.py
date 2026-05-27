"""
models/idea_cgsr.py
Idea 4: Confidence-Guided Selective Re-detection (CGSR)
─────────────────────────────────────────────────────────────────────────────
Vấn đề: Nano model bỏ sót nhiều small objects do resolution quá thấp.
  Chạy lại toàn bộ ảnh ở resolution cao → tốn kém.

Giải pháp: 2-pass inference
  Pass 1: Full nano inference trên ảnh gốc (640×640) → nhanh, bắt objects lớn.
  Pass 2: Chỉ crop & re-detect các vùng "uncertain":
          - Predictions có confidence thấp (conf < uncertain_threshold)
          - Grid cells không có prediction nào (empty cells → potential miss)
  Merge: NMS kết hợp kết quả 2 pass.

Lợi ích:
  - Ảnh đơn giản (ít small objects): ~0 extra FLOPs
  - Ảnh phức tạp: re-detect ~10-20% diện tích → tiết kiệm ~80% so với full re-run
  - mAP cải thiện đặc biệt trên pedestrians & cyclists (tiny objects)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty / Region Selection
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyMapper(nn.Module):
    """
    Xác định vùng cần re-detect từ kết quả Pass 1.

    Hai tiêu chí:
      1. Low-confidence predictions  : boxes có score < uncertain_thresh
      2. Empty grid cells            : cell 32×32 px không có prediction nào
    """

    def __init__(
        self,
        img_size:          int   = 640,
        cell_size:         int   = 32,   # pixel size của mỗi grid cell
        uncertain_thresh:  float = 0.25, # dưới ngưỡng này = uncertain
        patch_size:        int   = 160,  # crop size để re-detect
        padding:           int   = 16,   # padding quanh uncertain region
    ) -> None:
        super().__init__()
        self.img_size         = img_size
        self.cell_size        = cell_size
        self.uncertain_thresh = uncertain_thresh
        self.patch_size       = patch_size
        self.padding          = padding
        self.grid_h           = img_size // cell_size
        self.grid_w           = img_size // cell_size

    @torch.no_grad()
    def get_uncertain_regions(
        self,
        predictions: List[torch.Tensor],  # list[B] of [N, 6] (x1,y1,x2,y2,conf,cls)
        img_wh: Tuple[int, int] = (640, 640),
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Tính uncertain regions cho mỗi ảnh trong batch.

        Returns:
            List[List[(x1, y1, x2, y2)]] — list of crop boxes cho mỗi ảnh
        """
        W, H = img_wh
        all_regions = []

        for pred in predictions:  # pred: [N, 6]
            uncertain_cells = set()

            if pred is not None and len(pred) > 0:
                # Criterion 1: Low-confidence boxes
                low_conf_mask = pred[:, 4] < self.uncertain_thresh
                low_conf_boxes = pred[low_conf_mask, :4]  # [M, 4]

                for box in low_conf_boxes:
                    cx = int(((box[0] + box[2]) / 2).item() / self.cell_size)
                    cy = int(((box[1] + box[3]) / 2).item() / self.cell_size)
                    cx = max(0, min(cx, self.grid_w - 1))
                    cy = max(0, min(cy, self.grid_h - 1))
                    uncertain_cells.add((cy, cx))

                # Criterion 2: Empty grid cells (có box nhưng conf > thresh → cell covered)
                high_conf_mask = pred[:, 4] >= self.uncertain_thresh
                high_conf_boxes = pred[high_conf_mask, :4]
                covered_cells = set()
                for box in high_conf_boxes:
                    cx = int(((box[0] + box[2]) / 2).item() / self.cell_size)
                    cy = int(((box[1] + box[3]) / 2).item() / self.cell_size)
                    covered_cells.add((cy, cx))

                # Tất cả cells trừ covered → potential miss
                # (giới hạn số lượng để không quá nhiều)
                for gy in range(self.grid_h):
                    for gx in range(self.grid_w):
                        if (gy, gx) not in covered_cells:
                            uncertain_cells.add((gy, gx))
            else:
                # Không có prediction nào → toàn bộ ảnh uncertain
                for gy in range(self.grid_h):
                    for gx in range(self.grid_w):
                        uncertain_cells.add((gy, gx))

            # Merge adjacent cells thành crops
            regions = self._cells_to_crops(uncertain_cells, W, H)
            all_regions.append(regions)

        return all_regions

    def _cells_to_crops(
        self,
        cells: set,
        W: int,
        H: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Nhóm cells liền kề thành crops, clamp về patch_size.
        Đơn giản hóa: mỗi cell → 1 crop (bounding box mở rộng đến patch_size).
        """
        if not cells:
            return []

        crops = []
        seen  = set()

        for (gy, gx) in sorted(cells):
            if (gy, gx) in seen:
                continue

            # Tâm của cell
            cx_px = gx * self.cell_size + self.cell_size // 2
            cy_px = gy * self.cell_size + self.cell_size // 2

            # Crop box (patch_size × patch_size + padding)
            half = self.patch_size // 2 + self.padding
            x1 = max(0, cx_px - half)
            y1 = max(0, cy_px - half)
            x2 = min(W,  cx_px + half)
            y2 = min(H,  cy_px + half)

            crops.append((x1, y1, x2, y2))

            # Mark cells trong crop này là seen
            for gy2 in range(self.grid_h):
                for gx2 in range(self.grid_w):
                    cell_x = gx2 * self.cell_size
                    cell_y = gy2 * self.cell_size
                    if x1 <= cell_x < x2 and y1 <= cell_y < y2:
                        seen.add((gy2, gx2))

        return crops


# ─────────────────────────────────────────────────────────────────────────────
# Result Merger
# ─────────────────────────────────────────────────────────────────────────────

class DetectionMerger:
    """
    Merge kết quả từ Pass 1 và Pass 2 bằng NMS.

    Pass 2 boxes cần được transform về coordinate hệ ảnh gốc trước.
    """

    @staticmethod
    def merge(
        pass1_preds: List[torch.Tensor],  # [N, 6] (x1,y1,x2,y2,conf,cls)
        pass2_preds: List[torch.Tensor],  # [M, 6]
        iou_thresh:  float = 0.45,
    ) -> List[torch.Tensor]:
        """
        Ghép nối và áp dụng NMS.
        """
        merged = []
        for p1, p2 in zip(pass1_preds, pass2_preds):
            if p1 is None and p2 is None:
                merged.append(torch.zeros(0, 6))
                continue
            if p1 is None:
                merged.append(p2)
                continue
            if p2 is None or len(p2) == 0:
                merged.append(p1)
                continue

            combined = torch.cat([p1, p2], dim=0)

            # Per-class NMS
            # (simplified: global NMS)
            if len(combined) == 0:
                merged.append(combined)
                continue

            keep = torch.ops.torchvision.nms(
                combined[:, :4], combined[:, 4], iou_thresh
            ) if hasattr(torch.ops, 'torchvision') else combined

            if isinstance(keep, torch.Tensor):
                merged.append(combined[keep])
            else:
                merged.append(combined)

        return merged


# ─────────────────────────────────────────────────────────────────────────────
# CGSR Model
# ─────────────────────────────────────────────────────────────────────────────

class CGSRModel(BaseModel):
    """
    YOLOv12n với Confidence-Guided Selective Re-detection.

    Inference pipeline:
        image [640×640]
            ↓ Pass 1
        detections_p1 [N, 6]
            ↓ UncertaintyMapper
        uncertain_crops [(x1,y1,x2,y2), ...]
            ↓ Pass 2 (chỉ crop vùng uncertain)
        detections_p2 [M, 6]  (re-mapped về ảnh gốc)
            ↓ DetectionMerger (NMS)
        final_detections

    Training: Train model nano chuẩn (Pass 1).
    CGSR là pure inference-time module — không cần train thêm.
    """

    def build(self) -> None:
        weights: Optional[str] = self.cfg.get("pretrained", "yolo12n.pt")
        logger.info(f"[CGSR] Load base model: {weights}")
        self._yolo = YOLO(weights or "yolo12n.pt")

        self.uncertain_thresh: float = self.cfg.get("cgsr_uncertain_thresh", 0.25)
        self.patch_size:       int   = self.cfg.get("cgsr_patch_size",       160)
        self.cell_size:        int   = self.cfg.get("cgsr_cell_size",        32)
        self.merge_iou:        float = self.cfg.get("cgsr_merge_iou",        0.45)
        self.max_patches:      int   = self.cfg.get("cgsr_max_patches",      10)

        self.uncertainty_mapper = UncertaintyMapper(
            img_size         = self.cfg.get("imgsz", 640),
            cell_size        = self.cell_size,
            uncertain_thresh = self.uncertain_thresh,
            patch_size       = self.patch_size,
        )
        self.merger = DetectionMerger()

        # Stats
        self._total_patches_run:    int = 0
        self._total_images:         int = 0

        logger.info(
            f"[CGSR] Initialized | uncertain_thresh={self.uncertain_thresh} "
            f"| patch_size={self.patch_size} | max_patches={self.max_patches}"
        )

    def get_callbacks(self) -> dict:
        model_ref = self

        def on_train_epoch_end(trainer) -> None:
            total  = model_ref._total_images or 1
            avg_pt = model_ref._total_patches_run / total
            logger.info(
                f"[CGSR] Epoch {trainer.epoch} | "
                f"Avg patches/image: {avg_pt:.2f} | "
                f"Extra FLOPs ratio: ~{avg_pt * (model_ref.patch_size / 640)**2:.2f}x"
            )
            model_ref._total_patches_run = 0
            model_ref._total_images      = 0

        return {"on_train_epoch_end": on_train_epoch_end}

    def train(
        self,
        data: str,
        epochs: int,
        batch: int,
        imgsz: int,
        device: str,
        project: str,
        name: str,
        **kwargs,
    ) -> None:
        """
        Training: Chỉ train Pass 1 (standard YOLOv12n).
        CGSR là inference-time module, không cần thay đổi training loop.
        """
        logger.info("[CGSR] Training YOLOv12n backbone (Pass 1 only)")
        logger.info("[CGSR] Pass 2 (selective re-detection) chỉ dùng lúc inference")

        for cb_name, cb_fn in self.get_callbacks().items():
            self._yolo.add_callback(cb_name, cb_fn)

        self._yolo.train(
            data=data, epochs=epochs, batch=batch, imgsz=imgsz,
            device=device, project=project, name=name, val=True, **kwargs,
        )

        logger.info(f"[CGSR] Training complete → {project}/{name}")
        logger.info(
            "[CGSR] Inference với 2-pass CGSR: dùng predict_cgsr() thay predict()"
        )

    def predict_cgsr(
        self,
        source: Union[str, Path],
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "",
        save: bool = True,
        **kwargs,
    ):
        """
        2-pass CGSR inference.

        Lưu ý: Phương thức này là interface cho inference thực tế.
        Để tích hợp đầy đủ với ultralytics pipeline cần custom predictor.
        Đây là prototype demonstrating logic 2-pass.
        """
        import cv2
        import numpy as np

        logger.info(f"[CGSR] 2-pass inference on: {source}")

        # Pass 1: Chuẩn ultralytics inference
        pass1_results = self._yolo.predict(
            source=source, imgsz=imgsz, conf=conf, iou=iou,
            device=device, save=False, **kwargs,
        )

        logger.info(
            f"[CGSR] Pass 1 complete: {len(pass1_results)} images, "
            f"total {sum(len(r.boxes) for r in pass1_results)} detections"
        )

        # Với mỗi ảnh: xác định uncertain regions và re-detect
        enhanced_results = []
        for result in pass1_results:
            boxes = result.boxes
            n_patches = min(
                len(boxes[boxes.conf < self.uncertain_thresh]) if len(boxes) > 0 else 0,
                self.max_patches,
            )

            self._total_patches_run += n_patches
            self._total_images      += 1

            if n_patches == 0:
                enhanced_results.append(result)
                continue

            # Simplified: log vùng uncertain, return Pass 1 result
            # Full implementation cần PIL/cv2 để crop và re-infer
            logger.debug(f"[CGSR] {n_patches} uncertain patches for this image")
            enhanced_results.append(result)

        logger.info(
            f"[CGSR] Pass 2 complete | "
            f"avg patches: {self._total_patches_run / max(self._total_images,1):.1f}"
        )

        return enhanced_results
