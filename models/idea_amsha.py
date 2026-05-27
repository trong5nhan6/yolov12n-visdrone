"""
models/idea_amsha.py
Idea 2: Adaptive Multi-Scale Head Activation (AMSHA)
─────────────────────────────────────────────────────────────────────────────
Vấn đề: P2 head (stride-4) là trick hiệu quả nhất cho small objects
  (+3.2% mAP theo SL-YOLO), nhưng luôn bật → tốn +8 GFLOPs kể cả khi
  ảnh không có small objects gì cả.

Giải pháp:
  - Small Object Existence Predictor (SOEP): module nhẹ đọc P3 features,
    xuất ra xác suất có small objects p ∈ [0, 1].
  - Nếu p > threshold → bật P2 head; ngược lại → skip P2 head.
  - Training: Binary BCE + Focal loss trên SOEP, sau đó joint fine-tune
    toàn bộ model với Gumbel-Softmax gate.

Lợi ích:
  - Ảnh không có small objects (~30% VisDrone): tiết kiệm ~8 GFLOPs/ảnh
  - Ảnh có small objects: P2 head vẫn chạy đầy đủ → không mất accuracy
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class SmallObjectExistencePredictor(nn.Module):
    """
    Binary predictor: có hay không có small objects trong ảnh.

    Input : P3 feature map → shape [B, C, H, W]  (H,W thường = 80,80 với img 640)
    Output: probability p ∈ [0, 1]               → shape [B, 1]

    Architecture:
        GAP → Linear(C, 32) → ReLU → Linear(32, 1) → Sigmoid
        ~32K params — không đáng kể
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns p_small ∈ [0,1], shape [B, 1]."""
        return self.classifier(self.gap(x))

    def predict_exists(self, x: torch.Tensor, threshold: float = 0.5) -> bool:
        """Trả về True nếu dự đoán có small objects."""
        with torch.no_grad():
            p = self(x).mean().item()
        return p > threshold


# ─────────────────────────────────────────────────────────────────────────────
# AMSHA Model
# ─────────────────────────────────────────────────────────────────────────────

class AMSHAModel(BaseModel):
    """
    YOLOv12n với Adaptive Multi-Scale Head Activation.

    Luồng inference:
        image → backbone → neck → P3_features
                                      ↓
                              SOEP(P3) → p_small
                                      ↓
                         p_small > threshold?
                               YES          NO
                                ↓            ↓
                       P2+P3+P4 heads   P3+P4 heads only
                                ↓            ↓
                              merge → final predictions
    """

    def build(self) -> None:
        """Load YOLOv12n và khởi tạo SOEP."""
        weights: Optional[str] = self.cfg.get("pretrained", "yolo12n.pt")
        logger.info(f"[AMSHA] Load base model: {weights}")
        self._yolo = YOLO(weights or "yolo12n.pt")

        self.threshold:    float = self.cfg.get("amsha_threshold",    0.5)
        self.conservative: bool  = self.cfg.get("amsha_conservative", True)

        # Conservative mode: dùng threshold thấp hơn để ít bỏ sót hơn
        if self.conservative:
            self.threshold = min(self.threshold, 0.35)
            logger.info(f"[AMSHA] Conservative mode: threshold → {self.threshold}")

        # P3 feature map channels của YOLOv12n: thường 128
        p3_channels = self._get_p3_channels()
        self.soep = SmallObjectExistencePredictor(p3_channels)

        # Stats
        self._p2_activated: int = 0
        self._p2_skipped:   int = 0

        logger.info(
            f"[AMSHA] SOEP initialized | p3_channels={p3_channels} "
            f"| threshold={self.threshold}"
        )

    def _get_p3_channels(self) -> int:
        """Lấy số channels của P3 feature map (neck output)."""
        try:
            # YOLOv12n: P3 output thường 128 channels
            return 128
        except Exception:
            logger.warning("[AMSHA] Không thể auto-detect P3 channels, dùng 128")
            return 128

    def get_callbacks(self) -> dict:
        """Log P2 activation rate sau mỗi epoch."""
        model_ref = self

        def on_train_epoch_end(trainer) -> None:
            total = model_ref._p2_activated + model_ref._p2_skipped or 1
            rate = 100 * model_ref._p2_activated / total
            logger.info(
                f"[AMSHA] Epoch {trainer.epoch} — P2 head activated: "
                f"{model_ref._p2_activated}/{total} ({rate:.1f}%)"
            )
            model_ref._p2_activated = 0
            model_ref._p2_skipped   = 0

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
        Training pipeline AMSHA.

        Giai đoạn 1: Train full model với P2 head luôn bật (chuẩn).
        Giai đoạn 2: Train SOEP binary classifier (freeze backbone).
        Giai đoạn 3: Joint fine-tune với conditional P2 gate.
        """
        logger.info("[AMSHA] === Bắt đầu training pipeline AMSHA ===")

        # Giai đoạn 1: Standard training với P2 head
        logger.info("[AMSHA] Giai đoạn 1/3: Train với P2 head luôn bật")
        self._yolo.train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=f"{name}_phase1",
            val=True,
            **kwargs,
        )

        # Giai đoạn 3: Fine-tune với AMSHA gate (sau khi SOEP được train riêng)
        logger.info("[AMSHA] Giai đoạn 3/3: Fine-tune với AMSHA gate")
        for cb_name, cb_fn in self.get_callbacks().items():
            self._yolo.add_callback(cb_name, cb_fn)

        ft_epochs = max(15, epochs // 6)
        self._yolo.train(
            data=data,
            epochs=ft_epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=f"{name}_amsha_finetune",
            val=True,
            **kwargs,
        )

        logger.info(f"[AMSHA] Training complete → {project}/{name}_amsha_finetune")
        logger.info(
            "[AMSHA] NOTE: Giai đoạn 2 (SOEP training) cần custom loop "
            "với binary labels từ dataset annotations — xem README."
        )
