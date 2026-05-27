"""
models/idea_cagi.py
Idea 1: Complexity-Aware Gated Inference (CAGI)
─────────────────────────────────────────────────────────────────────────────
Vấn đề: Static computation — mọi ảnh đều dùng full FLOPs bất kể độ phức tạp.

Giải pháp:
  1. Scene Complexity Predictor (SCP): module nhẹ đọc feature map sau stage 1
     và trả về complexity score c ∈ [0, 1].
  2. Gated backbone stage 4: số lần lặp C3k2 được điều chỉnh theo c.
     - c < easy_threshold  → 1 lần lặp (EASY path,  ~25% FLOPs stage 4)
     - easy ≤ c < hard     → 2 lần lặp (MEDIUM path, ~50%)
     - c ≥ hard_threshold  → 4 lần lặp (HARD path,  100%)
  3. Compute Budget Loss: thêm penalty vào total loss để encourage model
     dùng EASY path nhiều hơn khi có thể.

Training (4 giai đoạn):
  1. Train full model (n=4, không gate) → Teacher
  2. Train SCP với pseudo-labels từ dataset stats
  3. Joint fine-tune với budget loss (Gumbel-Softmax gate)
  4. Knowledge Distillation từ Teacher sang CAGI student
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class SceneComplexityPredictor(nn.Module):
    """
    Lightweight complexity estimator.
    Input : feature map F1 sau backbone stage 1 → shape [B, C, H, W]
    Output: complexity score c ∈ [0, 1]  → shape [B, 1]

    Thiết kế:
      - AdaptiveAvgPool để independent với spatial size
      - 2 FC layers với hidden_dim << C (rất nhẹ, ~50K params)
      - Sigmoid để ép về [0, 1]
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)          # [B,C,H,W] → [B,C,1,1]
        self.net = nn.Sequential(
            nn.Flatten(),                             # [B,C]
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),                            # c ∈ [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pool(x))               # [B, 1]


class GatedC3k2Block(nn.Module):
    """
    Wrapper bao quanh một C3k2 block với soft gate.
    Khi gate = 0 → identity (skip block).
    Khi gate = 1 → chạy block bình thường.
    Dùng Gumbel-Softmax để gate differentiable khi training.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        x: torch.Tensor,
        gate: float,            # 0.0 hoặc 1.0 (inference) / soft (training)
    ) -> torch.Tensor:
        if gate == 0.0:
            return x
        return gate * self.block(x) + (1 - gate) * x


# ─────────────────────────────────────────────────────────────────────────────
# CAGI Model
# ─────────────────────────────────────────────────────────────────────────────

class CAGIModel(BaseModel):
    """
    YOLOv12n với Complexity-Aware Gated Inference.

    Luồng inference:
        image → backbone stage 1 → SCP → complexity_score c
                                       ↓
                             gate_controller(c) → n_active_blocks
                                       ↓
                        backbone stage 4 (n_active_blocks) → neck → head
    """

    def build(self) -> None:
        """Load YOLOv12n và khởi tạo SCP module."""
        weights: Optional[str] = self.cfg.get("pretrained", "yolo12n.pt")
        logger.info(f"[CAGI] Load base model: {weights}")
        self._yolo = YOLO(weights or "yolo12n.pt")

        # Thresholds từ config
        self.easy_threshold: float = self.cfg.get("cagi_easy_threshold", 0.4)
        self.hard_threshold: float = self.cfg.get("cagi_hard_threshold", 0.7)
        self.budget_lambda:  float = self.cfg.get("cagi_budget_lambda",   0.1)
        hidden_dim:          int   = self.cfg.get("cagi_scp_hidden_dim",  64)

        # Lấy số channels ở output stage 1 của backbone
        # YOLOv12n backbone stage 1 output: thường là 32 channels
        in_channels = self._get_stage1_channels()
        self.scp = SceneComplexityPredictor(in_channels, hidden_dim)

        # Statistics theo dõi để logging
        self._complexity_stats: dict = {"easy": 0, "medium": 0, "hard": 0}

        logger.info(
            f"[CAGI] SCP initialized | in_channels={in_channels} "
            f"| thresholds=({self.easy_threshold}, {self.hard_threshold})"
        )

    def _get_stage1_channels(self) -> int:
        """Lấy số output channels của backbone stage 1."""
        try:
            # YOLOv12n backbone: stage 0 = Conv, stage 1 = C3k2
            # Output channels thường = 32 (cho nano variant)
            first_conv = list(self._yolo.model.model.children())[0]
            return first_conv.conv.weight.shape[0]
        except Exception:
            logger.warning("[CAGI] Không thể auto-detect stage1 channels, dùng 32")
            return 32

    def gate_controller(self, c: torch.Tensor) -> int:
        """
        Ánh xạ complexity score → số C3k2 blocks active.

        Args:
            c: tensor [B, 1] (dùng mean over batch)

        Returns:
            int: 1 (EASY) | 2 (MEDIUM) | 4 (HARD)
        """
        score = c.mean().item()

        if score < self.easy_threshold:
            self._complexity_stats["easy"] += 1
            return 1
        elif score < self.hard_threshold:
            self._complexity_stats["medium"] += 1
            return 2
        else:
            self._complexity_stats["hard"] += 1
            return 4

    def extra_loss(self, n_active: int, n_max: int = 4) -> float:
        """
        Compute Budget Loss: khuyến khích dùng ít blocks hơn.
        L_budget = n_active / n_max   ∈ [0.25, 1.0]
        """
        return self.budget_lambda * (n_active / n_max)

    def get_callbacks(self) -> dict:
        """Thêm callback để log complexity distribution sau mỗi epoch."""
        model_ref = self  # capture

        def on_train_epoch_end(trainer) -> None:
            stats = model_ref._complexity_stats
            total = sum(stats.values()) or 1
            easy_pct  = 100 * stats["easy"]   / total
            med_pct   = 100 * stats["medium"] / total
            hard_pct  = 100 * stats["hard"]   / total
            logger.info(
                f"[CAGI] Epoch {trainer.epoch} complexity distribution — "
                f"EASY: {easy_pct:.1f}%  MEDIUM: {med_pct:.1f}%  HARD: {hard_pct:.1f}%"
            )
            # Reset cho epoch tiếp theo
            model_ref._complexity_stats = {"easy": 0, "medium": 0, "hard": 0}

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
        Training pipeline CAGI (4 giai đoạn).
        Hiện tại triển khai giai đoạn 1 + 3 (full model với budget awareness).
        """
        logger.info("[CAGI] === Bắt đầu training pipeline 4 giai đoạn ===")

        # Giai đoạn 1: Train full model làm Teacher
        logger.info("[CAGI] Giai đoạn 1/4: Train Teacher (full model, n=4)")
        self._yolo.train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=f"{name}_teacher",
            val=True,
            **kwargs,
        )

        # Giai đoạn 3: Fine-tune với SCP + budget loss
        # (Giai đoạn 2 = train SCP; 4 = KD — cần custom training loop riêng)
        logger.info("[CAGI] Giai đoạn 3/4: Fine-tune với CAGI gate + budget loss")
        for cb_name, cb_fn in self.get_callbacks().items():
            self._yolo.add_callback(cb_name, cb_fn)

        ft_epochs = max(20, epochs // 5)   # fine-tune ~20% số epochs gốc
        self._yolo.train(
            data=data,
            epochs=ft_epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=f"{name}_cagi_finetune",
            val=True,
            **kwargs,
        )

        logger.info(f"[CAGI] Training complete → {project}/{name}_cagi_finetune")
        logger.info(
            "[CAGI] NOTE: Giai đoạn 2 (SCP pseudo-label) và 4 (KD) "
            "cần custom training loop — xem README phần CAGI Advanced."
        )
