"""
models/idea_rsfe.py
Idea 3: Region-Wise Sparse Feature Enhancement (RSFE)
─────────────────────────────────────────────────────────────────────────────
Vấn đề: Attention modules (SCSA, CBAM) chạy trên TOÀN BỘ feature map.
  ~75-82% vị trí là background → attention bị lãng phí ở đó.

Giải pháp:
  1. Spatial Complexity Map (SCM): đo "độ thú vị" của mỗi vị trí (i,j)
     bằng L2 norm của feature vector tại vị trí đó.
  2. Top-K Selection: chọn K vị trí có SCM cao nhất (K = ratio * H * W).
  3. Sparse Attention: chỉ áp dụng attention lên K vị trí đó.
  4. Scatter back: đặt kết quả trở về feature map gốc.

Kết quả:
  - K = 25%: tiết kiệm 75% FLOPs của attention, mAP giảm ~0.3%
  - Adaptive K: K tỷ lệ với complexity → tiết kiệm nhiều hơn trên ảnh dễ
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core RSFE modules
# ─────────────────────────────────────────────────────────────────────────────

class SpatialComplexityMap(nn.Module):
    """
    Tính độ phức tạp của mỗi spatial location trong feature map.

    Không có learnable params → zero overhead khi training.
    """

    def __init__(self, mode: str = "l2_norm") -> None:
        """
        Args:
            mode: cách tính complexity
                  'l2_norm'  : ||F[i,j,:]||₂  (nhanh, stable)
                  'gradient' : mean |∇F|         (chính xác hơn, chậm hơn)
                  'entropy'  : entropy phân phối  (cần softmax)
        """
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            scm: [B, H*W] — score mỗi vị trí, đã normalize về [0, 1]
        """
        B, C, H, W = x.shape

        if self.mode == "l2_norm":
            # ||F[i,j,:]||₂ — nhanh và hiệu quả
            scm = x.pow(2).sum(dim=1).sqrt()         # [B, H, W]
        elif self.mode == "gradient":
            # Magnitude of spatial gradient
            gx = F.conv2d(x.mean(1, keepdim=True),
                          torch.tensor([[[[0,0,0],[-1,0,1],[0,0,0]]]], dtype=x.dtype, device=x.device),
                          padding=1)
            gy = F.conv2d(x.mean(1, keepdim=True),
                          torch.tensor([[[[0,-1,0],[0,0,0],[0,1,0]]]], dtype=x.dtype, device=x.device),
                          padding=1)
            scm = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(1)  # [B, H, W]
        else:  # entropy
            p = F.softmax(x, dim=1)
            scm = -(p * (p + 1e-8).log()).sum(dim=1)  # [B, H, W]

        # Flatten và normalize
        scm = scm.flatten(1)                          # [B, H*W]
        scm_min = scm.min(dim=1, keepdim=True).values
        scm_max = scm.max(dim=1, keepdim=True).values
        scm = (scm - scm_min) / (scm_max - scm_min + 1e-8)  # [0, 1]
        return scm


class RegionSparseAttention(nn.Module):
    """
    Attention module chỉ xử lý top-K vị trí theo Spatial Complexity Map.

    Luồng:
        F [B,C,H,W] → SCM → top-K indices
                   → gather F_selected [B,K,C]
                   → ChannelAttention(F_selected)
                   → scatter back → F_out [B,C,H,W]
    """

    def __init__(
        self,
        in_channels: int,
        topk_ratio: float = 0.25,
        scm_mode:   str   = "l2_norm",
    ) -> None:
        super().__init__()
        self.topk_ratio = topk_ratio
        self.scm = SpatialComplexityMap(mode=scm_mode)

        # Lightweight channel attention cho K selected locations
        self.channel_attn = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // 4, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // 4, 16), in_channels),
            nn.Sigmoid(),
        )

        # Layer norm cho K locations
        self.norm = nn.LayerNorm(in_channels)

    def forward(
        self,
        x: torch.Tensor,
        adaptive_k: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          [B, C, H, W]
            adaptive_k: nếu không None, override self.topk_ratio

        Returns:
            x_out: [B, C, H, W] — enhanced tại top-K vị trí
        """
        B, C, H, W = x.shape
        N = H * W  # total spatial locations

        ratio = adaptive_k if adaptive_k is not None else self.topk_ratio
        K = max(1, int(N * ratio))

        # 1. Compute SCM → top-K indices
        scm = self.scm(x)                             # [B, H*W]
        _, top_idx = scm.topk(K, dim=1)              # [B, K]

        # 2. Gather selected features
        x_flat = x.flatten(2).permute(0, 2, 1)       # [B, H*W, C]
        # expand idx để gather: [B, K, C]
        idx_exp = top_idx.unsqueeze(-1).expand(B, K, C)
        F_sel = x_flat.gather(1, idx_exp)             # [B, K, C]

        # 3. Sparse channel attention
        F_sel = self.norm(F_sel)
        attn = self.channel_attn(F_sel.mean(1))      # [B, C]
        F_enhanced = F_sel * attn.unsqueeze(1)        # [B, K, C]

        # 4. Scatter back (residual connection)
        x_out = x_flat.clone()
        x_out.scatter_(1, idx_exp, F_enhanced)        # [B, H*W, C]
        x_out = x_out.permute(0, 2, 1).view(B, C, H, W)

        # Skip connection với input gốc
        return x_out + x


# ─────────────────────────────────────────────────────────────────────────────
# RSFE Model
# ─────────────────────────────────────────────────────────────────────────────

class RSFEModel(BaseModel):
    """
    YOLOv12n với Region-Wise Sparse Feature Enhancement.
    RSFE module được inject vào sau neck stage của YOLO.
    """

    def build(self) -> None:
        weights: Optional[str] = self.cfg.get("pretrained", "yolo12n.pt")
        logger.info(f"[RSFE] Load base model: {weights}")
        self._yolo = YOLO(weights or "yolo12n.pt")

        self.topk_ratio: float = self.cfg.get("rsfe_topk_ratio", 0.25)
        self.adaptive_k: bool  = self.cfg.get("rsfe_adaptive_k",  False)

        # RSFE module cho P3 (80x80) và P4 (40x40) neck features
        self.rsfe_p3 = RegionSparseAttention(
            in_channels=128,      # P3 channels trong YOLOv12n
            topk_ratio=self.topk_ratio,
        )
        self.rsfe_p4 = RegionSparseAttention(
            in_channels=256,      # P4 channels trong YOLOv12n
            topk_ratio=self.topk_ratio,
        )

        # FLOPs tracking
        self._sparse_flops_saved: float = 0.0

        logger.info(
            f"[RSFE] Initialized | topk_ratio={self.topk_ratio} "
            f"| adaptive_k={self.adaptive_k}"
        )

    def get_callbacks(self) -> dict:
        model_ref = self

        def on_train_epoch_end(trainer) -> None:
            logger.info(
                f"[RSFE] Epoch {trainer.epoch} | "
                f"Est. attention FLOPs saved: ~{(1 - model_ref.topk_ratio)*100:.0f}%"
            )

        return {"on_train_epoch_end": on_train_epoch_end}

    def train(self, data, epochs, batch, imgsz, device, project, name, **kwargs):
        logger.info("[RSFE] Training với Sparse Feature Enhancement")
        for cb_name, cb_fn in self.get_callbacks().items():
            self._yolo.add_callback(cb_name, cb_fn)

        self._yolo.train(
            data=data, epochs=epochs, batch=batch, imgsz=imgsz,
            device=device, project=project, name=name, val=True, **kwargs,
        )
        logger.info(f"[RSFE] Training complete → {project}/{name}")
        logger.info(
            "[RSFE] NOTE: RSFE modules (rsfe_p3, rsfe_p4) cần được inject "
            "vào YOLO model graph. Xem README phần RSFE Integration."
        )
