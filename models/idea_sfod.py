"""
models/idea_sfod.py
Idea: SFOD — Small Feature-enhanced Object Detector
─────────────────────────────────────────────────────────────────────────────
Vấn đề: YOLOv12n baseline mAP50=0.263, nhưng các class nhỏ rất thấp:
  - people:   0.104  (pedestrian bị khuất, rất nhỏ)
  - bicycle:  0.064  (vài pixel trên ảnh drone)
  - motor:    0.226  (nhỏ, nhiều)

Nguyên nhân:
  YOLOv12n có 3 detection heads ở P3(80×80), P4(40×40), P5(20×20).
  Vật thể rất nhỏ (<8px) cần feature map 160×160 (P2) để có đủ spatial detail.
  Neck PAN dùng A2C2f nặng trong các layer downsample — nhiều params dư thừa.

Giải pháp (SFOD):
  1. P2 Detection Head (stride=4, 160×160)
     └─ Upsample P3 features → fuse với backbone P2 (layer[2], 64ch)
     └─ GhostConv 128→32ch (lightweight, ~1/4 params so với Conv+BN)
     └─ 4th detect head: bắt được objects nhỏ ≥ 4×4 px
     Paper: TPH-YOLOv5 (ICCVW 2021), BGF-YOLOv10 (2024)

  2. GhostConv Neck (PAN bottom-up path)
     └─ Thay A2C2f trong PAN bằng GhostConv (cheap linear feature generation)
     └─ Giảm ~40% params trong neck PAN, giảm GFLOPs
     Paper: GhostNet (CVPR 2020), BGF-YOLOv10, SL-YOLO (arxiv 2411.11477)

  3. Pretrained backbone transfer
     └─ Load yolo12n.pt, transfer backbone layers 0-8 (params shape khớp hoàn toàn)
     └─ FPN layers 9-14 transferred một phần (A2C2f args giống baseline)
     └─ P2 branch + PAN GhostConv: random init → học từ đầu

Expected vs baseline (0.2628 mAP50):
  - mAP50:    +8~15% → 0.285~0.305
  - people:   +25~40% → 0.13~0.15
  - bicycle:  +30~50% → 0.083~0.096
  - Params:   +0.2M net (P2 head +0.5M, GhostConv PAN -0.3M)
  - GFLOPs:   +1~2G (P2 at 160×160 adds compute, GhostConv saves some)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class SFODModel(BaseModel):
    """
    YOLOv12n với SFOD: 4-head detection (P2+P3+P4+P5) + GhostConv PAN neck.

    Luồng kiến trúc:
        Backbone (layers 0-8) — identical to YOLOv12n
            ↓
        FPN top-down (layers 9-14)
            ↓ (thêm mới)
        P2 branch: Upsample P3 → Concat backbone[2] → GhostConv(32ch)
            ↓
        PAN bottom-up: GhostConv (thay A2C2f)
            ↓
        Detect([P2=32ch, P3=64ch, P4=128ch, P5=256ch]) — 4 heads
    """

    def build(self) -> None:
        arch_yaml = self.cfg.get("sfod_arch_yaml", "configs/yolov12n_sfod.yaml")
        pretrained = self.cfg.get("pretrained", "yolo12n.pt")

        logger.info(f"[SFOD] Build model from: {arch_yaml}")
        self._yolo = YOLO(arch_yaml)

        # Transfer pretrained backbone + partial neck weights
        if pretrained:
            transferred = self._transfer_pretrained_weights(pretrained)
            logger.info(
                f"[SFOD] Weight transfer from '{pretrained}': "
                f"{transferred} tensors matched"
            )

        # Optionally freeze backbone
        if self.cfg.get("sfod_freeze_backbone", False):
            self._freeze_backbone()

        n_params = sum(p.numel() for p in self._yolo.model.parameters())
        logger.info(f"[SFOD] Model built | params={n_params:,}")

    # ── Weight transfer ───────────────────────────────────────────────────────

    def _transfer_pretrained_weights(self, pretrained_path: str) -> int:
        """
        Load yolo12n.pt và transfer tất cả layer có shape khớp vào SFOD model.

        Returns:
            Số lượng tensors được transfer thành công.
        """
        p = Path(pretrained_path)
        if not p.exists():
            logger.warning(
                f"[SFOD] Pretrained file '{pretrained_path}' không tồn tại — "
                "train from random init."
            )
            return 0

        try:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        except Exception:
            # fallback cho torch phiên bản cũ
            ckpt = torch.load(str(p), map_location="cpu")

        # Lấy state_dict từ checkpoint (ultralytics format)
        if isinstance(ckpt, dict):
            pretrained_sd = ckpt.get("model", ckpt)
            if hasattr(pretrained_sd, "float"):
                pretrained_sd = pretrained_sd.float()
            if hasattr(pretrained_sd, "state_dict"):
                pretrained_sd = pretrained_sd.state_dict()
        else:
            pretrained_sd = ckpt.state_dict()

        current_sd = self._yolo.model.state_dict()

        matched = {}
        for k, v in pretrained_sd.items():
            if k in current_sd and current_sd[k].shape == v.shape:
                matched[k] = v

        if matched:
            current_sd.update(matched)
            self._yolo.model.load_state_dict(current_sd, strict=False)
            logger.info(
                f"[SFOD] Transferred {len(matched)}/{len(pretrained_sd)} tensors "
                f"({100*len(matched)/max(len(pretrained_sd),1):.1f}%)"
            )
        else:
            logger.warning("[SFOD] Không có tensor nào khớp — check arch yaml.")

        return len(matched)

    def _freeze_backbone(self) -> None:
        """Freeze backbone layers 0-8 (chỉ train head + neck)."""
        try:
            for i, layer in enumerate(self._yolo.model.model[:9]):
                for p in layer.parameters():
                    p.requires_grad = False
            n_frozen = sum(
                p.numel()
                for p in self._yolo.model.parameters()
                if not p.requires_grad
            )
            logger.info(f"[SFOD] Backbone frozen | {n_frozen:,} params frozen")
        except Exception as e:
            logger.warning(f"[SFOD] Freeze backbone failed: {e}")

    # ── Train / Val ───────────────────────────────────────────────────────────

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
        logger.info(
            f"[SFOD] Training | epochs={epochs} batch={batch} "
            f"imgsz={imgsz} device={device}"
        )
        self._yolo.train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            val=True,
            **kwargs,
        )

    def val(
        self,
        data: str,
        split: str = "val",
        imgsz: int = 640,
        device: str = "",
        conf: float = 0.001,
        iou: float = 0.6,
        **kwargs,
    ):
        return self._yolo.val(
            data=data,
            split=split,
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            **kwargs,
        )
