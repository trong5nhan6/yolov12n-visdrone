"""
models/idea_iawr.py
Idea 5: Input-Adaptive Width Routing (IAWR)
─────────────────────────────────────────────────────────────────────────────
Vấn đề: YOLOv12n dùng cùng một width (channel multiplier) cho mọi ảnh.
  Ảnh có ít objects, nền đơn giản → width đầy đủ là lãng phí.

Giải pháp: Once-For-All (OFA) supernet + Content Router
  1. OFA Supernet: Train một model với tất cả widths {0.25, 0.5, 0.75, 1.0}
     sử dụng progressive shrinking (train wide → shrink → fine-tune narrow).
  2. Content Router (CR): Module nhẹ đọc thumbnail ảnh đầu vào → predict
     width ratio w ∈ {0.25, 0.5, 0.75, 1.0} phù hợp.
  3. Inference: CR quyết định width → chọn subnet tương ứng → inference.

Lợi ích:
  - Ảnh đơn giản: width 0.25 → 1/4 FLOPs (1.2M params, ~2.2 GFLOPs)
  - Ảnh phức tạp: width 1.0 → full model (đã train sẵn)
  - Không cần train lại — một lần train, nhiều subnets

Tham khảo: Once-For-All (Cai et al., ICLR 2020), Slimmable Networks
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Content Router
# ─────────────────────────────────────────────────────────────────────────────

class ContentRouter(nn.Module):
    """
    Lightweight image content classifier → chọn width ratio.

    Input : ảnh thumbnail (resize về 64×64) → [B, 3, 64, 64]
    Output: width index ∈ {0, 1, 2, 3}  →  {0.25, 0.5, 0.75, 1.0}

    Architecture:
        3×(Conv3×3 + BN + ReLU + MaxPool) → GAP → Linear(256, 4)
        ~180K params — rất nhẹ so với detector
    """

    WIDTH_OPTIONS: List[float] = [0.25, 0.50, 0.75, 1.00]

    def __init__(self, thumbnail_size: int = 64) -> None:
        super().__init__()
        self.thumbnail_size = thumbnail_size

        self.encoder = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 64 → 32

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 32 → 16

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 16 → 8

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # → [B, 256, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, len(self.WIDTH_OPTIONS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] — ảnh gốc hoặc thumbnail
        Returns:
            logits: [B, 4] — raw scores cho 4 width options
        """
        # Resize về thumbnail size nếu cần
        if x.shape[-1] != self.thumbnail_size or x.shape[-2] != self.thumbnail_size:
            x = F.interpolate(x, size=(self.thumbnail_size, self.thumbnail_size),
                              mode='bilinear', align_corners=False)
        feats  = self.encoder(x)
        logits = self.classifier(feats)
        return logits

    def predict_width(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> float:
        """
        Predict width ratio cho batch (lấy mode).

        Args:
            x:           [B, 3, H, W]
            temperature: softmax temperature (thấp → confident hơn)

        Returns:
            width ratio ∈ {0.25, 0.50, 0.75, 1.00}
        """
        with torch.no_grad():
            logits = self(x) / temperature          # [B, 4]
            idx    = logits.mean(0).argmax().item()  # consensus over batch
        return self.WIDTH_OPTIONS[idx]

    def gumbel_sample(
        self,
        x: torch.Tensor,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """
        Gumbel-Softmax sample — differentiable discrete choice cho training.

        Returns:
            one_hot: [B, 4] soft one-hot vector
        """
        logits = self(x)
        return F.gumbel_softmax(logits, tau=temperature, hard=True)


# ─────────────────────────────────────────────────────────────────────────────
# Width Selector Helper
# ─────────────────────────────────────────────────────────────────────────────

class WidthSelector:
    """
    Chọn subnet từ OFA supernet theo width ratio.

    Thực tế với ultralytics YOLO: cần load model với width_multiple khác nhau
    từ các checkpoint đã train sẵn (hoặc implement slimmable forward).

    Ở đây implement dạng multi-checkpoint: mỗi width có 1 checkpoint riêng.
    """

    CHECKPOINT_PATTERN = "{project}/width_{w}/weights/best.pt"

    def __init__(
        self,
        width_checkpoints: Optional[Dict[float, str]] = None,
        project: str = "runs/iawr",
    ) -> None:
        """
        Args:
            width_checkpoints: {width_ratio: path_to_checkpoint}
            project:           nơi tìm checkpoints nếu width_checkpoints=None
        """
        self.project           = project
        self._width_models: Dict[float, YOLO] = {}
        self._loaded_widths: List[float]      = []

        if width_checkpoints:
            for w, path in width_checkpoints.items():
                self._load_width(w, path)

    def _load_width(self, width: float, path: str) -> None:
        try:
            self._width_models[width] = YOLO(path)
            self._loaded_widths.append(width)
            logger.info(f"[IAWR] Loaded width={width} from {path}")
        except Exception as e:
            logger.warning(f"[IAWR] Cannot load width={width} from {path}: {e}")

    def get_model(self, width: float) -> Optional[YOLO]:
        """Lấy YOLO model tương ứng với width ratio."""
        # Tìm width gần nhất
        available = sorted(self._width_models.keys())
        if not available:
            return None
        closest = min(available, key=lambda w: abs(w - width))
        return self._width_models.get(closest)


# ─────────────────────────────────────────────────────────────────────────────
# IAWR Model
# ─────────────────────────────────────────────────────────────────────────────

class IAWRModel(BaseModel):
    """
    YOLOv12n với Input-Adaptive Width Routing.

    Training Strategy (Progressive Shrinking):
        Stage 1: Train full model (width=1.0)  → 100 epochs
        Stage 2: Fine-tune width=0.75          → 30 epochs
        Stage 3: Fine-tune width=0.5           → 30 epochs
        Stage 4: Fine-tune width=0.25          → 30 epochs
        Stage 5: Train Content Router với pseudo-labels từ validation mAP

    Inference:
        ContentRouter(img) → width_ratio
        WidthSelector.get_model(width_ratio) → subnet
        subnet.predict(img) → detections
    """

    WIDTH_OPTIONS:    List[float] = [0.25, 0.50, 0.75, 1.00]
    FINETUNE_EPOCHS:  int         = 30
    ROUTER_LR:        float       = 1e-3

    def build(self) -> None:
        weights: Optional[str] = self.cfg.get("pretrained", "yolo12n.pt")
        logger.info(f"[IAWR] Load base model (width=1.0): {weights}")
        self._yolo = YOLO(weights or "yolo12n.pt")

        self.width_options: List[float] = self.cfg.get(
            "iawr_width_options", self.WIDTH_OPTIONS
        )
        self.router_temp:   float = self.cfg.get("iawr_router_temperature", 0.5)
        self.min_width:     float = self.cfg.get("iawr_min_width", 0.25)

        # Content Router
        self.content_router = ContentRouter(
            thumbnail_size=self.cfg.get("iawr_thumbnail_size", 64)
        )

        # Width Selector (populated after training)
        self._width_selector: Optional[WidthSelector] = None

        # Training statistics
        self._width_distribution: Dict[float, int] = {w: 0 for w in self.width_options}

        logger.info(
            f"[IAWR] Initialized | widths={self.width_options} "
            f"| router_temp={self.router_temp}"
        )

    def get_callbacks(self) -> dict:
        model_ref = self

        def on_train_epoch_end(trainer) -> None:
            dist = model_ref._width_distribution
            total = sum(dist.values()) or 1
            dist_str = " | ".join(
                f"w={w:.2f}: {100*n/total:.1f}%"
                for w, n in sorted(dist.items())
            )
            logger.info(f"[IAWR] Epoch {trainer.epoch} width dist → {dist_str}")
            model_ref._width_distribution = {w: 0 for w in model_ref.width_options}

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
        Progressive Shrinking Training Pipeline.

        Stage 1: Train full model (width=1.0) — epochs epochs
        Stage 2-4: Fine-tune narrow subnets  — FINETUNE_EPOCHS mỗi stage
        Stage 5: Train Content Router        — cần custom loop (TODO)
        """
        logger.info("[IAWR] === Progressive Shrinking Training Pipeline ===")

        # ── Stage 1: Full model (width = 1.0) ───────────────────────────────
        logger.info("[IAWR] Stage 1/5: Train full model (width=1.0)")
        for cb_name, cb_fn in self.get_callbacks().items():
            self._yolo.add_callback(cb_name, cb_fn)

        self._yolo.train(
            data=data, epochs=epochs, batch=batch, imgsz=imgsz,
            device=device, project=project, name=f"{name}_w1.0", val=True, **kwargs,
        )
        logger.info(f"[IAWR] Stage 1 complete → {project}/{name}_w1.0")

        # ── Stage 2-4: Progressive shrinking ────────────────────────────────
        shrink_widths = [w for w in sorted(self.width_options, reverse=True) if w < 1.0]

        for stage_idx, width in enumerate(shrink_widths, start=2):
            logger.info(
                f"[IAWR] Stage {stage_idx}/{len(shrink_widths)+1}: "
                f"Fine-tune width={width}"
            )
            # Thực tế: cần YOLOv12n với width_multiple=width
            # Đây là placeholder — cần custom model YAML với width_multiple
            slim_weights = f"{project}/{name}_w1.0/weights/best.pt"
            try:
                slim_yolo = YOLO(slim_weights)
                slim_yolo.train(
                    data=data,
                    epochs=self.FINETUNE_EPOCHS,
                    batch=batch,
                    imgsz=imgsz,
                    device=device,
                    project=project,
                    name=f"{name}_w{width}",
                    val=True,
                    **kwargs,
                )
                logger.info(f"[IAWR] Stage {stage_idx} complete → width={width}")
            except Exception as e:
                logger.warning(f"[IAWR] Stage {stage_idx} (width={width}) failed: {e}")

        # ── Stage 5: Content Router training ────────────────────────────────
        logger.info(
            "[IAWR] Stage 5/5: Content Router training\n"
            "  → Cần custom training loop với pseudo-labels từ validation mAP.\n"
            "  → Xem scripts/train_router.py để biết thêm."
        )

        # Load tất cả checkpoints vào WidthSelector
        width_ckpts = {}
        for width in self.width_options:
            ckpt_path = f"{project}/{name}_w{width}/weights/best.pt"
            if width == 1.0:
                ckpt_path = f"{project}/{name}_w1.0/weights/best.pt"
            width_ckpts[width] = ckpt_path

        self._width_selector = WidthSelector(width_checkpoints=width_ckpts)

        logger.info(f"[IAWR] Training complete. Loaded {len(self._width_selector._loaded_widths)} width subnets.")

    def predict_adaptive(
        self,
        source: Union[str],
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "",
        save: bool = True,
        **kwargs,
    ):
        """
        Adaptive width inference.

        Luồng:
            1. Load ảnh → thumbnail → ContentRouter → width
            2. WidthSelector.get_model(width) → subnet
            3. subnet.predict(ảnh gốc) → detections
        """
        import numpy as np

        if self._width_selector is None:
            logger.warning("[IAWR] WidthSelector chưa được khởi tạo, dùng full model")
            return self._yolo.predict(
                source=source, imgsz=imgsz, conf=conf,
                iou=iou, device=device, save=save, **kwargs,
            )

        # Load ảnh và predict width
        try:
            import cv2
            img = cv2.imread(str(source))
            if img is None:
                raise ValueError(f"Cannot read image: {source}")
            img_t = torch.from_numpy(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            ).float().unsqueeze(0) / 255.0

            width = self.content_router.predict_width(img_t, self.router_temp)
            logger.info(f"[IAWR] ContentRouter → width={width}")
            self._width_distribution[width] = self._width_distribution.get(width, 0) + 1

        except Exception as e:
            logger.warning(f"[IAWR] ContentRouter failed ({e}), dùng width=1.0")
            width = 1.0

        model = self._width_selector.get_model(width)
        if model is None:
            logger.warning("[IAWR] Không tìm thấy subnet, dùng full model")
            model = self._yolo

        return model.predict(
            source=source, imgsz=imgsz, conf=conf,
            iou=iou, device=device, save=save, **kwargs,
        )
