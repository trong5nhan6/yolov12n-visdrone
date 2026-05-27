"""
models/yolov12n.py
YOLOv12n baseline — load pretrained, train/val/predict chuẩn ultralytics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class YOLOv12nModel(BaseModel):
    """
    Baseline YOLOv12n — không thêm bất kỳ modification nào.
    Dùng làm điểm so sánh (reference) cho tất cả ideas.

    Ultralytics tự động:
      - Validate sau mỗi epoch (val=True)
      - Lưu best.pt và last.pt
      - Vẽ curves, confusion matrix
    """

    def build(self) -> None:
        """Load YOLOv12n pretrained weights."""
        weights: Optional[str] = self.cfg.get("pretrained", "yolo12n.pt")

        if weights and Path(weights).exists():
            logger.info(f"Load pretrained weights từ: {weights}")
            self._yolo = YOLO(weights)
        else:
            # Ultralytics tự download nếu tên là model string chuẩn
            logger.info(f"Load / download weights: {weights}")
            self._yolo = YOLO(weights or "yolo12n.pt")

        logger.info(
            f"Model loaded: {self._yolo.model.__class__.__name__} | "
            f"params={sum(p.numel() for p in self._yolo.model.parameters()):,}"
        )

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
        """Train với ultralytics default settings."""
        logger.info(f"[YOLOv12n] Starting training | epochs={epochs} batch={batch}")

        self._yolo.train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            # Per-epoch val đã được bật mặc định trong ultralytics
            val=True,
            **kwargs,
        )

        logger.info(f"[YOLOv12n] Training complete. Results saved to: {project}/{name}")
