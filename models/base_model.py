"""
models/base_model.py
Abstract base class cho tất cả model/idea trong project.
Mọi idea đều kế thừa class này để đảm bảo interface nhất quán.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class — định nghĩa interface bắt buộc.

    Subclass phải implement:
        - build()   : khởi tạo model
        - train()   : training loop (thường delegate sang ultralytics)
        - val()     : validation
        - predict() : inference trên ảnh/video
        - export()  : xuất sang ONNX, TensorRT,...

    Subclass nên override (không bắt buộc):
        - get_callbacks() : trả về dict callback cho ultralytics trainer
        - extra_loss()    : custom loss term (ví dụ CAGI budget loss)
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: dict config đã merge từ base.yaml + experiment yaml
        """
        self.cfg = cfg
        self.model_name: str = cfg.get("model", "yolov12n")
        self.idea: str       = cfg.get("idea", "baseline")
        self.pretrained: Optional[str] = cfg.get("pretrained", "yolo12n.pt")

        # ultralytics YOLO instance — được gán trong build()
        self._yolo = None

        logger.info(f"[{self.__class__.__name__}] model={self.model_name}, idea={self.idea}")
        self.build()

    # ── Abstract methods ────────────────────────────────────────────────────

    @abstractmethod
    def build(self) -> None:
        """Khởi tạo self._yolo (ultralytics YOLO) và custom modules."""
        ...

    # ── Concrete methods (có thể override) ─────────────────────────────────

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
        Chạy training. Mặc định delegate hoàn toàn sang ultralytics.
        CAGI/AMSHA/... override để thêm custom logic.
        """
        assert self._yolo is not None, "Gọi build() trước khi train()"
        self._yolo.train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
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
    ) -> dict:
        """
        Chạy validation/test.
        Returns dict kết quả: {"mAP50": ..., "mAP50-95": ...}
        """
        assert self._yolo is not None
        results = self._yolo.val(
            data=data,
            split=split,
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            **kwargs,
        )
        return results

    def predict(
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
        Chạy inference trên ảnh/video/thư mục.
        """
        assert self._yolo is not None
        return self._yolo.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=save,
            **kwargs,
        )

    def export(self, format: str = "onnx", **kwargs):
        """Xuất model sang định dạng khác (ONNX, TensorRT, CoreML,...)."""
        assert self._yolo is not None
        return self._yolo.export(format=format, **kwargs)

    def get_callbacks(self) -> dict:
        """
        Trả về custom callbacks để đăng ký với ultralytics trainer.
        Override trong subclass để thêm callback riêng (ví dụ: log complexity score).

        Returns:
            dict: {event_name: callable}  — xem ultralytics docs để biết event names
        """
        return {}

    def extra_loss(self, *args, **kwargs):
        """
        Custom loss term (ví dụ: compute budget loss của CAGI).
        Mặc định trả về 0.0 (không thêm gì).
        Override trong subclass nếu cần.
        """
        return 0.0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def yolo(self):
        """Trả về ultralytics YOLO instance."""
        return self._yolo

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"idea={self.idea!r}, "
            f"pretrained={self.pretrained!r})"
        )
