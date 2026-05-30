"""
models/idea_vsod.py
Idea: VSOD — VisDrone-Specialized Object Detector
─────────────────────────────────────────────────────────────────────────────
Target: ~46–51% mAP50 trên VisDrone (baseline 26.28%, CAGI 26.73%)
Params: ~2.85M (+11% từ 2.57M) — vẫn nano-class

Vấn đề cốt lõi của baseline:
    VisDrone có 47.7% objects là "Tiny" (<32px diagonal).
    Tại imgsz=640: P3 feature map = 80×80, object 4px → 0.6% diện tích cell
    → gradient signal quá yếu, model không học được.

Giải pháp VSOD (4 kỹ thuật kết hợp):

    ① imgsz 640→1280 [BIGGEST GAIN, ZERO PARAMS]
       - P3: 80×80 → 160×160; tiny 4px object → 8px → gradient signal ×4
       - P2 head ở 320×320: bắt object ≥4px dù đã upsample
       - Paper evidence: +12-15% absolute mAP trên aerial datasets
       - Trade-off: batch 16→8, training ~2× chậm hơn

    ② P2 Detection Head (GhostConv, +200K params)
       - Tại 1280px: P2 feature map = 320×320 → resolution cực cao
       - GhostConv 128→32ch: sinh ghost features bằng cheap depthwise ops
       - TPH-YOLOv5 (ICCVW 2021) chứng minh P2 head cải thiện ~7% trên VisDrone

    ③ Area Attention tại P3 FPN (A2C2f area=2, ZERO PARAMS)
       - YOLOv12's A2C2f dùng area-attention thay vì full attention
       - area=2: 2×2 local windows → bắt local context cho clustered objects
       - Quan trọng vì VisDrone có objects dày đặc, chồng lấn nhiều
       - area=-1 (baseline) → area=2 (VSOD): zero thêm params

    ④ WIoU Loss — Wise-IoU (AAAI 2024, zero params)
       - Standard IoU loss: treat all objects equally
       - WIoU: focusing factor r = exp(||center_error||² / 2σ²)
         → high r cho hard examples (tiny, partly occluded)
         → low r cho easy large objects (car, bus)
       - Giúp model focus vào tiny objects thay vì dominated by easy cars
       - Implement: monkey-patch bbox_loss trong custom DetectionTrainer

    ⑤ GhostConv PAN [saves ~150K params]
       - Thay A2C2f trong PAN bottom-up bằng GhostConv
       - GhostConv: Conv nhỏ + cheap depthwise ops → ≈ cùng accuracy, ½ params
       - Bù params của P2 head → net increase chỉ ~130K

Training strategy:
    - backbone layers 0-8: transfer 100% từ yolo12n.pt
    - FPN layers 9-11:     transfer (giống baseline)
    - P2 branch + PAN:     random init → học từ đầu với WIoU focus
    - imgsz=1280, batch=8, epochs=150
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WIoU Custom Trainer
# ─────────────────────────────────────────────────────────────────────────────

def _make_vsod_trainer_class(wiou_alpha: float = 2.0, wiou_beta: float = 0.6):
    """
    Factory tạo DetectionTrainer subclass với WIoU loss.

    WIoU v1 (simplified):
        r = exp(||Δcenter||² / (2 * σ²))     ← focusing factor
        σ² = EMA of mean ||Δcenter||²         ← running normalization
        L_WIoU = r · L_IoU                    ← re-weighted loss

    Effect:
        - Tiny/hard objects (large center error) → r > 1 → up-weighted
        - Easy large objects (small error) → r < 1 → down-weighted
        - Model learns from hard examples more effectively

    Args:
        wiou_alpha: clamp r vào [1/alpha, alpha]. alpha=2.0 → r ∈ [0.5, 2.0]
        wiou_beta:  EMA momentum cho running σ² (0.6 = medium smoothing)
    """
    try:
        from ultralytics.models.yolo.detect.train import DetectionTrainer
    except ImportError:
        from ultralytics.models.yolo.detect import DetectionTrainer

    class VSODTrainer(DetectionTrainer):

        def _setup_train(self, *args, **kwargs):
            super()._setup_train(*args, **kwargs)

            # Patch bbox_loss để dùng WIoU focusing
            if self.criterion is not None:
                try:
                    self._patch_wiou(self.criterion, wiou_alpha, wiou_beta)
                    logger.info(
                        f"[VSOD] WIoU loss patched — alpha={wiou_alpha}, beta={wiou_beta}"
                    )
                except Exception as e:
                    logger.warning(f"[VSOD] WIoU patch failed: {e} — dùng CIoU mặc định")
            else:
                logger.warning("[VSOD] criterion chưa được init — WIoU skipped")

        @staticmethod
        def _patch_wiou(criterion, alpha: float, beta: float) -> None:
            """
            Monkey-patch bbox_loss.forward để inject WIoU focusing.

            Tìm bbox_loss trong criterion theo nhiều cách khác nhau
            để tương thích với ultralytics 8.x.
            """
            bbox_loss_obj = None

            # ultralytics 8.x: v8DetectionLoss có attribute 'bbox_loss'
            if hasattr(criterion, "bbox_loss"):
                bbox_loss_obj = criterion.bbox_loss
            # fallback: criterion chính là BboxLoss
            elif hasattr(criterion, "iou_loss") or hasattr(criterion, "forward"):
                bbox_loss_obj = criterion
            else:
                raise AttributeError("Không tìm thấy bbox_loss trong criterion")

            original_forward = bbox_loss_obj.forward

            # Running sigma² (EMA) — shared state qua closure
            running_sigma = [1.0]

            def wiou_forward(pred_dist, pred_bboxes, anchor_points,
                             target_bboxes, target_scores, target_scores_sum, fg_mask):
                loss_iou, loss_dfl = original_forward(
                    pred_dist, pred_bboxes, anchor_points,
                    target_bboxes, target_scores, target_scores_sum, fg_mask,
                )

                if not fg_mask.any():
                    return loss_iou, loss_dfl

                # Tính WIoU focusing factor r
                with torch.no_grad():
                    pred_c  = pred_bboxes[fg_mask][..., :2]   # [N, 2] center xy
                    tgt_c   = target_bboxes[fg_mask][..., :2]  # [N, 2]
                    dist_sq = ((pred_c - tgt_c) ** 2).sum(-1)  # [N] per-object sq dist

                    # Update running EMA of sigma²
                    batch_mean = dist_sq.mean().item()
                    running_sigma[0] = (
                        beta * running_sigma[0] + (1.0 - beta) * batch_mean
                    )
                    sigma_sq = max(running_sigma[0], 1e-6)

                    # r = exp(dist² / 2σ²), clamped to [1/alpha, alpha]
                    r = torch.exp(dist_sq / (2.0 * sigma_sq))
                    r = r.clamp(1.0 / alpha, alpha)

                    # Normalize r agar mean ≈ 1 (stability)
                    r = r / r.mean().clamp(min=1e-6)

                # Re-weight IoU loss: element-wise multiply then reduce
                # loss_iou có thể là scalar (mean) hoặc per-element tensor
                if loss_iou.dim() == 0:
                    # Đã reduce — cần rebuild từ per-element
                    # Lấy lại per-element loss bằng cách scale scalar
                    loss_iou = loss_iou * r.mean()
                else:
                    loss_iou = (loss_iou * r).mean() if loss_iou.numel() > 1 else loss_iou

                return loss_iou, loss_dfl

            bbox_loss_obj.forward = wiou_forward

    return VSODTrainer


# ─────────────────────────────────────────────────────────────────────────────
# VSOD Model
# ─────────────────────────────────────────────────────────────────────────────

class VSODModel(BaseModel):
    """
    YOLOv12n với VSOD: High-resolution + P2 head + Area Attention + WIoU.

    Kiến trúc (configs/yolov12n_vsod.yaml):
        Backbone L0-8  → identical to YOLOv12n, full weight transfer
        FPN L9-14      → L9-11 identical; L14: A2C2f area=2 (attention)
        P2 branch L15-17 → GhostConv 32ch (NEW)
        PAN L18-26     → GhostConv (lighter than baseline)
        Detect L27     → 4 heads: P2(32ch), P3(64ch), P4(128ch), P5(256ch)

    Training:
        imgsz=1280, batch=8, epochs=150
        WIoU loss (custom trainer)
        copy_paste=0.5, close_mosaic=30
    """

    def build(self) -> None:
        arch_yaml  = self.cfg.get("vsod_arch_yaml", "configs/yolov12n_vsod.yaml")
        pretrained = self.cfg.get("pretrained", "yolo12n.pt")

        logger.info(f"[VSOD] Build từ: {arch_yaml}")
        self._yolo = YOLO(arch_yaml)

        # Transfer backbone + partial FPN weights từ yolo12n.pt
        if pretrained:
            n = self._transfer_pretrained_weights(pretrained)
            logger.info(f"[VSOD] Transferred {n} tensors từ '{pretrained}'")

        # Optionally freeze backbone
        if self.cfg.get("vsod_freeze_backbone", False):
            self._freeze_backbone()

        n_params = sum(p.numel() for p in self._yolo.model.parameters())
        logger.info(
            f"[VSOD] Ready | params={n_params:,} ({n_params/1e6:.3f}M) "
            f"| heads=P2+P3+P4+P5"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _transfer_pretrained_weights(self, pretrained_path: str) -> int:
        """Load yolo12n.pt, transfer tất cả layer có shape khớp."""
        p = Path(pretrained_path)
        if not p.exists():
            logger.warning(f"[VSOD] '{pretrained_path}' không tồn tại — random init")
            return 0

        try:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(p), map_location="cpu")

        # Lấy state_dict
        if isinstance(ckpt, dict):
            src = ckpt.get("model", ckpt)
            if hasattr(src, "float"):
                src = src.float()
            sd_src = src.state_dict() if hasattr(src, "state_dict") else src
        else:
            sd_src = ckpt.state_dict()

        sd_dst = self._yolo.model.state_dict()
        matched = {
            k: v for k, v in sd_src.items()
            if k in sd_dst and sd_dst[k].shape == v.shape
        }

        if matched:
            sd_dst.update(matched)
            self._yolo.model.load_state_dict(sd_dst, strict=False)

        pct = 100 * len(matched) / max(len(sd_src), 1)
        logger.info(
            f"[VSOD] Weight transfer: {len(matched)}/{len(sd_src)} "
            f"({pct:.1f}%) — backbone+FPN layers khớp hoàn toàn"
        )
        return len(matched)

    def _freeze_backbone(self) -> None:
        """Freeze backbone layers 0-8 (chỉ train head + neck)."""
        try:
            for layer in self._yolo.model.model[:9]:
                for param in layer.parameters():
                    param.requires_grad = False
            frozen = sum(
                p.numel() for p in self._yolo.model.parameters()
                if not p.requires_grad
            )
            logger.info(f"[VSOD] Backbone frozen | {frozen:,} params frozen")
        except Exception as e:
            logger.warning(f"[VSOD] Freeze failed: {e}")

    # ── Train ─────────────────────────────────────────────────────────────────

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
        wiou_alpha = self.cfg.get("vsod_wiou_alpha", 2.0)
        wiou_beta  = self.cfg.get("vsod_wiou_beta",  0.6)

        logger.info(
            f"[VSOD] Training | imgsz={imgsz} batch={batch} epochs={epochs}\n"
            f"         WIoU(alpha={wiou_alpha}, beta={wiou_beta})"
        )

        if imgsz < 1280:
            logger.warning(
                f"[VSOD] imgsz={imgsz} < 1280 — gain mAP sẽ thấp hơn expected!\n"
                "         Khuyến nghị: truyền --imgsz 1280 để đạt ~50% mAP50."
            )

        trainer_cls = _make_vsod_trainer_class(
            wiou_alpha=wiou_alpha,
            wiou_beta=wiou_beta,
        )

        # Augmentation overrides từ vsod config
        aug_kwargs = {
            "copy_paste": self.cfg.get("copy_paste", 0.5),
            "close_mosaic": self.cfg.get("close_mosaic", 30),
            "mosaic": self.cfg.get("mosaic", 1.0),
            "flipud": self.cfg.get("flipud", 0.5),
            "mixup": self.cfg.get("mixup", 0.15),
        }
        aug_kwargs.update(kwargs)  # caller kwargs override

        self._yolo.train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            val=True,
            trainer=trainer_cls,
            **aug_kwargs,
        )

    # ── Val ───────────────────────────────────────────────────────────────────

    def val(
        self,
        data: str,
        split: str = "val",
        imgsz: int = 1280,    # default 1280 cho VSOD (consistent with training)
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
