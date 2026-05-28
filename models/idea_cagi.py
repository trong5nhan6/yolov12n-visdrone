"""
models/idea_cagi.py
Idea 1: Complexity-Aware Gated Inference (CAGI)
─────────────────────────────────────────────────────────────────────────────
Vấn đề: Static computation — mọi ảnh đều dùng full FLOPs bất kể độ phức tạp.

Giải pháp:
  1. Scene Complexity Predictor (SCP): module nhẹ đọc feature map sau layer[0]
     và trả về complexity score c ∈ [0, 1].
  2. Gated backbone stage (mặc định layer[4]): block được wrap bởi
     GatedC3k2Block.  current_gate được set bởi SCP hook trước mỗi forward:
     - c < easy_threshold  → gate = 0.25 (~25% compute, EASY path)
     - easy ≤ c < hard     → gate = 0.50 (~50% compute, MEDIUM path)
     - c ≥ hard_threshold  → gate = 1.00 (full compute,  HARD path)
  3. Compute Budget Loss: λ·c.mean() cộng vào detection loss để encourage
     model dùng EASY path nhiều hơn.

Training (4 giai đoạn):
  1. Train Teacher (full YOLOv12n, không gate)
  2. (future) Train SCP với pseudo-labels từ dataset stats
  3. Joint fine-tune với budget loss — gradient flows qua SCP → backbone
  4. (future) Knowledge Distillation từ Teacher sang CAGI student
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from ultralytics import YOLO

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class SceneComplexityPredictor(nn.Module):
    """
    Lightweight complexity estimator.
    Input : feature map → [B, C, H, W]
    Output: complexity score c ∈ [0, 1] → [B, 1]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        easy_threshold: float = 0.4,
        hard_threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pool(x))


class GatedC3k2Block(nn.Module):
    """
    Wrapper bao quanh một block với soft gate.
    gate ≤ 0 → identity (skip); gate ≥ 1 → full block; else → interpolation.

    `current_gate` được set bởi CAGIModel._scp_gate_hook trước mỗi forward.
    Mặc định = 1.0 (full compute) — an toàn khi hook chưa chạy.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.current_gate: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.current_gate
        if g <= 0.0:
            return x
        if g >= 1.0:
            return self.block(x)
        return g * self.block(x) + (1.0 - g) * x


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 helpers: custom trainer với budget loss
# ─────────────────────────────────────────────────────────────────────────────

def _make_finetune_trainer_class(
    scp: SceneComplexityPredictor,
    budget_lambda: float,
    complexity_stats: dict,
):
    """
    Factory: tạo DetectionTrainer subclass với budget loss.

    - Forward hook trên layer[0] bắt features F1 (không detach → gradient
      flows qua SCP → backbone).
    - _BudgetCriterion cộng λ·SCP(F1).mean() vào detection loss.
    - SCP có Adam optimizer riêng, step sau mỗi batch.
    """
    try:
        from ultralytics.models.yolo.detect.train import DetectionTrainer
    except ImportError:
        from ultralytics.models.yolo.detect import DetectionTrainer

    stage1_ref    = [None]
    scp_optim_ref = [None]

    class _BudgetCriterion:
        def __init__(self, base_criterion):
            self.base = base_criterion

        def __call__(self, preds, batch):
            det_loss, loss_items = self.base(preds, batch)

            if stage1_ref[0] is not None:
                c = scp(stage1_ref[0])
                total_loss = det_loss + budget_lambda * c.mean()

                score = c.detach().mean().item()
                if score < scp.easy_threshold:
                    complexity_stats["easy"] += 1
                elif score < scp.hard_threshold:
                    complexity_stats["medium"] += 1
                else:
                    complexity_stats["hard"] += 1

                stage1_ref[0] = None
                return total_loss, loss_items

            return det_loss, loss_items

    class CAGIFineTuneTrainer(DetectionTrainer):
        def _setup_train(self, world_size):
            super()._setup_train(world_size)

            dev = next(self.model.parameters()).device
            scp.to(dev)
            scp_optim_ref[0] = torch.optim.Adam(scp.parameters(), lr=1e-4)
            logger.info(f"[CAGI Stage3] SCP moved to {dev}, Adam lr=1e-4")

            self.criterion = _BudgetCriterion(self.criterion)
            logger.info(
                f"[CAGI Stage3] BudgetCriterion wrapped — lambda={budget_lambda}"
            )

            try:
                first_layer = self.model.model[0]
                self._cagi_hook = first_layer.register_forward_hook(
                    lambda m, inp, out: stage1_ref.__setitem__(0, out)
                )
                logger.info(
                    f"[CAGI Stage3] Hook on layer[0]: {type(first_layer).__name__}"
                )
            except Exception as e:
                logger.warning(
                    f"[CAGI Stage3] Hook failed: {e} — budget loss inactive"
                )
                self._cagi_hook = None

            def _on_batch_end(trainer):
                if scp_optim_ref[0] is not None:
                    scp_optim_ref[0].step()
                    scp_optim_ref[0].zero_grad()

            def _on_epoch_end(trainer):
                stats = complexity_stats
                total = sum(stats.values()) or 1
                ep    = getattr(trainer, "epoch", "?")
                logger.info(
                    f"[CAGI Stage3] Epoch {ep} — "
                    f"EASY: {100*stats['easy']/total:.1f}%  "
                    f"MEDIUM: {100*stats['medium']/total:.1f}%  "
                    f"HARD: {100*stats['hard']/total:.1f}%"
                )
                complexity_stats.update({"easy": 0, "medium": 0, "hard": 0})

            self.add_callback("on_train_batch_end", _on_batch_end)
            self.add_callback("on_train_epoch_end", _on_epoch_end)

        def final_eval(self):
            if hasattr(self, "_cagi_hook") and self._cagi_hook is not None:
                self._cagi_hook.remove()
                self._cagi_hook = None
            return super().final_eval()

    return CAGIFineTuneTrainer


# ─────────────────────────────────────────────────────────────────────────────
# CAGI Model
# ─────────────────────────────────────────────────────────────────────────────

class CAGIModel(BaseModel):
    """
    YOLOv12n với Complexity-Aware Gated Inference.

    Luồng inference:
        image → layer[0] → SCP hook → gate_value
                                           ↓
                        layer[gated_stage_idx] (GatedC3k2Block) → neck → head
    """

    def build(self) -> None:
        weights: Optional[str] = self.cfg.get("pretrained", "yolo12n.pt")
        logger.info(f"[CAGI] Load base model: {weights}")
        self._yolo = YOLO(weights or "yolo12n.pt")

        self.easy_threshold:   float         = self.cfg.get("cagi_easy_threshold",  0.4)
        self.hard_threshold:   float         = self.cfg.get("cagi_hard_threshold",  0.7)
        self.budget_lambda:    float         = self.cfg.get("cagi_budget_lambda",   0.1)
        hidden_dim:            int           = self.cfg.get("cagi_scp_hidden_dim",  64)
        self._gated_stage_idx: int           = self.cfg.get("cagi_gated_stage_idx", 4)
        # Path tới best.pt của teacher đã train sẵn.
        # Nếu được set và file tồn tại → bỏ qua stage 1, dùng luôn weights này.
        self._teacher_weights: Optional[str] = self.cfg.get("cagi_teacher_weights", None)

        in_channels = self._get_stage1_channels()
        self.scp = SceneComplexityPredictor(
            in_channels, hidden_dim,
            self.easy_threshold, self.hard_threshold,
        )

        self._complexity_stats: dict               = {"easy": 0, "medium": 0, "hard": 0}
        self._gated_blocks:     List[GatedC3k2Block] = []
        self._scp_hook                             = None

        self._wrap_gated_stage()
        self._register_scp_hook()

        logger.info(
            f"[CAGI] Ready | in_channels={in_channels} "
            f"| thresholds=({self.easy_threshold}, {self.hard_threshold}) "
            f"| gated_blocks={len(self._gated_blocks)}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_teacher_weights(
        self,
        data: str,
        epochs: int,
        batch: int,
        imgsz: int,
        device: str,
        project: str,
        name: str,
        kwargs: dict,
    ) -> Optional[Path]:
        """
        Trả về Path tới teacher best.pt theo logic:
          - cagi_teacher_weights được set và file tồn tại
                → dùng luôn, bỏ qua stage 1
          - cagi_teacher_weights được set nhưng file KHÔNG tồn tại
                → báo lỗi rõ ràng, dừng lại (không train nhầm từ đầu)
          - cagi_teacher_weights = None
                → train stage 1 từ đầu, trả về path vừa train
        """
        tw = self._teacher_weights

        # ── Có config path → dùng sẵn, skip stage 1 ─────────────────────────
        if tw is not None:
            p = Path(tw)
            if p.exists():
                logger.info(
                    f"[CAGI] cagi_teacher_weights={p} — bỏ qua stage 1, "
                    "dùng teacher đã train sẵn."
                )
                return p
            # File không tồn tại → dừng hẳn, không âm thầm train lại
            raise FileNotFoundError(
                f"[CAGI] cagi_teacher_weights='{tw}' nhưng file không tồn tại. "
                "Kiểm tra lại path hoặc đặt cagi_teacher_weights=null để train từ đầu."
            )

        # ── Không có config → train stage 1 ──────────────────────────────────
        logger.info("[CAGI] Giai đoạn 1/4: Train Teacher (full model)")
        self._yolo.train(
            data=data, epochs=epochs, batch=batch, imgsz=imgsz,
            device=device, project=project, name=f"{name}_teacher",
            val=True, **kwargs,
        )
        # Lấy path thực tế từ trainer (xử lý đúng khi ultralytics tự increment tên)
        teacher_save_dir = Path(self._yolo.trainer.save_dir)
        teacher_best     = teacher_save_dir / "weights" / "best.pt"
        logger.info(f"[CAGI] Stage 1 done → {teacher_save_dir}")
        return teacher_best

    def _get_stage1_channels(self) -> int:
        try:
            first_layer = self._yolo.model.model[0]
            return first_layer.conv.weight.shape[0]
        except Exception:
            logger.warning("[CAGI] Cannot auto-detect stage1 channels, using 16")
            return 16

    def _wrap_gated_stage(self) -> None:
        """Replace block tại gated_stage_idx với GatedC3k2Block."""
        try:
            backbone = self._yolo.model.model
            idx = self._gated_stage_idx
            if idx >= len(backbone):
                logger.warning(
                    f"[CAGI] gated_stage_idx={idx} out of range "
                    f"(backbone has {len(backbone)} layers) — gating inactive"
                )
                return
            block = backbone[idx]
            if isinstance(block, GatedC3k2Block):
                self._gated_blocks = [block]
                return
            gated = GatedC3k2Block(block)
            backbone[idx] = gated
            self._gated_blocks = [gated]
            logger.info(
                f"[CAGI] Wrapped layer[{idx}] ({type(block).__name__}) "
                "with GatedC3k2Block"
            )
        except Exception as e:
            logger.warning(f"[CAGI] _wrap_gated_stage failed: {e} — gating inactive")

    def _register_scp_hook(self) -> None:
        """
        Forward hook sau layer[0]: chạy SCP (no_grad) → set current_gate
        trên tất cả GatedC3k2Block trước khi chúng được gọi.
        """
        if not self._gated_blocks:
            return
        if self._scp_hook is not None:
            self._scp_hook.remove()
            self._scp_hook = None

        try:
            hook_layer = self._yolo.model.model[0]

            def _scp_gate_hook(module, inp, out):
                with torch.no_grad():
                    self.scp.to(out.device)
                    c = self.scp(out)
                gate = self._score_to_gate(c.mean().item())
                for gb in self._gated_blocks:
                    gb.current_gate = gate

            self._scp_hook = hook_layer.register_forward_hook(_scp_gate_hook)
            logger.info("[CAGI] SCP gate hook registered on layer[0]")
        except Exception as e:
            logger.warning(f"[CAGI] _register_scp_hook failed: {e}")

    def _score_to_gate(self, score: float) -> float:
        """Map complexity score → soft gate ∈ {0.25, 0.5, 1.0}."""
        if score < self.easy_threshold:
            self._complexity_stats["easy"] += 1
            return 0.25
        elif score < self.hard_threshold:
            self._complexity_stats["medium"] += 1
            return 0.5
        else:
            self._complexity_stats["hard"] += 1
            return 1.0

    def _remove_scp_hook(self) -> None:
        if self._scp_hook is not None:
            self._scp_hook.remove()
            self._scp_hook = None

    # ── Public API ───────────────────────────────────────────────────────────

    def gate_controller(self, c: torch.Tensor) -> int:
        """Ánh xạ complexity score → số blocks active (1 | 2 | 4)."""
        score = c.mean().item()
        if score < self.easy_threshold:
            return 1
        elif score < self.hard_threshold:
            return 2
        return 4

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
        logger.info("[CAGI] === Bắt đầu training pipeline ===")

        # Gỡ inference hook trước khi train (trainer tự quản lý hook riêng)
        self._remove_scp_hook()

        # ── Giai đoạn 1: Train Teacher (hoặc load sẵn) ───────────────────────
        teacher_best = self._resolve_teacher_weights(
            data=data, epochs=epochs, batch=batch, imgsz=imgsz,
            device=device, project=project, name=name, kwargs=kwargs,
        )

        if teacher_best is not None and teacher_best.exists():
            logger.info(f"[CAGI] Load teacher weights: {teacher_best}")
            self._yolo = YOLO(str(teacher_best))
            # Re-wire gating vào model vừa load (model cũ đã bị replace)
            self._gated_blocks = []
            self._wrap_gated_stage()
        else:
            logger.warning("[CAGI] Không có teacher weights — dùng model hiện tại.")

        # ── Giai đoạn 3: Fine-tune với budget loss ───────────────────────────
        ft_epochs = max(20, epochs // 5)
        logger.info(
            f"[CAGI] Giai đoạn 3/4: Fine-tune với budget loss "
            f"({ft_epochs} epochs, lambda={self.budget_lambda})"
        )

        self._complexity_stats = {"easy": 0, "medium": 0, "hard": 0}

        trainer_cls = _make_finetune_trainer_class(
            scp              = self.scp,
            budget_lambda    = self.budget_lambda,
            complexity_stats = self._complexity_stats,
        )

        self._yolo.train(
            data=data, epochs=ft_epochs, batch=batch, imgsz=imgsz,
            device=device, project=project, name=f"{name}_cagi_finetune",
            val=True, trainer=trainer_cls,
            **{**kwargs, "exist_ok": True},
        )
        # Lấy path thực tế của stage 3 để lưu SCP đúng chỗ
        ft_save_dir = Path(self._yolo.trainer.save_dir)

        # ── Lưu SCP weights vào cùng thư mục model ───────────────────────────
        scp_save_dir = ft_save_dir / "weights"
        scp_save_dir.mkdir(parents=True, exist_ok=True)
        scp_path = scp_save_dir / "scp.pt"
        torch.save(self.scp.state_dict(), scp_path)
        logger.info(f"[CAGI] SCP weights saved → {scp_path}")

        # Re-register inference hook sau khi train xong
        self._register_scp_hook()

        logger.info(
            "[CAGI] NOTE: Giai đoạn 2 (SCP pseudo-label) và 4 (KD) "
            "cần custom training loop — xem README phần CAGI Advanced."
        )
        logger.info(f"[CAGI] Training complete → {ft_save_dir / 'weights' / 'best.pt'}")

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
            data=data, split=split, imgsz=imgsz,
            device=device, conf=conf, iou=iou, **kwargs,
        )
