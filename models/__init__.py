"""
models/__init__.py
Model registry — map (model_name, idea) → class
"""

from models.yolov12n import YOLOv12nModel
from models.idea_cagi import CAGIModel
from models.idea_amsha import AMSHAModel
from models.idea_rsfe import RSFEModel
from models.idea_cgsr import CGSRModel
from models.idea_iawr import IAWRModel
from models.idea_sfod import SFODModel
from models.idea_vsod import VSODModel

# ---------------------------------------------------------------------------
# Registry: key = (model, idea)  →  value = model class
# Thêm model/idea mới: chỉ cần thêm entry vào dict này
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict = {
    ("yolov12n", "baseline"): YOLOv12nModel,
    ("yolov12n", "cagi"):     CAGIModel,
    ("yolov12n", "amsha"):    AMSHAModel,
    ("yolov12n", "rsfe"):     RSFEModel,
    ("yolov12n", "cgsr"):     CGSRModel,
    ("yolov12n", "iawr"):     IAWRModel,
    ("yolov12n", "sfod"):     SFODModel,   # Small Feature-enhanced Object Detector
    ("yolov12n", "vsod"):     VSODModel,   # VisDrone-Specialized Object Detector (target ~50%)
}


def build_model(model_name: str, idea: str, cfg: dict):
    """
    Khởi tạo model từ registry.

    Args:
        model_name: tên model, ví dụ 'yolov12n'
        idea:       tên idea, ví dụ 'baseline' | 'cagi' | ...
        cfg:        dict config đã merge (base + experiment)

    Returns:
        instance của BaseModel subclass tương ứng

    Raises:
        KeyError nếu (model_name, idea) chưa được đăng ký
    """
    key = (model_name.lower(), idea.lower())
    if key not in MODEL_REGISTRY:
        available = ", ".join(f"({m},{i})" for m, i in MODEL_REGISTRY)
        raise KeyError(
            f"Combination ({model_name!r}, {idea!r}) chưa được đăng ký.\n"
            f"Các lựa chọn có sẵn: {available}"
        )
    cls = MODEL_REGISTRY[key]
    return cls(cfg)


__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "YOLOv12nModel",
    "CAGIModel",
    "AMSHAModel",
    "RSFEModel",
    "CGSRModel",
    "IAWRModel",
    "SFODModel",
    "VSODModel",
]
