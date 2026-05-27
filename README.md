# YOLOv12n × VisDrone — Adaptive Computation Research

> **ACCV 2026** — Nghiên cứu làm nhẹ YOLOv12n cho phát hiện đối tượng từ UAV  
> Core idea: **Dynamic computation** thay vì static — dễ hay khó đều qua hết model là lãng phí.

---

## Tổng quan project

Project này triển khai và so sánh **5 research ideas** về adaptive computation trên YOLOv12n × VisDrone2019-DET:

| Idea | Tên đầy đủ | Cơ chế | FLOPs savings (est.) |
|------|-----------|--------|----------------------|
| `baseline` | YOLOv12n (reference) | — | — |
| `cagi` | Complexity-Aware Gated Inference | Gate C3k2 blocks theo scene complexity | 25–60% |
| `amsha` | Adaptive Multi-Scale Head Activation | Bật/tắt P2 head theo small-object predictor | ~8 GFLOPs/img |
| `rsfe` | Region-Wise Sparse Feature Enhancement | Sparse attention chỉ trên top-K vị trí | 75% attention FLOPs |
| `cgsr` | Confidence-Guided Selective Re-detection | 2-pass: full nano → re-detect uncertain patches | ~80% vs full re-run |
| `iawr` | Input-Adaptive Width Routing | OFA supernet + content router → subnet selection | 25–75% |

---

## Cài đặt môi trường

### Yêu cầu
- Python 3.9+
- CUDA 11.8+ (khuyến nghị)
- 8GB+ VRAM (training), 4GB+ (inference)

### Cài đặt

```bash
# 1. Clone / mở project
cd yolov12n-visdrone

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Cài dependencies
pip install -r requirements.txt
```

### Kiểm tra cài đặt

```bash
python -c "import ultralytics; ultralytics.checks()"
```

---

## Chuẩn bị VisDrone dataset

### Bước 1: Download

Download VisDrone2019-DET từ [GitHub chính thức](https://github.com/VisDrone/VisDrone-Dataset):
- `VisDrone2019-DET-train.zip`
- `VisDrone2019-DET-val.zip`
- `VisDrone2019-DET-test-dev.zip`

Giải nén vào `datasets/VisDrone_raw/`:

```
datasets/VisDrone_raw/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
├── VisDrone2019-DET-val/
└── VisDrone2019-DET-test-dev/
```

### Bước 2: Convert sang YOLO format

```bash
python scripts/prepare_visdrone.py \
    --visdrone-root datasets/VisDrone_raw \
    --output-root   datasets/VisDrone
```

### Bước 3: Kiểm tra dataset

```bash
python scripts/check_dataset.py --dataset-root datasets/VisDrone
```

Kết quả mong đợi:
```
  train:  6471 images, 6471 labels
  val:     548 images,  548 labels
  test:   1610 images, 1610 labels
  ✓ Dataset check PASSED
```

### Bước 4: Cập nhật config

Mở `configs/visdrone.yaml` và đảm bảo `path` trỏ đúng:

```yaml
path: datasets/VisDrone    # hoặc đường dẫn tuyệt đối
```

---

## Training

### Baseline (YOLOv12n chuẩn)

```bash
python train.py --model yolov12n --idea baseline
```

### Các ideas

```bash
# CAGI — Complexity-Aware Gated Inference
python train.py --model yolov12n --idea cagi

# AMSHA — Adaptive Multi-Scale Head Activation
python train.py --model yolov12n --idea amsha

# RSFE — Region-Wise Sparse Feature Enhancement
python train.py --model yolov12n --idea rsfe

# CGSR — Confidence-Guided Selective Re-detection
python train.py --model yolov12n --idea cgsr

# IAWR — Input-Adaptive Width Routing
python train.py --model yolov12n --idea iawr
```

### Tùy chỉnh hyperparameters

```bash
# Override epochs, batch size, device
python train.py --model yolov12n --idea cagi \
    --epochs 150 --batch 32 --device cuda:0

# Custom config file
python train.py --config configs/my_experiment.yaml

# Override bất kỳ param nào (omegaconf dotlist)
python train.py --idea rsfe epochs=200 rsfe_topk_ratio=0.3

# Test sau mỗi 5 epochs (kèm full metrics report)
python train.py --idea baseline --test-every 5
```

### Val & test tự động

- **Sau mỗi epoch**: ultralytics tự chạy val (`val=True` — mặc định bật).
- **Sau mỗi N epochs**: test split chạy tự động qua callback và in **full metrics report** (xem mục Metrics bên dưới).
- **CSV tích lũy**: mỗi lần test tự động ghi kết quả vào `runs/metrics.csv` để so sánh experiments.

---

## Validation

```bash
# Val split (mặc định) — in full metrics report
python val.py \
    --weights runs/train/yolov12n_baseline/weights/best.pt

# Test split
python val.py \
    --weights runs/train/yolov12n_baseline/weights/best.pt \
    --split test

# Tùy chỉnh threshold
python val.py --weights best.pt --conf 0.25 --iou 0.5

# Không đo efficiency (nhanh hơn, bỏ FPS/latency)
python val.py --weights best.pt --no-efficiency

# Lưu kết quả vào CSV để so sánh nhiều experiments
python val.py --weights best.pt --save-csv runs/results.csv \
    --model-name YOLOv12n --idea cagi
```

### Output của val.py

```
══════════════════════════════════════════════════════════
  EVALUATION REPORT  [TEST]  —  YOLOv12n / cagi
══════════════════════════════════════════════════════════
  1. ACCURACY METRICS
  ──────────────────────────────────────────────────────
  mAP50                            0.2312
  mAP50-95                         0.1187
  Precision (mean)                 0.3841
  Recall / AR (mean)               0.2956

  Per-class AP50:
    pedestrian           0.3124  ████████████░░░░░░░░░░░░░
    people               0.1843  ███████░░░░░░░░░░░░░░░░░░
    bicycle              0.0921  ████░░░░░░░░░░░░░░░░░░░░░
    car                  0.5671  █████████████████████░░░░
    van                  0.3205  █████████████░░░░░░░░░░░░
    truck                0.2841  ███████████░░░░░░░░░░░░░░
    tricycle             0.1532  ██████░░░░░░░░░░░░░░░░░░░
    awning-tricycle      0.1124  █████░░░░░░░░░░░░░░░░░░░░
    bus                  0.4267  █████████████████░░░░░░░░
    motor                0.2318  █████████░░░░░░░░░░░░░░░░

  Per-class AP50-95:
    pedestrian           0.1203
    car                  0.2891
    ...

  2. EFFICIENCY METRICS
  ──────────────────────────────────────────────────────
  Parameters                       2.623 M
  GFLOPs                           8.70 G
  Model size                       5.42 MB
  Latency (batch=1)                12.30 ms/img
  FPS (batch=1)                    81.3 fps
  Device                           cuda:0

  3. EFFICIENCY RATIOS
  ──────────────────────────────────────────────────────
  mAP50 / GFLOPs                   0.0266
  mAP50 / Params (M)               0.0881
  mAP50 × FPS                      18.78
══════════════════════════════════════════════════════════
```

---

## Metrics được đo

### Accuracy metrics

| Metric | Mô tả | Ghi chú |
|--------|-------|---------|
| **mAP50** | Mean Average Precision @ IoU=0.5 | Tiêu chuẩn chính VisDrone |
| **mAP50-95** | mAP trung bình @ IoU=0.5:0.95 | Đánh giá tổng quát hơn |
| **Precision** | Tỷ lệ dự đoán đúng / tổng dự đoán | Mean across classes |
| **Recall / AR** | Tỷ lệ bắt được / tổng ground truth | Mean across classes |
| **AP50 per-class** | AP@0.5 cho từng class (10 classes) | Phân tích class imbalance |
| **AP50-95 per-class** | AP@0.5:0.95 cho từng class | Chi tiết hơn per-class |

### Efficiency metrics

| Metric | Mô tả | Ghi chú |
|--------|-------|---------|
| **Parameters (M)** | Số lượng tham số model | Càng nhỏ càng tốt |
| **GFLOPs** | Floating-point operations | Đo tính toán lý thuyết |
| **Model size (MB)** | Dung lượng file `.pt` | Quan trọng với edge deploy |
| **Latency (ms/img)** | Thời gian inference / ảnh (batch=1) | Đo trên device thực tế |
| **FPS (batch=1)** | Frames per second (real-time) | 1000 / latency |

### Efficiency ratios

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **mAP50 / GFLOPs** | mAP50 ÷ GFLOPs | Accuracy per unit compute |
| **mAP50 / Params (M)** | mAP50 ÷ Params | Accuracy per parameter |
| **mAP50 × FPS** | mAP50 × FPS | Combined accuracy–speed score |

> **Với adaptive models (CAGI, IAWR)**: GFLOPs được đo là *average* trên test set, không phải worst-case — đây là điểm mạnh cần nhấn mạnh trong paper.

---

## So sánh experiments với CSV

Mỗi lần chạy `val.py --save-csv` hoặc mỗi lần test callback kích hoạt trong training, kết quả được ghi vào CSV:

```bash
# Xem kết quả tất cả experiments
cat runs/metrics.csv

# Hoặc mở bằng Excel / pandas
python -c "
import pandas as pd
df = pd.read_csv('runs/metrics.csv')
cols = ['model', 'idea', 'split', 'acc_map50', 'acc_map50_95',
        'eff_gflops', 'eff_params_m', 'eff_fps',
        'ratio_map50_per_gflop', 'ratio_map50_x_fps']
print(df[cols].to_string(index=False))
"
```

Ví dụ output CSV so sánh:

```
model     idea      split  acc_map50  acc_map50_95  eff_gflops  eff_params_m  eff_fps  ratio_map50_per_gflop
YOLOv12n  baseline  test   0.2301     0.1182        8.70        2.623         81.3     0.0265
YOLOv12n  cagi      test   0.2287     0.1171        5.20        2.701         94.7     0.0440
YOLOv12n  amsha     test   0.2318     0.1193        7.10        2.631         87.2     0.0326
YOLOv12n  rsfe      test   0.2334     0.1204        8.70        2.718         79.8     0.0268
```

---

## Inference / Predict

```bash
# Một ảnh
python predict.py \
    --weights runs/train/yolov12n_baseline/weights/best.pt \
    --source  path/to/image.jpg

# Thư mục ảnh
python predict.py \
    --weights best.pt \
    --source  datasets/VisDrone/images/test/

# CGSR 2-pass inference
python predict.py \
    --weights best.pt \
    --source  image.jpg \
    --idea    cgsr

# IAWR adaptive width inference
python predict.py \
    --weights best.pt \
    --source  image.jpg \
    --idea    iawr

# Không lưu ảnh output
python predict.py --weights best.pt --source img.jpg --no-save
```

---

## Cấu trúc project

```
yolov12n-visdrone/
│
├── configs/
│   ├── base.yaml            # Master config (epochs, lr, aug, metrics, ...)
│   ├── visdrone.yaml        # Dataset: paths, classes
│   ├── baseline.yaml        # Experiment: baseline override
│   ├── idea_cagi.yaml       # Experiment: CAGI override
│   ├── idea_amsha.yaml      # Experiment: AMSHA override
│   ├── idea_rsfe.yaml       # Experiment: RSFE override
│   ├── idea_cgsr.yaml       # Experiment: CGSR override
│   └── idea_iawr.yaml       # Experiment: IAWR override
│
├── models/
│   ├── __init__.py          # MODEL_REGISTRY + build_model()
│   ├── base_model.py        # Abstract BaseModel
│   ├── yolov12n.py          # Baseline YOLOv12n
│   ├── idea_cagi.py         # SceneComplexityPredictor + GatedC3k2
│   ├── idea_amsha.py        # SmallObjectExistencePredictor
│   ├── idea_rsfe.py         # SpatialComplexityMap + RegionSparseAttention
│   ├── idea_cgsr.py         # UncertaintyMapper + 2-pass inference
│   └── idea_iawr.py         # ContentRouter + WidthSelector (OFA)
│
├── utils/
│   ├── __init__.py          # Re-export các hàm chính
│   └── metrics.py           # Accuracy + Efficiency + Ratio metrics
│                            #   compute_efficiency_metrics()
│                            #   parse_accuracy_metrics()
│                            #   compute_ratios()
│                            #   print_full_report()
│                            #   save_metrics_csv()
│
├── scripts/
│   ├── prepare_visdrone.py  # Convert VisDrone → YOLO format
│   └── check_dataset.py     # Kiểm tra dataset integrity
│
├── datasets/
│   ├── README.md            # Hướng dẫn chuẩn bị dataset
│   ├── VisDrone_raw/        # ← giải nén VisDrone gốc vào đây
│   └── VisDrone/            # ← output sau khi convert (auto-tạo)
│
├── logs/                    # Log files (auto-tạo)
├── runs/
│   ├── train/               # Kết quả training (auto-tạo bởi ultralytics)
│   ├── val/                 # Kết quả validation
│   └── metrics.csv          # ← CSV tổng hợp tất cả experiments
│
├── train.py                 # Main training entry point
├── val.py                   # Standalone validation + full metrics
├── predict.py               # Inference
├── requirements.txt
└── README.md
```

---

## Config system

Project dùng **OmegaConf** để merge configs với priority:

```
CLI overrides  >  experiment YAML  >  base.yaml
```

Các params quan trọng trong `configs/base.yaml`:

```yaml
epochs:                   100
batch:                    16
imgsz:                    640
lr0:                      0.01
device:                   ""       # "" = auto, "cpu", "0", "0,1"

# Test scheduling
test_every_n_epochs:      10       # 0 = disable test
val_split:                val
test_split:               test

# Metrics
test_measure_efficiency:  true     # đo FPS/latency khi test (false = bỏ qua)
metrics_csv:              runs/metrics.csv
```

---

## Tips & Troubleshooting

**Lỗi: `yolo12n.pt` không download được**
```bash
# Download thủ công từ ultralytics GitHub releases
# hoặc dùng YOLOv8n/v10n làm pretrained tạm:
python train.py --pretrained yolov8n.pt --idea baseline
```

**Out of memory**
```bash
# Giảm batch size
python train.py --batch 8 --idea baseline

# Giảm image size
python train.py --imgsz 416
```

**Training quá chậm do đo efficiency**
```bash
# Tắt đo efficiency trong test callback
# Trong base.yaml: test_measure_efficiency: false
# Hoặc khi chạy val.py thủ công:
python val.py --weights best.pt --no-efficiency
```

**Muốn resume training**
```bash
python train.py --idea baseline --exist-ok \
    --pretrained runs/train/yolov12n_baseline/weights/last.pt
```

**GFLOPs đo ra None**
```bash
# Cài thêm thop (optional fallback)
pip install thop
```

---

## Kết quả mong đợi (VisDrone val)

| Model | mAP50 | mAP50-95 | GFLOPs (avg) | Params | FPS (est.) | mAP/GFLOPs |
|-------|-------|----------|--------------|--------|------------|------------|
| YOLOv12n baseline | ~23.0% | ~12.0% | 8.7 | 2.6M | ~80 | 0.026 |
| + CAGI (avg) | ~22.8% | ~11.7% | ~5.2 | 2.7M | ~95 | 0.044 |
| + AMSHA | ~23.2% | ~11.9% | ~7.1 | 2.6M | ~87 | 0.033 |
| + RSFE | ~23.3% | ~12.2% | 8.7* | 2.7M | ~80 | 0.027 |
| + CGSR (inference) | ~24.0% | ~13.0% | ~11** | 2.6M | ~65 | 0.022 |
| + IAWR (avg w=0.5) | ~22.0% | ~11.5% | ~4.5 | 0.7M | ~130 | 0.049 |

*RSFE tiết kiệm FLOPs trong attention, không phải toàn model  
**CGSR thêm FLOPs nhưng tăng mAP đáng kể cho tiny objects

---

## Citation

Nếu dùng code này trong nghiên cứu, vui lòng cite:

```bibtex
@article{your2026adaptive,
  title   = {Adaptive Computation for Lightweight UAV Object Detection},
  author  = {Your Name},
  journal = {ACCV},
  year    = {2026}
}
```

---

## License

MIT License — xem `LICENSE` để biết thêm.
