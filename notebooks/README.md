# Notebooks — Hướng dẫn sử dụng trên Google Colab

Thư mục này chứa **6 Jupyter notebooks**, mỗi notebook dành cho một idea nghiên cứu.
Tất cả được thiết kế để chạy trực tiếp trên **Google Colab** với GPU miễn phí (T4).

---

## Danh sách notebooks

| File | Idea | Mô tả ngắn | Thời gian ước tính |
|------|------|------------|-------------------|
| `train_baseline.ipynb` | Baseline | YOLOv12n chuẩn, không modification | ~2–3h (T4) |
| `train_cagi.ipynb` | CAGI | Gate C3k2 blocks theo scene complexity | ~3–4h (T4) |
| `train_amsha.ipynb` | AMSHA | Bật/tắt P2 head theo small-object predictor | ~2.5–3.5h (T4) |
| `train_rsfe.ipynb` | RSFE | Sparse attention trên top-K spatial locations | ~2–3h (T4) |
| `train_cgsr.ipynb` | CGSR | 2-pass inference: nano + selective re-detect | ~2–3h (T4) |
| `train_iawr.ipynb` | IAWR | OFA supernet 4 widths + Content Router | ~6–8h (T4) |

> **Khuyến nghị**: Chạy `train_baseline.ipynb` trước để có reference mAP,
> sau đó lần lượt chạy các ideas để so sánh.

---

## Yêu cầu trước khi bắt đầu

### 1. Tài khoản Google
- Google account bình thường: có T4 GPU miễn phí (~12h/ngày)
- **Colab Pro** (~$10/tháng): A100 GPU, runtime lâu hơn → khuyến nghị cho IAWR

### 2. Google Drive
Cần ~**30 GB** dung lượng Drive trống cho:
- Dataset VisDrone (~5 GB)
- 6 experiment results (~3–5 GB mỗi cái)

### 3. Project code
Chuẩn bị **một trong ba cách** upload code lên Colab (xem chi tiết ở mục bên dưới):
- File `.zip` của thư mục `yolov12n-visdrone/`
- Repo GitHub
- Copy sẵn lên Google Drive

### 4. Dataset VisDrone (đã convert sẵn)
Dataset đã được convert sang YOLO format và zip lại. Upload `VisDrone.zip` lên Drive tại:
```
MyDrive/yolov12n-visdrone/datasets/VisDrone.zip
```
Notebook sẽ tự động giải nén — **không cần tải lại từ VisDrone GitHub**.

---

## Hướng dẫn từng bước

### Bước 1 — Mở notebook trên Colab

1. Vào [colab.research.google.com](https://colab.research.google.com)
2. Click **File** → **Upload notebook**
3. Chọn file `.ipynb` muốn chạy (ví dụ: `train_baseline.ipynb`)

Hoặc nếu đã có notebook trên Drive:

1. Vào [colab.research.google.com](https://colab.research.google.com)
2. Click tab **Google Drive** → chọn file `.ipynb`

---

### Bước 2 — Bật GPU

1. Click **Runtime** (thanh menu trên cùng)
2. Chọn **Change runtime type**
3. Chọn **T4 GPU** (miễn phí) hoặc **A100** (Colab Pro)
4. Click **Save**

Kiểm tra GPU đã sẵn sàng: cell đầu tiên sẽ in thông tin `nvidia-smi`.

---

### Bước 3 — Chạy các cells theo thứ tự

Mỗi notebook có **11 section**, chạy từ trên xuống dưới:

```
Section 1  — Kiểm tra GPU & CUDA
Section 2  — Mount Google Drive
Section 3  — Cài đặt dependencies
Section 4  — Upload project code      ← cần chọn 1 trong 3 cách
Section 5  — Chuẩn bị VisDrone dataset
Section 6  — Cấu hình training        ← có thể điều chỉnh epochs/batch
Section 7  — Training                 ← cell chạy lâu nhất
Section 7b — (Idea-specific) phân tích riêng từng idea
Section 8  — Đánh giá full metrics
Section 9  — Visualize curves
Section 10 — Lưu kết quả lên Drive
Section 11 — Demo inference
```

Để chạy tất cả cùng lúc: **Runtime** → **Run all** (không khuyến nghị lần đầu).

---

### Bước 4 — Upload project code (Section 4)

Chọn **một trong ba cách** và bỏ comment dòng tương ứng trong cell:

**Cách 1 — Upload file zip** (đơn giản nhất):
```python
from google.colab import files
up = files.upload()           # chọn file yolov12n-visdrone.zip
with zipfile.ZipFile(list(up.keys())[0]) as z:
    z.extractall('/content/')
```
Để tạo file zip: nén toàn bộ thư mục `yolov12n-visdrone/` thành `yolov12n-visdrone.zip`.

**Cách 2 — Copy từ Google Drive** (khuyến nghị nếu chạy nhiều lần):
```python
shutil.copytree('/content/drive/MyDrive/yolov12n-visdrone-code',
                PROJECT_DIR, dirs_exist_ok=True)
```
Upload thư mục `yolov12n-visdrone/` lên Drive trước, đặt tên `yolov12n-visdrone-code`.

**Cách 3 — Git clone** (nếu project đã push lên GitHub):
```bash
!git clone https://github.com/your-username/yolov12n-visdrone /content/yolov12n-visdrone
```

---

### Bước 5 — Load dataset từ Google Drive (Section 5)

Dataset đã được **convert sẵn sang YOLO format** và **zip thành `VisDrone.zip`**.
Bạn chỉ cần đảm bảo file này đã được upload lên Drive trước khi chạy notebook.

**Yêu cầu trước khi chạy:**
Upload `VisDrone.zip` lên Drive tại đường dẫn:
```
MyDrive/yolov12n-visdrone/datasets/VisDrone.zip
```

> File này được tạo bằng cách chạy `scripts/prepare_visdrone.py` rồi zip thư mục `datasets/VisDrone/` lại.
> **Không cần chạy lại `prepare_visdrone.py`** — notebook sẽ tự giải nén.

Notebook xử lý theo thứ tự ưu tiên:
```
VisDrone/ đã tồn tại trong Colab?
    ✓ → Dùng ngay (nhanh nhất)
    ✗ → Drive có VisDrone.zip?
            ✓ → Giải nén (~60s)
            ✗ → Drive có thư mục VisDrone/ (đã giải nén)?
                    ✓ → Tạo symlink (~2s)
                    ✗ → Báo lỗi, kiểm tra lại đường dẫn Drive
```

---

### Bước 6 — Điều chỉnh config training (Section 6)

Thay đổi các biến trong cell params:

```python
EPOCHS      = 100   # tăng lên 150-200 nếu có Colab Pro
BATCH       = 16    # giảm xuống 8 nếu bị lỗi CUDA out of memory
IMGSZ       = 640   # có thể giảm xuống 416 để train nhanh hơn
DEVICE      = '0'   # GPU 0 (không cần đổi trên Colab)
TEST_EVERY  = 10    # chạy test split sau mỗi 10 epochs
```

---

### Bước 7 — Training (Section 7)

Cell training chạy lâu nhất. Trong quá trình chạy:

- **Mỗi epoch**: ultralytics tự chạy val và in mAP50
- **Mỗi `TEST_EVERY` epochs**: chạy test split + in full metrics report (Accuracy + Efficiency + Ratios)
- **Log**: ghi vào `logs/{EXP_NAME}.log`

Nếu Colab bị ngắt kết nối giữa chừng:
```python
# Trong cell config, đổi pretrained weights sang last.pt:
# Thêm vào cuối lệnh train:
# --pretrained runs/train/yolov12n_baseline/weights/last.pt
# --exist-ok
```

---

### Bước 8 — Đọc kết quả metrics (Section 8)

Sau khi training xong, Section 8 chạy val và test, in ra:

```
══════════════════════════════════════════════════════════
  EVALUATION REPORT  [TEST]  —  YOLOv12n / baseline
══════════════════════════════════════════════════════════
  1. ACCURACY METRICS
  mAP50                            0.2301
  mAP50-95                         0.1182
  Precision (mean)                 0.3756
  Recall / AR (mean)               0.2901

  Per-class AP50:
    pedestrian           0.3021  ████████████░░░░░░░░░
    car                  0.5512  █████████████████████

  2. EFFICIENCY METRICS
  Parameters                       2.623 M
  GFLOPs                           8.70 G
  Model size                       5.42 MB
  Latency (batch=1)                12.30 ms/img
  FPS (batch=1)                    81.3 fps

  3. EFFICIENCY RATIOS
  mAP50 / GFLOPs                   0.0265
  mAP50 / Params (M)               0.0877
  mAP50 x FPS                      18.70
══════════════════════════════════════════════════════════
```

Kết quả được tự động ghi vào `runs/metrics.csv` để so sánh nhiều experiments.

---

### Bước 9 — Lưu kết quả (Section 10)

Section 10 tự động:
1. Copy toàn bộ experiment folder lên Drive: `MyDrive/yolov12n-visdrone/runs/{EXP_NAME}/`
2. Backup `metrics.csv` lên Drive
3. (Tùy chọn) Download `best.pt` về máy tính

---

## So sánh kết quả nhiều experiments

Sau khi chạy xong nhiều notebooks, đọc CSV tổng hợp:

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/yolov12n-visdrone/metrics.csv')
cols = ['model', 'idea', 'split', 'acc_map50', 'acc_map50_95',
        'eff_gflops', 'eff_params_m', 'eff_fps',
        'ratio_map50_per_gflop', 'ratio_map50_x_fps']
print(df[cols].to_string(index=False))
```

Ví dụ output:

```
model     idea      split  acc_map50  acc_map50_95  eff_gflops  eff_params_m  eff_fps  ratio_map50_per_gflop
YOLOv12n  baseline  test      0.2301        0.1182        8.70         2.623     81.3                 0.0265
YOLOv12n  cagi      test      0.2289        0.1174        5.20         2.701     94.7                 0.0440
YOLOv12n  amsha     test      0.2318        0.1193        7.10         2.631     87.2                 0.0326
YOLOv12n  rsfe      test      0.2334        0.1204        8.70         2.718     79.8                 0.0268
```

---

## Xử lý lỗi thường gặp

**CUDA out of memory**
```python
# Trong Section 6, giảm batch size:
BATCH = 8   # hoặc 4
```

**Runtime bị ngắt (Colab disconnect)**

Colab miễn phí giới hạn ~12h liên tục. Nếu bị ngắt:
1. Reconnect runtime
2. Chạy lại từ Section 1 đến Section 5 (nhanh vì dataset đã có trên Drive)
3. Trong Section 6, thêm flag resume vào lệnh train bằng cách sửa cell train:
```bash
!python train.py \
    --model   {MODEL} \
    --idea    {IDEA} \
    ...
    --pretrained {PROJECT_OUT}/{EXP_NAME}/weights/last.pt \
    --exist-ok
```

**Notebook chạy nhưng mAP = 0**

Kiểm tra dataset đã được convert đúng chưa:
```bash
!python scripts/check_dataset.py --dataset-root datasets/VisDrone
```

**GFLOPs hiển thị N/A**

Cài thêm `thop`:
```bash
%pip install thop
```

**`yolo12n.pt` không download được**

Dùng YOLOv8n làm pretrained tạm:
```python
# Trong Section 6:
# Thêm vào lệnh train: --pretrained yolov8n.pt
```

---

## Thứ tự chạy được khuyến nghị

```
1. train_baseline.ipynb   → lấy reference mAP (~23%) và FPS (~80)
2. train_rsfe.ipynb       → nhẹ nhất, ít thay đổi nhất, dễ so sánh
3. train_amsha.ipynb      → tiết kiệm FLOPs rõ rệt, dễ train
4. train_cagi.ipynb       → phức tạp hơn, cần theo dõi complexity log
5. train_cgsr.ipynb       → train như baseline, khác biệt ở inference
6. train_iawr.ipynb       → phức tạp nhất, train lâu nhất (dùng Colab Pro)
```

---

## Cấu trúc thư mục sau khi chạy xong trên Drive

```
MyDrive/yolov12n-visdrone/
├── datasets/
│   ├── VisDrone_raw/        # Dataset gốc (giải nén từ zip)
│   └── VisDrone/            # YOLO format (auto-tạo)
├── runs/
│   ├── yolov12n_baseline/   # Kết quả training baseline
│   │   ├── weights/
│   │   │   ├── best.pt      # Model tốt nhất
│   │   │   └── last.pt      # Model epoch cuối
│   │   ├── results.png      # Training curves
│   │   └── results.csv      # Per-epoch metrics
│   ├── yolov12n_cagi/
│   ├── yolov12n_amsha/
│   └── ...
└── metrics.csv              # Tổng hợp tất cả experiments
```
