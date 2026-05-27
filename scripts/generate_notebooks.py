# -*- coding: utf-8 -*-
"""Generate 6 Colab notebooks for each model idea."""
import json
from pathlib import Path

def md(s): return {"cell_type":"markdown","metadata":{},"source":s.strip()}
def code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip()}
def nb(cells): return {
    "nbformat":4,"nbformat_minor":5,
    "metadata":{
        "accelerator":"GPU",
        "colab":{"provenance":[],"gpuType":"T4"},
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.10.0"},
    },
    "cells":cells,
}

# ── shared cells ──────────────────────────────────────────────────────────────

GPU_CHECK = [
    md("## 1. Kiem tra GPU"),
    code(
        "import subprocess, torch\n"
        "r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)\n"
        "print(r.stdout if r.returncode==0 else 'No GPU')\n"
        "print(f'PyTorch : {torch.__version__}')\n"
        "print(f'CUDA    : {torch.version.cuda}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
        "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')\n"
    ),
]

DRIVE_MOUNT = [
    md("## 2. Mount Google Drive"),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "import os\n"
        "DRIVE_ROOT = '/content/drive/MyDrive/yolov12n-visdrone'\n"
        "os.makedirs(DRIVE_ROOT, exist_ok=True)\n"
        "print(f'Drive: {DRIVE_ROOT}')\n"
    ),
]

INSTALL = [
    md("## 3. Cai dat dependencies"),
    code(
        "%pip install -q ultralytics>=8.3.0 omegaconf>=2.3.0 rich>=13.0.0 thop\n"
        "import ultralytics; ultralytics.checks()\n"
        "print(f'ultralytics: {ultralytics.__version__}')\n"
    ),
]

UPLOAD = [
    md(
        "## 4. Upload project code\n\n"
        "Chon **mot trong 3 cach**:\n\n"
        "- Cach 1: Upload file `.zip` tu may tinh\n"
        "- Cach 2: Copy tu Google Drive\n"
        "- Cach 3: `git clone` tu GitHub\n"
    ),
    code(
        "import os, shutil, zipfile\n"
        "PROJECT_DIR = '/content/yolov12n-visdrone'\n"
        "\n"
        "# -- Cach 1: Upload zip -------------------------------------------------\n"
        "# from google.colab import files\n"
        "# up = files.upload()\n"
        "# with zipfile.ZipFile(list(up.keys())[0]) as z: z.extractall('/content/')\n"
        "\n"
        "# -- Cach 2: Tu Drive ---------------------------------------------------\n"
        "# shutil.copytree('/content/drive/MyDrive/yolov12n-visdrone-code',\n"
        "#                 PROJECT_DIR, dirs_exist_ok=True)\n"
        "\n"
        "# -- Cach 3: Git clone --------------------------------------------------\n"
        "# !git clone https://github.com/your-username/yolov12n-visdrone {PROJECT_DIR}\n"
        "\n"
        "print('OK' if os.path.exists(PROJECT_DIR) else 'Project chua duoc upload')\n"
    ),
    code(
        "import os\n"
        "os.chdir('/content/yolov12n-visdrone')\n"
        "print(os.getcwd(), os.listdir('.'))\n"
    ),
]

DATASET = [
    md(
        "## 5. Load VisDrone dataset tu Google Drive\n\n"
        "> Dataset da duoc convert sang YOLO format, zip lai va upload len Drive.\n"
        "> Cell nay chi can giai nen hoac tao symlink -- **khong can chay lai prepare_visdrone.py**.\n\n"
        "**Duong dan Drive du kien:**\n"
        "```\n"
        "MyDrive/yolov12n-visdrone/datasets/VisDrone.zip   <- zip toan bo thu muc VisDrone/\n"
        "hoac\n"
        "MyDrive/yolov12n-visdrone/datasets/VisDrone/      <- thu muc da giai nen san\n"
        "```\n"
    ),
    code(
        "import os, shutil, zipfile, time\n"
        "\n"
        "YOLO     = '/content/yolov12n-visdrone/datasets/VisDrone'\n"
        "DY_DIR   = '/content/drive/MyDrive/yolov12n-visdrone/datasets/VisDrone'\n"
        "DY_ZIP   = '/content/drive/MyDrive/yolov12n-visdrone/datasets/VisDrone.zip'\n"
        "\n"
        "os.makedirs('/content/yolov12n-visdrone/datasets', exist_ok=True)\n"
        "\n"
        "if os.path.exists(YOLO):\n"
        "    print('Dataset da ton tai:', YOLO)\n"
        "\n"
        "elif os.path.exists(DY_ZIP):\n"
        "    # Giai nen tu file zip tren Drive (nhanh hon copy thu muc)\n"
        "    print(f'Giai nen {DY_ZIP} ...')\n"
        "    t0 = time.time()\n"
        "    with zipfile.ZipFile(DY_ZIP, 'r') as z:\n"
        "        z.extractall('/content/yolov12n-visdrone/datasets/')\n"
        "    print(f'Xong! ({time.time()-t0:.0f}s)')\n"
        "\n"
        "elif os.path.exists(DY_DIR):\n"
        "    # Symlink truc tiep thu muc tren Drive (khong can copy)\n"
        "    print(f'Symlink tu Drive: {DY_DIR}')\n"
        "    os.symlink(DY_DIR, YOLO)\n"
        "    print('Symlink OK')\n"
        "\n"
        "else:\n"
        "    print('KHONG TIM THAY DATASET!')\n"
        "    print('Hay dam bao da upload len Drive theo mot trong hai cach:')\n"
        "    print('  1. MyDrive/yolov12n-visdrone/datasets/VisDrone.zip')\n"
        "    print('  2. MyDrive/yolov12n-visdrone/datasets/VisDrone/')\n"
        "    raise FileNotFoundError('Dataset not found on Drive')\n"
        "\n"
        "# Kiem tra nhanh\n"
        "for split in ['train', 'val', 'test']:\n"
        "    img_dir = os.path.join(YOLO, 'images', split)\n"
        "    n = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0\n"
        "    print(f'  {split:5s}: {n} images')\n"
    ),
    code(
        "# Kiem tra toan ven dataset (tuy chon)\n"
        "!python scripts/check_dataset.py --dataset-root datasets/VisDrone\n"
    ),
]

SAVE_DRIVE = [
    md("## 10. Luu ket qua len Google Drive"),
    code(
        "import shutil, os\n"
        "DRIVE_RUNS = '/content/drive/MyDrive/yolov12n-visdrone/runs'\n"
        "os.makedirs(DRIVE_RUNS, exist_ok=True)\n"
        "src = f'{PROJECT_OUT}/{EXP_NAME}'\n"
        "dst = f'{DRIVE_RUNS}/{EXP_NAME}'\n"
        "if os.path.exists(src):\n"
        "    shutil.copytree(src, dst, dirs_exist_ok=True)\n"
        "    sz = sum(os.path.getsize(os.path.join(d,f))\n"
        "             for d,_,fs in os.walk(dst) for f in fs)\n"
        "    print(f'Saved: {dst}  ({sz/1e6:.1f} MB)')\n"
        "else:\n"
        "    print('Not found:', src)\n"
    ),
    code(
        "import shutil, os\n"
        "src = 'runs/metrics.csv'\n"
        "dst = '/content/drive/MyDrive/yolov12n-visdrone/metrics.csv'\n"
        "if os.path.exists(src): shutil.copy2(src, dst); print('metrics.csv saved')\n"
    ),
    code(
        "from google.colab import files\n"
        "import os\n"
        "pt = f'{PROJECT_OUT}/{EXP_NAME}/weights/best.pt'\n"
        "if os.path.exists(pt): files.download(pt)\n"
        "else: print('best.pt not found')\n"
    ),
]

PREDICT_DEMO = [
    md("## 11. Demo Inference (tuy chon)"),
    code(
        "import glob, random\n"
        "imgs = glob.glob('datasets/VisDrone/images/test/*.jpg')\n"
        "if imgs:\n"
        "    s = random.choice(imgs)\n"
        "    print('Demo:', s)\n"
        "    os.system(f'python predict.py --weights {WEIGHTS} --source \"{s}\"'\n"
        "              f' --conf 0.25 --imgsz {IMGSZ} --device {DEVICE}'\n"
        "              f' --project runs/predict --name demo_{IDEA}')\n"
    ),
    code(
        "from IPython.display import Image, display\n"
        "import glob\n"
        "for p in glob.glob(f'runs/predict/demo_{IDEA}/*.jpg')[:3]:\n"
        "    display(Image(p, width=820))\n"
    ),
]

VISUALIZE = [
    md("## 9. Visualize ket qua training"),
    code(
        "from IPython.display import Image, display\n"
        "import os\n"
        "exp_dir = f'{PROJECT_OUT}/{EXP_NAME}'\n"
        "for name in ['results.png','confusion_matrix.png','PR_curve.png','F1_curve.png']:\n"
        "    p = os.path.join(exp_dir, name)\n"
        "    if os.path.exists(p):\n"
        "        print(name); display(Image(p, width=820))\n"
    ),
    code(
        "import pandas as pd, os\n"
        "rcsv = f'{PROJECT_OUT}/{EXP_NAME}/results.csv'\n"
        "if os.path.exists(rcsv):\n"
        "    df = pd.read_csv(rcsv); df.columns = df.columns.str.strip()\n"
        "    col = 'metrics/mAP50(B)'\n"
        "    if col in df.columns:\n"
        "        best = df.loc[df[col].idxmax()]\n"
        "        print(f'Best epoch : {int(best.get(\"epoch\",0))}')\n"
        "        print(f'mAP50      : {best[col]:.4f}')\n"
        "        print(f'mAP50-95   : {best.get(\"metrics/mAP50-95(B)\",0):.4f}')\n"
        "        print(f'Precision  : {best.get(\"metrics/precision(B)\",0):.4f}')\n"
        "        print(f'Recall     : {best.get(\"metrics/recall(B)\",0):.4f}')\n"
        "    else:\n"
        "        print(df.tail(3).to_string())\n"
    ),
]

EVAL = [
    md(
        "## 8. Danh gia sau training -- Full Metrics Report\n\n"
        "3 phan:\n"
        "1. **Accuracy**: mAP50, mAP50-95, Precision, Recall, AP per-class\n"
        "2. **Efficiency**: GFLOPs, Params, Model size, Latency, FPS\n"
        "3. **Efficiency Ratios**: mAP50/GFLOPs, mAP50/Params, mAP50*FPS\n"
    ),
    code(
        "import os\n"
        "WEIGHTS = f'{PROJECT_OUT}/{EXP_NAME}/weights/best.pt'\n"
        "if not os.path.exists(WEIGHTS):\n"
        "    WEIGHTS = f'{PROJECT_OUT}/{EXP_NAME}/weights/last.pt'\n"
        "    print(f'best.pt not found, using: {WEIGHTS}')\n"
        "else:\n"
        "    print(f'Weights: {WEIGHTS}')\n"
    ),
    code(
        "# -- Val split ---------------------------------------------------------\n"
        "!python val.py \\\n"
        "    --weights        {WEIGHTS} \\\n"
        "    --data           configs/visdrone.yaml \\\n"
        "    --split          val \\\n"
        "    --imgsz          {IMGSZ} \\\n"
        "    --device         {DEVICE} \\\n"
        "    --model-name     {MODEL} \\\n"
        "    --idea           {IDEA} \\\n"
        "    --save-csv       runs/metrics.csv \\\n"
        "    --benchmark-runs 30\n"
    ),
    code(
        "# -- Test split --------------------------------------------------------\n"
        "!python val.py \\\n"
        "    --weights        {WEIGHTS} \\\n"
        "    --data           configs/visdrone.yaml \\\n"
        "    --split          test \\\n"
        "    --imgsz          {IMGSZ} \\\n"
        "    --device         {DEVICE} \\\n"
        "    --model-name     {MODEL} \\\n"
        "    --idea           {IDEA} \\\n"
        "    --save-csv       runs/metrics.csv \\\n"
        "    --benchmark-runs 30\n"
    ),
    code(
        "import pandas as pd, os\n"
        "if os.path.exists('runs/metrics.csv'):\n"
        "    df = pd.read_csv('runs/metrics.csv')\n"
        "    want = ['timestamp','model','idea','split',\n"
        "            'acc_map50','acc_map50_95','acc_precision','acc_recall',\n"
        "            'eff_gflops','eff_params_m','eff_fps','eff_latency_ms',\n"
        "            'ratio_map50_per_gflop','ratio_map50_x_fps']\n"
        "    cols = [c for c in want if c in df.columns]\n"
        "    pd.set_option('display.max_columns', None)\n"
        "    pd.set_option('display.width', 200)\n"
        "    print(df[cols].to_string(index=False))\n"
        "else:\n"
        "    print('metrics.csv not found')\n"
    ),
]

# ── idea-specific extra cells ─────────────────────────────────────────────────

EXTRA = {
    "baseline": [],
    "cagi": [
        md("### 7b. CAGI -- Phan tich complexity distribution"),
        code(
            "import re, os\n"
            "log = f'logs/{EXP_NAME}.log'\n"
            "easy, med, hard = [], [], []\n"
            "pat = r'EASY: ([\\d.]+)%.*?MEDIUM: ([\\d.]+)%.*?HARD: ([\\d.]+)%'\n"
            "if os.path.exists(log):\n"
            "    for line in open(log):\n"
            "        m = re.search(pat, line)\n"
            "        if m: easy.append(float(m.group(1))); med.append(float(m.group(2))); hard.append(float(m.group(3)))\n"
            "if easy:\n"
            "    e,m_,h = sum(easy)/len(easy), sum(med)/len(med), sum(hard)/len(hard)\n"
            "    ratio  = (e*0.25 + m_*0.5 + h*1.0)/100\n"
            "    print(f'Avg complexity ({len(easy)} epochs):')\n"
            "    print(f'  EASY  : {e:.1f}%  (1 block)')\n"
            "    print(f'  MEDIUM: {m_:.1f}%  (2 blocks)')\n"
            "    print(f'  HARD  : {h:.1f}%  (4 blocks)')\n"
            "    print(f'Est. avg FLOPs stage4: {ratio*100:.1f}%')\n"
            "    print(f'Est. FLOPs saved     : {(1-ratio)*100:.1f}%')\n"
            "else:\n"
            "    print('No CAGI log found')\n"
        ),
    ],
    "amsha": [
        md("### 7b. AMSHA -- P2 head activation rate"),
        code(
            "import re, os\n"
            "log = f'logs/{EXP_NAME}.log'\n"
            "rates = []\n"
            "pat = r'P2 head activated: \\d+/\\d+ \\(([\\d.]+)%\\)'\n"
            "if os.path.exists(log):\n"
            "    for line in open(log):\n"
            "        m = re.search(pat, line)\n"
            "        if m: rates.append(float(m.group(1)))\n"
            "if rates:\n"
            "    avg = sum(rates)/len(rates)\n"
            "    saved = (100-avg)/100*8.0\n"
            "    print(f'P2 activation rate ({len(rates)} epochs):')\n"
            "    print(f'  Mean: {avg:.1f}%  Min: {min(rates):.1f}%  Max: {max(rates):.1f}%')\n"
            "    print(f'Est. FLOPs saved: {saved:.1f} GFLOPs/img ({100-avg:.1f}% images skip P2)')\n"
            "else:\n"
            "    print('No AMSHA log found')\n"
        ),
    ],
    "rsfe": [
        md("### 7b. RSFE -- Sparse attention FLOPs estimate"),
        code(
            "import yaml\n"
            "with open('configs/idea_rsfe.yaml') as f: cfg = yaml.safe_load(f)\n"
            "topk = cfg.get('rsfe_topk_ratio', 0.25)\n"
            "print(f'topk_ratio : {topk}  ({topk*100:.0f}% locations processed)')\n"
            "print(f'Attention FLOPs saved: {(1-topk)*100:.0f}%')\n"
        ),
    ],
    "cgsr": [
        md("### 7b. CGSR -- Demo 2-pass selective re-detection"),
        code(
            "import glob, random, os\n"
            "imgs = glob.glob('datasets/VisDrone/images/test/*.jpg')\n"
            "if imgs:\n"
            "    s = random.choice(imgs)\n"
            "    print('2-pass demo:', s)\n"
            "    os.system(f'python predict.py --weights {WEIGHTS} --source \"{s}\"'\n"
            "              f' --idea cgsr --conf 0.25 --device {DEVICE}'\n"
            "              f' --project runs/predict --name demo_cgsr_2pass')\n"
        ),
    ],
    "iawr": [
        md("### 7b. IAWR -- Kiem tra width checkpoints"),
        code(
            "import os\n"
            "for w in [1.0, 0.75, 0.5, 0.25]:\n"
            "    ck = f'{PROJECT_OUT}/{EXP_NAME}_w{w}/weights/best.pt'\n"
            "    print(f'  {\"ok\" if os.path.exists(ck) else \"missing\"} w={w}: {ck}')\n"
        ),
        code(
            "import glob, random, os\n"
            "imgs = glob.glob('datasets/VisDrone/images/test/*.jpg')\n"
            "if imgs:\n"
            "    s = random.choice(imgs)\n"
            "    os.system(f'python predict.py --weights {WEIGHTS} --source \"{s}\"'\n"
            "              f' --idea iawr --conf 0.25 --device {DEVICE}'\n"
            "              f' --project runs/predict --name demo_iawr')\n"
        ),
    ],
}

# ── idea definitions ─────────────────────────────────────────────────────────

IDEAS = [
    ("baseline", "YOLOv12n Baseline",
     "YOLOv12n pretrained chuan, khong modification. Dung lam reference.\n\n"
     "> Thoi gian: ~2-3h tren Colab T4 (100 epochs)"),

    ("cagi", "Complexity-Aware Gated Inference (CAGI)",
     "Scene Complexity Predictor quyet dinh so C3k2 blocks: 1/2/4 theo EASY/MEDIUM/HARD.\n\n"
     "> Thoi gian: ~3-4h tren Colab T4"),

    ("amsha", "Adaptive Multi-Scale Head Activation (AMSHA)",
     "SOEP du doan xac suat co small objects. Thap -> bo qua P2 head -> tiet kiem ~8 GFLOPs/img.\n\n"
     "> Thoi gian: ~2.5-3.5h tren Colab T4"),

    ("rsfe", "Region-Wise Sparse Feature Enhancement (RSFE)",
     "Top-K sparse attention: chi xu ly 25% spatial locations -> tiet kiem 75% FLOPs attention.\n\n"
     "> Thoi gian: ~2-3h tren Colab T4"),

    ("cgsr", "Confidence-Guided Selective Re-detection (CGSR)",
     "2-pass inference: full nano pass -> re-detect vung uncertain. Tang recall tiny objects.\n\n"
     "> Thoi gian: ~2-3h training"),

    ("iawr", "Input-Adaptive Width Routing (IAWR)",
     "OFA supernet 4 widths {0.25,0.5,0.75,1.0} + Content Router chon subnet theo anh.\n\n"
     "> Thoi gian: ~6-8h tren Colab T4. Khuyen nghi Colab Pro + A100."),
]

# ── generate ─────────────────────────────────────────────────────────────────

def make_config_cells(key):
    cfg_file = "baseline.yaml" if key == "baseline" else f"idea_{key}.yaml"
    cfg_path = f"configs/{cfg_file}"
    return [
        md("## 6. Cau hinh training"),
        code(
            "import yaml\n"
            "with open('configs/base.yaml') as f: base = yaml.safe_load(f)\n"
            f"idea_cfg_path = '{cfg_path}'\n"
            "try:\n"
            "    with open(idea_cfg_path) as f: idea = yaml.safe_load(f)\n"
            "    print('Experiment config:', idea_cfg_path)\n"
            "    print(yaml.dump(idea, default_flow_style=False))\n"
            "except FileNotFoundError:\n"
            "    idea = {}; print('Using base config only')\n"
            "merged = {**base, **idea}\n"
            "print('Key params:')\n"
            "for k in ['epochs','batch','imgsz','lr0','device','test_every_n_epochs']:\n"
            "    print(f'  {k}: {merged.get(k,\"N/A\")}')\n"
        ),
        code(
            f"IDEA        = '{key}'\n"
            "MODEL       = 'yolov12n'\n"
            "EPOCHS      = 100   # change here\n"
            "BATCH       = 16    # reduce to 8 if OOM\n"
            "IMGSZ       = 640\n"
            "DEVICE      = '0'   # GPU 0\n"
            "TEST_EVERY  = 10\n"
            "PROJECT_OUT = 'runs/train'\n"
            "EXP_NAME    = f'{MODEL}_{IDEA}'\n"
            "print(f'Model: {MODEL} | Idea: {IDEA} | Epochs: {EPOCHS} | Batch: {BATCH}')\n"
            "print(f'Output: {PROJECT_OUT}/{EXP_NAME}/')\n"
        ),
    ]

def make_train_cells():
    return [
        md(
            "## 7. Training\n\n"
            "- Val tu dong sau moi epoch (ultralytics val=True)\n"
            "- Test + full metrics report sau moi TEST_EVERY epochs\n"
        ),
        code(
            "import os; os.makedirs('logs', exist_ok=True)\n"
            "!python train.py \\\n"
            "    --model      {MODEL} \\\n"
            "    --idea       {IDEA} \\\n"
            "    --epochs     {EPOCHS} \\\n"
            "    --batch      {BATCH} \\\n"
            "    --imgsz      {IMGSZ} \\\n"
            "    --device     {DEVICE} \\\n"
            "    --project    {PROJECT_OUT} \\\n"
            "    --name       {EXP_NAME} \\\n"
            "    --test-every {TEST_EVERY} \\\n"
            "    --log-file   logs/{EXP_NAME}.log\n"
        ),
    ]

def generate(key, name, desc, out_dir):
    header = md(
        f"# YOLOv12n x VisDrone -- `{key}` Notebook\n\n"
        f"> ACCV 2026 | Adaptive Computation for Lightweight UAV Object Detection\n\n"
        f"**Idea**: {name}\n\n"
        f"**Mo ta**: {desc}\n\n"
        "---\n\n"
        "### Huong dan\n"
        "1. Runtime -> Change runtime type -> T4 GPU\n"
        "2. Chay tung cell tu tren xuong duoi\n"
        "3. Ket qua luu vao Drive: MyDrive/yolov12n-visdrone/runs/\n"
        "4. Metrics: MyDrive/yolov12n-visdrone/metrics.csv\n"
    )
    cells = (
        [header]
        + GPU_CHECK
        + DRIVE_MOUNT
        + INSTALL
        + UPLOAD
        + DATASET
        + make_config_cells(key)
        + make_train_cells()
        + EXTRA.get(key, [])
        + EVAL
        + VISUALIZE
        + SAVE_DRIVE
        + PREDICT_DEMO
    )
    path = out_dir / f"train_{key}.ipynb"
    path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    return path

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    out_dir.mkdir(exist_ok=True)
    print(f"Generating {len(IDEAS)} notebooks -> {out_dir}/\n")
    for key, name, desc in IDEAS:
        p = generate(key, name, desc, out_dir)
        print(f"  ok  {p.name:<32}  ({p.stat().st_size/1024:.1f} KB)")
    print(f"\nDone!")
