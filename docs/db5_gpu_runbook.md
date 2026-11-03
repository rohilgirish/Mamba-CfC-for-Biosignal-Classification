# Ninapro DB5 GPU Training Runbook (Google Colab / Kaggle)

This runbook details the exact, reproducible protocol to execute full training of all 5 neural network models for the Ninapro DB5 attribution study on a remote GPU (such as Google Colab or Kaggle with a T4 GPU).

---

## 1. Files to Upload to Google Drive

Before starting the Colab session, upload the processed DB5 dataset files from your local workspace to Google Drive:

- Local directory: `Capstone/DB5/Mamba-CfC-for-Biosignal-Classification/data/processed_db5/`
- Target Drive directory: `My Drive/Ninapro_DB5/data/` (or `/content/drive/MyDrive/Ninapro_DB5/data/`)

### Exact File List and Sizes
| File Name | Exact Size (Bytes) | Size (MB) | Purpose |
| :--- | :--- | :--- | :--- |
| `train.h5` | 168,966,339 B | 161.14 MB | Training split (Subjects 9, 2, 6, 1, 8, 3; 186,942 windows) |
| `val.h5` | 52,251,803 B | 49.83 MB | Validation split (Subjects 10, 5; 65,414 windows) |
| `test.h5` | 57,632,713 B | 54.96 MB | Held-out test split (Subjects 4, 7; 65,237 windows) |
| `norm_stats_db5.json` | 978 B | 0.001 MB | Frozen train-set channel-wise $\mu$ and $\sigma$ normalization constants |
| **Total** | **278,851,833 B** | **~265.93 MB** | *(Under 300 MB; uploads in ~1-2 minutes)* |

*(Note: Raw `.mat` files are NOT needed on Colab; only the preprocessed `.h5` files and normalization stats are required).*

---

## 2. Environment Preparation in Google Colab

1. **Open Google Colab** and create a new notebook or upload `notebooks/db5_train_colab.ipynb`.
2. **Select GPU Hardware Accelerator**:
   - Menu: **Runtime** $\to$ **Change runtime type**
   - Hardware accelerator: **T4 GPU**
   - Click **Save**.

---

## 3. Step-by-Step Execution Guide

### Step 1: Ingest Repository Code
Choose **Option A** (Git clone on `main`) or **Option B** (Code Zip from Drive):

#### Option A: Clone from GitHub (`main` branch)

**If Public Repository**:
```bash
!git clone https://github.com/rohilgirish/Mamba-CfC-for-Biosignal-Classification.git
%cd Mamba-CfC-for-Biosignal-Classification
```

**If Private Repository (via Colab Secrets)**:
1. In Colab's left sidebar, click the **Key icon** (Secrets).
2. Add a new secret: Name = `GITHUB_TOKEN`, Value = your GitHub Personal Access Token (read permission).
3. Toggle "Notebook access" ON for `GITHUB_TOKEN`.
4. Run:
```python
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/rohilgirish/Mamba-CfC-for-Biosignal-Classification.git
%cd Mamba-CfC-for-Biosignal-Classification
```

#### Option B: Unzip Code from Google Drive (Fallback)
If git is not accessible on the instance, upload `code_db5.zip` to `MyDrive/Ninapro_DB5/code_db5.zip`:
```python
from google.colab import drive
import os
drive.mount('/content/drive')

!unzip -q /content/drive/MyDrive/Ninapro_DB5/code_db5.zip -d /content/ninapro_db5
%cd /content/ninapro_db5
```

---

### Step 2: Install Required Dependencies
Colab provides PyTorch and CUDA by default. Install the specific required packages:
```bash
!pip install --quiet h5py==3.16.0 ncps==1.0.1 scikit-learn pyyaml
```

Verify GPU environment:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
```

---

### Step 3: Mount Google Drive and Copy Processed Data
```python
from google.colab import drive
import os

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

DRIVE_DATA_DIR = '/content/drive/MyDrive/Ninapro_DB5/data'
LOCAL_DATA_DIR = 'data/processed_db5'

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Copy to local NVMe SSD storage for maximal I/O throughput
for fname in ['train.h5', 'val.h5', 'test.h5', 'norm_stats_db5.json']:
    src = os.path.join(DRIVE_DATA_DIR, fname)
    dst = os.path.join(LOCAL_DATA_DIR, fname)
    if os.path.exists(src) and not os.path.exists(dst):
        !cp "$src" "$dst"
        print(f"Copied {fname} to local storage.")
    elif os.path.exists(dst):
        print(f"{fname} ready locally.")
    else:
        print(f"[!] Warning: {src} not found on Drive!")
```

---

### Step 4: Run GPU Smoke Test & Latency Profiling
Measure real seconds per batch on the T4 GPU before launching the full run. Smoke tests are completely isolated in `smoke/`:
```bash
# Isolated smoke test across models in protocol order
!python scripts/train_db5.py --model cfc-only --smoke-test --device cuda
!python scripts/train_db5.py --model mamba-only --smoke-test --device cuda
!python scripts/train_db5.py --model mamba-cfc --smoke-test --device cuda
!python scripts/train_db5.py --model mamba-width --smoke-test --device cuda
!python scripts/train_db5.py --model mamba-3l --smoke-test --device cuda
```

---

### Step 5: Full Training Run (All 5 Models)
Run all 5 models sequentially in the required protocol order:
**`cfc-only` $\to$ `mamba-only` $\to$ `mamba-cfc` $\to$ `mamba-width` $\to$ `mamba-3l`**

- Optimizer: `Adam(lr=1e-3)`
- Loss: `CrossEntropyLoss()`
- Batch size: `256`
- Sampler: `WeightedRandomSampler` on train labels
- Gradient clipping: `max_norm=1.0`
- Early stopping: `patience=5` on validation Macro-F1
- Max epochs: `50`
- Seed: `42` (default)

```bash
%%bash
# Required model order
MODELS=("cfc-only" "mamba-only" "mamba-cfc" "mamba-width" "mamba-3l")

mkdir -p /content/drive/MyDrive/Ninapro_DB5/checkpoints
mkdir -p /content/drive/MyDrive/Ninapro_DB5/reports

for model in "${MODELS[@]}"; do
    echo "=========================================================="
    echo "Starting training for: $model"
    echo "=========================================================="
    python scripts/train_db5.py \
        --model "$model" \
        --data-dir data/processed_db5 \
        --out-dir . \
        --max-epochs 50 \
        --batch-size 256 \
        --lr 1e-3 \
        --patience 5 \
        --seed 42 \
        --device cuda \
        --resume
        
    # Backup checkpoints and reports immediately after each model finishes
    cp -r checkpoints/* /content/drive/MyDrive/Ninapro_DB5/checkpoints/
    cp -r reports/* /content/drive/MyDrive/Ninapro_DB5/reports/
    echo "Backed up checkpoints and reports for $model to Google Drive."
done
```

#### Optional Multi-Seed Loop (Seeds 42, 43, 44):
To run multi-seed replication with automated namespacing:
```bash
%%bash
MODELS=("cfc-only" "mamba-only" "mamba-cfc" "mamba-width" "mamba-3l")
SEEDS=(42 43 44)

for seed in "${SEEDS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Training $model with seed $seed..."
        python scripts/train_db5.py \
            --model "$model" \
            --seed "$seed" \
            --device cuda \
            --resume
    done
done
```

---

### Step 6: Post-Training Evaluation on Held-Out Test Set
Evaluate all best checkpoints on the unseen test subjects (`test.h5`, 65,237 windows):
```bash
!python scripts/evaluate_db5.py \
    --models cfc-only mamba-only mamba-cfc mamba-width mamba-3l \
    --data-dir data/processed_db5 \
    --ckpt-dir checkpoints \
    --out-dir reports \
    --seed 42 \
    --device cuda
```

Display results table:
```python
import json
import pandas as pd

with open('reports/final_results_db5.json', 'r') as f:
    results = json.load(f)

# Unpack results keyed by model and seed
flat_results = []
for model, seeds in results.items():
    for seed, metrics in seeds.items():
        flat_results.append({
            'Model': model,
            'Seed': seed,
            'Params': f"{metrics['parameters']:,}",
            'Macro-F1': f"{metrics['macro_f1']:.4f}",
            'Balanced Accuracy': f"{metrics['balanced_accuracy']:.4f}"
        })

df = pd.DataFrame(flat_results)
print(df.to_markdown(index=False))
```

---

### Step 7: Backup Final Artifacts to Drive
```bash
!cp reports/final_results_db5.json /content/drive/MyDrive/Ninapro_DB5/reports/final_results_db5.json
!cp reports/*_history_db5*.json /content/drive/MyDrive/Ninapro_DB5/reports/
!cp checkpoints/*_db5_*best.pt /content/drive/MyDrive/Ninapro_DB5/checkpoints/
!cp checkpoints/*_db5_*latest.pt /content/drive/MyDrive/Ninapro_DB5/checkpoints/
```

---

## 4. Handling Disconnections and Resuming

Each model automatically saves a resume checkpoint (`checkpoints/<model>_db5_seed<N>_latest.pt`) after **every epoch**, including optimizer state, best validation F1 score, epochs without improvement, and history.

If the Colab runtime disconnects:
1. Re-mount Google Drive.
2. Copy `checkpoints/` and `reports/` from Google Drive back into your working directory.
3. Rerun the training command with `--resume`.
4. The script detects `checkpoints/<model>_db5_seed<N>_latest.pt` and automatically resumes from the exact epoch where it was interrupted.
5. Note: A safety guard prevents resuming real runs from checkpoints generated in `--smoke-test` mode.
