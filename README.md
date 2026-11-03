# Controlled Attribution Study of Mamba and CfC Neural Networks for sEMG Gesture Recognition (Ninapro DB5)

This repository implements a controlled architectural attribution study comparing **Mamba (State Space Model)**, **CfC (Closed-form Continuous-time Recurrent Neural Network)**, and a **Mamba-CfC Hybrid** on the **Ninapro DB5** surface electromyography (sEMG) benchmark.

This study systematically replicates the experimental protocol established on Ninapro DB1 to determine whether the representational strengths of state-space temporal sequence modeling and closed-form continuous-time dynamics translate from on-chip envelope sEMG (DB1) to raw double-armband bipolar sEMG (DB5).

---

## 1. Ninapro DB5 Dataset Facts & Protocol Specifications

- **Dataset**: Ninapro DB5 (Double Myo Gesture Dataset).
- **Sensors & Channels**: 16 raw bipolar sEMG channels recorded simultaneously from two 8-channel Thalmic Myo armbands (8 proximal, 8 distal around the right forearm).
- **Sampling Frequency**: **200 Hz** (verified from MATLAB file headers; 5 ms inter-sample period).
- **Recordings & Subjects**: 10 intact subjects $\times$ 3 exercise sets (`E1`: 12 basic finger movements, `E2`: 17 isometric/wrist configurations, `E3`: 23 grasping and functional movements).
- **Label Alignment**:
  - Unified contiguous label mapping from 0 to 52 (53 classes total).
  - Rest label ($0 \to 0$); Exercise 1 ($1 \dots 12$); Exercise 2 ($13 \dots 29$ via $+12$ offset); Exercise 3 ($30 \dots 52$ via $+29$ offset).
  - Uses `restimulus` and `rerepetition` to preserve accurate movement onset/offset boundaries.
- **Windowing**:
  - Window length: **300 ms** ($60\text{ samples}$ at $200\text{ Hz}$).
  - Stride: **100 ms** ($20\text{ samples}$ at $200\text{ Hz}$).
  - Last-sample label rule: Each window's label is assigned from the final sample of the window.
- **Subject-Wise Disjoint Split (6 / 2 / 2)**:
  - Strict subject-wise partitioning using `np.random.RandomState(42).permutation(10)`:
    - **Train (6 subjects)**: `[9, 2, 6, 1, 8, 3]` (186,942 windows; 62.24% Rest).
    - **Val (2 subjects)**: `[10, 5]` (65,414 windows; 62.98% Rest).
    - **Test (2 subjects)**: `[4, 7]` (65,237 windows; 62.59% Rest).
  - Zero subject overlap across splits.
- **Normalization**:
  - Channel-wise Z-score normalization computed **strictly on the training subjects**, frozen, and applied out-of-sample to validation and test sets (zero data leakage).

---

## 2. Model Architectures & Parameter Budgeting

The benchmark evaluates 5 neural network architectures designed with strict controlled parameter budgeting:

| Model | Backbone Configuration | Parameters | Parameter Ratio vs Hybrid |
| :--- | :--- | :--- | :--- |
| **`cfc-only`** | Linear($16 \to 64$) + 1 CfC layer ($64$ units, $\tau$ frozen at $1.0$) + Linear head | **54,069** | $-56.3\%$ |
| **`mamba-only`** | Linear($16 \to 64$) + 2 Mamba layers ($d=64, \text{state}=16, E=2$) + Linear head | **69,941** | $-43.4\%$ |
| **`mamba-3l`** | Linear($16 \to 64$) + 3 Mamba layers ($d=64$) + Linear head (depth control) | **102,645** | $-17.0\%$ |
| **`mamba-width`** | Linear($16 \to 88$) + 2 Mamba layers ($W=88$) + Linear head (parameter-matched control) | **122,901** | **$-0.60\%$** |
| **`mamba-cfc`** | Linear($16 \to 64$) + 2 Mamba + Linear($64 \to 64$) + 1 CfC ($64$) + Linear head | **123,637** | Baseline |

- **Extra Hybrid Parameters**: The hybrid adds exactly $53,696$ parameters over `mamba-only` ($\text{Linear}_{64\to64} [4,160] + \text{CfC}_{64} [49,536]$).
- **Pooling**: Last-timestep pooling (`x[:, -1, :]`).

---

## 3. How to Run

### Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt # or: pip install h5py ncps scikit-learn pyyaml torch pytest
```

### Preprocessing & Dataset Generation
To process the raw MATLAB files from the parent directory `../s1` ... `../s10`:
```bash
python src/data/preprocess_db5.py
```
This generates `train.h5` (186,942 windows), `val.h5` (65,414 windows), `test.h5` (65,237 windows), and `norm_stats_db5.json` in `data/processed_db5/`.

### Running Tests
Execute the full pytest verification suite:
```bash
pytest tests/test_data_db5.py tests/test_models_db5.py -v
```

### Training Models
Train individual models or run the protocol sequence:
```bash
# Recommended execution order:
python scripts/train_db5.py --model cfc-only --device cuda
python scripts/train_db5.py --model mamba-only --device cuda
python scripts/train_db5.py --model mamba-cfc --device cuda
python scripts/train_db5.py --model mamba-width --device cuda
python scripts/train_db5.py --model mamba-3l --device cuda
```

Key Training Hyperparameters:
- **Optimizer**: Adam ($1 \times 10^{-3}$)
- **Batch Size**: 256
- **Sampler**: `WeightedRandomSampler` on training labels
- **Gradient Clipping**: `max_norm=1.0`
- **Early Stopping**: Patience 5 on validation Macro-F1
- **Max Epochs**: 50
- **Seed**: 42 (optional multi-seed: `--seeds 42 43 44`)

### Evaluation on Held-Out Test Subjects
Evaluate best saved checkpoints on the 65,237 held-out test windows:
```bash
python scripts/evaluate_db5.py --device cuda
```

---

## 4. Acknowledgements

The pure-PyTorch Mamba fallback implementation (`src/models/mamba_fallback.py` and `pscan.py`) and the hybrid architecture definitions derive from the Ninapro DB1 baseline study by rohanth3 ([https://github.com/rohanth3/-Mamba-CfC-for-Biosignal-Classification](https://github.com/rohanth3/-Mamba-CfC-for-Biosignal-Classification)). These files were modified for this project to support Ninapro DB5 data structures and 53 target gesture classes.
Closed-form Continuous-time (CfC) neural models are powered by the official `ncps` package.

