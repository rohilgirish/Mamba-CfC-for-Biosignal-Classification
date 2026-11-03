import os
import glob
import json
import yaml
import h5py
import hashlib
import time
import numpy as np
import scipy.io as sio
import sys
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view

# Adjust Python path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.utils_db5 import get_db5_unified_label, get_subject_splits_db5

# Candidate locations for raw DB5 (contains s1..s10 folders)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
POSSIBLE_RAW_DIRS = [
    os.path.abspath(os.path.join(REPO_ROOT, "..")),          # Capstone/DB5 when repo is Capstone/DB5/<repo>
    os.path.abspath(os.path.join(REPO_ROOT, "../DB5")),      # Adjacent DB5/ folder
    os.path.abspath(os.path.join(REPO_ROOT, "../../DB5")),    # DB1 style relative path
    os.path.abspath(os.path.join(os.getcwd(), "..")),
]

RAW_DIR = None
for p in POSSIBLE_RAW_DIRS:
    if os.path.exists(p) and os.path.isdir(p) and os.path.exists(os.path.join(p, "s1")):
        RAW_DIR = p
        break

if RAW_DIR is None:
    raise FileNotFoundError(f"Could not locate DB5 raw directory in: {POSSIBLE_RAW_DIRS}")

PROCESSED_DIR = os.path.join("data", "processed_db5")
SPLITS_DIR = "splits"
REPORTS_DIR = "reports"
CONFIG_PATH = os.path.join("configs", "preprocess_db5.yaml")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def parse_filename(f_path):
    """
    Extracts subject and exercise from filename e.g. S1_E2_A1.mat
    Returns (subject_id, exercise_id)
    """
    basename = os.path.basename(f_path).replace('.mat', '')
    parts = basename.split('_')
    sub_id = int(parts[0][1:])
    ex_id = int(parts[1][1:])
    return sub_id, ex_id

def compute_norm_stats_db5(train_subs, mat_files):
    """
    Computes mean and std strictly on raw continuous EMG of training subjects.
    Zero leakage check: raises error if any non-training file is touched.
    """
    train_emg_list = []
    loaded_subjects = set()
    
    for f_path in mat_files:
        sub_id, ex_id = parse_filename(f_path)
        if sub_id in train_subs:
            loaded_subjects.add(sub_id)
            data = sio.loadmat(f_path)
            emg = data['emg'] # continuous samples (N, 16)
            train_emg_list.append(emg)
            
    assert loaded_subjects == set(train_subs), f"Mismatch in loaded train subjects: {loaded_subjects} vs {train_subs}"
    
    all_train_emg = np.vstack(train_emg_list)
    mean_vals = np.mean(all_train_emg, axis=0)
    std_vals = np.std(all_train_emg, axis=0)
    std_vals[std_vals == 0] = 1.0  # avoid division by zero
    
    return {
        "mean": mean_vals.tolist(),
        "std": std_vals.tolist()
    }

def main():
    print(f"=== DB5 Preprocessing Pipeline ===")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")
    
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config_str = f.read()
        CONFIG = yaml.safe_load(config_str)
        
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    # Discover files
    mat_files = sorted(glob.glob(os.path.join(RAW_DIR, "s*", "*.mat")))
    if not mat_files:
        mat_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.mat")))
        
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {RAW_DIR}")
        
    print(f"Discovered {len(mat_files)} .mat files.")
    
    # Subject discovery
    subjects = sorted(list(set([parse_filename(f)[0] for f in mat_files])))
    print(f"Discovered {len(subjects)} subjects: {subjects}")
    
    # Split subjects: 6 Train / 2 Val / 2 Test (Seed 42)
    train_subs, val_subs, test_subs = get_subject_splits_db5(
        subjects,
        CONFIG["train_size"],
        CONFIG["val_size"],
        CONFIG["test_size"],
        CONFIG["seed"]
    )
    
    # Assert disjointness
    assert set(train_subs).isdisjoint(val_subs), "Train and Val overlap!"
    assert set(train_subs).isdisjoint(test_subs), "Train and Test overlap!"
    assert set(val_subs).isdisjoint(test_subs), "Val and Test overlap!"
    assert sorted(train_subs + val_subs + test_subs) == subjects, "Split does not partition all subjects!"
    
    splits_dict = {
        "train": train_subs,
        "val": val_subs,
        "test": test_subs
    }
    splits_path = os.path.join(SPLITS_DIR, "subject_splits_db5.json")
    with open(splits_path, "w") as f:
        json.dump(splits_dict, f, indent=4)
    print(f"Saved splits to {splits_path}")
    print(f"  Train ({len(train_subs)}): {train_subs}")
    print(f"  Val   ({len(val_subs)}): {val_subs}")
    print(f"  Test  ({len(test_subs)}): {test_subs}")
    
    # Compute normalization statistics strictly on training subjects
    print("\nComputing normalization statistics (channel-wise Z-score) on Train subjects only...")
    stats = compute_norm_stats_db5(train_subs, mat_files)
    mean_vals = np.array(stats["mean"], dtype=np.float32)
    std_vals = np.array(stats["std"], dtype=np.float32)
    
    stats_path = os.path.join(PROCESSED_DIR, "norm_stats_db5.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved normalization stats to {stats_path}")
    print(f"Train Mean per channel: {np.round(mean_vals, 3)}")
    print(f"Train Std per channel:  {np.round(std_vals, 3)}")
    
    # Process each split
    wl = CONFIG["window_length"] # 60 samples
    st = CONFIG["stride"]        # 20 samples
    print(f"\nWindowing parameters: length={wl} samples (300 ms), stride={st} samples (100 ms) at 200 Hz")
    
    split_summaries = {}
    
    def process_split(split_name, subs):
        print(f"\nProcessing split: {split_name} (Subjects: {subs})...", flush=True)
        X_list = []
        Y_list = []
        
        split_files = [f for f in mat_files if parse_filename(f)[0] in subs]
        print(f"  Found {len(split_files)} files for split {split_name}.")
        
        for f_path in split_files:
            sub_id, ex_id = parse_filename(f_path)
            data = sio.loadmat(f_path)
            
            emg = data['emg'].astype(np.float32)
            restimulus = data['restimulus'].squeeze()
            rerepetition = data['rerepetition'].squeeze()
            
            # Apply FROZEN training normalization stats
            emg = (emg - mean_vals) / std_vals
            
            if emg.shape[0] < wl:
                continue
                
            # Vectorized sliding window
            emg_windows = sliding_window_view(emg, window_shape=wl, axis=0)[::st]
            emg_windows = np.swapaxes(emg_windows, 1, 2) # (num_windows, wl, channels)
            
            restimulus_windows = sliding_window_view(restimulus, window_shape=wl, axis=0)[::st]
            rerepetition_windows = sliding_window_view(rerepetition, window_shape=wl, axis=0)[::st]
            
            num_windows = emg_windows.shape[0]
            
            # Last sample label rule
            final_labels_raw = restimulus_windows[:, -1]
            final_reps = rerepetition_windows[:, -1]
            
            # Label purity across window
            matches = (restimulus_windows == final_labels_raw[:, None])
            purities = np.sum(matches, axis=1) / float(wl)
            
            # Unified label mapping: Rest 0->0, E1 1..12, E2 13..29, E3 30..52
            mapped_labels = np.array([get_db5_unified_label(ex_id, int(l)) for l in final_labels_raw], dtype=np.float32)
            
            start_indices = np.arange(0, num_windows * st, st, dtype=np.float32)
            subj_col = np.full(num_windows, sub_id, dtype=np.float32)
            exer_col = np.full(num_windows, ex_id, dtype=np.float32)
            
            # Meta array: [subject, exercise, repetition, label, label_purity, start_idx]
            meta_array = np.column_stack([
                subj_col, exer_col, final_reps.astype(np.float32), mapped_labels, purities.astype(np.float32), start_indices
            ])
            
            X_list.append(emg_windows)
            Y_list.append(meta_array)
            
        if not X_list:
            print(f"No windows generated for split {split_name}!")
            return 0
            
        X_all = np.vstack(X_list).astype(np.float32)
        Y_all = np.vstack(Y_list).astype(np.float32)
        
        h5_path = os.path.join(PROCESSED_DIR, f"{split_name}.h5")
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("X", data=X_all, compression="gzip")
            hf.create_dataset("Y", data=Y_all, compression="gzip")
            
        n_windows = X_all.shape[0]
        labels = Y_all[:, 3].astype(int)
        rest_count = int(np.sum(labels == 0))
        rest_pct = (rest_count / n_windows) * 100.0
        unique_classes = len(np.unique(labels))
        
        # Check normalized data mean and std on flattened samples
        sample_mean = float(np.mean(X_all))
        sample_std = float(np.std(X_all))
        
        split_summaries[split_name] = {
            "num_windows": n_windows,
            "rest_count": rest_count,
            "rest_pct": rest_pct,
            "unique_classes": unique_classes,
            "overall_mean": sample_mean,
            "overall_std": sample_std,
            "subjects": subs
        }
        
        print(f"Saved {h5_path}:")
        print(f"  Shape X: {X_all.shape}, Shape Y: {Y_all.shape}")
        print(f"  Total Windows: {n_windows:,}")
        print(f"  Rest Class (0): {rest_count:,} ({rest_pct:.2f}%)")
        print(f"  Active Classes: {n_windows - rest_count:,} ({100.0 - rest_pct:.2f}%)")
        print(f"  Unique Classes present: {unique_classes}/53")
        return n_windows

    n_train = process_split("train", train_subs)
    n_val = process_split("val", val_subs)
    n_test = process_split("test", test_subs)
    
    # Save run info
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": config_hash,
        "seed": CONFIG["seed"],
        "sampling_rate": 200,
        "window_length": wl,
        "stride": st,
        "package_versions": {
            "numpy": np.__version__,
            "h5py": h5py.__version__
        },
        "subjects_processed": len(subjects),
        "files_processed": len(mat_files),
        "windows_generated": {
            "train": n_train,
            "val": n_val,
            "test": n_test
        },
        "split_summaries": split_summaries
    }
    with open(os.path.join(REPORTS_DIR, "run_info_db5.json"), "w") as f:
        json.dump(run_info, f, indent=4)
        
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION REPORT")
    print("=" * 60)
    print(f"{'Split':<8} | {'Subjects':<18} | {'Windows':<10} | {'Rest %':<8} | {'Classes':<8} | {'Mean':<8} | {'Std':<8}")
    print("-" * 76)
    for sp, sinfo in split_summaries.items():
        sub_str = str(sinfo['subjects'])
        print(f"{sp:<8} | {sub_str:<18} | {sinfo['num_windows']:<10,d} | {sinfo['rest_pct']:<7.2f}% | {sinfo['unique_classes']:<8} | {sinfo['overall_mean']:<8.4f} | {sinfo['overall_std']:<8.4f}")
    print("=" * 60)
    print("DB5 Preprocessing complete!")

if __name__ == "__main__":
    main()

