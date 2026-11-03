import os
import json
import h5py
import yaml
import numpy as np
import scipy.io as sio
import sys
import tempfile
import shutil
import glob

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.utils_db5 import get_db5_unified_label, get_subject_splits_db5
from src.data.preprocess_db5 import compute_norm_stats_db5, parse_filename

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed_db5")
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "preprocess_db5.yaml")

# Candidate locations for raw DB5
POSSIBLE_RAW_DIRS = [
    os.path.abspath(os.path.join(REPO_ROOT, "..")),          # Capstone/DB5 when repo is Capstone/DB5/<repo>
    os.path.abspath(os.path.join(REPO_ROOT, "../DB5")),      # Adjacent DB5/ folder
    os.path.abspath(os.path.join(REPO_ROOT, "../../DB5")),    # DB1 style relative path
]
RAW_DIR = None
for p in POSSIBLE_RAW_DIRS:
    if os.path.exists(p) and os.path.isdir(p) and os.path.exists(os.path.join(p, "s1")):
        RAW_DIR = p
        break

def test_split_generation_db5():
    """Same seed gives same split, different gives different; union exactly subjects 1..10."""
    subjects = list(range(1, 11))
    t1, v1, ts1 = get_subject_splits_db5(subjects, 6, 2, 2, seed=42)
    t2, v2, ts2 = get_subject_splits_db5(subjects, 6, 2, 2, seed=42)
    t3, v3, ts3 = get_subject_splits_db5(subjects, 6, 2, 2, seed=999)
    assert t1 == t2, "Same seed must give same split"
    assert t1 != t3, "Different seed must give different split"

    assert len(t1) == 6
    assert len(v1) == 2
    assert len(ts1) == 2
    assert len(set(t1) & set(v1)) == 0, "Train and Val must be disjoint"
    assert len(set(t1) & set(ts1)) == 0, "Train and Test must be disjoint"
    assert len(set(v1) & set(ts1)) == 0, "Val and Test must be disjoint"
    assert set(t1 + v1 + ts1) == set(subjects), "Union must be exactly subjects 1..10"

def test_exercise_label_mapping_and_boundaries():
    """
    Asserts max raw label in E1 files is 12, E2 is 17, E3 is 23.
    Asserts mapped classes form contiguous 0..52 space with no gaps or overlap.
    """
    assert RAW_DIR is not None, "Raw DB5 directory not found"
    mat_files = sorted(glob.glob(os.path.join(RAW_DIR, "s*", "*.mat")))
    
    e1_max_labels = set()
    e2_max_labels = set()
    e3_max_labels = set()
    
    for f in mat_files:
        sub_id, ex_id = parse_filename(f)
        mat = sio.loadmat(f)
        restim = mat['restimulus'].squeeze()
        max_lbl = int(restim.max())
        if ex_id == 1:
            e1_max_labels.add(max_lbl)
        elif ex_id == 2:
            e2_max_labels.add(max_lbl)
        elif ex_id == 3:
            e3_max_labels.add(max_lbl)
            
    assert e1_max_labels == {12}, f"E1 files must have max label 12, got {e1_max_labels}"
    assert e2_max_labels == {17}, f"E2 files must have max label 17, got {e2_max_labels}"
    assert e3_max_labels == {23}, f"E3 files must have max label 23, got {e3_max_labels}"
    
    # Test mapped label space
    rest_mapped = get_db5_unified_label(1, 0)
    assert rest_mapped == 0
    
    e1_mapped = [get_db5_unified_label(1, l) for l in range(1, 13)]
    e2_mapped = [get_db5_unified_label(2, l) for l in range(1, 18)]
    e3_mapped = [get_db5_unified_label(3, l) for l in range(1, 24)]
    
    assert e1_mapped == list(range(1, 13)), "E1 must map to 1..12"
    assert e2_mapped == list(range(13, 30)), "E2 must map to 13..29"
    assert e3_mapped == list(range(30, 53)), "E3 must map to 30..52"
    
    all_mapped = [0] + e1_mapped + e2_mapped + e3_mapped
    assert all_mapped == list(range(53)), "Mapped labels must be exactly contiguous 0..52"
    assert len(set(all_mapped)) == 53, "Must have exactly 53 unique classes"

def test_shapes_labels_disjoint_db5():
    """No window overlaps boundaries. Correct counts and starts."""
    assert os.path.exists(os.path.join(PROCESSED_DIR, "train.h5"))
    assert os.path.exists(os.path.join(PROCESSED_DIR, "val.h5"))
    assert os.path.exists(os.path.join(PROCESSED_DIR, "test.h5"))
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    wl = config["window_length"] # 60
    stride = config["stride"]    # 20

    train_subs = set()
    val_subs = set()
    test_subs = set()

    for split, sub_set in [("train", train_subs), ("val", val_subs), ("test", test_subs)]:
        with h5py.File(os.path.join(PROCESSED_DIR, f"{split}.h5"), "r") as hf:
            Y = hf["Y"][:]
            X = hf["X"][:]
            
            assert X.shape[1] == wl, f"Window length mismatch in {split}: {X.shape[1]} vs {wl}"
            assert X.shape[2] == 16, f"Channel mismatch in {split}: {X.shape[2]} vs 16"
            assert not np.isnan(X).any(), f"NaNs found in {split}"
            assert not np.isinf(X).any(), f"Infs found in {split}"
            
            labels = Y[:, 3]
            assert labels.min() >= 0 and labels.max() <= 52, f"Labels out of bounds in {split}"
            
            unique_subs = set(Y[:, 0])
            sub_set.update(unique_subs)

            # Check starts against true recording length
            recs = list(set((int(row[0]), int(row[1])) for row in Y))
            for sub, ex in recs:
                rec_rows = Y[(Y[:,0] == sub) & (Y[:,1] == ex)]
                starts = rec_rows[:, 5]
                expected_starts = np.arange(0, len(starts) * stride, stride)
                assert np.all(starts == expected_starts), "Starts must be 0, stride, 2*stride..."
                
                # Fetch real length from raw .mat
                f_path = os.path.join(RAW_DIR, f"s{sub}", f"S{sub}_E{ex}_A1.mat")
                mat = sio.loadmat(f_path)
                L = mat['emg'].shape[0]
                
                expected_window_count = (L - wl) // stride + 1
                assert len(starts) == expected_window_count, f"Window count mismatch in S{sub}_E{ex}"
                assert starts[-1] + wl <= L, "Last window exceeds recording bounds"
            
    assert len(train_subs & val_subs) == 0, "Leakage: Train and Val have overlapping subjects at window level"
    assert len(train_subs & test_subs) == 0, "Leakage: Train and Test have overlapping subjects at window level"
    assert len(val_subs & test_subs) == 0, "Leakage: Val and Test have overlapping subjects at window level"
    
    # Explicit test split guard: exactly 65,237 windows and all 53 classes present
    with h5py.File(os.path.join(PROCESSED_DIR, "test.h5"), "r") as hf:
        test_Y = hf["Y"][:]
        assert len(test_Y) == 65237, f"Expected 65,237 test windows, got {len(test_Y)}"
        assert len(np.unique(test_Y[:, 3])) == 53, f"Expected 53 unique classes in test split, got {len(np.unique(test_Y[:, 3]))}"

def test_data_integrity_db5():
    """For 1000 random windows, assert X equals scaled raw data and labels are correct."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    wl = config["window_length"]
    
    with open(os.path.join(PROCESSED_DIR, "norm_stats_db5.json"), "r") as f:
        stats = json.load(f)
    train_mean = np.array(stats["mean"], dtype=np.float32)
    train_std = np.array(stats["std"], dtype=np.float32)

    mat_cache = {}

    for split in ["train", "val", "test"]:
        with h5py.File(os.path.join(PROCESSED_DIR, f"{split}.h5"), "r") as hf:
            Y = hf["Y"][:]
            X = hf["X"][:]
            
            np.random.seed(42)
            n_samples = min(334, len(Y))
            indices = np.random.choice(len(Y), n_samples, replace=False)
            
            for idx in indices:
                sub = int(Y[idx][0])
                ex = int(Y[idx][1])
                rep = int(Y[idx][2])
                label = int(Y[idx][3])
                purity = float(Y[idx][4])
                start = int(Y[idx][5])
                
                x_val = X[idx]
                
                cache_key = (sub, ex)
                if cache_key not in mat_cache:
                    f_path = os.path.join(RAW_DIR, f"s{sub}", f"S{sub}_E{ex}_A1.mat")
                    mat = sio.loadmat(f_path)
                    mat_cache[cache_key] = (mat['emg'].astype(np.float32), mat['restimulus'].squeeze())
                    
                raw_emg, restim = mat_cache[cache_key]
                
                raw_window = raw_emg[start:start+wl]
                scaled_window = (raw_window - train_mean) / train_std
                assert np.allclose(x_val, scaled_window, atol=1e-4), "Data scaling integrity failed"
                
                # Verify label at start + wl - 1 (last-sample rule)
                raw_last_label = int(restim[start + wl - 1])
                expected_label = get_db5_unified_label(ex, raw_last_label)
                assert expected_label == label, f"Label mapping integrity failed at idx {idx}"

def test_subject_ids_match_splits_db5():
    """Test subject IDs in h5 match splits/subject_splits_db5.json exactly."""
    with open(os.path.join(SPLITS_DIR, "subject_splits_db5.json"), "r") as f:
        splits = json.load(f)
    
    total_recs = 0
    for split_name in ["train", "val", "test"]:
        expected_subs = set(splits[split_name])
        with h5py.File(os.path.join(PROCESSED_DIR, f"{split_name}.h5"), "r") as hf:
            Y = hf["Y"][:]
            actual_subs = set(int(s) for s in np.unique(Y[:, 0]))
            assert expected_subs == actual_subs, f"Subject mismatch in {split_name}: expected {expected_subs}, got {actual_subs}"
            recs = set((int(row[0]), int(row[1])) for row in Y)
            total_recs += len(recs)
            
    assert total_recs == 30, f"Exactly 30 (sub, ex) recordings must be present, got {total_recs}"

def test_compute_norm_stats_integration_db5():
    """Recompute compute_norm_stats_db5 and assert matches saved json."""
    with open(os.path.join(SPLITS_DIR, "subject_splits_db5.json"), "r") as f:
        splits = json.load(f)
    train_subs = splits["train"]
    
    mat_files = sorted(glob.glob(os.path.join(RAW_DIR, "s*", "*.mat")))
    stats = compute_norm_stats_db5(train_subs, mat_files)
    
    with open(os.path.join(PROCESSED_DIR, "norm_stats_db5.json"), "r") as f:
        saved_stats = json.load(f)
        
    assert np.allclose(stats["mean"], saved_stats["mean"], atol=1e-5), "Integration: mean changed"
    assert np.allclose(stats["std"], saved_stats["std"], atol=1e-5), "Integration: std changed"

def test_train_normalization_sanity_db5():
    """Check normalized train data has mean ~0 and std ~1 across channels."""
    train_path = os.path.join(PROCESSED_DIR, "train.h5")
    with h5py.File(train_path, "r") as hf:
        # Sample random windows to avoid aliasing the periodic 10-second repetition cycle
        rng = np.random.RandomState(42)
        n_windows = len(hf["X"])
        idx = np.sort(rng.choice(n_windows, 2000, replace=False))
        X_sample = hf["X"][idx]
        X_flat = X_sample.reshape(-1, 16)
        sample_mean = np.mean(X_flat, axis=0)
        sample_std = np.std(X_flat, axis=0)
        
        assert np.allclose(sample_mean, 0, atol=0.05), f"Train mean per channel not close to 0: {sample_mean}"
        assert np.allclose(sample_std, 1, atol=0.1), f"Train std per channel not close to 1: {sample_std}"

def test_no_normalization_leakage_db5():
    """Fits stats on train, perturbs val, refits, stats identical. Also perturb train changes stats."""
    temp_dir = tempfile.mkdtemp()
    try:
        train_subs = [1, 2]
        val_subs = [3]
        
        train_files = []
        for s in train_subs:
            f_path = os.path.join(temp_dir, f"S{s}_E1_A1.mat")
            sio.savemat(f_path, {'emg': np.random.randn(100, 16).astype(np.float32)})
            train_files.append(f_path)
            
        val_file = os.path.join(temp_dir, "S3_E1_A1.mat")
        sio.savemat(val_file, {'emg': np.random.randn(100, 16).astype(np.float32)})
        
        all_files = train_files + [val_file]
        
        loaded_files = []
        original_loadmat = sio.loadmat
        def mock_loadmat(f_path):
            loaded_files.append(f_path)
            return original_loadmat(f_path)
            
        import unittest.mock
        with unittest.mock.patch('scipy.io.loadmat', side_effect=mock_loadmat):
            stats_before = compute_norm_stats_db5(train_subs, all_files)
            
        assert val_file not in loaded_files, "Leakage: Validation file was loaded by compute_norm_stats_db5!"
        for tf in train_files:
            assert tf in loaded_files, "Train file was not loaded by compute_norm_stats_db5!"
        
        # Perturb val heavily
        sio.savemat(val_file, {'emg': np.random.randn(100, 16).astype(np.float32) * 1000 + 500})
        stats_after_val_perturb = compute_norm_stats_db5(train_subs, all_files)
        
        assert np.allclose(stats_before["mean"], stats_after_val_perturb["mean"]), "Leakage: mean changed!"
        assert np.allclose(stats_before["std"], stats_after_val_perturb["std"]), "Leakage: std changed!"
        
        # Positive control: Perturb train file
        sio.savemat(train_files[0], {'emg': np.random.randn(100, 16).astype(np.float32) * 1000 + 500})
        stats_after_train_perturb = compute_norm_stats_db5(train_subs, all_files)
        
        assert not np.allclose(stats_before["mean"], stats_after_train_perturb["mean"]), "Positive control failed: mean identical after train perturb"
        
    finally:
        shutil.rmtree(temp_dir)
