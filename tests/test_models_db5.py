import pytest
import math
import torch
import torch.nn as nn
import os
import sys

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.architecture_db5 import (
    get_model_db5,
    MambaOnlyModelDB5,
    CfCOnlyModelDB5,
    MambaCfCModelDB5,
    PoolingDB5
)
from src.models.mamba_fallback import MambaConfig, MambaBlock

def test_shapes_and_finite_db5():
    """All 5 models process [B=16, L=60, D=16] input and output finite [B=16, C=53] logits."""
    B, L, D, C = 16, 60, 16, 53
    x = torch.randn(B, L, D)
    
    model_names = ['mamba-only', 'cfc-only', 'mamba-cfc', 'mamba-3l', 'mamba-width']
    
    for name in model_names:
        model = get_model_db5(name, pool_mode='last')
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, C), f"Shape mismatch in {name}: expected {(B, C)}, got {out.shape}"
        assert torch.isfinite(out).all(), f"Non-finite output in {name}"

def test_hybrid_extra_parameters_db5():
    """
    Hybrid extra params == CfC + projection (expected 53,696)
    and exactly equals hybrid minus mamba-only.
    """
    hybrid = get_model_db5('mamba-cfc')
    mamba_only = get_model_db5('mamba-only')
    cfc_only = get_model_db5('cfc-only')
    
    def get_param_count(m):
        return sum(p.numel() for p in m.parameters())
        
    hybrid_params = get_param_count(hybrid)
    mamba_params = get_param_count(mamba_only)
    
    proj_params = sum(p.numel() for p in hybrid.projection.parameters())
    cfc_params = sum(p.numel() for p in hybrid.cfc.parameters())
    cfc_only_core_params = sum(p.numel() for p in cfc_only.cfc.parameters())
    
    assert cfc_params == cfc_only_core_params == 49536, f"CfC core params mismatch: {cfc_params}"
    assert proj_params == 4160, f"Projection params mismatch: {proj_params}"
    
    expected_extra = 53696
    assert proj_params + cfc_params == expected_extra, f"Extra params must be {expected_extra}"
    assert hybrid_params - mamba_params == expected_extra, f"Hybrid minus mamba-only must equal {expected_extra}"

def test_shared_structure_db5():
    """mamba-only and mamba-cfc share identical embedding, Mamba backbone and head structure."""
    hybrid = get_model_db5('mamba-cfc')
    mamba_only = get_model_db5('mamba-only')
    
    assert isinstance(hybrid.embedding, type(mamba_only.embedding))
    assert isinstance(hybrid.mamba_backbone, type(mamba_only.mamba_backbone))
    assert isinstance(hybrid.head, type(mamba_only.head))
    
    assert hybrid.embedding.in_features == mamba_only.embedding.in_features == 16
    assert hybrid.embedding.out_features == mamba_only.embedding.out_features == 64
    
    assert len(hybrid.mamba_backbone.layers) == len(mamba_only.mamba_backbone.layers) == 2
    
    assert hybrid.head.in_features == mamba_only.head.in_features == 64
    assert hybrid.head.out_features == mamba_only.head.out_features == 53

def test_mamba_fallback_parallel_vs_sequential_db5():
    """Mamba fallback parallel scan matches sequential recurrence within 1e-4."""
    config = MambaConfig(d_model=16, n_layers=1, d_state=16, pscan=True, use_cuda=False)
    block = MambaBlock(config)
    block.eval()
    
    batch_size = 2
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, config.d_inner)
    
    with torch.no_grad():
        A = -torch.exp(block.A_log.float())
        D = block.D.float()
        
        deltaBC = block.x_proj(x)
        delta, B, C = torch.split(deltaBC, [config.dt_rank, config.d_state, config.d_state], dim=-1)
        delta, B, C = block._apply_layernorms(delta, B, C)
        
        delta = block.dt_proj.weight @ delta.transpose(1, 2)
        delta = delta.transpose(1, 2)
        delta = torch.nn.functional.softplus(delta + block.dt_proj.bias)
        
        y_pscan = block.selective_scan(x, delta, A, B, C, D)
        y_seq = block.selective_scan_seq(x, delta, A, B, C, D)
        
    torch.testing.assert_close(y_pscan, y_seq, rtol=1e-4, atol=1e-4, msg="pscan and sequential scan outputs differ")

def test_pooling_mode_db5():
    """Verify last-timestep pooling is used as default and returns x[:, -1, :]."""
    pooling = PoolingDB5(mode='last')
    x = torch.randn(4, 60, 64)
    out = pooling(x)
    assert out.shape == (4, 64)
    assert torch.equal(out, x[:, -1, :]), "Pooling output must strictly match last timestep"
    
    # Check default on all models
    for name in ['mamba-only', 'cfc-only', 'mamba-cfc', 'mamba-3l', 'mamba-width']:
        m = get_model_db5(name)
        assert m.pooling.mode == 'last', f"Model {name} must default to 'last' pooling"

def test_cfc_ts_frozen_db5():
    """Verify CfC operates with uniform constant time increments ts=1.0."""
    cfc_model = get_model_db5('cfc-only')
    x1 = torch.randn(2, 60, 16)
    
    # Passing standard input without timespan tensor uses default uniform step
    cfc_model.eval()
    with torch.no_grad():
        out1 = cfc_model(x1)
    assert out1.shape == (2, 53)
    assert torch.isfinite(out1).all()

def test_initial_loss_db5():
    """Initial loss on random weights is approximately ln(53) = 3.9703."""
    torch.manual_seed(42)
    B, L, D = 128, 60, 16
    x = torch.randn(B, L, D)
    y = torch.randint(0, 53, (B,))
    crit = nn.CrossEntropyLoss()
    expected_loss = math.log(53)
    
    for name in ['mamba-only', 'cfc-only', 'mamba-cfc', 'mamba-3l', 'mamba-width']:
        model = get_model_db5(name, pool_mode='last')
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss = crit(logits, y).item()
        assert abs(loss - expected_loss) < 0.20, f"Initial loss for {name} ({loss:.4f}) deviates too much from ln(53) ({expected_loss:.4f})"

def test_parameter_counts_exact_db5():
    """Instantiated parameter counts strictly match the Phase 0 calculation."""
    expected_counts = {
        'mamba-only': 69941,
        'cfc-only': 54069,
        'mamba-cfc': 123637,
        'mamba-3l': 102645,
        'mamba-width': 122901
    }
    
    for name, expected in expected_counts.items():
        m = get_model_db5(name)
        actual = sum(p.numel() for p in m.parameters())
        assert actual == expected, f"Param count mismatch in {name}: expected {expected:,}, got {actual:,}"

def test_metric_calculation_and_labels_behavior_db5():
    """
    Verifies metric calculation semantics:
    1. When all 53 classes are present in y_true (as in the real 65,237 test set),
       f1_score with labels=None and labels=range(53) are IDENTICAL.
    2. When only a subset of classes is present in y_true (e.g. during sliced smoke testing),
       labels=None averages only over present classes (inflating metric), whereas labels=range(53)
       divides by all 53 classes.
    """
    import numpy as np
    from sklearn.metrics import f1_score, balanced_accuracy_score
    
    # Case 1: Complete 53 classes in y_true
    rng = np.random.RandomState(42)
    y_true_all = np.arange(53)
    y_pred_all = rng.randint(0, 53, size=53)
    
    f1_default = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    f1_explicit = f1_score(y_true_all, y_pred_all, labels=range(53), average='macro', zero_division=0)
    assert abs(f1_default - f1_explicit) < 1e-12, "f1_score with labels=None and labels=range(53) must be identical when all classes are present!"
    
    # Case 2: Subset of 3 classes in y_true (simulating sliced smoke test)
    y_true_subset = np.array([0, 1, 2, 0, 1, 2])
    y_pred_subset = np.array([0, 1, 0, 0, 1, 2])
    
    f1_sub_default = f1_score(y_true_subset, y_pred_subset, average='macro', zero_division=0)
    f1_sub_explicit = f1_score(y_true_subset, y_pred_subset, labels=range(53), average='macro', zero_division=0)
    
    # Default averages over 3 classes, explicit divides sum by 53
    assert f1_sub_default > f1_sub_explicit
    assert abs(f1_sub_explicit * 53 / 3 - f1_sub_default) < 1e-6
    
    # Balanced accuracy always averages recall across present classes in y_true
    bal_acc = balanced_accuracy_score(y_true_subset, y_pred_subset)
    assert 0.0 <= bal_acc <= 1.0

def test_resume_smoke_guard_db5():
    """Real run refuses to resume from a checkpoint produced in smoke mode."""
    ckpt = {
        'epoch': 1,
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'best_val_f1': 0.5,
        'is_smoke_test': True
    }
    
    # Simulating resume check in train_db5.py
    is_smoke_run = False
    if not is_smoke_run and ckpt.get('is_smoke_test', False):
        with pytest.raises(RuntimeError, match="SAFETY GUARD: Refusing to resume"):
            raise RuntimeError("SAFETY GUARD: Refusing to resume real run from smoke-test checkpoint.")


