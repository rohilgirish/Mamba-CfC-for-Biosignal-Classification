import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset_db5 import NinaproDB5Dataset
from src.models.architecture_db5 import get_model_db5

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    return float(macro_f1), float(bal_acc)

def main():
    parser = argparse.ArgumentParser(description="Evaluate DB5 Models on Test Set")
    parser.add_argument('--models', nargs='+', default=['mamba-only', 'cfc-only', 'mamba-cfc', 'mamba-3l', 'mamba-width'],
                        help="Models to evaluate")
    parser.add_argument('--data-dir', type=str, default='data/processed_db5',
                        help="Directory containing test.h5")
    parser.add_argument('--ckpt-dir', type=str, default=None,
                        help="Directory containing model checkpoints (defaults to checkpoints or smoke/checkpoints)")
    parser.add_argument('--out-dir', type=str, default=None,
                        help="Directory to save final_results_db5.json (defaults to reports or smoke/reports)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Evaluation seed to lookup (default 42)")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help="Optional list of seeds to evaluate (e.g. 42 43 44)")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use ('cuda' or 'cpu')")
    parser.add_argument('--smoke-test', action='store_true',
                        help="Evaluate in smoke test mode (evaluates on 1,000 samples in separate smoke directory)")
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using evaluation device: {device}", flush=True)
    
    base_dir = 'smoke' if args.smoke_test else '.'
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else os.path.join(base_dir, 'checkpoints')
    reports_dir = args.out_dir if args.out_dir is not None else os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    test_h5 = os.path.join(args.data_dir, 'test.h5')
    if not os.path.exists(test_h5):
        raise FileNotFoundError(f"Missing {test_h5}")
        
    print(f"Loading test dataset from {test_h5}...", flush=True)
    max_samples = 1000 if args.smoke_test else None
    test_dataset = NinaproDB5Dataset(test_h5, max_samples=max_samples)
    
    # Real test evaluation guard: must use all 65,237 windows and contain all 53 classes
    if not args.smoke_test:
        assert len(test_dataset) == 65237, (
            f"GUARD VIOLATION: Real test evaluation must use exactly 65,237 windows, got {len(test_dataset):,}"
        )
        unique_classes = np.unique(test_dataset.labels.numpy())
        assert len(unique_classes) == 53, (
            f"GUARD VIOLATION: Real test evaluation must contain all 53 classes in y_true, got {len(unique_classes)}"
        )
        print(f"[*] Verified test split integrity: 65,237 windows, 53 classes present.")
    else:
        print(f"[*] Smoke test mode active: evaluated on {len(test_dataset):,} samples.")
        
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    results_path = os.path.join(reports_dir, 'final_results_db5.json')
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
        except Exception:
            results = {}
            
    seeds_to_eval = args.seeds if args.seeds is not None else [args.seed]
    
    print("\n" + "=" * 75)
    print(f"{'Model':<14} | {'Seed':<6} | {'Params':<10} | {'Macro-F1':<10} | {'Balanced Accuracy':<18}")
    print("-" * 75)
    
    for model_name in args.models:
        for seed in seeds_to_eval:
            # Check for seed-specific checkpoint first, fallback to standard
            seed_ckpt = os.path.join(ckpt_dir, f"{model_name}_db5_seed{seed}_best.pt")
            std_ckpt = os.path.join(ckpt_dir, f"{model_name}_db5_best.pt")
            
            if os.path.exists(seed_ckpt):
                ckpt_path = seed_ckpt
            elif os.path.exists(std_ckpt):
                ckpt_path = std_ckpt
            else:
                print(f"{model_name:<14} | {seed:<6} | Checkpoint not found at {seed_ckpt}")
                continue
                
            model = get_model_db5(model_name, pool_mode='last', device=device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            params = sum(p.numel() for p in model.parameters())
            
            macro_f1, bal_acc = evaluate_model(model, test_loader, device)
            
            if model_name not in results:
                results[model_name] = {}
                
            results[model_name][f"seed{seed}"] = {
                'macro_f1': macro_f1,
                'balanced_accuracy': bal_acc,
                'parameters': params
            }
            print(f"{model_name:<14} | {seed:<6} | {params:<10,d} | {macro_f1:<10.4f} | {bal_acc:<18.4f}")
            
    print("=" * 75)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved updated results to {results_path}")

if __name__ == '__main__':
    main()
