import os
import argparse
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score
import sys

# Ensure repo root is in python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset_db5 import NinaproDB5Dataset, get_balanced_sampler_db5
from src.models.architecture_db5 import get_model_db5

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_epoch(model, dataloader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    n_samples = 0
    for b_idx, (batch_x, batch_y) in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break
            
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        # Gradient clipping for stability (DB1 used max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        bs = batch_x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())
        
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    avg_loss = total_loss / max(1, n_samples)
    
    return avg_loss, macro_f1, bal_acc

def eval_epoch(model, dataloader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    n_samples = 0
    with torch.no_grad():
        for b_idx, (batch_x, batch_y) in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
                
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    avg_loss = total_loss / max(1, n_samples)
    
    return avg_loss, macro_f1, bal_acc

def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
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

def run_training_for_seed(args, seed: int, device: torch.device):
    set_seed(seed)
    
    base_out_dir = os.path.join(args.out_dir, 'smoke') if args.smoke_test else args.out_dir
    ckpt_dir = os.path.join(base_out_dir, 'checkpoints')
    reports_dir = os.path.join(base_out_dir, 'reports')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print(f"\n" + "=" * 65)
    print(f"DB5 Training Execution: {args.model} | Seed: {seed}")
    print(f"Device: {device}")
    print(f"Package versions: PyTorch={torch.__version__}, NumPy={np.__version__}")
    try:
        import ncps
        print(f"ncps={ncps.__version__}")
    except Exception:
        pass
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {base_out_dir}")
    print(f"Batch size: {args.batch_size} | LR: {args.lr} | Patience: {args.patience} | Max Epochs: {args.max_epochs}")
    print("=" * 65)
    
    train_h5 = os.path.join(args.data_dir, 'train.h5')
    val_h5 = os.path.join(args.data_dir, 'val.h5')
    test_h5 = os.path.join(args.data_dir, 'test.h5')
    
    if not os.path.exists(train_h5) or not os.path.exists(val_h5):
        raise FileNotFoundError(f"Missing train.h5 or val.h5 in {args.data_dir}")
        
    max_train_batches = 20 if args.smoke_test else None
    max_val_batches = 10 if args.smoke_test else None
    max_samples = 2000 if args.smoke_test else None
    
    max_epochs = min(args.max_epochs, 2) if args.smoke_test else args.max_epochs
    if args.smoke_test:
        print(f"[*] Smoke test mode active: saving to {base_out_dir}, limit train to {max_train_batches} batches, epochs={max_epochs}")
        
    print("Loading datasets...", flush=True)
    train_dataset = NinaproDB5Dataset(train_h5, max_samples=max_samples)
    val_dataset = NinaproDB5Dataset(val_h5, max_samples=max_samples)
    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")
    
    train_sampler = get_balanced_sampler_db5(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = get_model_db5(args.model, pool_mode='last', device=device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model {args.model} instantiated with {param_count:,} parameters.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_ckpt_path = os.path.join(ckpt_dir, f"{args.model}_db5_seed{seed}_best.pt")
    latest_ckpt_path = os.path.join(ckpt_dir, f"{args.model}_db5_seed{seed}_latest.pt")
    history_path = os.path.join(reports_dir, f"{args.model}_history_db5_seed{seed}.json")
    
    # Also save standard unseeded alias if seed == 42
    std_best_ckpt_path = os.path.join(ckpt_dir, f"{args.model}_db5_best.pt") if seed == 42 else None
    
    start_epoch = 0
    best_val_f1 = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    
    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"[*] Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        
        # Safety guard: refuse to resume a real training run from a smoke-test checkpoint
        ckpt_is_smoke = checkpoint.get('is_smoke_test', False)
        if not args.smoke_test and ckpt_is_smoke:
            raise RuntimeError(
                f"SAFETY GUARD: Refusing to resume real run from smoke-test checkpoint '{latest_ckpt_path}'. "
                f"Delete smoke checkpoints or run with --smoke-test."
            )
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        history = checkpoint.get('history', history)
        print(f"[*] Resumed at epoch {start_epoch + 1} with best_val_f1={best_val_f1:.4f}")
        
    print(f"\nStarting training loop from epoch {start_epoch + 1} to {max_epochs}...")
    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, max_batches=max_train_batches
        )
        val_loss, val_f1, val_acc = eval_epoch(
            model, val_loader, criterion, device, max_batches=max_val_batches
        )
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{max_epochs:02d} [{elapsed:.1f}s]")
        print(f"  Train: Loss={train_loss:.4f} | Macro-F1={train_f1:.4f} | Bal-Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f} | Macro-F1={val_f1:.4f} | Bal-Acc={val_acc:.4f}", flush=True)
        
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            if std_best_ckpt_path:
                torch.save(model.state_dict(), std_best_ckpt_path)
            print(f"  [*] Saved new best model checkpoint (Val F1: {val_f1:.4f}) -> {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  [-] No improvement for {epochs_no_improve}/{args.patience} epochs.")
            
        # Save resume checkpoint with explicit is_smoke_test flag
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1,
            'epochs_no_improve': epochs_no_improve,
            'history': history,
            'seed': seed,
            'is_smoke_test': bool(args.smoke_test)
        }, latest_ckpt_path)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
            
        if epochs_no_improve >= args.patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}.", flush=True)
            break
            
    # Evaluation on held-out test subjects ONLY after training finishes
    print("\n" + "=" * 65)
    print("Evaluating held-out TEST split using best checkpoint...")
    print("=" * 65)
    if os.path.exists(test_h5) and os.path.exists(best_ckpt_path):
        test_dataset = NinaproDB5Dataset(test_h5, max_samples=max_samples)
        
        # Real test evaluation guard: must contain all 65,237 windows and all 53 classes
        if not args.smoke_test:
            assert len(test_dataset) == 65237, (
                f"GUARD VIOLATION: Real test evaluation must use exactly 65,237 windows, got {len(test_dataset):,}"
            )
            unique_test_classes = np.unique(test_dataset.labels.numpy())
            assert len(unique_test_classes) == 53, (
                f"GUARD VIOLATION: Real test evaluation must contain all 53 classes in y_true, got {len(unique_test_classes)}"
            )
            print(f"[*] Verified test split integrity: 65,237 windows, 53 classes present.")
            
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        best_model = get_model_db5(args.model, pool_mode='last', device=device)
        best_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        
        test_macro_f1, test_bal_acc = evaluate_test_set(best_model, test_loader, device)
        print(f"Model: {args.model} | Seed: {seed}")
        print(f"  Test Macro-F1:          {test_macro_f1:.4f}")
        print(f"  Test Balanced Accuracy: {test_bal_acc:.4f}")
        print(f"  Parameter Count:        {param_count:,}")
        
        final_results_path = os.path.join(reports_dir, "final_results_db5.json")
        final_results = {}
        if os.path.exists(final_results_path):
            try:
                with open(final_results_path, 'r') as f:
                    final_results = json.load(f)
            except Exception:
                final_results = {}
                
        if args.model not in final_results:
            final_results[args.model] = {}
            
        final_results[args.model][f"seed{seed}"] = {
            "macro_f1": test_macro_f1,
            "balanced_accuracy": test_bal_acc,
            "parameters": param_count,
            "best_val_f1": best_val_f1,
            "epochs_trained": len(history['val_f1'])
        }
        
        with open(final_results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"Saved evaluation results to {final_results_path}")
    else:
        print("[!] Test h5 or best checkpoint not found. Skipping test evaluation.")

def main():
    parser = argparse.ArgumentParser(description="Train DB5 Models")
    parser.add_argument('--model', type=str, required=True,
                        choices=['mamba-only', 'cfc-only', 'mamba-cfc', 'mamba-3l', 'mamba-width'])
    parser.add_argument('--data-dir', type=str, default='data/processed_db5',
                        help="Path to processed DB5 data containing train.h5, val.h5, test.h5")
    parser.add_argument('--out-dir', type=str, default='.',
                        help="Output directory base for checkpoints and reports")
    parser.add_argument('--max-epochs', type=int, default=50,
                        help="Maximum training epochs (default 50, matching DB1 Phase 4)")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size (default 256, matching DB1 Phase 4)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for Adam optimizer (default 1e-3)")
    parser.add_argument('--patience', type=int, default=5,
                        help="Early stopping patience on validation Macro-F1 (default 5)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help="Optional list of seeds to run sequentially (e.g. --seeds 42 43 44)")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use ('cuda' or 'cpu'). Auto-detected if not set.")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from latest checkpoint if available")
    parser.add_argument('--smoke-test', action='store_true',
                        help="Run small smoke test in separate out-dir/smoke/ without touching real runs")
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds_to_run = args.seeds if args.seeds is not None else [args.seed]
    
    for s in seeds_to_run:
        run_training_for_seed(args, s, device)

if __name__ == '__main__':
    main()
