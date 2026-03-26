import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Ensure src is importable
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from src.dataset import KFoldStructuralDataset, get_val_transform, build_full_df
from src.model import TripleStreamEfficientNet

def log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if y_true.ndim == 1:
        yt = np.zeros_like(y_pred)
        yt[np.arange(len(y_true)), y_true.astype(int)] = 1
        y_true = yt
    loss = -(y_true * np.log(y_pred)).sum(axis=1)
    return np.mean(loss)

def run_analysis():
    data_dir = "data"
    ckpt_path = "checkpoints/best_fold1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Analyzing with device: {device}")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    # Create dummy samples if needed for validation of script, 
    # but here we rely on data/ existing.
    
    # Load model
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TripleStreamEfficientNet(num_classes=2, pretrained=False).to(device)
    # Check if state_dict keys match (sometimes they have 'module.' prefix)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Get Fold 1 Val Data
    print("Building dataset...")
    full_df = build_full_df(data_dir)
    labels_arr = np.array([0 if l == "stable" else 1 for l in full_df["label"]])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    val_idx = None
    for i, (tr, val) in enumerate(skf.split(np.zeros(len(full_df)), labels_arr), 1):
        if i == 1:
            val_idx = val.tolist()
            break
            
    val_tf = get_val_transform(224)
    val_ds = KFoldStructuralDataset(data_dir, val_idx, full_df, is_train=False, transform=val_tf)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # Collect Logits
    all_logits = []
    all_labels = []
    print(f"Collecting logits for {len(val_ds)} samples...")
    with torch.no_grad():
        for f, t, d, y in loader:
            out = model(f.to(device), t.to(device), d.to(device))
            all_logits.append(out.cpu().numpy())
            # y might be soft [s, u] or hard label index
            all_labels.append(y.cpu().numpy())
            
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    # Grid Search
    alphas = [1.0, 1.05, 1.1, 1.15, 1.2]
    Ts = [0.5, 0.8, 1.0, 1.2, 1.5]
    
    print(f"\n{'alpha':>6} | {'T':>5} | {'LogLoss':>10}")
    print("-" * 30)
    
    results = []
    for alpha in alphas:
        for T in Ts:
            # 1. T-Scaling
            scaled_logits = all_logits / T
            # 2. Softmax
            exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            # 3. Power Scaling
            probs = np.power(probs, alpha)
            # 4. Re-norm
            probs = probs / probs.sum(axis=1, keepdims=True)
            # 5. Score
            score = log_loss(all_labels, probs)
            results.append((alpha, T, score))
            print(f"{alpha:6.2f} | {T:5.1f} | {score:10.6f}")
            
    df = pd.DataFrame(results, columns=["alpha", "T", "logloss"])
    best = df.loc[df["logloss"].idxmin()]
    
    print("\n" + "="*40)
    print(f"GOLDEN_COMBINATION: alpha={best['alpha']:.2f}, T={best['T']:.1f}")
    print(f"Minimal Val LogLoss: {best['logloss']:.6f}")
    print("="*40)
    
    # Simulation Projection
    # Fold 1 Val Loss is usually a good proxy, but LB has slightly different distribution.
    # Typically, a shift of 0.02-0.05 is possible.
    low = max(0.001, best['logloss'] - 0.01)
    high = best['logloss'] + 0.03
    print(f"Expected LB Range: {low:.3f} ~ {high:.3f}")

if __name__ == "__main__":
    run_analysis()
