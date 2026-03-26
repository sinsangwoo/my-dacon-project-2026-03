"""
Multi-Stage Pseudo-Labeling v2
===============================
현재 모델(best_fold*.pth 앙상블)로 test 데이터를 예측하고,
unstable_prob < 0.01 OR > 0.99 인 고신뢰 샘플만 pseudo_v2.csv 로 추출.

사용법:
    python src/make_pseudo_v2.py --data_dir data
    python src/make_pseudo_v2.py --threshold 0.01   # 기본값
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import StructuralDataset, get_val_transform
from src.model import TripleStreamEfficientNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--output",     default="data/pseudo_v2.csv")
    p.add_argument("--img_size",   type=int,   default=224)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--n_folds",    type=int,   default=5)
    p.add_argument("--threshold",  type=float, default=0.01,
                   help="unstable_prob < threshold  또는 > 1-threshold 인 샘플만 추출")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    val_tf       = get_val_transform(args.img_size)
    test_dataset = StructuralDataset(
        data_dir=args.data_dir, split="test",
        transform=val_tf, img_size=args.img_size,
    )
    test_loader = DataLoader(test_dataset, args.batch_size,
                             shuffle=False, num_workers=0)

    # ── 5-Fold 앙상블 예측 ────────────────────────────────────────
    ensemble_probs = None
    final_ids      = None
    loaded          = 0

    for fold_idx in range(1, args.n_folds + 1):
        ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  {ckpt_path} 없음, 건너뜀")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        sa    = ckpt.get("args", {})
        model = TripleStreamEfficientNet(
            num_classes=2, pretrained=False,
            dropout=sa.get("dropout", 0.4),
            stoch_depth_p=0.0,

        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(memory_format=torch.channels_last)
        model.eval()
        print(f"  ✅ Fold {fold_idx} loaded (epoch={ckpt['epoch']})")

        step_ids, step_probs = [], []
        with torch.no_grad():
            for front, top, diff, ids in test_loader:
                out   = model(
                    front.to(device, memory_format=torch.channels_last), 
                    top.to(device, memory_format=torch.channels_last), 
                    diff.to(device, memory_format=torch.channels_last)
                )
                probs = torch.softmax(out, dim=1).cpu().numpy()
                step_probs.append(probs)
                if loaded == 0:
                    step_ids.extend(ids)

        step_probs = np.concatenate(step_probs, axis=0)
        if ensemble_probs is None:
            ensemble_probs = step_probs
            final_ids      = step_ids
        else:
            ensemble_probs += step_probs
        loaded += 1

    if loaded == 0:
        raise RuntimeError("Checkpoint 가 없습니다.")

    ensemble_probs /= loaded
    u_prob = ensemble_probs[:, 1]

    # ── 확신도 필터링 ─────────────────────────────────────────────
    thr   = args.threshold
    mask  = (u_prob < thr) | (u_prob > 1.0 - thr)
    n_all = len(u_prob)
    n_sel = mask.sum()

    print(f"\n🔍 Confidence filter (threshold={thr})")
    print(f"   Total test samples : {n_all}")
    print(f"   High-confidence    : {n_sel} ({100*n_sel/n_all:.1f}%)")
    print(f"     → stable  (< {thr})   : {(u_prob < thr).sum()}")
    print(f"     → unstable(> {1-thr:.2f}) : {(u_prob > 1-thr).sum()}")

    pseudo_df = pd.DataFrame({
        "id":            [final_ids[i] for i in range(n_all) if mask[i]],
        "unstable_prob": u_prob[mask],
        "stable_prob":   ensemble_probs[:, 0][mask],
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pseudo_df.to_csv(args.output, index=False)
    print(f"💾 pseudo_v2.csv saved: {args.output}  ({len(pseudo_df)} rows)")
    print("✨ Done! 다음 단계: python src/train.py --use_pseudo_v2")


if __name__ == "__main__":
    main()
