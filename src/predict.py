"""
Dacon 구조 안정성 분류 - 앙상블 예측 스크립트 (v5)
=====================================================
5개 Fold Best Model 을 Soft-Voting 으로 앙상블 + TTA 적용.

사용법:
    python src/predict.py --data_dir data --n_folds 5
    python src/predict.py --tta_steps 3   # 원본 + H-Flip + ColorJitter
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import StructuralDataset, get_val_transform
from src.model import TripleStreamConvNeXt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--output_csv", default="submission.csv")
    p.add_argument("--img_size",   type=int, default=268)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_folds",    type=int, default=5)
    p.add_argument("--tta_steps",  type=int, default=3,
                   help="TTA 변환 수 (1=원본만, 3=원본+HFlip+ColorJitter)")
    return p.parse_args()


def build_tta_transforms(img_size, n_steps):
    base = get_val_transform(img_size)
    tfs  = [base]
    if n_steps >= 2:
        tfs.append(transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            base,
        ]))
    if n_steps >= 3:
        tfs.append(transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            base,
        ]))
    return tfs


def predict_with_model(model, dataset, batch_size, device, tta_transforms):
    """단일 모델 + TTA 예측 → (ids, probs(N,2))"""
    all_ids     = None
    sum_probs   = None

    for t_idx, tf in enumerate(tta_transforms):
        dataset.transform = tf
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
        step_ids, step_probs = [], []
        with torch.no_grad():
            for front, top, diff, ids in loader:
                out   = model(front.to(device), top.to(device), diff.to(device))
                probs = torch.softmax(out, dim=1).cpu().numpy()
                step_probs.append(probs)
                if t_idx == 0:
                    step_ids.extend(ids)
        step_probs = np.concatenate(step_probs, axis=0)
        if sum_probs is None:
            sum_probs = step_probs
            all_ids   = step_ids
        else:
            sum_probs += step_probs

    return all_ids, sum_probs / len(tta_transforms)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    val_tf  = get_val_transform(args.img_size)
    tta_tfs = build_tta_transforms(args.img_size, args.tta_steps)
    print(f"🔄 TTA steps: {len(tta_tfs)}")

    test_dataset = StructuralDataset(
        data_dir=args.data_dir, split="test",
        transform=val_tf, img_size=args.img_size,
    )

    ensemble_probs = None
    final_ids      = None
    loaded_folds   = 0

    for fold_idx in range(1, args.n_folds + 1):
        ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  {ckpt_path} 없음, 건너뜀")
            continue

        ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})
        model      = TripleStreamConvNeXt(
            num_classes=2, pretrained=False,
            dropout=saved_args.get("dropout", 0.3),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  ✅ Fold {fold_idx} loaded "
              f"(epoch={ckpt['epoch']}, Loss={ckpt.get('dev_loss',0):.4f}, "
              f"AUC={ckpt.get('dev_auc',0):.4f})")

        ids, probs = predict_with_model(
            model, test_dataset, args.batch_size, device, tta_tfs)

        if ensemble_probs is None:
            ensemble_probs = probs
            final_ids      = ids
        else:
            ensemble_probs += probs
        loaded_folds += 1

    if loaded_folds == 0:
        raise RuntimeError("유효한 checkpoint 가 없습니다. train.py 를 먼저 실행하세요.")

    ensemble_probs /= loaded_folds
    print(f"\n🗳️  Ensemble from {loaded_folds} fold(s)")

    submission = pd.DataFrame({
        "id":            final_ids,
        "unstable_prob": ensemble_probs[:, 1],
        "stable_prob":   ensemble_probs[:, 0],
    })
    submission.to_csv(args.output_csv, index=False)
    print(f"💾 Submission: {args.output_csv}")

    u = ensemble_probs[:, 1]
    print(f"   Avg unstable_prob : {u.mean():.4f}")
    print(f"   Unstable (≥0.5)   : {(u >= 0.5).sum()}")
    print(f"   Stable   (<0.5)   : {(u < 0.5).sum()}")

    # 검증
    sums     = ensemble_probs.sum(axis=1)
    in_range = (u.min() >= 0.0) and (u.max() <= 1.0)
    print(f"   Sum==1.0          : {np.allclose(sums, 1.0)}")
    print(f"   In [0,1]          : {in_range}")
    print("✨ Done!")


if __name__ == "__main__":
    main()
