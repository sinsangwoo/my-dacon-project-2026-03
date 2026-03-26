"""
Dacon 구조 안정성 — 앙상블 예측 스크립트 (v6)
================================================
best_fold{k}.pth (best-ECE) + best_fold{k}_swa.pth (SWA)
두 버전을 Rank Averaging 으로 앙상블. TTA 선택 가능.

사용법:
    python src/predict.py --data_dir data --n_folds 5
    python src/predict.py --tta_steps 3 --use_swa
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
from src.model import TripleStreamEfficientNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--output_csv", default="submission.csv")
    p.add_argument("--img_size",   type=int, default=240)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_folds",    type=int, default=5)
    p.add_argument("--tta_steps",  type=int, default=3,
                   help="TTA 수 (1=원본, 3=원본+HFlip+ColorJitter)")
    p.add_argument("--use_swa",    action="store_true", default=True,
                   help="SWA 모델도 앙상블에 포함 (기본: True)")
    return p.parse_args()


def build_tta_transforms(img_size, n):
    base = get_val_transform(img_size)
    tfs  = [base]
    if n >= 2:
        tfs.append(transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0), base
        ]))
    if n >= 3:
        tfs.append(transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1), base
        ]))
    return tfs


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa    = ckpt.get("args", {})
    model = TripleStreamEfficientNet(
        num_classes=2,
        pretrained=False,
        dropout=sa.get("dropout", 0.4),
        stoch_depth_p=0.0,   # 추론 시 SD 비활성
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    temperature = ckpt.get("temperature", 1.0)
    temperature = np.clip(temperature, 0.1, 3.0)
    return model, temperature, ckpt.get("epoch", "?"), ckpt.get("dev_ece", float("nan"))


@torch.no_grad()
def predict_with_model(model, dataset, batch_size, device, tta_tfs, temperature=1.0):
    """단일 모델 + TTA 예측 → (ids, probs(N,2))"""
    sum_probs = None
    all_ids   = None

    for t_idx, tf in enumerate(tta_tfs):
        dataset.transform = tf
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
        step_probs, step_ids = [], []
        for front, top, diff, ids in loader:
            logits = model(
                front.to(device), top.to(device), diff.to(device)
            ) / max(temperature, 0.1)   # Temperature Scaling 적용
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            step_probs.append(probs)
            if t_idx == 0:
                step_ids.extend(ids)

        sp = np.concatenate(step_probs, axis=0)
        if sum_probs is None:
            sum_probs = sp
            all_ids   = step_ids
        else:
            sum_probs += sp

    return all_ids, sum_probs / len(tta_tfs)


def ensemble_mean(probs_list):
    """Calibrated Mean Ensemble — LogLoss 에 더 적합함.
    Rank Averaging 은 LogLoss 에서 불리하므로 제거.
    """
    mean_probs = np.mean(probs_list, axis=0)
    # LogLoss 방어: 극단적인 확률값 클리핑 (1e-6 ~ 1-1e-6)
    mean_probs = np.clip(mean_probs, 1e-6, 1.0 - 1e-6)
    # 합계 1로 재정규화
    mean_probs = mean_probs / mean_probs.sum(axis=1, keepdims=True)
    return mean_probs


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

    all_probs_list = []
    final_ids      = None

    for fold_idx in range(1, args.n_folds + 1):
        # ── best-ECE 버전 ──
        ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
        if os.path.exists(ckpt_path):
            model, T, ep, ece = load_model(ckpt_path, device)
            print(f"  ✅ Fold {fold_idx}  epoch={ep}  ECE={ece:.4f}  T={T:.3f}")
            ids, probs = predict_with_model(
                model, test_dataset, args.batch_size,
                device, tta_tfs, temperature=T
            )
            all_probs_list.append(probs)
            if final_ids is None:
                final_ids = ids
        else:
            print(f"  ⚠️  {ckpt_path} 없음, 건너뜀")

        # ── SWA 버전 ──
        if args.use_swa:
            swa_path = os.path.join(
                args.save_dir, f"best_fold{fold_idx}_swa.pth"
            )
            if os.path.exists(swa_path):
                model_s, T_s, _, ece_s = load_model(swa_path, device)
                print(f"  📦 Fold {fold_idx} SWA  ECE={ece_s:.4f}")
                _, probs_s = predict_with_model(
                    model_s, test_dataset, args.batch_size,
                    device, tta_tfs, temperature=T_s
                )
                all_probs_list.append(probs_s)

    if not all_probs_list:
        raise RuntimeError("유효한 checkpoint 가 없습니다. train.py 를 먼저 실행하세요.")

    print(f"\n🗳️  Ensemble Mean of {len(all_probs_list)} predictions...")
    ensemble_probs = ensemble_mean(all_probs_list)

    submission = pd.DataFrame({
        "id":            final_ids,
        "unstable_prob": ensemble_probs[:, 1],
        "stable_prob":   ensemble_probs[:, 0],
    })

    # 원본 sample_submission.csv 순서와 100% 일치하도록 정렬 강제
    sample_sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    submission = submission.set_index("id").reindex(sample_sub["id"]).reset_index()

    submission.to_csv(args.output_csv, index=False)
    print(f"💾 Submission: {args.output_csv}")

    u = ensemble_probs[:, 1]
    print(f"   Avg unstable_prob : {u.mean():.4f}")
    print(f"   Unstable (≥0.5)   : {(u >= 0.5).sum()}")
    print(f"   Stable   (<0.5)   : {(u < 0.5).sum()}")
    print(f"   Sum==1.0          : {np.allclose(ensemble_probs.sum(1), 1.0)}")
    print("✨ Done!")


if __name__ == "__main__":
    main()
