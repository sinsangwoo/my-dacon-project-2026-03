"""
Dacon 구조 안정성 — 앙상블 예측 스크립트 (v8-fix)
"""

import argparse
import os
import sys
import warnings
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)
import sys
import os

# Optional: Add project root to sys.path for direct script execution from nested dirs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import StructuralDataset
from src.model import DualStreamEfficientNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--output_csv", default="submission.csv")
    p.add_argument("--img_size",   type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_folds",    type=int, default=5)
    p.add_argument("--tta_steps",  type=int, default=3)
    p.add_argument("--use_swa",    action="store_true", default=False)
    return p.parse_args()


def build_tta_transforms(img_size: int, n: int) -> list:
    _normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    step0 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        _normalize,
    ])
    tfs = [step0]

    if n >= 2:
        step1 = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            _normalize,
        ])
        tfs.append(step1)

    if n >= 3:
        step2 = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            _normalize,
        ])
        tfs.append(step2)

    return tfs


def load_model(ckpt_path: str, device, override_img_size: int = None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt.get("args", {})

    ckpt_img_size = sa.get("img_size", 224)
    img_size_used = override_img_size if override_img_size is not None else ckpt_img_size

    model = DualStreamEfficientNet(
        num_classes=2,
        pretrained=False,
        dropout=sa.get("dropout", 0.4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(memory_format=torch.channels_last)
    model.eval()

    # [FIX] 체크포인트에 저장된 temperature 사용, 없으면 1.0 (무보정)
    temperature = ckpt.get("temperature", 1.0)
    # 안전 범위 클램핑 (너무 극단적인 T 방지)
    temperature = max(0.5, min(temperature, 3.0))

    print(
        f"     img_size={img_size_used}  "
        f"T={temperature:.3f}  "
        f"ckpt_img_size={ckpt_img_size}"
    )
    return (
        model, temperature,
        ckpt.get("epoch", "?"), ckpt.get("dev_ece", float("nan")),
        img_size_used,
    )


def natural_key(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


def check_id_consistency(data_dir, csv_ids):
    test_path = os.path.join(data_dir, "test")
    if not os.path.exists(test_path) or not os.path.isdir(test_path):
        return
    folder_ids = sorted(os.listdir(test_path), key=natural_key)
    csv_ids_list = list(csv_ids)
    if len(folder_ids) != len(csv_ids_list):
        print(f"  ❗ WARNING: ID count mismatch! CSV={len(csv_ids_list)}, Folders={len(folder_ids)}")
    else:
        print(f"  ✅ ID count match: {len(csv_ids_list)}")


@torch.no_grad()
def predict_with_model(model, dataset, batch_size, device, tta_tfs, temperature: float = 1.0):
    sum_probs = None
    all_ids = None

    for t_idx, tf in enumerate(tta_tfs):
        dataset.transform = tf
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        step_probs, step_ids = [], []
        for front, top, ids in loader:
            front = front.to(device, memory_format=torch.channels_last)
            top = top.to(device, memory_format=torch.channels_last)
            logits = model(front, top) / max(temperature, 0.1)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            step_probs.append(probs)
            if t_idx == 0:
                step_ids.extend(ids)

        sp = np.concatenate(step_probs, axis=0)
        if sum_probs is None:
            sum_probs = sp
            all_ids = step_ids
        else:
            sum_probs = sum_probs + sp

    return all_ids, sum_probs / len(tta_tfs)


def ensemble_mean(probs_list: list) -> np.ndarray:
    """[FIX] 순수 산술 평균 + 보수적 클리핑. Sharpening 완전 제거."""
    mean_p = np.mean(probs_list, axis=0)  # (N, 2)

    # [FIX] alpha sharpening 완전 제거 (alpha=1.0)
    # 과도한 sharpening이 LogLoss 폭발의 주범

    # 보수적 클리핑: [0.005, 0.995]
    mean_p = np.clip(mean_p, 0.005, 0.995)
    mean_p = mean_p / mean_p.sum(axis=1, keepdims=True)
    return mean_p


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    sample_sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    print(f"📋 Loaded sample_submission.csv: {len(sample_sub)} samples")
    check_id_consistency(args.data_dir, sample_sub["id"])

    all_probs_list = []
    final_ids = None

    # [FIX] 모든 Fold 사용 (1~n_folds)
    for fold_idx in range(1, args.n_folds + 1):

        # best-ECE 체크포인트
        ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
        if os.path.exists(ckpt_path):
            model, T, ep, ece, img_size = load_model(
                ckpt_path, device, override_img_size=args.img_size
            )
            print(f"  ✅ Fold {fold_idx}  epoch={ep}  ECE={ece:.4f}  T={T:.3f}")

            tta_tfs = build_tta_transforms(img_size, args.tta_steps)
            test_dataset = StructuralDataset(
                data_dir=args.data_dir, split="test",
                transform=tta_tfs[0], img_size=img_size,
            )
            print(f"  🔄 TTA steps: {len(tta_tfs)}  img_size={img_size}")

            ids, probs = predict_with_model(
                model, test_dataset, args.batch_size,
                device, tta_tfs, temperature=T
            )
            all_probs_list.append(probs)
            if final_ids is None:
                final_ids = ids
        else:
            print(f"  ⚠️  {ckpt_path} 없음, 건너뜀")

        # SWA 체크포인트
        if args.use_swa:
            swa_path = os.path.join(args.save_dir, f"best_fold{fold_idx}_swa.pth")
            if os.path.exists(swa_path):
                model_s, T_s, _, ece_s, img_size_s = load_model(
                    swa_path, device, override_img_size=args.img_size
                )
                print(f"  📦 Fold {fold_idx} SWA  ECE={ece_s:.4f}  T={T_s:.3f}")
                tta_tfs_s = build_tta_transforms(img_size_s, args.tta_steps)
                test_ds_s = StructuralDataset(
                    data_dir=args.data_dir, split="test",
                    transform=tta_tfs_s[0], img_size=img_size_s,
                )
                _, probs_s = predict_with_model(
                    model_s, test_ds_s, args.batch_size,
                    device, tta_tfs_s, temperature=T_s
                )
                all_probs_list.append(probs_s)

    if not all_probs_list:
        raise RuntimeError("유효한 checkpoint가 없습니다. train.py를 먼저 실행하세요.")

    print(f"\n🗳️  Ensemble Mean of {len(all_probs_list)} predictions...")
    ensemble_probs = ensemble_mean(all_probs_list)

    submission = pd.DataFrame({
        "id":            final_ids,
        "unstable_prob": ensemble_probs[:, 1],
        "stable_prob":   ensemble_probs[:, 0],
    })
    submission = submission.set_index("id").reindex(sample_sub["id"]).reset_index()
    submission.to_csv(args.output_csv, index=False)
    print(f"💾 Submission: {args.output_csv}")

    u = ensemble_probs[:, 1]
    print(f"\n📊 unstable_prob Distribution:")
    print(f"   Min: {u.min():.6f} | Max: {u.max():.6f} | Mean: {u.mean():.6f}")

    counts, bins = np.histogram(u, bins=10, range=(0, 1))
    print("   [Histogram]")
    for i in range(len(counts)):
        bar = "█" * int(counts[i] / max(counts) * 20) if max(counts) > 0 else ""
        print(f"   {bins[i]:.1f}-{bins[i+1]:.1f} | {counts[i]:4d} | {bar}")

    p_extreme = ((u < 0.1) | (u > 0.9)).mean() * 100
    p_middle = ((u >= 0.4) & (u <= 0.6)).mean() * 100
    print(f"\n   Extreme (0-0.1 or 0.9-1.0): {p_extreme:.1f}%")
    print(f"   Middle  (0.4-0.6)         : {p_middle:.1f}%")

    print(f"\n   Unstable (≥0.5): {(u >= 0.5).sum()} / {len(u)}")
    print(f"   Stable   (<0.5): {(u < 0.5).sum()}")
    print(f"   Sum==1.0 check : {np.allclose(ensemble_probs.sum(1), 1.0)}")

    if final_ids is not None:
        is_order_correct = (list(final_ids) == list(sample_sub["id"]))
        print(f"   ID Order Match : {is_order_correct}")

    print("✨ Done!")


if __name__ == "__main__":
    main()