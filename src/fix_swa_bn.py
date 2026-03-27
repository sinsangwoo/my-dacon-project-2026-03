"""
Fold 2의 SWA 가중치를 다시 로드하여 BN 통계치만 업데이트하는 스크립트.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    KFoldStructuralDataset,
    build_full_df,
    get_train_transform,
)
from src.model import TripleStreamEfficientNet
from src.train import update_bn_custom

def fix_bn():
    p = argparse.ArgumentParser()
    p.add_argument("--fold_idx", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_suffix", type=str, default="_fixed_bn")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = f"checkpoints/best_fold{args.fold_idx}_swa.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    print(f"📦 Loading SWA checkpoint: {ckpt_path}")
    # weights_only=False for loading dict with args
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # saved_args 복원
    saved_args_dict = ckpt.get("args", {})
    img_size = saved_args_dict.get("img_size", 224)
    data_dir = saved_args_dict.get("data_dir", "data")
    seed     = saved_args_dict.get("seed", 42)
    
    # 모델 초기화
    model = TripleStreamEfficientNet(
        num_classes=2,
        pretrained=False,
        dropout=saved_args_dict.get("dropout", 0.4),
        stoch_depth_p=saved_args_dict.get("stoch_depth_p", 0.1),
    ).to(device)
    
    model = model.to(memory_format=torch.channels_last)
    
    # 가중치 로드
    model.load_state_dict(ckpt["model_state_dict"])
    print("✅ Model weights loaded.")

    # 데이터셋 준비 (K-Fold 인덱스 재현)
    full_df = build_full_df(data_dir)
    labels_arr = np.array([0 if l == "stable" else 1 for l in full_df["label"].tolist()])
    skf = StratifiedKFold(n_splits=saved_args_dict.get("n_folds", 5), shuffle=True, random_state=seed)
    
    folds = list(skf.split(np.zeros(len(full_df)), labels_arr))
    # 1-indexed args.fold_idx -> 0-indexed folds list
    tr_idx, _ = folds[args.fold_idx - 1]
    
    print(f"📊 Fold {args.fold_idx} train samples: {len(tr_idx)}")

    train_tf = get_train_transform(img_size)
    train_ds = KFoldStructuralDataset(
        data_dir, tr_idx.tolist(), full_df,
        is_train=True, transform=train_tf,
        img_size=img_size,
    )
    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    
    print(f"🚀 Updating BN statistics using {len(train_ds)} samples...")
    # TripleStream 대응 커스텀 함수 호출
    update_bn_custom(train_loader, model, device=device)
    
    # 저장
    save_path = ckpt_path.replace(".pth", f"{args.save_suffix}.pth")
    ckpt["model_state_dict"] = model.state_dict()
    # ECE 값은 BN 업데이트 후 달라지므로 None 처리 또는 유지 (여기선 유지)
    
    torch.save(ckpt, save_path)
    print(f"✨ Done! Fixed checkpoint saved to: {save_path}")

if __name__ == "__main__":
    # torchvison transform 관련 경고 방지
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    fix_bn()
