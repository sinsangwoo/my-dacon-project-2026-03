"""
src/dataset.py
===============
v8 — 리팩토링:
  1. Diff 스트림 완전 제거 (test에 simulation.mp4 없음 → 학습-추론 불일치 해소)
  2. Dual-Stream (front + top) 전용으로 단순화
  3. 도메인 갭 대응 증강 대폭 강화 (train=고정조명 → test=랜덤조명/카메라)
  4. Dev-Aware 분리를 위한 split 필드 유지
"""

import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {"stable": 0, "unstable": 1}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# ──────────────────────────────────────────────────────────────────
#  KFoldStructuralDataset (train/val)
# ──────────────────────────────────────────────────────────────────

class KFoldStructuralDataset(Dataset):
    """5-Fold CV 용 Dataset — Dual Stream (front + top)."""

    def __init__(
        self,
        data_dir: str,
        fold_indices: list,
        all_df: pd.DataFrame,
        is_train: bool = True,
        transform=None,
        img_size: int = 224,
        pseudo_df: pd.DataFrame = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size

        fold_df = all_df.iloc[fold_indices].copy().reset_index(drop=True)
        if is_train and pseudo_df is not None and len(pseudo_df) > 0:
            existing_ids = set(all_df["id"])
            p_df = pseudo_df[~pseudo_df["id"].isin(existing_ids)].copy()
            if len(p_df) < len(pseudo_df):
                print(f"   ⚠️  Filtered {len(pseudo_df) - len(p_df)} overlapping IDs from pseudo_df")
            fold_df = pd.concat([fold_df, p_df], ignore_index=True)

        self.df = fold_df
        self.ids = fold_df["id"].tolist()

        self.labels = []
        for _, row in fold_df.iterrows():
            if "unstable_prob" in row and not pd.isna(row.get("unstable_prob", float("nan"))):
                self.labels.append(float(row["unstable_prob"]))
            else:
                self.labels.append(LABEL_MAP[row["label"]])

        self.splits = fold_df["split"].tolist()
        self.transform = transform if transform is not None else get_val_transform(img_size)

    def __len__(self):
        return len(self.ids)

    def _img_root(self, idx):
        sp = self.splits[idx]
        return os.path.join(self.data_dir, sp if sp in ("train", "dev") else "test")

    def __getitem__(self, idx):
        sid = self.ids[idx]
        folder = os.path.join(self._img_root(idx), sid)
        seed = np.random.randint(2_147_483_647)

        def _apply(img):
            random.seed(seed)
            torch.manual_seed(seed)
            return self.transform(img)

        front = _apply(Image.open(os.path.join(folder, "front.png")).convert("RGB"))
        top = _apply(Image.open(os.path.join(folder, "top.png")).convert("RGB"))

        target = self.labels[idx]
        if isinstance(target, (int, np.integer)):
            target = torch.tensor(target, dtype=torch.long)
        else:
            target = torch.tensor([1.0 - target, target], dtype=torch.float)

        return front, top, target


# ──────────────────────────────────────────────────────────────────
#  StructuralDataset (test 전용)
# ──────────────────────────────────────────────────────────────────

class StructuralDataset(Dataset):
    """test 데이터 예측 전용 — Dual Stream (front + top)."""

    def __init__(self, data_dir, split="test", transform=None, img_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        if split != "test":
            raise ValueError("StructuralDataset 은 'test' split 만 지원합니다.")
        self.df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
        self.ids = self.df["id"].tolist()
        self.transform = transform if transform is not None else get_val_transform(img_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        folder = os.path.join(self.data_dir, "test", sid)
        seed = np.random.randint(2_147_483_647)

        def _apply(img):
            random.seed(seed)
            torch.manual_seed(seed)
            return self.transform(img)

        front = _apply(Image.open(os.path.join(folder, "front.png")).convert("RGB"))
        top = _apply(Image.open(os.path.join(folder, "top.png")).convert("RGB"))

        return front, top, sid


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

def build_full_df(data_dir: str) -> pd.DataFrame:
    """train.csv + dev.csv → id / label / split DataFrame."""
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    dv = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    tr["split"] = "train"
    dv["split"] = "dev"
    return pd.concat([tr, dv], ignore_index=True).reset_index(drop=True)


def load_pseudo_v2(data_dir: str, threshold: float = 0.01) -> pd.DataFrame:
    """pseudo_v2.csv 에서 확신도 높은 샘플을 soft label과 함께 반환."""
    path = os.path.join(data_dir, "pseudo_v2.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "label", "split", "unstable_prob"])
    df = pd.read_csv(path)
    mask = (df["unstable_prob"] < threshold) | (df["unstable_prob"] > 1.0 - threshold)
    df = df[mask].copy()
    df["label"] = df["unstable_prob"].apply(lambda p: "unstable" if p > 0.5 else "stable")
    df["split"] = "test"
    return df[["id", "label", "split", "unstable_prob"]].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────
#  Transforms — 도메인 갭 대응 강화
# ──────────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 224):
    """도메인 적응 강화 학습 증강.

    Train(고정 조명/카메라) → Test(랜덤 조명/카메라) 갭 해소를 위해
    조명·시점·색상 변화를 공격적으로 시뮬레이션.
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop((img_size, img_size)),

        # 시점 변화 시뮬레이션
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(
            degrees=0, translate=(0.08, 0.08),
            scale=(0.9, 1.1), shear=8,
        ),

        # 조명 변화 시뮬레이션 (공격적)
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.3, hue=0.08,
        ),
        transforms.RandomGrayscale(p=0.08),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
        transforms.RandomAutocontrast(p=0.2),
        transforms.RandomEqualize(p=0.15),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.12, scale=(0.02, 0.12)),
    ])


def get_val_transform(img_size: int = 224):
    """검증/테스트용 (augmentation 없음)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
