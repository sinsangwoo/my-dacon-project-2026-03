"""
Dacon 구조 안정성 분류 — Dataset (v6)
=======================================
변경점:
  - img_size 기본값 240 (EfficientNet-B1 네이티브 해상도)
  - 나머지 구조 유지 (KFoldStructuralDataset, StructuralDataset,
    build_full_df, load_pseudo_v2, transforms)
"""

import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP     = {"stable": 0, "unstable": 1}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# ──────────────────────────────────────────────────────────────────
#  KFoldStructuralDataset
# ──────────────────────────────────────────────────────────────────

class KFoldStructuralDataset(Dataset):
    """5-Fold CV 용 Dataset.

    train.csv + dev.csv 를 통합한 full_df 에서
    fold_indices 로 슬라이싱. pseudo_df 는 train fold 에만 병합.
    """

    def __init__(
        self,
        data_dir:    str,
        fold_indices: list,
        all_df:      pd.DataFrame,
        is_train:    bool = True,
        transform=None,
        img_size:    int  = 240,
        pseudo_df:   pd.DataFrame = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size

        fold_df = all_df.iloc[fold_indices].copy().reset_index(drop=True)
        if is_train and pseudo_df is not None and len(pseudo_df) > 0:
            # Leakage Check: all_df (train+dev) 에 이미 존재하는 ID 가 pseudo_df 에 있으면 제외
            existing_ids = set(all_df["id"])
            p_df = pseudo_df[~pseudo_df["id"].isin(existing_ids)].copy()
            if len(p_df) < len(pseudo_df):
                 print(f"   ⚠️  Filtered {len(pseudo_df) - len(p_df)} overlapping IDs from pseudo_df")
            fold_df = pd.concat([fold_df, p_df], ignore_index=True)

        self.df     = fold_df
        self.ids    = fold_df["id"].tolist()
        self.labels = [LABEL_MAP[l] for l in fold_df["label"].tolist()]
        self.splits = fold_df["split"].tolist()
        self.transform = transform if transform is not None else get_val_transform(img_size)

    def __len__(self):
        return len(self.ids)

    def _img_root(self, idx):
        sp = self.splits[idx]
        return os.path.join(self.data_dir, sp if sp in ("train", "dev") else "test")

    def _get_diff_map(self, video_path):
        blank = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        if not os.path.exists(video_path):
            return blank
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return blank
        ret, fs = cap.read()
        if not ret:
            cap.release()
            return blank
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
            ret, fe = cap.read()
            if not ret:
                fe = fs
        else:
            fe = fs
        cap.release()
        diff = cv2.absdiff(fs, fe)
        return Image.fromarray(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        sid    = self.ids[idx]
        folder = os.path.join(self._img_root(idx), sid)
        seed   = np.random.randint(2_147_483_647)

        def _apply(img):
            random.seed(seed)
            torch.manual_seed(seed)
            return self.transform(img)

        front = _apply(Image.open(os.path.join(folder, "front.png")).convert("RGB"))
        top   = _apply(Image.open(os.path.join(folder, "top.png")).convert("RGB"))
        diff  = _apply(self._get_diff_map(os.path.join(folder, "simulation.mp4")))
        return front, top, diff, self.labels[idx]


# ──────────────────────────────────────────────────────────────────
#  StructuralDataset  (test 전용)
# ──────────────────────────────────────────────────────────────────

class StructuralDataset(Dataset):
    """test 데이터 예측 전용."""

    def __init__(self, data_dir, split="test", transform=None, img_size=240):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        if split != "test":
            raise ValueError("StructuralDataset 은 'test' split 만 지원합니다.")
        self.df        = pd.read_csv(
            os.path.join(data_dir, "sample_submission.csv")
        )
        self.ids       = self.df["id"].tolist()
        self.transform = transform if transform is not None else get_val_transform(img_size)

    def __len__(self):
        return len(self.ids)

    def _get_diff_map(self, video_path):
        blank = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        if not os.path.exists(video_path):
            return blank
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return blank
        ret, fs = cap.read()
        if not ret:
            cap.release()
            return blank
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
            ret, fe = cap.read()
            if not ret:
                fe = fs
        else:
            fe = fs
        cap.release()
        diff = cv2.absdiff(fs, fe)
        return Image.fromarray(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        sid    = self.ids[idx]
        folder = os.path.join(self.data_dir, "test", sid)
        seed   = np.random.randint(2_147_483_647)

        def _apply(img):
            random.seed(seed)
            torch.manual_seed(seed)
            return self.transform(img)

        front = _apply(Image.open(os.path.join(folder, "front.png")).convert("RGB"))
        top   = _apply(Image.open(os.path.join(folder, "top.png")).convert("RGB"))
        diff  = _apply(self._get_diff_map(os.path.join(folder, "simulation.mp4")))
        return front, top, diff, sid


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


def load_pseudo_v2(
    data_dir: str, threshold: float = 0.01
) -> pd.DataFrame:
    """pseudo_v2.csv 에서 확신도 높은 샘플만 반환."""
    path = os.path.join(data_dir, "pseudo_v2.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "label", "split"])
    df = pd.read_csv(path)
    mask = (df["unstable_prob"] < threshold) | \
           (df["unstable_prob"] > 1.0 - threshold)
    df   = df[mask].copy()
    df["label"] = df["unstable_prob"].apply(
        lambda p: "unstable" if p > 0.5 else "stable"
    )
    df["split"] = "test"
    return df[["id", "label", "split"]].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────
#  Transforms  (EfficientNet-B1: 240 기본)
# ──────────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 240):
    """Physics-Aware 학습 증강."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
        ),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transform(img_size: int = 240):
    """검증/테스트용."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])
