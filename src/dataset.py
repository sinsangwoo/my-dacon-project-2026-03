"""
Dacon 구조 안정성 분류 — Dataset (v7)
=======================================
수정 사항 (v6 → v7):
  [#4] StructuralDataset.__getitem__ 에서 diff 스트림에 대한
       물리적으로 부적절한 ColorJitter 적용 문제 해소.

       diff = |last_frame - first_frame| 은 동역학 신호(모션 맵)이므로
       색상/밝기 왜곡(ColorJitter, GaussianBlur 등)을 가하면
       물리적 신호가 오염됨.

       해결:
         KFoldStructuralDataset / StructuralDataset 모두
         __getitem__에서 transform을 split 처리:
           - front/top  : self.transform 전체 적용 (기하학 + 색상)
           - diff       : self.transform_geom_only 적용
                          (Resize + 기하학 변환 + ToTensor + Normalize만)

         transform 교체(TTA)는 self.transform 만 외부에서 바꾸면 됨.
         transform_geom_only 는 transform에서 ColorJitter/GaussianBlur/
         RandomGrayscale/RandomErasing 을 자동 제거하여 생성.

  부수 수정:
       - img_size 기본값을 224로 통일 (train.py 기본값과 일치)
       - dataset.py 독스트링 업데이트
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

# ColorJitter 등 색상 관련 transform 타입 목록 — diff 스트림에서 제거 대상
_COLOR_TRANSFORM_TYPES = (
    transforms.ColorJitter,
    transforms.RandomGrayscale,
    transforms.GaussianBlur,
    transforms.RandomErasing,
)


def _make_geom_only_transform(tf) -> transforms.Compose:
    """Compose 객체에서 색상 관련 변환만 제거해 기하학 전용 Compose 반환.

    diff 스트림(동역학 모션 맵)은 기하학 변환(HFlip, Rotation, Affine)은
    허용하되 색상 왜곡(ColorJitter, Blur 등)은 물리적으로 부적절하므로 제거.

    Args:
        tf: transforms.Compose 또는 단일 transform
    Returns:
        색상 변환이 제거된 transforms.Compose
    """
    if isinstance(tf, transforms.Compose):
        filtered = [
            t for t in tf.transforms
            if not isinstance(t, _COLOR_TRANSFORM_TYPES)
        ]
        return transforms.Compose(filtered)
    # 단일 transform인 경우
    if isinstance(tf, _COLOR_TRANSFORM_TYPES):
        # 색상 transform 자체인 경우 — Identity로 대체
        return transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std =[0.229, 0.224, 0.225])])
    return tf


# ──────────────────────────────────────────────────────────────────
#  KFoldStructuralDataset
# ──────────────────────────────────────────────────────────────────

class KFoldStructuralDataset(Dataset):
    """5-Fold CV 용 Dataset.

    train.csv + dev.csv 를 통합한 full_df 에서
    fold_indices 로 슬라이싱. pseudo_df 는 train fold 에만 병합.

    diff 스트림은 기하학 변환만 적용 (ColorJitter 등 색상 변환 제외).
    """

    def __init__(
        self,
        data_dir:     str,
        fold_indices:  list,
        all_df:        pd.DataFrame,
        is_train:      bool = True,
        transform=None,
        img_size:      int  = 224,
        pseudo_df:     pd.DataFrame = None,
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

        self.df     = fold_df
        self.ids    = fold_df["id"].tolist()
        
        # 라벨 처리 — soft label 인 경우 float 그대로, hard label 인 경우 int 변환
        self.labels = []
        for _, row in fold_df.iterrows():
            if "unstable_prob" in row and not pd.isna(row["unstable_prob"]):
                # Soft Label (Distillation target)
                self.labels.append(float(row["unstable_prob"]))
            else:
                # Hard Label
                self.labels.append(LABEL_MAP[row["label"]])

        self.splits = fold_df["split"].tolist()

        # transform 설정 — 외부에서 주입하거나 기본값 사용
        self.transform = transform if transform is not None else get_val_transform(img_size)
        # [#4 수정] diff 전용 기하학 transform 자동 생성
        self.transform_geom_only = _make_geom_only_transform(self.transform)

    # transform 교체 시 transform_geom_only 도 자동 동기화
    def _set_transform(self, tf):
        self._transform = tf
        self.transform_geom_only = _make_geom_only_transform(tf)

    def _get_transform(self):
        return self._transform

    transform = property(_get_transform, _set_transform)

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

        def _apply(img, use_geom_only: bool = False):
            """세 스트림 모두 동일 seed로 기하학 변환을 동기화.
            diff 스트림은 use_geom_only=True 로 색상 변환 제거.
            """
            tf = self.transform_geom_only if use_geom_only else self.transform
            random.seed(seed)
            torch.manual_seed(seed)
            return tf(img)

        front = _apply(Image.open(os.path.join(folder, "front.png")).convert("RGB"))
        top   = _apply(Image.open(os.path.join(folder, "top.png")).convert("RGB"))
        # [#4 수정] diff는 기하학 변환만 — 색상 왜곡 없음
        diff  = _apply(
            self._get_diff_map(os.path.join(folder, "simulation.mp4")),
            use_geom_only=True,
        )
        # target 이 float 이면 soft label, int 이면 hard label
        target = self.labels[idx]
        if isinstance(target, (int, np.integer)):
            target = torch.tensor(target, dtype=torch.long)
        else:
            # Distillation [stable_prob, unstable_prob]
            target = torch.tensor([1.0 - target, target], dtype=torch.float)

        return front, top, diff, target


# ──────────────────────────────────────────────────────────────────
#  StructuralDataset  (test 전용)
# ──────────────────────────────────────────────────────────────────

class StructuralDataset(Dataset):
    """test 데이터 예측 전용.

    diff 스트림은 기하학 변환만 적용 (KFoldStructuralDataset 와 동일 정책).
    """

    def __init__(self, data_dir, split="test", transform=None, img_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        if split != "test":
            raise ValueError("StructuralDataset 은 'test' split 만 지원합니다.")
        self.df  = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
        self.ids = self.df["id"].tolist()

        self.transform = transform if transform is not None else get_val_transform(img_size)
        # [#4 수정] diff 전용 기하학 transform 자동 생성
        self.transform_geom_only = _make_geom_only_transform(self.transform)

    # transform 교체 시 transform_geom_only 도 자동 동기화
    def _set_transform(self, tf):
        self._transform = tf
        self.transform_geom_only = _make_geom_only_transform(tf)

    def _get_transform(self):
        return self._transform

    transform = property(_get_transform, _set_transform)

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

        def _apply(img, use_geom_only: bool = False):
            tf = self.transform_geom_only if use_geom_only else self.transform
            random.seed(seed)
            torch.manual_seed(seed)
            return tf(img)

        front = _apply(Image.open(os.path.join(folder, "front.png")).convert("RGB"))
        top   = _apply(Image.open(os.path.join(folder, "top.png")).convert("RGB"))
        # [#4 수정] diff는 기하학 변환만
        diff  = _apply(
            self._get_diff_map(os.path.join(folder, "simulation.mp4")),
            use_geom_only=True,
        )
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
    """pseudo_v2.csv 에서 확신도 높은 샘플을 soft label 과 함께 반환."""
    path = os.path.join(data_dir, "pseudo_v2.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "label", "split", "unstable_prob"])
    df = pd.read_csv(path)
    mask = (
        (df["unstable_prob"] < threshold) |
        (df["unstable_prob"] > 1.0 - threshold)
    )
    df = df[mask].copy()
    # label 컬럼은 기존 호환성을 위해 유지 (hard label)
    df["label"] = df["unstable_prob"].apply(
        lambda p: "unstable" if p > 0.5 else "stable"
    )
    df["split"] = "test"
    return df[["id", "label", "split", "unstable_prob"]].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────
#  Transforms
#  img_size 기본값: 224 (train.py 기본값과 통일)
# ──────────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 224):
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
            std =[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transform(img_size: int = 224):
    """검증/테스트용 (augmentation 없음)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])
