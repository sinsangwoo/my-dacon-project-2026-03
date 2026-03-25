"""
Dacon 구조 안정성 분류 - Dataset (v5: 5-Fold CV 대응)
=======================================================
주요 변경:
  - KFoldStructuralDataset: train+dev 전체를 통합하여 fold 인덱스로 분할
  - pseudo_v2.csv 지원 (확신도 0.01/0.99 필터링된 고신뢰 샘플)
  - img_size 파라미터로 입력 해상도 유연 조정 (권장: 268 = 224 * 1.2)
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
#  KFoldStructuralDataset  (5-Fold CV 전용)
# ──────────────────────────────────────────────────────────────────

class KFoldStructuralDataset(Dataset):
    """5-Fold Cross-Validation 용 Dataset.

    train.csv + dev.csv 를 통합한 전체 라벨 데이터를 fold_indices 로 분할.
    pseudo_v2.csv 가 존재하면 train fold 에만 추가 병합.

    Args:
        data_dir:     데이터 루트 디렉토리
        fold_indices: 이 Dataset 이 사용할 샘플 인덱스 리스트
        all_df:       전체 통합 DataFrame (id, label, split 컬럼 필수)
        is_train:     True → 학습용 (augmentation 적용)
        transform:    None 이면 기본 transform 사용
        img_size:     입력 해상도 (권장 268)
        pseudo_df:    고신뢰 pseudo label DataFrame (is_train=True 일 때만 병합)
    """

    def __init__(
        self,
        data_dir: str,
        fold_indices: list,
        all_df: pd.DataFrame,
        is_train: bool = True,
        transform=None,
        img_size: int = 268,
        pseudo_df: pd.DataFrame = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train

        # fold 인덱스로 슬라이싱
        fold_df = all_df.iloc[fold_indices].copy().reset_index(drop=True)

        # train fold 에만 pseudo 병합
        if is_train and pseudo_df is not None and len(pseudo_df) > 0:
            fold_df = pd.concat([fold_df, pseudo_df], ignore_index=True)

        self.df     = fold_df
        self.ids    = fold_df["id"].tolist()
        self.labels = [LABEL_MAP[lbl] for lbl in fold_df["label"].tolist()]
        self.splits = fold_df["split"].tolist()   # 'train' | 'dev' | 'test'(pseudo)

        self.transform = transform if transform is not None else get_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.ids)

    def _img_root(self, idx: int) -> str:
        sp = self.splits[idx]
        return os.path.join(self.data_dir, sp if sp in ("train", "dev") else "test")

    def _get_diff_map(self, video_path: str) -> Image.Image:
        blank = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        if not os.path.exists(video_path):
            return blank
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return blank
        ret, frame_start = cap.read()
        if not ret:
            cap.release()
            return blank
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
            ret, frame_end = cap.read()
            if not ret:
                frame_end = frame_start
        else:
            frame_end = frame_start
        cap.release()
        diff = cv2.absdiff(frame_start, frame_end)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        return Image.fromarray(diff)

    def __getitem__(self, idx: int):
        sid    = self.ids[idx]
        folder = os.path.join(self._img_root(idx), sid)

        front_img = Image.open(os.path.join(folder, "front.png")).convert("RGB")
        top_img   = Image.open(os.path.join(folder, "top.png")).convert("RGB")
        diff_img  = self._get_diff_map(os.path.join(folder, "simulation.mp4"))

        # 세 스트림에 동일한 랜덤 변환 적용 (공유 seed)
        seed = np.random.randint(2_147_483_647)

        def _apply(img):
            random.seed(seed)
            torch.manual_seed(seed)
            return self.transform(img)

        return _apply(front_img), _apply(top_img), _apply(diff_img), self.labels[idx]


# ──────────────────────────────────────────────────────────────────
#  StructuralDataset  (test 예측 전용 / 하위 호환)
# ──────────────────────────────────────────────────────────────────

class StructuralDataset(Dataset):
    """test 데이터 예측 전용 Dataset.

    split='test' 만 지원. predict.py 에서 사용.
    """

    def __init__(self, data_dir, split="test", transform=None, img_size=268):
        super().__init__()
        self.data_dir = data_dir
        self.split    = split
        self.img_size = img_size

        if split == "test":
            csv_path = os.path.join(data_dir, "sample_submission.csv")
        else:
            raise ValueError(f"StructuralDataset은 'test' split만 지원합니다. 학습에는 KFoldStructuralDataset 사용.")

        self.df        = pd.read_csv(csv_path)
        self.ids       = self.df["id"].tolist()
        self.has_label = False
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
#  헬퍼: 전체 통합 DataFrame 빌드
# ──────────────────────────────────────────────────────────────────

def build_full_df(data_dir: str) -> pd.DataFrame:
    """train.csv + dev.csv 를 통합하여 id / label / split 컬럼을 가진
    DataFrame 을 반환한다. 5-Fold 분할의 기반 데이터."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    dev_df   = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    train_df["split"] = "train"
    dev_df["split"]   = "dev"
    return pd.concat([train_df, dev_df], ignore_index=True).reset_index(drop=True)


def load_pseudo_v2(data_dir: str, threshold: float = 0.01) -> pd.DataFrame:
    """pseudo_v2.csv 에서 확신도 높은 샘플만 필터링하여 반환.

    pseudo_v2.csv 컬럼: id, unstable_prob, stable_prob
    threshold:  unstable_prob < threshold → stable,
                unstable_prob > 1-threshold → unstable 로 라벨링
    """
    path = os.path.join(data_dir, "pseudo_v2.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "label", "split"])

    df = pd.read_csv(path)
    stable_mask   = df["unstable_prob"] < threshold
    unstable_mask = df["unstable_prob"] > (1.0 - threshold)
    df = df[stable_mask | unstable_mask].copy()

    df["label"] = df["unstable_prob"].apply(
        lambda p: "unstable" if p > 0.5 else "stable")
    df["split"] = "test"   # 이미지 경로가 test/ 아래에 있음
    return df[["id", "label", "split"]].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 268):
    """Physics-Aware 학습 증강. img_size=268 (224 * 1.2) 권장."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transform(img_size: int = 268):
    """검증/테스트용 (augmentation 없음)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
