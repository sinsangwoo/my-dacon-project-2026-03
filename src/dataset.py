import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np

LABEL_MAP = {"stable": 0, "unstable": 1}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

class StructuralDataset(Dataset):
    """구조 안정성 이진 분류 Dataset (Triple-Stream용).

    각 샘플 폴더에서 front.png, top.png와 simulation.mp4(Diff)를 로드하여
    **3개의 텐서** (3, H, W)를 반환합니다.

    Returns:
        (front_tensor, top_tensor, diff_tensor, label)  — train/dev
        (front_tensor, top_tensor, diff_tensor, sample_id) — test
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        img_size: int = 224,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size

        # ── CSV 로드 ──────────────────────────────────────────────
        if split == "train":
            csv_path = os.path.join(data_dir, "train.csv")
            self.df = pd.read_csv(csv_path)
        elif split == "dev":
            csv_path = os.path.join(data_dir, "dev.csv")
            self.df = pd.read_csv(csv_path)
        elif split == "train_merged":
            train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
            dev_df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
            train_df["split"] = "train"
            dev_df["split"] = "dev"
            self.df = pd.concat([train_df, dev_df], ignore_index=True)
        elif split == "train_merged_pseudo":
            train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
            dev_df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
            train_df["split"] = "train"
            dev_df["split"] = "dev"
            dfs = [train_df, dev_df]
            
            pseudo_path = os.path.join(data_dir, "pseudo_train.csv")
            if os.path.exists(pseudo_path):
                pseudo_df = pd.read_csv(pseudo_path)
                pseudo_df["split"] = "test"
                dfs.append(pseudo_df)
                
            self.df = pd.concat(dfs, ignore_index=True)
        elif split == "test":
            csv_path = os.path.join(data_dir, "sample_submission.csv")
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.ids = self.df["id"].tolist()

        # ── 라벨 ─────────────────────────────────────────────────
        self.has_label = split in ("train", "dev", "train_merged", "train_merged_pseudo")
        if self.has_label:
            self.labels = [LABEL_MAP[lbl] for lbl in self.df["label"].tolist()]

        # ── 이미지 폴더 경로 ──────────────────────────────────────
        self.split_dir_map = {"train": "train", "dev": "dev", "test": "test"}

        # ── Transform ─────────────────────────────────────────────
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.ids)

    def _get_diff_map(self, video_path):
        """simulation.mp4에서 시작 프레임과 끝 프레임의 차이맵 추출."""
        if not os.path.exists(video_path):
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
            
        # 첫 프레임
        ret, frame_start = cap.read()
        if not ret:
            cap.release()
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
            
        # 마지막 프레임 찾기
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, frame_end = cap.read()
            if not ret:
                frame_end = frame_start
        else:
            frame_end = frame_start
            
        cap.release()
        
        # Difference Map 계산
        diff = cv2.absdiff(frame_start, frame_end)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        return Image.fromarray(diff)

    def __getitem__(self, idx: int):
        sample_id = self.ids[idx]
        
        if self.split in ("train_merged", "train_merged_pseudo"):
            orig_split = self.df.iloc[idx]["split"]
            img_root = os.path.join(self.data_dir, self.split_dir_map[orig_split])
        else:
            img_root = os.path.join(self.data_dir, self.split_dir_map[self.split])
            
        folder = os.path.join(img_root, sample_id)

        # 1. Front/Top 이미지 로드
        front_img = Image.open(os.path.join(folder, "front.png")).convert("RGB")
        top_img = Image.open(os.path.join(folder, "top.png")).convert("RGB")
        
        # 2. Difference Map (Temporal Feature)
        video_path = os.path.join(folder, "simulation.mp4")
        diff_img = self._get_diff_map(video_path)

        # 3. 일관된 증강 기술 (Random Seed 공유)
        seed = np.random.randint(2147483647)
        
        def apply_transform(img, seed):
            import random
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return self.transform(img)

        front_tensor = apply_transform(front_img, seed)
        top_tensor   = apply_transform(top_img, seed)
        diff_tensor  = apply_transform(diff_img, seed)

        if self.has_label:
            y = self.labels[idx]
            return front_tensor, top_tensor, diff_tensor, y
        else:
            return front_tensor, top_tensor, diff_tensor, sample_id


# ═══════════════════════════════════════════════════════════════════
#  Physics-Aware Augmentation
# ═══════════════════════════════════════════════════════════════════

def get_train_transform(img_size: int = 224):
    """학습용 Physics-Aware Augmentation.

    - 구조적 형태를 크게 해치지 않는 범위의 회전 (±15°)
    - ColorJitter 강화 (밝기/대비/채도/색상)
    - GaussianBlur (시뮬레이션 렌더링 노이즈 대응)
    - RandomHorizontalFlip
    - RandomAffine (미세한 이동/스케일 변동)
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05),
        ),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05,
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transform(img_size: int = 224):
    """검증/테스트용 transform (augmentation 없음)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
