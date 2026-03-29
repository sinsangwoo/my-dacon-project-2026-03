"""
src/extract_features.py
========================
DINOv2 + CLIP 사전학습 모델로 train/dev/test 이미지의
특징 벡터를 1회 추출하여 data/features/ 에 캐싱.

사용법:
    python src/extract_features.py
    python src/extract_features.py --data_dir data --batch_size 32
    python src/extract_features.py --model dinov2          # DINOv2만
    python src/extract_features.py --model clip            # CLIP만
    python src/extract_features.py --model all             # 둘 다 (기본)
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--output_dir", default="data/features")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--model",      default="all", choices=["dinov2", "clip", "all"])
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Simple Image Dataset (front + top 각각)
# ──────────────────────────────────────────────────────────────────

class ImageListDataset(Dataset):
    """이미지 경로 리스트 → transform 적용 후 텐서 반환."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), idx


# ──────────────────────────────────────────────────────────────────
#  데이터 수집: train/dev/test 의 모든 샘플 ID와 이미지 경로
# ──────────────────────────────────────────────────────────────────

def collect_samples(data_dir: str):
    """모든 split의 샘플 정보를 수집.

    Returns:
        list of dict: [{"id": str, "split": str, "front": path, "top": path}, ...]
    """
    samples = []

    # Train
    train_csv = os.path.join(data_dir, "train.csv")
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        for sid in df["id"]:
            folder = os.path.join(data_dir, "train", str(sid))
            samples.append({
                "id": str(sid), "split": "train",
                "front": os.path.join(folder, "front.png"),
                "top": os.path.join(folder, "top.png"),
            })

    # Dev
    dev_csv = os.path.join(data_dir, "dev.csv")
    if os.path.exists(dev_csv):
        df = pd.read_csv(dev_csv)
        for sid in df["id"]:
            folder = os.path.join(data_dir, "dev", str(sid))
            samples.append({
                "id": str(sid), "split": "dev",
                "front": os.path.join(folder, "front.png"),
                "top": os.path.join(folder, "top.png"),
            })

    # Test
    sub_csv = os.path.join(data_dir, "sample_submission.csv")
    if os.path.exists(sub_csv):
        df = pd.read_csv(sub_csv)
        for sid in df["id"]:
            folder = os.path.join(data_dir, "test", str(sid))
            samples.append({
                "id": str(sid), "split": "test",
                "front": os.path.join(folder, "front.png"),
                "top": os.path.join(folder, "top.png"),
            })

    return samples


# ──────────────────────────────────────────────────────────────────
#  DINOv2 Feature Extractor
# ──────────────────────────────────────────────────────────────────

def extract_dinov2(samples, output_dir, batch_size=32, img_size=224):
    """DINOv2 ViT-S/14 로 front/top 각각의 특징 벡터 추출 후 concat 저장.

    출력: {output_dir}/dinov2/{sample_id}.npy  — shape (768,)
          front(384) + top(384) = 768 차원
    """
    print("\n" + "=" * 60)
    print("🦕 DINOv2 Feature Extraction (ViT-S/14, 384-d per view)")
    print("=" * 60)

    save_dir = os.path.join(output_dir, "dinov2")
    os.makedirs(save_dir, exist_ok=True)

    # 이미 추출 완료된 샘플 스킵 확인
    existing = set(
        f.replace(".npy", "") for f in os.listdir(save_dir) if f.endswith(".npy")
    )
    todo = [s for s in samples if s["id"] not in existing]
    if not todo:
        print(f"  ✅ All {len(samples)} samples already extracted. Skipping.")
        return
    print(f"  📊 Total: {len(samples)} | Already done: {len(existing)} | Todo: {len(todo)}")

    # 모델 로드
    print("  📦 Loading DINOv2 ViT-S/14...")
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    except Exception as e:
        print(f"  ⚠️  torch.hub 실패: {e}")
        print("  💡 pip install 후 재시도: pip install timm")
        return

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  🖥️  Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # front / top 각각 추출
    for view in ["front", "top"]:
        print(f"\n  🔍 Extracting [{view}] features...")
        paths = [s[view] for s in todo]
        ids = [s["id"] for s in todo]

        dataset = ImageListDataset(paths, transform)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

        all_feats = []
        with torch.no_grad():
            for batch_imgs, batch_idxs in loader:
                feats = model(batch_imgs.to(device))  # (B, 384)
                all_feats.append(feats.cpu().numpy())

        all_feats = np.concatenate(all_feats, axis=0)  # (N, 384)

        # view별 임시 저장
        if view == "front":
            front_feats = {ids[i]: all_feats[i] for i in range(len(ids))}
        else:
            top_feats = {ids[i]: all_feats[i] for i in range(len(ids))}

        print(f"    ✅ {view}: {all_feats.shape}")

    # front + top concat → 저장
    print(f"\n  💾 Saving concatenated features to {save_dir}/")
    for sid in front_feats:
        combined = np.concatenate([front_feats[sid], top_feats[sid]])  # (768,)
        np.save(os.path.join(save_dir, f"{sid}.npy"), combined)

    print(f"  ✅ DINOv2 done: {len(todo)} samples saved ({combined.shape[0]}-d)")


# ──────────────────────────────────────────────────────────────────
#  CLIP Feature Extractor
# ──────────────────────────────────────────────────────────────────

def extract_clip(samples, output_dir, batch_size=32, img_size=224):
    """CLIP ViT-B/32 로 front/top 각각의 특징 벡터 추출 후 concat 저장.

    출력: {output_dir}/clip/{sample_id}.npy  — shape (1024,)
          front(512) + top(512) = 1024 차원
    """
    print("\n" + "=" * 60)
    print("📎 CLIP Feature Extraction (ViT-B/32, 512-d per view)")
    print("=" * 60)

    save_dir = os.path.join(output_dir, "clip")
    os.makedirs(save_dir, exist_ok=True)

    existing = set(
        f.replace(".npy", "") for f in os.listdir(save_dir) if f.endswith(".npy")
    )
    todo = [s for s in samples if s["id"] not in existing]
    if not todo:
        print(f"  ✅ All {len(samples)} samples already extracted. Skipping.")
        return
    print(f"  📊 Total: {len(samples)} | Already done: {len(existing)} | Todo: {len(todo)}")

    # CLIP 로드 시도
    print("  📦 Loading CLIP ViT-B/32...")
    try:
        import clip as clip_module
        model, preprocess = clip_module.load("ViT-B/32", device="cpu")
    except ImportError:
        print("  ⚠️  clip 패키지 없음. 설치: pip install git+https://github.com/openai/CLIP.git")
        print("  💡 CLIP 없이 DINOv2만으로도 진행 가능합니다.")
        return
    except Exception as e:
        print(f"  ⚠️  CLIP 로드 실패: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"  🖥️  Device: {device}")

    # CLIP 전용 transform (clip.load가 반환하는 preprocess 사용)
    for view in ["front", "top"]:
        print(f"\n  🔍 Extracting [{view}] features...")
        paths = [s[view] for s in todo]
        ids = [s["id"] for s in todo]

        dataset = ImageListDataset(paths, preprocess)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

        all_feats = []
        with torch.no_grad():
            for batch_imgs, batch_idxs in loader:
                feats = model.encode_image(batch_imgs.to(device))  # (B, 512)
                feats = feats.float().cpu().numpy()
                # L2 정규화
                norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
                feats = feats / norms
                all_feats.append(feats)

        all_feats = np.concatenate(all_feats, axis=0)

        if view == "front":
            front_feats = {ids[i]: all_feats[i] for i in range(len(ids))}
        else:
            top_feats = {ids[i]: all_feats[i] for i in range(len(ids))}

        print(f"    ✅ {view}: {all_feats.shape}")

    print(f"\n  💾 Saving concatenated features to {save_dir}/")
    for sid in front_feats:
        combined = np.concatenate([front_feats[sid], top_feats[sid]])  # (1024,)
        np.save(os.path.join(save_dir, f"{sid}.npy"), combined)

    print(f"  ✅ CLIP done: {len(todo)} samples saved ({combined.shape[0]}-d)")


# ──────────────────────────────────────────────────────────────────
#  Feature Loading Utility (train_lgbm.py 등에서 사용)
# ──────────────────────────────────────────────────────────────────

def load_features(
    data_dir: str,
    feature_dir: str = "data/features",
    model_name: str = "dinov2",
    splits: list = None,
):
    """캐싱된 특징 벡터를 로드하여 (X, y, ids, splits) 반환.

    Args:
        data_dir:     data/ 경로
        feature_dir:  features 캐시 경로
        model_name:   "dinov2" | "clip" | "all"
        splits:       ["train", "dev"] 등. None이면 전체.

    Returns:
        X:      np.ndarray (N, D)
        y:      np.ndarray (N,) — test는 -1
        ids:    list of str
        split_labels: list of str
    """
    if splits is None:
        splits = ["train", "dev", "test"]

    # 라벨 로드
    label_map = {}
    for csv_name, split_name in [("train.csv", "train"), ("dev.csv", "dev")]:
        csv_path = os.path.join(data_dir, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                label_map[str(row["id"])] = 1 if row["label"] == "unstable" else 0

    # 전체 샘플 수집
    all_samples = collect_samples(data_dir)
    filtered = [s for s in all_samples if s["split"] in splits]

    X_list, y_list, id_list, split_list = [], [], [], []

    for s in filtered:
        sid = s["id"]
        feats = []

        if model_name in ("dinov2", "all"):
            path = os.path.join(feature_dir, "dinov2", f"{sid}.npy")
            if os.path.exists(path):
                feats.append(np.load(path))

        if model_name in ("clip", "all"):
            path = os.path.join(feature_dir, "clip", f"{sid}.npy")
            if os.path.exists(path):
                feats.append(np.load(path))

        if not feats:
            continue

        combined = np.concatenate(feats)
        X_list.append(combined)
        y_list.append(label_map.get(sid, -1))
        id_list.append(sid)
        split_list.append(s["split"])

    X = np.stack(X_list, axis=0) if X_list else np.array([])
    y = np.array(y_list)

    return X, y, id_list, split_list


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("🚀 Feature Extraction Pipeline")
    print(f"   data_dir:   {args.data_dir}")
    print(f"   output_dir: {args.output_dir}")
    print(f"   model:      {args.model}")
    print(f"   batch_size: {args.batch_size}")
    print(f"   img_size:   {args.img_size}")

    samples = collect_samples(args.data_dir)
    print(f"\n📊 Total samples found: {len(samples)}")

    split_counts = {}
    for s in samples:
        split_counts[s["split"]] = split_counts.get(s["split"], 0) + 1
    for sp, cnt in sorted(split_counts.items()):
        print(f"   {sp}: {cnt}")

    # 이미지 존재 여부 검증
    missing = 0
    for s in samples:
        for view in ["front", "top"]:
            if not os.path.exists(s[view]):
                missing += 1
    if missing > 0:
        print(f"\n  ⚠️  Missing images: {missing}")
    else:
        print(f"\n  ✅ All images exist.")

    # 추출 실행
    if args.model in ("dinov2", "all"):
        extract_dinov2(samples, args.output_dir, args.batch_size, args.img_size)

    if args.model in ("clip", "all"):
        extract_clip(samples, args.output_dir, args.batch_size, args.img_size)

    # 최종 검증
    print("\n" + "=" * 60)
    print("📋 Feature Cache Summary")
    print("=" * 60)
    for model_name in ["dinov2", "clip"]:
        feat_dir = os.path.join(args.output_dir, model_name)
        if os.path.exists(feat_dir):
            files = [f for f in os.listdir(feat_dir) if f.endswith(".npy")]
            if files:
                sample_feat = np.load(os.path.join(feat_dir, files[0]))
                print(f"  {model_name}: {len(files)} files, dim={sample_feat.shape[0]}")
            else:
                print(f"  {model_name}: empty")
        else:
            print(f"  {model_name}: not extracted")

    print("\n✨ Done! Next step: python src/train_lgbm.py")


if __name__ == "__main__":
    main()