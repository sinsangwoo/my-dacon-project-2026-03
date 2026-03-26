"""
Best model을 사용하여 test 데이터 중 확실한(확률 <0.1 또는 >0.9) 샘플만 골라
pseudo_label로 추출하여 pseudo_train.csv 생성 (Self-Training용)
"""
import os, sys
import warnings
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import StructuralDataset, get_val_transform
from src.model import TripleStreamEfficientNet

def main():
    data_dir = "data"
    ckpt_path = "checkpoints/best_model.pth"
    pseudo_csv = os.path.join(data_dir, "pseudo_train.csv")
    img_size = 224
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # 체크포인트 로드
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        return
        
    print(f"📦 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved_args = ckpt.get("args", {})
    
    # 모델 생성 & 가중치 로드
    model = TripleStreamEfficientNet(
        num_classes=2,
        pretrained=False,
        dropout=saved_args.get("dropout", 0.4),
        stoch_depth_p=0.0,

    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    print("✅ Model loaded successfully")

    # 테스트 데이터셋 로드
    val_transform = get_val_transform(img_size)
    test_dataset = StructuralDataset(
        data_dir=data_dir, split="test",
        transform=val_transform, img_size=img_size,
    )
    
    # ── TTA (Test Time Augmentation) 설정 ─────────────────────────
    tta_transforms = [
        val_transform,
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), val_transform]),
        transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1), 
            val_transform
        ]),
    ]
    print(f"🔄 TTA enabled for Pseudo-Labeling: {len(tta_transforms)} augmentations")

    # 예측
    all_ids = []
    final_probs = None

    for t_idx, tta_tf in enumerate(tta_transforms):
        print(f"   - TTA step {t_idx+1}/{len(tta_transforms)}...")
        test_dataset.transform = tta_tf
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )
        
        step_probs = []
        step_ids = []
        with torch.no_grad():
            for front, top, diff, sample_ids in test_loader:
                front = front.to(device, memory_format=torch.channels_last)
                top   = top.to(device, memory_format=torch.channels_last)
                diff  = diff.to(device, memory_format=torch.channels_last)
                outputs = model(front, top, diff)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()  # (B, 2)
                step_probs.append(probs)
                if t_idx == 0:
                    step_ids.extend(sample_ids)
        
        step_probs = np.concatenate(step_probs, axis=0)
        if final_probs is None:
            final_probs = step_probs
            all_ids = step_ids
        else:
            final_probs += step_probs

    # TTA 평균 계산
    all_probs = final_probs / len(tta_transforms)
    unstable_probs = all_probs[:, 1]

    # ── Pseudo-Label 필터링 (임계값: < 0.1, > 0.9) ─────────────────
    threshold_low = 0.1
    threshold_high = 0.9
    
    pseudo_ids = []
    pseudo_labels = []
    
    for i in range(len(all_ids)):
        prob = unstable_probs[i]
        if prob < threshold_low:
            pseudo_ids.append(all_ids[i])
            pseudo_labels.append("stable")
        elif prob > threshold_high:
            pseudo_ids.append(all_ids[i])
            pseudo_labels.append("unstable")
            
    print(f"\n📊 Total test samples: {len(all_ids)}")
    print(f"🎯 Selected highly confident samples: {len(pseudo_ids)} "
          f"({len(pseudo_ids)/len(all_ids)*100:.1f}%)")
    
    if len(pseudo_ids) == 0:
        print("⚠️ No samples met the threshold criteria. Skipping pseudo_train.csv creation.")
        return
        
    stable_count = sum(1 for label in pseudo_labels if label == "stable")
    unstable_count = sum(1 for label in pseudo_labels if label == "unstable")
    print(f"   - Stable: {stable_count}")
    print(f"   - Unstable: {unstable_count}")

    # ── pseudo_train.csv 저장 ────────────────────────────────────
    pseudo_df = pd.DataFrame({
        "id": pseudo_ids,
        "label": pseudo_labels
    })
    
    pseudo_df.to_csv(pseudo_csv, index=False)
    print(f"💾 Pseudo labels saved: {pseudo_csv}")
    print("✨ Successfully generated pseudo labels for self-training!")

if __name__ == "__main__":
    main()
