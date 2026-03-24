"""
Best model 로드 후 test 데이터 예측 → submission.csv 생성
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
from src.model import TripleStreamConvNeXt

def main():
    data_dir = "data"
    ckpt_path = "checkpoints/best_model.pth"
    output_csv = "submission.csv"
    img_size = 224
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # 체크포인트 로드
    print(f"📦 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved_args = ckpt.get("args", {})
    best_metric_str = f"Dev Loss: {ckpt.get('dev_loss', 'N/A'):.4f}" if 'dev_loss' in ckpt else f"Dev AUC: {ckpt.get('dev_auc', 'N/A'):.4f}"
    print(f"   Saved at epoch {ckpt['epoch']} with {best_metric_str}")

    # 모델 생성 & 가중치 로드
    model = TripleStreamConvNeXt(
        num_classes=2,
        pretrained=False,
        dropout=saved_args.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("✅ Model loaded successfully")

    # 테스트 데이터셋
    val_transform = get_val_transform(img_size)
    test_dataset = StructuralDataset(
        data_dir=data_dir, split="test",
        transform=val_transform, img_size=img_size,
    )
    
    # ── TTA (Test Time Augmentation) 설정 ─────────────────────────
    # 원본 + 좌우반전 + 미세한 Brightness/Contrast (수직 반전 제외)
    tta_transforms = [
        val_transform,
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), val_transform]),
        transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1), 
            val_transform
        ]),
    ]
    print(f"🔄 TTA enabled: {len(tta_transforms)} augmentations (Orig, H-Flip, ColorJitter)")

    # 예측
    all_ids = []
    final_probs = None

    for t_idx, tta_tf in enumerate(tta_transforms):
        print(f"   - TTA step {t_idx+1}/{len(tta_transforms)}...")
        # 임시 데이터셋/로더 (transform만 교체)
        test_dataset.transform = tta_tf
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )
        
        step_probs = []
        step_ids = []
        with torch.no_grad():
            for front, top, diff, sample_ids in test_loader:
                front = front.to(device)
                top = top.to(device)
                diff = diff.to(device)
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

    # ── 확률 정규화 (Clipping 제거) ──────────────────────────────
    # 합계 1.0 확인 및 개별 확률 추출
    unstable_probs = all_probs[:, 1]
    stable_probs = all_probs[:, 0]

    # ── submission.csv 생성 (id, unstable_prob, stable_prob 순서) ──
    submission = pd.DataFrame({
        "id": all_ids,
        "unstable_prob": unstable_probs,
        "stable_prob": stable_probs,
    })
    
    # 컬럼 순서 재확인 (혹시 모를 상황 대비)
    submission = submission[["id", "unstable_prob", "stable_prob"]]
    
    submission.to_csv(output_csv, index=False)
    print(f"\n💾 Submission saved: {output_csv}")
    print(f"   Total predictions: {len(submission)}")
    
    # ── 검증 ──────────────────────────────────────────────────
    print("\n🔍 Validating submission...")
    # 1. 합계 검증
    sums = submission["unstable_prob"] + submission["stable_prob"]
    is_sum_one = np.allclose(sums, 1.0)
    print(f"   - Sum of probabilities is 1.0: {'TRUE' if is_sum_one else 'FALSE'}")
    
    # 2. 범위 검증 (Clipping 제거되었으므로 0~1 사이인지 확인)
    is_in_range = (submission["unstable_prob"].min() >= 0.0) and (submission["unstable_prob"].max() <= 1.0)
    print(f"   - Probabilities within [0.0, 1.0]: {'TRUE' if is_in_range else 'FALSE'}")
    
    # 3. 컬럼 순서 검증
    is_correct_order = list(submission.columns) == ["id", "unstable_prob", "stable_prob"]
    print(f"   - Column order is correct: {'TRUE' if is_correct_order else 'FALSE'}")

    print(f"\n📈 Statistics (TTA applied):")
    print(f"   Avg unstable_prob: {submission['unstable_prob'].mean():.4f}")
    print(f"   Unstable 예측 (prob >= 0.5): {(unstable_probs >= 0.5).sum()}")
    print(f"   Stable 예측 (prob < 0.5): {(unstable_probs < 0.5).sum()}")
    print("✨ Done!")

if __name__ == "__main__":
    main()
