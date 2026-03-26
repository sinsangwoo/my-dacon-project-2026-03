"""
Dacon 구조 안정성 — 앙상블 예측 스크립트 (v7)
================================================
수정 사항 (v6 → v7):
  [#1] img_size를 CLI 기본값(잘못된 240)이 아니라
       체크포인트에 저장된 args["img_size"] 에서 읽어옴.
       → 학습/추론 해상도 완전 일치 보장.
  [#2] build_tta_transforms()를 완전히 독립된 Compose 객체로 재구성.
       기존 코드는 base를 여러 Compose가 공유해 dataset.transform 교체 시
       side-effect 발생 → 완전 분리.
  [#3] load_model()이 사용 img_size를 로그에 출력하여 불일치 즉시 탐지.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import StructuralDataset, get_val_transform
from src.model import TripleStreamEfficientNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--output_csv", default="submission.csv")
    p.add_argument(
        "--img_size", type=int, default=None,
        help="추론 해상도 (미지정 시 각 체크포인트의 저장된 args.img_size 자동 사용)"
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_folds",    type=int, default=5)
    p.add_argument(
        "--tta_steps", type=int, default=3,
        help="TTA 스텝 수 (1=원본만, 2=+HFlip, 3=+HFlip+밝기)"
    )
    p.add_argument("--use_swa", action="store_true", default=True,
                   help="SWA 모델도 앙상블에 포함 (기본: True)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  [수정 #2] TTA Transforms — 완전히 독립된 Compose 객체
# ──────────────────────────────────────────────────────────────────

def build_tta_transforms(img_size: int, n: int) -> list:
    """각 TTA step을 완전히 독립된 Compose 객체로 반환.

    변경 전(버그):
        base = get_val_transform(img_size)
        tfs.append(Compose([HFlip, base]))   # base 공유 → side-effect

    변경 후(수정):
        각 step 내부에서 Normalize까지 포함한 완전한 파이프라인을 독립 생성.
        공유 참조 없음.
    """
    _normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    # Step 0: 원본 (augmentation 없음)
    step0 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        _normalize,
    ])

    tfs = [step0]

    if n >= 2:
        # Step 1: 수평 반전 — 구조물의 좌우 대칭성 활용
        step1 = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            _normalize,
        ])
        tfs.append(step1)

    if n >= 3:
        # Step 2: 밝기/대비 미세 변동 — 조명 편차에 대한 강건성
        #         (diff 스트림에도 독립적으로 적용되나 dataset.py 레벨에서
        #          diff는 기하학 변환만 받으므로 사실상 이 step에서 diff는
        #          color jitter 없이 resize만 통과함 — dataset.py #4 수정과 연동)
        step2 = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            _normalize,
        ])
        tfs.append(step2)

    return tfs


# ──────────────────────────────────────────────────────────────────
#  [수정 #1] load_model — img_size를 체크포인트에서 복원
# ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device, override_img_size: int = None):
    """체크포인트 로드. img_size는 저장된 args에서 읽는 것이 기본.

    Returns:
        model, temperature, epoch, dev_ece, img_size_used
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa   = ckpt.get("args", {})

    # [핵심 수정] 학습 시 img_size를 체크포인트에서 복원
    #   override_img_size가 명시된 경우(CLI --img_size)만 덮어씀
    ckpt_img_size = sa.get("img_size", 224)   # 학습 기본값 224
    img_size_used = override_img_size if override_img_size is not None else ckpt_img_size

    model = TripleStreamEfficientNet(
        num_classes=2,
        pretrained=False,
        dropout=sa.get("dropout", 0.4),
        stoch_depth_p=0.0,   # 추론 시 SD 완전 비활성
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(memory_format=torch.channels_last)
    model.eval()

    temperature = float(np.clip(ckpt.get("temperature", 1.0), 0.1, 3.0))

    print(
        f"     img_size={img_size_used}  "
        f"T={temperature:.3f}  "
        f"ckpt_img_size={ckpt_img_size}"
        + (" ⚠️  OVERRIDE" if override_img_size is not None
                              and override_img_size != ckpt_img_size else "")
    )
    return (
        model, temperature,
        ckpt.get("epoch", "?"), ckpt.get("dev_ece", float("nan")),
        img_size_used,
    )


# ──────────────────────────────────────────────────────────────────
#  예측 루틴
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_model(
    model, dataset, batch_size, device,
    tta_tfs, temperature: float = 1.0
):
    """단일 모델 + TTA → (ids, probs shape=(N,2))"""
    sum_probs: np.ndarray | None = None
    all_ids: list | None = None

    for t_idx, tf in enumerate(tta_tfs):
        # dataset.transform 교체 — 각 tf는 완전히 독립된 객체이므로 안전
        dataset.transform = tf
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=False
        )
        step_probs, step_ids = [], []
        for front, top, diff, ids in loader:
            logits = model(
                front.to(device), top.to(device), diff.to(device)
            ) / max(temperature, 0.1)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            step_probs.append(probs)
            if t_idx == 0:
                step_ids.extend(ids)

        sp = np.concatenate(step_probs, axis=0)   # (N, 2)
        if sum_probs is None:
            sum_probs = sp
            all_ids   = step_ids
        else:
            sum_probs = sum_probs + sp

    return all_ids, sum_probs / len(tta_tfs)   # type: ignore[operator]


def ensemble_mean(probs_list: list) -> np.ndarray:
    """Calibrated Mean Ensemble with Probability Sharpening.
    1) 평균 계산
    2) [Probability Sharpening] Power Scaling (alpha=1.1) 적용 — LogLoss 최적화
    3) [Re-normalization] axis=1 기준 합계 1 강제
    4) [Clipping] Dacon 기준 1e-15 클리핑 방어
    """
    mean_p = np.mean(probs_list, axis=0)              # (N, 2)
    
    # [1. Probability Sharpener]
    # 극단적인 확률값의 변동을 완화하면서도 정답에 대한 확신도를 정교하게 보정
    alpha = 1.1
    mean_p = np.power(mean_p, alpha)
    mean_p = mean_p / mean_p.sum(axis=1, keepdims=True)
    
    # [2. Final Defense]
    mean_p = np.clip(mean_p, 1e-15, 1.0 - 1e-15)
    mean_p = mean_p / mean_p.sum(axis=1, keepdims=True)
    return mean_p


# ──────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if args.img_size is not None:
        print(f"⚠️  --img_size={args.img_size} 강제 지정 (체크포인트 값 무시)")

    all_probs_list: list = []
    final_ids = None

    for fold_idx in range(1, args.n_folds + 1):

        # ── best-ECE 버전 ──────────────────────────────────────────
        ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
        if os.path.exists(ckpt_path):
            model, T, ep, ece, img_size = load_model(
                ckpt_path, device, override_img_size=args.img_size
            )
            print(f"  ✅ Fold {fold_idx}  epoch={ep}  ECE={ece:.4f}  T={T:.3f}")

            # img_size가 fold마다 달라질 수 있으므로 여기서 생성
            tta_tfs = build_tta_transforms(img_size, args.tta_steps)
            test_dataset = StructuralDataset(
                data_dir=args.data_dir, split="test",
                transform=tta_tfs[0], img_size=img_size,
            )
            print(f"🔄 TTA steps: {len(tta_tfs)}  img_size={img_size}")

            ids, probs = predict_with_model(
                model, test_dataset, args.batch_size,
                device, tta_tfs, temperature=T
            )
            all_probs_list.append(probs)
            if final_ids is None:
                final_ids = ids
        else:
            print(f"  ⚠️  {ckpt_path} 없음, 건너뜀")

        # ── SWA 버전 ──────────────────────────────────────────────
        if args.use_swa:
            swa_path = os.path.join(
                args.save_dir, f"best_fold{fold_idx}_swa.pth"
            )
            if os.path.exists(swa_path):
                model_s, T_s, _, ece_s, img_size_s = load_model(
                    swa_path, device, override_img_size=args.img_size
                )
                print(f"  📦 Fold {fold_idx} SWA  ECE={ece_s:.4f}")
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
    sample_sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    submission = submission.set_index("id").reindex(sample_sub["id"]).reset_index()
    submission.to_csv(args.output_csv, index=False)
    print(f"💾 Submission: {args.output_csv}")

    u = ensemble_probs[:, 1]
    print(f"   Avg unstable_prob : {u.mean():.4f}")
    print(f"   Unstable (≥0.5)   : {(u >= 0.5).sum()}  /  {len(u)}")
    print(f"   Stable   (<0.5)   : {(u < 0.5).sum()}")
    print(f"   Sum==1.0 check    : {np.allclose(ensemble_probs.sum(1), 1.0)}")
    print("✨ Done!")


if __name__ == "__main__":
    main()
