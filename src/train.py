"""
Dacon 구조 안정성 분류 - 학습 스크립트 (Physical Intelligence v4)
==================================================================
주요 변경점 (v4):
  - PhysicsConsistencyLoss를 훈련 Loss에 직접 반영 (--pcs_lambda)
  - 버그 수정: dev_probs 루프에서 model(front, top, diff) 3인자 호출
  - Best 모델 저장 기준: dev_loss - pcs_bonus 복합 점수
  - 로그 출력에 PCS Reg Loss 항목 추가

사용법:
    python src/train.py --data_dir data --epochs 30 --batch_size 16 --lr 3e-4
    python src/train.py --pcs_lambda 0.2   # PCS 정규화 가중치 강화
    python src/train.py --merge_dev        # Train+Dev 합쳐서 학습
    python src/train.py --use_pseudo       # Pseudo-Label까지 포함
"""

import argparse
import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import StructuralDataset, get_train_transform, get_val_transform
from src.model import TripleStreamConvNeXt, FocalLoss, PhysicsConsistencyLoss, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structural Stability Classification - Triple-Stream + PCS Training",
    )
    # 데이터
    parser.add_argument("--data_dir",    type=str,   default="data")
    parser.add_argument("--img_size",    type=int,   default=224)
    parser.add_argument("--num_workers", type=int,   default=0)

    # 학습
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # 모델
    parser.add_argument("--pretrained",     action="store_true",  default=True)
    parser.add_argument("--no_pretrained",  dest="pretrained",    action="store_false")
    parser.add_argument("--share_backbone", action="store_true",  default=False)
    parser.add_argument("--dropout",        type=float,           default=0.3)

    # Physics Consistency Regularization
    parser.add_argument(
        "--pcs_lambda", type=float, default=0.1,
        help="PhysicsConsistencyLoss 가중치. 0이면 비활성. (권장: 0.05~0.2)",
    )
    parser.add_argument(
        "--pcs_temperature", type=float, default=2.0,
        help="PhysicsConsistencyLoss KL divergence 온도 스케일",
    )

    # 증강
    parser.add_argument("--mixup_alpha",  type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--cutmix_prob",  type=float, default=0.5)

    # 출력
    parser.add_argument("--save_dir",    type=str, default="checkpoints")
    parser.add_argument("--output_csv",  type=str, default="submission.csv")
    parser.add_argument("--report_file", type=str, default="report.md")
    parser.add_argument("--seed",        type=int, default=42)

    # 데이터 전략
    parser.add_argument("--merge_dev",   action="store_true", default=False)
    parser.add_argument("--use_pseudo",  action="store_true", default=False)

    # 조기 종료
    parser.add_argument("--patience", type=int, default=7)

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  CutMix / Mixup
# ─────────────────────────────────────────────────────────────────────────────

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_data(x1, x2, x3, y, alpha, device):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x1.size(0)).to(device)
    mx1   = lam * x1 + (1 - lam) * x1[idx]
    mx2   = lam * x2 + (1 - lam) * x2[idx]
    mx3   = lam * x3 + (1 - lam) * x3[idx]
    return mx1, mx2, mx3, y, y[idx], lam


def cutmix_data(x1, x2, x3, y, alpha, device):
    lam          = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx          = torch.randperm(x1.size(0)).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x1.size(), lam)
    x1[:, :, bbx1:bbx2, bby1:bby2] = x1[idx, :, bbx1:bbx2, bby1:bby2]
    x2[:, :, bbx1:bbx2, bby1:bby2] = x2[idx, :, bbx1:bbx2, bby1:bby2]
    x3[:, :, bbx1:bbx2, bby1:bby2] = x3[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (x1.size(-1) * x1.size(-2))
    return x1, x2, x3, y, y[idx], lam


# ─────────────────────────────────────────────────────────────────────────────
#  Train / Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, pcs_loss_fn, optimizer, device, args):
    """1 에폭 학습.

    total_loss = focal_loss + pcs_lambda * physics_consistency_loss
    """
    model.train()
    running_loss      = 0.0
    running_pcs_loss  = 0.0
    correct           = 0
    total             = 0
    all_labels, all_probs = [], []

    use_aug = args.mixup_alpha > 0 or args.cutmix_alpha > 0

    for front, top, diff, labels in loader:
        front  = front.to(device)
        top    = top.to(device)
        diff   = diff.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # ── CutMix / Mixup ────────────────────────────────────────
        mixed = False
        if use_aug and np.random.rand() < args.cutmix_prob:
            mixed = True
            if np.random.rand() < 0.5 and args.mixup_alpha > 0:
                front, top, diff, y_a, y_b, lam = mixup_data(
                    front, top, diff, labels, args.mixup_alpha, device)
            elif args.cutmix_alpha > 0:
                front, top, diff, y_a, y_b, lam = cutmix_data(
                    front, top, diff, labels, args.cutmix_alpha, device)
            else:
                mixed = False

        logits = model(front, top, diff)

        # ── Classification Loss ───────────────────────────────────
        if mixed:
            cls_loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            cls_loss = criterion(logits, labels)

        # ── Physics Consistency Regularization ────────────────────
        pcs_reg = torch.tensor(0.0, device=device)
        if args.pcs_lambda > 0:
            pcs_reg = pcs_loss_fn(model, front, top, diff, logits)

        loss = cls_loss + args.pcs_lambda * pcs_reg
        loss.backward()
        optimizer.step()

        bs = front.size(0)
        running_loss     += loss.item()     * bs
        running_pcs_loss += pcs_reg.item()  * bs
        total            += bs

        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        _, predicted = logits.max(1)

        if mixed:
            correct += (lam * predicted.eq(y_a).sum().item()
                        + (1 - lam) * predicted.eq(y_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss     = running_loss     / total
    epoch_pcs_loss = running_pcs_loss / total
    epoch_acc      = 100.0 * correct  / total
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_acc, epoch_auc, epoch_pcs_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """검증 평가: Loss / AUC / PCS(물리 일관성 점수) 반환."""
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    all_labels, all_probs = [], []
    total_pcs, pcs_count  = 0.0, 0

    for front, top, diff, labels in loader:
        front  = front.to(device)
        top    = top.to(device)
        diff   = diff.to(device)
        labels = labels.to(device)

        logits = model(front, top, diff)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * front.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        _, predicted = logits.max(1)

        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

        # ── PCS: 좌우 반전 일관성 ─────────────────────────────────
        logits_flip = model(
            torch.flip(front, dims=[3]),
            torch.flip(top,   dims=[3]),
            torch.flip(diff,  dims=[3]),
        )
        probs_flip = torch.softmax(logits_flip, dim=1)[:, 1].cpu().numpy()
        total_pcs  += np.sum(1.0 - np.abs(probs - probs_flip))
        pcs_count  += len(labels)

    epoch_loss = running_loss / total
    epoch_acc  = 100.0 * correct / total
    epoch_pcs  = total_pcs / pcs_count if pcs_count > 0 else 0.0
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_acc, epoch_auc, epoch_pcs


@torch.no_grad()
def analyze_errors(model, dataset, device, top_k=5):
    """Dev 데이터셋에서 모델이 가장 혼동한 샘플 top_k개 분석."""
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    errors = []

    for front, top, diff, labels in loader:
        front  = front.to(device)
        top    = top.to(device)
        diff   = diff.to(device)
        labels = labels.to(device)

        logits = model(front, top, diff)
        probs  = torch.softmax(logits, dim=1)

        for i in range(len(labels)):
            true_label   = labels[i].item()
            unstable_p   = probs[i, 1].item()
            error_score  = (1.0 - unstable_p) if true_label == 1 else unstable_p
            pred_label   = "unstable" if unstable_p >= 0.5 else "stable"
            errors.append(dict(
                true_label   = "unstable" if true_label == 1 else "stable",
                pred_label   = pred_label,
                unstable_prob= unstable_p,
                error_score  = error_score,
            ))

    for i, err in enumerate(errors):
        err["sample_id"] = dataset.ids[i]

    errors.sort(key=lambda x: x["error_score"], reverse=True)
    return errors[:top_k]


@torch.no_grad()
def predict_test(model, loader, device):
    """테스트셋 예측 → (sample_ids, probs) 반환."""
    model.eval()
    all_ids, all_probs = [], []

    for front, top, diff, sample_ids in loader:
        front = front.to(device)
        top   = top.to(device)
        diff  = diff.to(device)

        logits = model(front, top, diff)
        probs  = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_ids.extend(sample_ids)

    return all_ids, np.concatenate(all_probs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(history, best_epoch, best_dev_loss, top_errors,
                    report_path, dev_probs, dev_dataset, best_pcs):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Automated Training Report (Physical Intelligence v4)\n\n")
        f.write("## 1. 학습 요약\n")
        f.write(f"- **Best Epoch**: {best_epoch}\n")
        f.write(f"- **Best Dev Loss**: {best_dev_loss:.4f}\n")
        f.write(f"- **Best Physical Consistency Score (PCS)**: {best_pcs:.4f}\n\n")

        f.write("## 2. Epoch별 추이\n")
        f.write("| Epoch | Train Loss | PCS Reg | Train AUC | Dev Loss | Dev AUC | PCS |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for h in history:
            f.write(
                f"| {h['epoch']} "
                f"| {h['train_loss']:.4f} "
                f"| {h.get('pcs_reg', 0):.4f} "
                f"| {h['train_auc']:.4f} "
                f"| {h['dev_loss']:.4f} "
                f"| {h['dev_auc']:.4f} "
                f"| {h.get('pcs', 0):.4f} |\n"
            )

        f.write("\n## 3. Top-5 Error Analysis\n")
        f.write("| Sample ID | True | Predicted | Unstable P | Error Score |\n")
        f.write("|---|---|---|---|---|\n")
        for err in top_errors:
            f.write(
                f"| `{err['sample_id']}` "
                f"| {err['true_label']} "
                f"| {err['pred_label']} "
                f"| {err['unstable_prob']:.4f} "
                f"| {err['error_score']:.4f} |\n"
            )

        f.write("\n## 4. 최종 Dev 예측 분포\n")
        pred_stable   = sum(1 for p in dev_probs if p < 0.5)
        pred_unstable = len(dev_probs) - pred_stable
        true_stable   = sum(1 for lbl in dev_dataset.labels if lbl == 0)
        true_unstable = len(dev_dataset.labels) - true_stable
        f.write(f"- 실제: Stable {true_stable} / Unstable {true_unstable}\n")
        f.write(f"- 예측: Stable {pred_stable} / Unstable {pred_unstable}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────
    train_transform = get_train_transform(args.img_size)
    val_transform   = get_val_transform(args.img_size)

    if args.use_pseudo:
        train_split = "train_merged_pseudo"
    else:
        train_split = "train_merged" if args.merge_dev else "train"

    train_dataset = StructuralDataset(args.data_dir, train_split, train_transform, args.img_size)
    dev_dataset   = StructuralDataset(args.data_dir, "dev",       val_transform,   args.img_size)
    test_dataset  = StructuralDataset(args.data_dir, "test",      val_transform,   args.img_size)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    dev_loader   = DataLoader(dev_dataset,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"📦 Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Test: {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────
    model = TripleStreamConvNeXt(
        num_classes=2,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    print(f"🧠 Model  : Triple-Stream ConvNeXt-Tiny")
    print(f"   Params : {count_parameters(model):,}")
    print(f"   Pretrained: {args.pretrained}")
    print(f"   PCS λ  : {args.pcs_lambda}  (temperature={args.pcs_temperature})")

    # ── Loss ──────────────────────────────────────────────────────
    criterion   = FocalLoss(gamma=2.0)
    pcs_loss_fn = PhysicsConsistencyLoss(temperature=args.pcs_temperature)
    print("🏷️  Loss  : FocalLoss(γ=2) + PhysicsConsistencyLoss")

    # ── Optimizer / Scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training Loop ─────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    # 복합 점수: dev_loss - PCS_BONUS_WEIGHT * pcs  (낮을수록 좋음)
    PCS_BONUS_WEIGHT = 0.05

    best_score        = float("inf")
    best_dev_loss     = float("inf")
    best_epoch        = 0
    best_pcs_val      = 0.0
    epochs_no_improve = 0
    history           = []

    hdr = (f"{'Epoch':>5} | {'TrainLoss':>9} | {'PCSReg':>7} | {'TrAUC':>6} | "
           f"{'DevLoss':>8} | {'DevAUC':>7} | {'PCS':>6} | {'Score':>8} | {'LR':>9} | Time")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_auc, pcs_reg = train_one_epoch(
            model, train_loader, criterion, pcs_loss_fn, optimizer, device, args)
        dev_loss, dev_acc, dev_auc, dev_pcs = evaluate(
            model, dev_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        # 복합 점수: dev_loss가 낮고 PCS가 높을수록 좋음
        score = dev_loss - PCS_BONUS_WEIGHT * dev_pcs

        print(
            f"{epoch:5d} | {train_loss:9.4f} | {pcs_reg:7.4f} | {train_auc:6.4f} | "
            f"{dev_loss:8.4f} | {dev_auc:7.4f} | {dev_pcs:6.4f} | {score:8.4f} | "
            f"{current_lr:9.2e} | {elapsed:.1f}s"
        )

        history.append(dict(
            epoch      = epoch,
            train_loss = train_loss,
            pcs_reg    = pcs_reg,
            train_auc  = train_auc,
            dev_loss   = dev_loss,
            dev_auc    = dev_auc,
            pcs        = dev_pcs,
        ))

        # Best 모델 저장 (복합 점수 기준)
        if score < best_score:
            best_score        = score
            best_dev_loss     = dev_loss
            best_pcs_val      = dev_pcs
            best_epoch        = epoch
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(dict(
                epoch              = epoch,
                model_state_dict   = model.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                dev_loss           = dev_loss,
                dev_auc            = dev_auc,
                dev_pcs            = dev_pcs,
                composite_score    = score,
                args               = vars(args),
            ), ckpt_path)
            print(f"  ✅ Best saved  Dev Loss={dev_loss:.4f}  PCS={dev_pcs:.4f}  Score={score:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️  No improvement {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}.")
            break

    print("=" * len(hdr))
    print(f"\n🏆 Best  Dev Loss={best_dev_loss:.4f}  PCS={best_pcs_val:.4f}  @ Epoch {best_epoch}")

    # ── Best 모델 로드 & Error Analysis ──────────────────────────
    print("\n🔍 Analyzing Top-5 Errors on Dev Dataset...")
    ckpt = torch.load(os.path.join(args.save_dir, "best_model.pth"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    top_errors = analyze_errors(model, dev_dataset, device, top_k=5)

    # ── Dev probs 수집 (버그 수정: 반드시 3인자 호출) ─────────────
    model.eval()
    dev_probs = []
    with torch.no_grad():
        for front, top, diff, _labels in dev_loader:          # diff 반드시 unpack
            logits = model(front.to(device), top.to(device), diff.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            dev_probs.extend(p)

    # ── Report ────────────────────────────────────────────────────
    report_path = os.path.join(os.getcwd(), args.report_file)
    best_pcs_in_history = max(h["pcs"] for h in history)
    generate_report(history, best_epoch, best_dev_loss, top_errors,
                    report_path, dev_probs, dev_dataset, best_pcs_in_history)
    print(f"📊 Report saved : {report_path}")

    # ── Test Prediction ───────────────────────────────────────────
    print("\n📝 Generating test predictions...")
    test_ids, test_probs = predict_test(model, test_loader, device)

    submission = pd.DataFrame({
        "id":            test_ids,
        "unstable_prob": test_probs[:, 1],
        "stable_prob":   test_probs[:, 0],
    })
    submission.to_csv(args.output_csv, index=False)
    print(f"💾 Submission saved: {args.output_csv}")
    print("✨ Done!")


if __name__ == "__main__":
    main()
