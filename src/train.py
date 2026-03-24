"""
Dacon 구조 안정성 분류 - 학습 스크립트 (v3: Evaluation Upgrade)
================================================================
사용법:
    python src/train.py --data_dir data --epochs 20 --batch_size 16 --lr 3e-4
    python src/train.py --share_backbone          # 가중치 공유 모드
    python src/train.py --label_smoothing 0.1     # Label Smoothing 적용
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

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import StructuralDataset, get_train_transform, get_val_transform
from src.model import TripleStreamConvNeXt, FocalLoss, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structural Stability Classification - Dual-Stream Training",
    )
    # 데이터
    parser.add_argument("--data_dir", type=str, default="data", help="데이터 루트 디렉토리")
    parser.add_argument("--img_size", type=int, default=224, help="이미지 리사이즈 크기")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 워커 수")

    # 학습
    parser.add_argument("--epochs", type=int, default=20, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=3e-4, help="학습률 (pretrained 모델 기준)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label Smoothing 계수 (0이면 비활성)")

    # 모델
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="ImageNet pretrained 가중치 사용")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--share_backbone", action="store_true", default=False,
                        help="두 스트림의 backbone 가중치 공유")
    parser.add_argument("--dropout", type=float, default=0.3, help="Classifier dropout")

    # 출력
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="체크포인트 저장 경로")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="제출 CSV 파일명")
    parser.add_argument("--report_file", type=str, default="report.md", help="리포트 마크다운 파일명")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--merge_dev", action="store_true", default=False,
                        help="Train과 Dev 데이터를 합쳐서 학습")
    parser.add_argument("--use_pseudo", action="store_true", default=False,
                        help="Pseudo-Label(pseudo_train.csv)까지 포함하여 학습 (Self-Training)")
    
    # 신규 증강 (CutMix, Mixup)
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha (0이면 비활성)")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="CutMix alpha (0이면 비활성)")
    parser.add_argument("--cutmix_prob", type=float, default=0.5, help="CutMix/Mixup 적용 확률")
    
    # 조기 종료 (Early Stopping)
    parser.add_argument("--patience", type=int, default=5, help="조기 종료 기준 에폭 수")
    
    return parser.parse_args()


def set_seed(seed: int):
    """재현성을 위한 시드 설정."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_seed(seed: int):
    """재현성을 위한 시드 설정."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def mixup_data(x1, x2, x3, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, mixed_x3, y_a, y_b, lam


def cutmix_data(x1, x2, x3, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x1.size(), lam)
    x1[:, :, bbx1:bbx2, bby1:bby2] = x1[index, :, bbx1:bbx2, bby1:bby2]
    x2[:, :, bbx1:bbx2, bby1:bby2] = x2[index, :, bbx1:bbx2, bby1:bby2]
    x3[:, :, bbx1:bbx2, bby1:bby2] = x3[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.size()[-1] * x1.size()[-2]))
    return x1, x2, x3, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_one_epoch(model, loader, criterion, optimizer, device, args):
    """1 에폭 학습 (CutMix/Mixup 포함)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probs = []

    for front, top, diff, labels in loader:
        front = front.to(device)
        top = top.to(device)
        diff = diff.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        r = np.random.rand(1)
        if r < args.cutmix_prob and (args.mixup_alpha > 0 or args.cutmix_alpha > 0):
            if np.random.rand(1) < 0.5 and args.mixup_alpha > 0:
                front, top, diff, y_a, y_b, lam = mixup_data(front, top, diff, labels, args.mixup_alpha, device)
            elif args.cutmix_alpha > 0:
                front, top, diff, y_a, y_b, lam = cutmix_data(front, top, diff, labels, args.cutmix_alpha, device)
            else:
                y_a, y_b, lam = labels, labels, 1.0
                
            outputs = model(front, top, diff)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(front, top, diff)
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * front.size(0)
        _, predicted = outputs.max(1)
        
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy() # unstable prob
        
        total += labels.size(0)
        
        # Mixup/CutMix 시 정확도 계산은 y_a 기준 (근사치)
        if r < args.cutmix_prob and (args.mixup_alpha > 0 or args.cutmix_alpha > 0):
             correct += (lam * predicted.eq(y_a).sum().item() + (1 - lam) * predicted.eq(y_b).sum().item())
        else:
             correct += predicted.eq(labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    # Label Smoothing을 적용했으므로 Train label을 float로 변환하여 AUC 계산 시 경고 방지
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5
        
    return epoch_loss, epoch_acc, epoch_auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """검증 평가 (AUC & PCS 계산 포함)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probs = []
    
    total_pcs = 0.0
    pcs_count = 0

    for front, top, diff, labels in loader:
        front = front.to(device)
        top = top.to(device)
        diff = diff.to(device)
        labels = labels.to(device)

        outputs = model(front, top, diff)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * front.size(0)
        _, predicted = outputs.max(1)
        
        probs = torch.softmax(outputs, dim=1)
        unstable_probs = probs[:, 1].cpu().numpy()
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(unstable_probs)
        
        # ── Physical Consistency Score (PCS) 계산 ────────────────
        # 입력을 좌우 반전시켰을 때의 예측 확률 차이를 측정 (물리적 대칭성 이해도)
        front_flip = torch.flip(front, dims=[3])
        top_flip = torch.flip(top, dims=[3])
        diff_flip = torch.flip(diff, dims=[3])
        
        outputs_flip = model(front_flip, top_flip, diff_flip)
        probs_flip = torch.softmax(outputs_flip, dim=1)
        unstable_probs_flip = probs_flip[:, 1].cpu().numpy()
        
        # PCS = 1 - Mean(|Prob_orig - Prob_flip|)
        consistency = 1.0 - np.abs(unstable_probs - unstable_probs_flip)
        total_pcs += np.sum(consistency)
        pcs_count += len(labels)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    epoch_pcs = total_pcs / pcs_count if pcs_count > 0 else 0.0
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5
        
    return epoch_loss, epoch_acc, epoch_auc, epoch_pcs


@torch.no_grad()
def analyze_errors(model, dataset, device, top_k=5):
    """검증 데이터셋에서 가장 헷갈려 하는 샘플 (Top Error) 분석."""
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    errors = []
    
    for front, top, diff, labels in loader:
        front = front.to(device)
        top = top.to(device)
        diff = diff.to(device)
        labels = labels.to(device)
        
        outputs = model(front, top, diff)
        probs = torch.softmax(outputs, dim=1)
        
        for i in range(len(labels)):
            true_label = labels[i].item()
            unstable_prob = probs[i, 1].item()
            
            # Ground truth probability
            if true_label == 1: # Unstable
                loss_proxy = 1.0 - unstable_prob # 예측이 틀릴수록 값이 큼
                pred_label = "unstable" if unstable_prob >= 0.5 else "stable"
            else: # Stable
                loss_proxy = unstable_prob # 예측이 틀릴수록 값이 큼
                pred_label = "unstable" if unstable_prob >= 0.5 else "stable"
                
            errors.append({
                "true_label": "unstable" if true_label == 1 else "stable",
                "pred_label": pred_label,
                "unstable_prob": unstable_prob,
                "error_score": loss_proxy
            })
            
    # 에러가 가장 큰 샘플 k개 추출
    # (ids 정보가 Dataset에 있으므로 인덱스로 매핑 가능)
    for i, err in enumerate(errors):
        err["sample_id"] = dataset.ids[i]
        
    errors.sort(key=lambda x: x["error_score"], reverse=True)
    return errors[:top_k]

@torch.no_grad()
def predict_test(model, loader, device):
    """테스트셋 예측 → (sample_ids, probabilities) 반환."""
    model.eval()
    all_ids = []
    all_probs = []

    for front, top, diff, sample_ids in loader:
        front = front.to(device)
        top = top.to(device)
        diff = diff.to(device)

        outputs = model(front, top, diff)
        probs = torch.softmax(outputs, dim=1)  # (B, 2)
        all_probs.append(probs.cpu().numpy())
        all_ids.extend(sample_ids)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_ids, all_probs

def generate_report(history, best_epoch, best_dev_loss, top_errors, report_path, dev_probs, dev_dataset, best_pcs):
    """학습이 완료된 후 report.md 생성"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Automated Training Report (Physical Intelligence v3)\n\n")
        f.write("## 1. 학습 요약\n")
        f.write(f"- **Best Epoch**: {best_epoch}\n")
        f.write(f"- **Best Dev Loss**: {best_dev_loss:.4f}\n")
        f.write(f"- **Best Physical Consistency Score (PCS)**: {best_pcs:.4f}\n\n")
        
        f.write("## 2. Epoch별 추이\n")
        f.write("| Epoch | Train Loss | Train AUC | Dev Loss | Dev AUC | PCS |\n")
        f.write("|---|---|---|---|---|---|\n")
        for log in history:
            f.write(f"| {log['epoch']} | {log['train_loss']:.4f} | {log['train_auc']:.4f} | {log['dev_loss']:.4f} | {log['dev_auc']:.4f} | {log.get('pcs', 0):.4f} |\n")
            
        f.write("\n## 3. Top-5 Error Analysis\n")
        f.write("모델이 가장 혼동한 검증(Dev) 데이터 샘플 Top 5입니다.\n\n")
        f.write("| Sample ID | True Label | Predicted Label | Unstable Prob | Error Score |\n")
        f.write("|---|---|---|---|---|\n")
        for err in top_errors:
            f.write(f"| `{err['sample_id']}` | {err['true_label']} | {err['pred_label']} | {err['unstable_prob']:.4f} | {err['error_score']:.4f} |\n")
            
        f.write("\n> 혼동 기준(Error Score)은 실제 정답일 확률이 낮을수록 (1에 가까울수록) 높게 계산되었습니다. 이는 안정과 불안정의 경계선(주변 환경 변수 복잡도, 영상 내의 미세한 흔들림/회전 등)에 있는 샘플일 가능성이 높습니다.\n\n")

        # Dev Prediction Summary
        f.write("## 4. 최종 Dev 데이터 예측 결과 요약\n")
        pred_stable = sum(1 for p in dev_probs if p < 0.5)
        pred_unstable = len(dev_probs) - pred_stable
        true_stable = sum(1 for lbl in dev_dataset.labels if lbl == 0)
        true_unstable = len(dev_dataset.labels) - true_stable
        
        f.write(f"- **실제 라벨 분포**: Stable {true_stable} / Unstable {true_unstable}\n")
        f.write(f"- **모델 예측 분포**: Stable {pred_stable} / Unstable {pred_unstable}\n")
        

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── 디바이스 설정 ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── 데이터셋 & DataLoader ─────────────────────────────────────
    train_transform = get_train_transform(args.img_size)
    val_transform = get_val_transform(args.img_size)

    if args.use_pseudo:
        train_split = "train_merged_pseudo"
    else:
        train_split = "train_merged" if args.merge_dev else "train"
        
    train_dataset = StructuralDataset(
        data_dir=args.data_dir, split=train_split,
        transform=train_transform, img_size=args.img_size,
    )
    dev_dataset = StructuralDataset(
        data_dir=args.data_dir, split="dev",
        transform=val_transform, img_size=args.img_size,
    )
    test_dataset = StructuralDataset(
        data_dir=args.data_dir, split="test",
        transform=val_transform, img_size=args.img_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    print(f"📦 Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Test: {len(test_dataset)}")

    # ── 모델 ──────────────────────────────────────────────────────
    model = TripleStreamConvNeXt(
        num_classes=2,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    print(f"🧠 Model: Triple-Stream ConvNeXt-Tiny")
    print(f"   Parameters: {count_parameters(model):,}")
    print(f"   Pretrained: {args.pretrained}")

    # ── Loss (Focal Loss 적용) ────────────────────────────────────
    criterion = FocalLoss(gamma=2.0)
    print("🏷️  Loss: Focal Loss (gamma=2.0)")

    # ── Optimizer / Scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ── 학습 루프 ─────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_dev_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    print("\n" + "=" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train AUC':>9} | "
          f"{'Dev Loss':>9} | {'Dev AUC':>8} | {'LR':>10} | {'Time':>6}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args
        )
        dev_loss, dev_acc, dev_auc, dev_pcs = evaluate(model, dev_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"{epoch:5d} | {train_loss:10.4f} | {train_auc:9.4f} | "
            f"{dev_loss:9.4f} | {dev_auc:8.4f} | {current_lr:10.2e} | {elapsed:5.1f}s | PCS: {dev_pcs:.4f}"
        )
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "dev_loss": dev_loss,
            "dev_auc": dev_auc,
            "pcs": dev_pcs,
        })

        # Best 모델 저장 (LogLoss 최소화 기준)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_loss": dev_loss,
                "dev_auc": dev_auc,
                "dev_pcs": dev_pcs,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✅ Best model saved (Dev Loss: {dev_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ No improvement for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= args.patience:
            print(f"\n⏹️ Early stopping triggered after {epoch} epochs.")
            break

    print("=" * 80)
    print(f"\n🏆 Best Dev Loss: {best_dev_loss:.4f} @ Epoch {best_epoch}")

    # ── Best 모델로 Error Analysis 수행 ───────────────────────────
    print("\n🔍 Analyzing Top-5 Errors on Dev Dataset...")
    best_ckpt = torch.load(
        os.path.join(args.save_dir, "best_model.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    top_errors = analyze_errors(model, dev_dataset, device, top_k=5)
    
    # Dev probs for summary
    _, dev_probs_full = predict_test(model, dev_loader, device) # Using predict_test logic structure (requires outputting tuple without labels)
    
    # We need a small manual loop for dev_probs for simplicity:
    model.eval()
    dev_probs = []
    with torch.no_grad():
        for front, top, _ in dev_loader:
            outputs = model(front.to(device), top.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            dev_probs.extend(probs)

    # 리포트 생성
    report_path = os.path.join(os.getcwd(), args.report_file)
    best_pcs = max([h['pcs'] for h in history])
    generate_report(history, best_epoch, best_dev_loss, top_errors, report_path, dev_probs, dev_dataset, best_pcs)
    print(f"📊 Training Report saved: {report_path}")

    # ── Best 모델로 테스트 예측 ───────────────────────────────────
    print("\n📝 Generating test predictions...")
    test_ids, test_probs = predict_test(model, test_loader, device)

    # sample_submission.csv 형식: id, unstable_prob, stable_prob
    submission = pd.DataFrame({
        "id": test_ids,
        "unstable_prob": test_probs[:, 1],  # index 1 = unstable
        "stable_prob": test_probs[:, 0],    # index 0 = stable
    })
    submission.to_csv(args.output_csv, index=False)
    print(f"💾 Submission saved: {args.output_csv}")
    print("✨ Done!")


if __name__ == "__main__":
    main()
