"""
Dacon 구조 안정성 분류 - 학습 스크립트 (v5: 5-Fold CV)
==========================================================
사용법:
    # 기본 5-Fold 학습
    python src/train.py --data_dir data --epochs 30 --batch_size 16

    # Pseudo v2 병합
    python src/train.py --use_pseudo_v2

    # PCS 정규화 강화
    python src/train.py --pcs_lambda 0.15
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    KFoldStructuralDataset,
    build_full_df,
    load_pseudo_v2,
    get_train_transform,
    get_val_transform,
)
from src.model import (
    TripleStreamConvNeXt,
    FocalLoss,
    PhysicsConsistencyLoss,
    compute_ece,
    compute_gradcam_consistency,
    count_parameters,
)


# ──────────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # 데이터
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--img_size",    type=int,   default=268,
                   help="입력 해상도 (224*1.2=268 권장)")
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--n_folds",     type=int,   default=5)
    p.add_argument("--use_pseudo_v2", action="store_true", default=False,
                   help="pseudo_v2.csv (확신도 0.01/0.99 필터) 를 train fold 에 병합")

    # 학습
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05,
                   help="AdamW weight decay (고체급 모델 과적합 방지: 0.05 권장)")

    # 모델
    p.add_argument("--pretrained",    action="store_true",  default=True)
    p.add_argument("--no_pretrained", dest="pretrained",    action="store_false")
    p.add_argument("--dropout",       type=float, default=0.3)

    # Loss
    p.add_argument("--focal_gamma",      type=float, default=3.0,
                   help="FocalLoss gamma (3.0 = 결정 경계 날카롭게)")
    p.add_argument("--label_smoothing",  type=float, default=0.05,
                   help="Label Smoothing (0.05 = 확신도 강화)")
    p.add_argument("--pcs_lambda",       type=float, default=0.1)
    p.add_argument("--pcs_temperature",  type=float, default=2.0)

    # 증강
    p.add_argument("--mixup_alpha",  type=float, default=0.2)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--cutmix_prob",  type=float, default=0.5)

    # 출력
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--report_file", default="report.md")
    p.add_argument("--seed",        type=int,   default=42)

    # 조기 종료
    p.add_argument("--patience", type=int, default=7)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Seed / Augmentation Helpers
# ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def rand_bbox(size, lam):
    W, H   = size[2], size[3]
    cut_w  = int(W * np.sqrt(1.0 - lam))
    cut_h  = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    return (
        np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H),
        np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H),
    )


def mixup(x1, x2, x3, y, alpha, dev):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x1.size(0)).to(dev)
    return (lam*x1 + (1-lam)*x1[idx],
            lam*x2 + (1-lam)*x2[idx],
            lam*x3 + (1-lam)*x3[idx],
            y, y[idx], lam)


def cutmix(x1, x2, x3, y, alpha, dev):
    lam              = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx              = torch.randperm(x1.size(0)).to(dev)
    b1,b2,b3,b4     = rand_bbox(x1.size(), lam)
    x1[:,:,b1:b3,b2:b4] = x1[idx,:,b1:b3,b2:b4]
    x2[:,:,b1:b3,b2:b4] = x2[idx,:,b1:b3,b2:b4]
    x3[:,:,b1:b3,b2:b4] = x3[idx,:,b1:b3,b2:b4]
    lam = 1 - (b3-b1)*(b4-b2) / (x1.size(-1)*x1.size(-2))
    return x1, x2, x3, y, y[idx], lam


# ──────────────────────────────────────────────────────────────────
#  Train / Evaluate
# ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, pcs_fn, optimizer, device, args):
    model.train()
    run_loss = run_pcs = total = correct = 0.0
    all_lbl, all_prob = [], []

    use_aug = (args.mixup_alpha > 0 or args.cutmix_alpha > 0)

    for front, top, diff, labels in loader:
        front, top, diff, labels = (
            front.to(device), top.to(device), diff.to(device), labels.to(device)
        )
        optimizer.zero_grad()

        mixed = False
        if use_aug and np.random.rand() < args.cutmix_prob:
            if np.random.rand() < 0.5 and args.mixup_alpha > 0:
                front, top, diff, ya, yb, lam = mixup(
                    front, top, diff, labels, args.mixup_alpha, device)
            elif args.cutmix_alpha > 0:
                front, top, diff, ya, yb, lam = cutmix(
                    front, top, diff, labels, args.cutmix_alpha, device)
            else:
                mixed = False
            mixed = True

        logits = model(front, top, diff)

        cls_loss = (
            lam * criterion(logits, ya) + (1-lam) * criterion(logits, yb)
            if mixed else criterion(logits, labels)
        )

        pcs_reg = torch.tensor(0.0, device=device)
        if args.pcs_lambda > 0:
            pcs_reg = pcs_fn(model, front, top, diff, logits)

        loss = cls_loss + args.pcs_lambda * pcs_reg
        loss.backward()
        optimizer.step()

        bs = front.size(0)
        run_loss += loss.item() * bs
        run_pcs  += pcs_reg.item() * bs
        total    += bs

        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        _, pred = logits.max(1)
        correct += (lam*pred.eq(ya).sum().item() + (1-lam)*pred.eq(yb).sum().item()
                    if mixed else pred.eq(labels).sum().item())
        all_lbl.extend(labels.cpu().numpy())
        all_prob.extend(probs)

    auc = roc_auc_score(all_lbl, all_prob) if len(set(all_lbl)) > 1 else 0.5
    return run_loss/total, 100*correct/total, auc, run_pcs/total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    run_loss = total = correct = 0.0
    all_lbl, all_prob = [], []
    pcs_sum = pcs_n = 0.0

    for front, top, diff, labels in loader:
        front, top, diff, labels = (
            front.to(device), top.to(device), diff.to(device), labels.to(device)
        )
        logits = model(front, top, diff)
        loss   = criterion(logits, labels)

        run_loss += loss.item() * front.size(0)
        probs    = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        _, pred  = logits.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
        all_lbl.extend(labels.cpu().numpy())
        all_prob.extend(probs)

        # PCS
        lf = torch.flip(front, [3])
        lt = torch.flip(top,   [3])
        ld = torch.flip(diff,  [3])
        probs_f = torch.softmax(model(lf, lt, ld), dim=1)[:, 1].cpu().numpy()
        pcs_sum += np.sum(1.0 - np.abs(probs - probs_f))
        pcs_n   += len(labels)

    auc = roc_auc_score(all_lbl, all_prob) if len(set(all_lbl)) > 1 else 0.5
    ece = compute_ece(np.array(all_prob), np.array(all_lbl))
    return (run_loss/total, 100*correct/total,
            auc, pcs_sum/pcs_n, ece,
            np.array(all_prob), np.array(all_lbl))


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_ids, all_probs = [], []
    for front, top, diff, ids in loader:
        out   = model(front.to(device), top.to(device), diff.to(device))
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_ids.extend(ids)
    return all_ids, np.concatenate(all_probs, axis=0)


# ──────────────────────────────────────────────────────────────────
#  Report
# ──────────────────────────────────────────────────────────────────

def generate_report(fold_results, report_path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Automated Training Report (Physical Intelligence v5 — 5-Fold CV)\n\n")

        f.write("## 1. Fold 요약\n")
        f.write("| Fold | Best Epoch | Best Dev Loss | Dev AUC | PCS | ECE |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in fold_results:
            f.write(f"| {r['fold']} | {r['best_epoch']} "
                    f"| {r['best_loss']:.4f} | {r['best_auc']:.4f} "
                    f"| {r['best_pcs']:.4f} | {r['best_ece']:.4f} |\n")

        mean_loss = np.mean([r['best_loss'] for r in fold_results])
        mean_auc  = np.mean([r['best_auc']  for r in fold_results])
        mean_pcs  = np.mean([r['best_pcs']  for r in fold_results])
        mean_ece  = np.mean([r['best_ece']  for r in fold_results])
        f.write(f"| **Mean** | — | **{mean_loss:.4f}** | **{mean_auc:.4f}** "
                f"| **{mean_pcs:.4f}** | **{mean_ece:.4f}** |\n")

        f.write("\n## 2. 지표 설명\n")
        f.write("- **PCS (Physical Consistency Score)**: 좌우 반전 시 예측 일관성. "
                "1.0 에 가까울수록 배경 노이즈가 아닌 구조 형상으로 판단.\n")
        f.write("- **ECE (Expected Calibration Error)**: 예측 확률과 실제 정확도의 "
                "괴리. 0.0 에 가까울수록 '근거 있는 자신감'.\n")
        f.write("- **GradCAM Consistency**: 원본/반전 이미지 Activation Map 의 "
                "Pearson r. 1.0 이면 동일 물리 영역에 주목.\n\n")

        f.write("## 3. Epoch별 추이 (Fold 1)\n")
        if fold_results:
            hist = fold_results[0].get("history", [])
            f.write("| Epoch | Train Loss | PCS Reg | Train AUC "
                    "| Dev Loss | Dev AUC | PCS | ECE |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for h in hist:
                f.write(f"| {h['epoch']} | {h['train_loss']:.4f} "
                        f"| {h.get('pcs_reg',0):.4f} | {h['train_auc']:.4f} "
                        f"| {h['dev_loss']:.4f} | {h['dev_auc']:.4f} "
                        f"| {h['pcs']:.4f} | {h['ece']:.4f} |\n")


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📐 Input size: {args.img_size}x{args.img_size}  "
          f"(weight_decay={args.weight_decay}, γ={args.focal_gamma}, "
          f"ls={args.label_smoothing})")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 전체 데이터 통합 ──────────────────────────────────────────
    full_df    = build_full_df(args.data_dir)
    pseudo_df  = load_pseudo_v2(args.data_dir) if args.use_pseudo_v2 else None
    if pseudo_df is not None and len(pseudo_df) > 0:
        print(f"🔖 Pseudo v2 samples: {len(pseudo_df)} "
              f"(threshold=0.01/0.99 필터)")

    labels_arr = np.array([0 if l == "stable" else 1
                           for l in full_df["label"].tolist()])

    # ── 5-Fold Stratified K-Fold ──────────────────────────────────
    skf          = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                   random_state=args.seed)
    fold_results = []
    fold_test_probs = []   # 각 fold 의 test 예측 (앙상블용)
    test_ids_ref    = None

    train_tf = get_train_transform(args.img_size)
    val_tf   = get_val_transform(args.img_size)

    # ── Test DataLoader (공통) ────────────────────────────────────
    from src.dataset import StructuralDataset
    test_dataset = StructuralDataset(
        data_dir=args.data_dir, split="test",
        transform=val_tf, img_size=args.img_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    for fold_idx, (tr_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(full_df)), labels_arr), start=1):

        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx}/{args.n_folds}  "
              f"| train={len(tr_idx)}  val={len(val_idx)}")
        print(f"{'='*70}")

        train_ds = KFoldStructuralDataset(
            args.data_dir, tr_idx.tolist(), full_df,
            is_train=True,  transform=train_tf,
            img_size=args.img_size, pseudo_df=pseudo_df,
        )
        val_ds = KFoldStructuralDataset(
            args.data_dir, val_idx.tolist(), full_df,
            is_train=False, transform=val_tf,
            img_size=args.img_size,
        )
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        # ── Model ────────────────────────────────────────────────
        model = TripleStreamConvNeXt(
            num_classes=2, pretrained=args.pretrained, dropout=args.dropout
        ).to(device)

        criterion = FocalLoss(
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
        pcs_fn = PhysicsConsistencyLoss(temperature=args.pcs_temperature)

        # AdamW weight_decay=0.05 (과적합 방지)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
        # CosineAnnealingWarmRestarts — Local Minima 탈출
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-6,
        )

        # ── Training Loop ────────────────────────────────────────
        PCS_BONUS = 0.05
        best_score        = float("inf")
        best_epoch        = 0
        best_loss_val     = float("inf")
        best_auc_val      = 0.0
        best_pcs_val      = 0.0
        best_ece_val      = 1.0
        epochs_no_improve = 0
        history           = []

        hdr = (f"{'Ep':>3} | {'TrLoss':>7} | {'PCSReg':>6} | "
               f"{'TrAUC':>6} | {'VLoss':>7} | {'VAUC':>6} | "
               f"{'PCS':>6} | {'ECE':>6} | {'Score':>7} | Time")
        print(hdr)
        print("-" * len(hdr))

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_auc, pcs_reg = train_one_epoch(
                model, train_loader, criterion, pcs_fn, optimizer, device, args)
            v_loss, v_acc, v_auc, v_pcs, v_ece, v_probs, v_lbls = evaluate(
                model, val_loader, criterion, device)
            scheduler.step(epoch - 1)   # CosineWarmRestarts 는 실제 step 수 기반

            score = v_loss - PCS_BONUS * v_pcs
            elapsed = time.time() - t0

            print(f"{epoch:3d} | {tr_loss:7.4f} | {pcs_reg:6.4f} | "
                  f"{tr_auc:6.4f} | {v_loss:7.4f} | {v_auc:6.4f} | "
                  f"{v_pcs:6.4f} | {v_ece:6.4f} | {score:7.4f} | {elapsed:.1f}s")

            history.append(dict(
                epoch=epoch, train_loss=tr_loss, pcs_reg=pcs_reg,
                train_auc=tr_auc, dev_loss=v_loss, dev_auc=v_auc,
                pcs=v_pcs, ece=v_ece,
            ))

            if score < best_score:
                best_score        = score
                best_loss_val     = v_loss
                best_auc_val      = v_auc
                best_pcs_val      = v_pcs
                best_ece_val      = v_ece
                best_epoch        = epoch
                epochs_no_improve = 0
                ckpt_path = os.path.join(args.save_dir,
                                         f"best_fold{fold_idx}.pth")
                torch.save(dict(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    dev_loss=v_loss, dev_auc=v_auc,
                    dev_pcs=v_pcs,  dev_ece=v_ece,
                    composite_score=score,
                    args=vars(args),
                ), ckpt_path)
                print(f"  ✅ Fold{fold_idx} best  "
                      f"Loss={v_loss:.4f} AUC={v_auc:.4f} "
                      f"PCS={v_pcs:.4f} ECE={v_ece:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"  ⏹️  Early stop @ epoch {epoch}")
                    break

        # ── Fold 종료: best model 로 test 예측 ──────────────────
        ckpt = torch.load(os.path.join(args.save_dir, f"best_fold{fold_idx}.pth"),
                          map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        # GradCAM Consistency (val set 첫 샘플)
        try:
            sample_front, sample_top, sample_diff, _ = val_ds[0]
            gc_score = compute_gradcam_consistency(
                model,
                sample_front.unsqueeze(0),
                sample_top.unsqueeze(0),
                sample_diff.unsqueeze(0),
                device,
            )
        except Exception:
            gc_score = float("nan")

        print(f"  🎨 GradCAM Consistency (val[0]): {gc_score:.4f}")

        fold_results.append(dict(
            fold=fold_idx,
            best_epoch=best_epoch,
            best_loss=best_loss_val,
            best_auc=best_auc_val,
            best_pcs=best_pcs_val,
            best_ece=best_ece_val,
            gradcam_consistency=gc_score,
            history=history,
        ))

        t_ids, t_probs = predict_loader(model, test_loader, device)
        fold_test_probs.append(t_probs)
        if test_ids_ref is None:
            test_ids_ref = t_ids

        print(f"  Fold {fold_idx} complete ─ "
              f"Best epoch={best_epoch}  Loss={best_loss_val:.4f}  "
              f"AUC={best_auc_val:.4f}  ECE={best_ece_val:.4f}")

    # ── 5-Fold Soft-Voting Ensemble ───────────────────────────────
    print("\n🗳️  Soft-Voting Ensemble (5 folds)...")
    ensemble_probs = np.mean(fold_test_probs, axis=0)   # (N, 2)

    submission = pd.DataFrame({
        "id":            test_ids_ref,
        "unstable_prob": ensemble_probs[:, 1],
        "stable_prob":   ensemble_probs[:, 0],
    })
    out_csv = os.path.join(os.getcwd(), "submission.csv")
    submission.to_csv(out_csv, index=False)
    print(f"💾 Submission saved: {out_csv}")

    # 예측 통계
    u_prob = ensemble_probs[:, 1]
    print(f"   Avg unstable_prob : {u_prob.mean():.4f}")
    print(f"   Unstable (≥0.5)   : {(u_prob >= 0.5).sum()}")
    print(f"   Stable   (<0.5)   : {(u_prob < 0.5).sum()}")

    # ── Report ───────────────────────────────────────────────────
    report_path = os.path.join(os.getcwd(), args.report_file)
    generate_report(fold_results, report_path)
    print(f"📊 Report saved: {report_path}")

    # Fold 평균 요약
    print("\n" + "="*50)
    print("📈 5-Fold CV Summary")
    print(f"   Mean Dev Loss : {np.mean([r['best_loss'] for r in fold_results]):.4f}")
    print(f"   Mean Dev AUC  : {np.mean([r['best_auc']  for r in fold_results]):.4f}")
    print(f"   Mean PCS      : {np.mean([r['best_pcs']  for r in fold_results]):.4f}")
    print(f"   Mean ECE      : {np.mean([r['best_ece']  for r in fold_results]):.4f}")
    print("✨ Done!")


if __name__ == "__main__":
    main()
