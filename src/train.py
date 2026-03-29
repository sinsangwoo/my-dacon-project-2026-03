"""
src/train.py
============
v8 — Dual-Stream EfficientNet-B0 학습 스크립트

핵심 변경:
  1. dataset.py / model.py 변경사항 반영 (front, top만 사용)
  2. diff 스트림 완전 제거
  3. CPU 환경 최적화
  4. 1 epoch 30분 이내를 목표로 간결화
  5. train+dev 통합 5-Fold CV 유지
  6. Temperature Scaling 저장
  7. submission.csv 자동 생성 제거 (predict.py / ensemble.py로 분리)
"""

import argparse
import os 
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.swa_utils import AveragedModel
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
    DualStreamEfficientNet,
    FocalLoss,
    PhysicsConsistencyLoss,
    SAM,
    TemperatureScaler,
    compute_ece,
    compute_gradcam_consistency,
    count_parameters,
)


# ──────────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Structural Stability — Dual-Stream EfficientNet-B0 v8")

    # 데이터
    p.add_argument("--data_dir",      default="data")
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--num_workers",   type=int, default=min(4, os.cpu_count() or 0))
    p.add_argument("--n_folds",       type=int, default=5)
    p.add_argument("--use_pseudo_v2", action="store_true", default=False)

    # 학습
    p.add_argument("--epochs",        type=int, default=20)
    p.add_argument("--batch_size",    type=int, default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=0.03)
    p.add_argument("--patience",      type=int, default=5)

    # 모델
    p.add_argument("--pretrained",    action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.add_argument("--dropout",       type=float, default=0.30)

    # Loss
    p.add_argument("--focal_gamma",      type=float, default=1.5)
    p.add_argument("--label_smoothing",  type=float, default=0.05)
    p.add_argument("--pcs_lambda",       type=float, default=0.05)
    p.add_argument("--pcs_temperature",  type=float, default=2.0)

    # SAM
    p.add_argument("--use_sam",   action="store_true", default=False)
    p.add_argument("--sam_rho",   type=float, default=0.05)

    # SWA
    p.add_argument("--swa_epochs", type=int, default=0)

    # 증강
    p.add_argument("--mixup_alpha", type=float, default=0.2)

    # 출력
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--report_file", default="report.md")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--model_v",     type=str, default="v8_dualstream_b0")
    p.add_argument("--fold_idx",    type=int, default=None,
                   help="Specific fold to train (1-based). If None, all folds.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Utils
# ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def mixup(x1, x2, y, alpha, dev):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x1.size(0)).to(dev)
    return (
        lam * x1 + (1 - lam) * x1[idx],
        lam * x2 + (1 - lam) * x2[idx],
        y, y[idx], lam
    )


def update_bn_custom(loader, model, device=None):
    """DualStream 대응 BN 업데이트."""
    model.train()
    with torch.no_grad():
        for front, top, _ in loader:
            front = front.to(device, memory_format=torch.channels_last)
            top = top.to(device, memory_format=torch.channels_last)
            model(front, top)


# ──────────────────────────────────────────────────────────────────
#  Train / Eval
# ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, pcs_fn, optimizer, scheduler,
                    device, args, use_sam=False):
    model.train()

    run_loss = 0.0
    run_pcs = 0.0
    total = 0
    correct = 0.0
    all_lbl, all_prob = [], []

    for front, top, labels in loader:
        front = front.to(device, memory_format=torch.channels_last)
        top = top.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)

        mixed = False
        if args.mixup_alpha > 0 and np.random.rand() < 0.5:
            front, top, ya, yb, lam = mixup(front, top, labels, args.mixup_alpha, device)
            mixed = True

        def _forward_loss():
            logits = model(front, top)
            cls = (
                lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                if mixed else criterion(logits, labels)
            )
            pcs = torch.tensor(0.0, device=device)
            if args.pcs_lambda > 0:
                pcs = pcs_fn(model, front, top, logits)
            return logits, cls + args.pcs_lambda * pcs, pcs

        if use_sam:
            optimizer.zero_grad()
            logits, loss, pcs_val = _forward_loss()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            _, loss2, _ = _forward_loss()
            loss2.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            logits, loss, pcs_val = _forward_loss()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = front.size(0)
        run_loss += loss.item() * bs
        run_pcs += pcs_val.item() * bs
        total += bs

        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        _, pred = logits.max(1)
        hard_labels = labels if labels.ndim == 1 else labels.max(1)[1]

        correct += (
            lam * pred.eq(ya).sum().item() + (1 - lam) * pred.eq(yb).sum().item()
            if mixed else pred.eq(hard_labels).sum().item()
        )
        all_lbl.extend(hard_labels.cpu().numpy())
        all_prob.extend(probs)

    auc = roc_auc_score(all_lbl, all_prob) if len(set(all_lbl)) > 1 else 0.5
    return run_loss / total, 100 * correct / total, auc, run_pcs / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    run_loss = total = correct = pcs_sum = pcs_n = 0.0
    all_lbl, all_prob = [], []

    for front, top, labels in loader:
        front = front.to(device, memory_format=torch.channels_last)
        top = top.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)

        logits = model(front, top)
        loss = criterion(logits, labels)

        run_loss += loss.item() * front.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        _, pred = logits.max(1)

        hard_labels = labels if labels.ndim == 1 else labels.max(1)[1]

        total += hard_labels.size(0)
        correct += pred.eq(hard_labels).sum().item()
        all_lbl.extend(hard_labels.cpu().numpy())
        all_prob.extend(probs)

        pf = torch.softmax(
            model(torch.flip(front, [3]), torch.flip(top, [3])),
            dim=1
        )[:, 1].cpu().numpy()
        pcs_sum += np.sum(1.0 - np.abs(probs - pf))
        pcs_n += len(hard_labels)

    auc = roc_auc_score(all_lbl, all_prob) if len(set(all_lbl)) > 1 else 0.5
    ece = compute_ece(np.array(all_prob), np.array(all_lbl))
    return (
        run_loss / total,
        100 * correct / total,
        auc,
        pcs_sum / pcs_n,
        ece,
        np.array(all_prob),
        np.array(all_lbl),
    )


# ──────────────────────────────────────────────────────────────────
#  Report
# ──────────────────────────────────────────────────────────────────

def generate_report(fold_results, report_path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Training Report (v8 — Dual-Stream EfficientNet-B0)\n\n")
        f.write("| Fold | Best Ep | Dev Loss | Dev AUC | PCS | ECE | T |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in fold_results:
            f.write(
                f"| {r['fold']} | {r['best_epoch']} | {r['best_loss']:.4f} "
                f"| {r['best_auc']:.4f} | {r['best_pcs']:.4f} "
                f"| {r['best_ece']:.4f} | {r['temperature']:.3f} |\n"
            )


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 72)
    print(f"🚀 Structural Stability Model {args.model_v}")
    print("=" * 72)
    print(f"🖥️  Device       : {device}")
    print(f"📐 img_size      : {args.img_size}")
    print(f"⚙️  SAM          : {args.use_sam} (rho={args.sam_rho})")
    print(f"📦 SWA epochs    : {args.swa_epochs}")
    print(f"🎯 FocalLoss     : gamma={args.focal_gamma}  ls={args.label_smoothing}")
    print("=" * 72)

    full_df = build_full_df(args.data_dir)
    pseudo_df = load_pseudo_v2(args.data_dir) if args.use_pseudo_v2 else None
    if pseudo_df is not None and len(pseudo_df) > 0:
        print(f"🔖 Pseudo v2: {len(pseudo_df)} samples")

    labels_arr = np.array([0 if l == "stable" else 1 for l in full_df["label"].tolist()])

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    train_tf = get_train_transform(args.img_size)
    val_tf = get_val_transform(args.img_size)

    fold_results = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(full_df)), labels_arr), start=1
    ):
        if args.fold_idx is not None and fold_idx != args.fold_idx:
            continue

        print(f"\n{'=' * 72}")
        print(f"FOLD {fold_idx}/{args.n_folds}  train={len(tr_idx)}  val={len(val_idx)}")
        print(f"{'=' * 72}")

        train_ds = KFoldStructuralDataset(
            args.data_dir, tr_idx.tolist(), full_df,
            is_train=True, transform=train_tf, img_size=args.img_size,
            pseudo_df=pseudo_df,
        )
        val_ds = KFoldStructuralDataset(
            args.data_dir, val_idx.tolist(), full_df,
            is_train=False, transform=val_tf, img_size=args.img_size,
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda")
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda")
        )

        model = DualStreamEfficientNet(
            num_classes=2,
            pretrained=args.pretrained,
            dropout=args.dropout,
        ).to(device)
        model = model.to(memory_format=torch.channels_last)
        print(f"   Params: {count_parameters(model):,}")

        criterion = FocalLoss(
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
        pcs_fn = PhysicsConsistencyLoss(temperature=args.pcs_temperature)

        base_opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        if args.use_sam:
            optimizer = SAM(model.parameters(), base_opt, rho=args.sam_rho)
            print("   Optimizer: SAM(AdamW)")
        else:
            optimizer = base_opt
            print("   Optimizer: AdamW")

        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            base_opt,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy="cos",
        )

        swa_model = AveragedModel(model) if args.swa_epochs > 0 else None
        swa_start = args.epochs - args.swa_epochs + 1 if args.swa_epochs > 0 else 10**9

        best_score = float("inf")
        best_epoch = 0
        best_loss_val = float("inf")
        best_auc_val = 0.0
        best_pcs_val = 0.0
        best_ece_val = 1.0
        epochs_no_improve = 0

        hdr = (f"{'Ep':>3} | {'TrLoss':>7} | {'TrAUC':>6} | "
               f"{'VLoss':>7} | {'VAUC':>6} | {'PCS':>6} | {'ECE':>6} | {'Score':>7} | Time")
        print(hdr)
        print("-" * len(hdr))

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc, tr_auc, pcs_reg = train_one_epoch(
                model, train_loader, criterion, pcs_fn,
                optimizer, scheduler, device, args,
                use_sam=args.use_sam
            )
            v_loss, v_acc, v_auc, v_pcs, v_ece, v_probs, v_lbls = evaluate(
                model, val_loader, criterion, device
            )

            if swa_model is not None and epoch >= swa_start:
                swa_model.update_parameters(model)

            score = v_loss - 0.05 * v_pcs
            elapsed = time.time() - t0

            print(
                f"{epoch:3d} | {tr_loss:7.4f} | {tr_auc:6.4f} | {v_loss:7.4f} | "
                f"{v_auc:6.4f} | {v_pcs:6.4f} | {v_ece:6.4f} | {score:7.4f} | {elapsed:.1f}s"
            )

            if score < best_score:
                best_score = score
                best_loss_val = v_loss
                best_auc_val = v_auc
                best_pcs_val = v_pcs
                best_ece_val = v_ece
                best_epoch = epoch
                epochs_no_improve = 0

                ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "dev_loss": v_loss,
                    "dev_auc": v_auc,
                    "dev_pcs": v_pcs,
                    "dev_ece": v_ece,
                    "composite_score": score,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  ✅ best updated: Loss={v_loss:.4f} AUC={v_auc:.4f} PCS={v_pcs:.4f} ECE={v_ece:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"  ⏹️  Early stop @ epoch {epoch}")
                    break

        # SWA 저장
        if swa_model is not None:
            print("   📦 Updating SWA BatchNorm statistics...")
            try:
                update_bn_custom(train_loader, swa_model, device=device)
                _, _, _, _, swa_ece, _, _ = evaluate(swa_model, val_loader, criterion, device)
                swa_path = os.path.join(args.save_dir, f"best_fold{fold_idx}_swa.pth")
                torch.save({
                    "model_state_dict": swa_model.module.state_dict(),
                    "dev_ece": swa_ece,
                    "args": vars(args),
                }, swa_path)
                print(f"   💾 SWA saved → {swa_path}  ECE={swa_ece:.4f}")
            except Exception as e:
                print(f"   ⚠️  SWA failed: {e}")

        # Best ckpt reload
        ckpt = torch.load(
            os.path.join(args.save_dir, f"best_fold{fold_idx}.pth"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        # Temperature Scaling
        scaler = TemperatureScaler(model).to(device)
        try:
            T_val = scaler.fit(val_loader, device)
        except Exception as e:
            print(f"   ⚠️  Temperature scaling failed: {e}")
            T_val = 1.0

        ckpt["temperature"] = T_val
        torch.save(ckpt, os.path.join(args.save_dir, f"best_fold{fold_idx}.pth"))

        # GradCAM Consistency
        try:
            sf, st, _ = val_ds[0]
            gc_score = compute_gradcam_consistency(
                model,
                sf.unsqueeze(0), st.unsqueeze(0),
                device,
            )
        except Exception:
            gc_score = float("nan")

        print(f"   🎨 GradCAM Consistency: {gc_score:.4f}")

        fold_results.append({
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "best_loss": best_loss_val,
            "best_auc": best_auc_val,
            "best_pcs": best_pcs_val,
            "best_ece": best_ece_val,
            "temperature": T_val,
            "gradcam_consistency": gc_score,
        })

        print(
            f"  Fold {fold_idx} done — epoch={best_epoch} "
            f"Loss={best_loss_val:.4f} AUC={best_auc_val:.4f} "
            f"ECE={best_ece_val:.4f} T={T_val:.3f}"
        )

    report_path = os.path.join(os.getcwd(), args.report_file)
    generate_report(fold_results, report_path)
    print(f"\n📊 Report saved: {report_path}")

    print("\n" + "=" * 55)
    print("📈 CV Summary")
    for k, label in [
        ("best_loss", "Mean Dev Loss"),
        ("best_auc", "Mean Dev AUC "),
        ("best_pcs", "Mean PCS     "),
        ("best_ece", "Mean ECE     "),
        ("temperature", "Mean Temp   "),
    ]:
        vals = [r.get(k, float("nan")) for r in fold_results]
        print(f"   {label}: {np.nanmean(vals):.4f}")
    print("✨ Done!")


if __name__ == "__main__":
    main()