"""
Dacon 구조 안정성 분류 — 학습 스크립트 (v6)
==============================================
주요 변경 (v6):
  - Backbone: EfficientNet-B1 (CPU 15분/epoch 목표)
  - SAM Optimizer (Flat Minima)
  - SWA: 마지막 swa_epochs 에포크 가중치 평균 → best_fold{k}_swa.pth
  - Temperature Scaling: fold 종료 후 val NLL 최소화로 T 탐색
  - 체크포인트 두 버전 저장: best ECE 기준 + SWA 적용
  - label_smoothing=0.1, focal_gamma=2.0

패치 (v6 → v6.1):
  [PATCH-1] cutmix(): in-place 수정 전 patch 영역을 clone()으로 먼저 추출
  [PATCH-2] train_one_epoch(): mixed=True 덮어쓰기 버그 수정
  [PATCH-3] Temperature Scaling torch.load(): weights_only=False 로 통일

사용법:
    python src/train.py --data_dir data --epochs 30 --batch_size 16
    python src/train.py --use_sam               # SAM optimizer 활성
    python src/train.py --swa_epochs 5          # SWA: 마지막 5 에포크
    python src/train.py --use_pseudo_v2         # Pseudo v2 병합
"""

import argparse
import copy
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
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    KFoldStructuralDataset,
    StructuralDataset,
    build_full_df,
    load_pseudo_v2,
    get_train_transform,
    get_val_transform,
)
from src.model import (
    TripleStreamEfficientNet,
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
    p = argparse.ArgumentParser(
        description="Structural Stability — Triple-Stream EfficientNet-B1 v6"
    )
    # 데이터
    p.add_argument("--img_size",      type=int,   default=224,
                   help="EfficientNet-B1 추천 규격 224 (속도 최적화)")
    p.add_argument("--num_workers",   type=int,   default=os.cpu_count() or 0)
    p.add_argument("--n_folds",       type=int,   default=5)
    p.add_argument("--use_pseudo_v2", action="store_true", default=False)

    # 학습
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=0.05)

    # 모델
    p.add_argument("--pretrained",    action="store_true",  default=True)
    p.add_argument("--no_pretrained", dest="pretrained",    action="store_false")
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--stoch_depth_p", type=float, default=0.1,
                   help="Stochastic Depth drop 확률 (0이면 비활성)")

    # Loss
    p.add_argument("--focal_gamma",      type=float, default=2.0)
    p.add_argument("--label_smoothing",  type=float, default=0.1)
    p.add_argument("--pcs_lambda",       type=float, default=0.1)
    p.add_argument("--pcs_temperature",  type=float, default=2.0)

    # SAM
    p.add_argument("--use_sam",   action="store_true", default=False,
                   help="SAM Optimizer 활성화 (Flat Minima, 느리지만 일반화↑)")
    p.add_argument("--sam_rho",   type=float, default=0.05,
                   help="SAM perturbation 반경 ρ")

    # SWA
    p.add_argument("--swa_epochs", type=int, default=5,
                   help="마지막 N 에포크 가중치를 SWA 로 평균. 0이면 비활성.")

    # 증강
    p.add_argument("--mixup_alpha",  type=float, default=0.2)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--cutmix_prob",  type=float, default=0.5)

    # 출력
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--report_file", default="report.md")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--patience",    type=int,   default=7)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
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
    return (
        lam*x1+(1-lam)*x1[idx], lam*x2+(1-lam)*x2[idx],
        lam*x3+(1-lam)*x3[idx], y, y[idx], lam
    )


def cutmix(x1, x2, x3, y, alpha, dev):
    lam         = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx         = torch.randperm(x1.size(0)).to(dev)
    b1,b2,b3,b4 = rand_bbox(x1.size(), lam)
    # [PATCH-1] in-place 수정 전에 교체할 영역을 clone()으로 먼저 추출.
    #   수정 전: xk[idx,:,b1:b3,b2:b4] 를 읽을 때 xk가 이미 오염되어
    #            idx[i]==i 인 원소에서 잘못된 값을 복사하는 통계 편향 발생.
    #   수정 후: patch를 미리 clone하여 in-place 수정 이전 값만 사용.
    for xk in (x1, x2, x3):
        patch = xk[idx, :, b1:b3, b2:b4].clone()
        xk[:, :, b1:b3, b2:b4] = patch
    lam = 1 - (b3-b1)*(b4-b2) / (x1.size(-1)*x1.size(-2))
    return x1, x2, x3, y, y[idx], lam


# ──────────────────────────────────────────────────────────────────
#  Train one epoch
# ──────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, pcs_fn, optimizer,
    device, args, use_sam: bool = False
):
    """1 에포크 학습.

    SAM 사용 시 first_step / second_step 2-pass.
    그 외엔 일반 backward.
    """
    model.train()
    run_loss = run_pcs = total = correct = 0.0
    all_lbl, all_prob = [], []
    use_aug = (args.mixup_alpha > 0 or args.cutmix_alpha > 0)

    for front, top, diff, labels in loader:
        front, top, diff, labels = (
            front.to(device), top.to(device),
            diff.to(device),  labels.to(device)
        )

        # ── CutMix / Mixup ──
        # [PATCH-2] mixed=True 덮어쓰기 버그 수정.
        #   수정 전: else 브랜치에서 mixed=False 를 세팅해도
        #            블록 마지막의 mixed=True 가 항상 덮어써서
        #            ya/yb 미정의 상태에서 mixed=True 가 되는 NameError 발생 가능.
        #   수정 후: mixed=True 를 성공적 augmentation 분기 내부로 이동.
        mixed = False
        if use_aug and np.random.rand() < args.cutmix_prob:
            if np.random.rand() < 0.5 and args.mixup_alpha > 0:
                front, top, diff, ya, yb, lam = mixup(
                    front, top, diff, labels, args.mixup_alpha, device)
                mixed = True
            elif args.cutmix_alpha > 0:
                front, top, diff, ya, yb, lam = cutmix(
                    front, top, diff, labels, args.cutmix_alpha, device)
                mixed = True
            # else: 두 alpha 모두 0 이면 mixed=False 유지

        def _forward_loss():
            """logits 계산 + total loss 반환. SAM 양쪽 pass 공유."""
            logits = model(front, top, diff)
            cls = (
                lam * criterion(logits, ya) + (1-lam) * criterion(logits, yb)
                if mixed else criterion(logits, labels)
            )
            pcs = torch.tensor(0.0, device=device)
            if args.pcs_lambda > 0:
                pcs = pcs_fn(model, front, top, diff, logits)
            return logits, cls + args.pcs_lambda * pcs, pcs

        if use_sam:
            # SAM first step
            logits, loss, pcs_val = _forward_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # SAM second step (perturbed 위치에서 gradient)
            _, loss2, _ = _forward_loss()
            loss2.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            logits, loss, pcs_val = _forward_loss()
            loss.backward()
            optimizer.step()

        bs = front.size(0)
        run_loss += loss.item() * bs
        run_pcs  += pcs_val.item() * bs
        total    += bs

        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        _, pred = logits.max(1)
        correct += (
            lam*pred.eq(ya).sum().item() + (1-lam)*pred.eq(yb).sum().item()
            if mixed else pred.eq(labels).sum().item()
        )
        all_lbl.extend(labels.cpu().numpy())
        all_prob.extend(probs)

    auc = roc_auc_score(all_lbl, all_prob) if len(set(all_lbl)) > 1 else 0.5
    return run_loss/total, 100*correct/total, auc, run_pcs/total


# ──────────────────────────────────────────────────────────────────
#  Evaluate
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    run_loss = total = correct = pcs_sum = pcs_n = 0.0
    all_lbl, all_prob = [], []

    for front, top, diff, labels in loader:
        front, top, diff, labels = (
            front.to(device), top.to(device),
            diff.to(device),  labels.to(device)
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
        pf = torch.softmax(
            model(torch.flip(front,[3]),
                  torch.flip(top,  [3]),
                  torch.flip(diff, [3])),
            dim=1
        )[:, 1].cpu().numpy()
        pcs_sum += np.sum(1.0 - np.abs(probs - pf))
        pcs_n   += len(labels)

    auc = roc_auc_score(all_lbl, all_prob) if len(set(all_lbl)) > 1 else 0.5
    ece = compute_ece(np.array(all_prob), np.array(all_lbl))
    return (
        run_loss/total, 100*correct/total,
        auc, pcs_sum/pcs_n, ece,
        np.array(all_prob), np.array(all_lbl)
    )


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
        f.write("# Training Report (v6.1 — EfficientNet-B1 / SWA / SAM)\n\n")

        f.write("## 1. Fold 요약\n")
        f.write("| Fold | Best Ep | Dev Loss | Dev AUC | PCS | ECE | T(temp) | SWA ECE |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in fold_results:
            f.write(
                f"| {r['fold']} | {r['best_epoch']} "
                f"| {r['best_loss']:.4f} | {r['best_auc']:.4f} "
                f"| {r['best_pcs']:.4f} | {r['best_ece']:.4f} "
                f"| {r.get('temperature', float('nan')):.3f} "
                f"| {r.get('swa_ece', float('nan')):.4f} |\n"
            )

        means = {
            k: np.nanmean([r.get(k, float('nan')) for r in fold_results])
            for k in ('best_loss','best_auc','best_pcs','best_ece','temperature','swa_ece')
        }
        f.write(
            f"| **Mean** | — "
            f"| **{means['best_loss']:.4f}** | **{means['best_auc']:.4f}** "
            f"| **{means['best_pcs']:.4f}** | **{means['best_ece']:.4f}** "
            f"| **{means['temperature']:.3f}** "
            f"| **{means['swa_ece']:.4f}** |\n"
        )

        f.write("\n## 2. 기법 설명\n")
        f.write("- **SWA**: 마지막 N 에포크 가중치 평균 → 더 평탄한 Loss 지형 → 일반화↑\n")
        f.write("- **SAM**: 2-pass gradient ascent/descent → Flat Minima 수렴\n")
        f.write("- **Temperature Scaling**: 학습 후 val NLL 최소화로 T 탐색 → ECE 교정\n")
        f.write("- **Stochastic Depth**: 레이어 무작위 생략 → 소규모 데이터 암기 방지\n")
        f.write("- **Rank Averaging**: 확률 순위 기반 앙상블 → LogLoss 폭탄 방어\n\n")

        f.write("## 3. Epoch 추이 (Fold 1)\n")
        if fold_results:
            hist = fold_results[0].get("history", [])
            f.write("| Ep | TrLoss | PCSReg | TrAUC | VLoss | VAUC | PCS | ECE |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for h in hist:
                f.write(
                    f"| {h['epoch']} | {h['train_loss']:.4f} "
                    f"| {h.get('pcs_reg',0):.4f} | {h['train_auc']:.4f} "
                    f"| {h['dev_loss']:.4f} | {h['dev_auc']:.4f} "
                    f"| {h['pcs']:.4f} | {h['ece']:.4f} |\n"
                )


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device      : {device}")
    print(f"📐 img_size     : {args.img_size}")
    print(f"⚙️  SAM         : {args.use_sam}  (ρ={args.sam_rho})")
    print(f"📦 SWA epochs   : {args.swa_epochs}")
    print(f"🎯 FocalLoss    : γ={args.focal_gamma}  ls={args.label_smoothing}")
    print(f"💧 StochDepth   : {args.stoch_depth_p}")

    os.makedirs(args.save_dir, exist_ok=True)

    full_df   = build_full_df(args.data_dir)
    pseudo_df = load_pseudo_v2(args.data_dir) if args.use_pseudo_v2 else None
    if pseudo_df is not None and len(pseudo_df) > 0:
        print(f"🔖 Pseudo v2   : {len(pseudo_df)} samples")

    labels_arr = np.array(
        [0 if l == "stable" else 1 for l in full_df["label"].tolist()]
    )

    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )
    train_tf = get_train_transform(args.img_size)
    val_tf   = get_val_transform(args.img_size)

    # Test DataLoader (공통)
    test_dataset = StructuralDataset(
        data_dir=args.data_dir, split="test",
        transform=val_tf, img_size=args.img_size,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    fold_results    = []
    fold_test_probs = []   # best-ECE 버전
    fold_swa_probs  = []   # SWA 버전
    test_ids_ref    = None

    for fold_idx, (tr_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(full_df)), labels_arr), start=1
    ):
        print(f"\n{'='*72}")
        print(f"  FOLD {fold_idx}/{args.n_folds}  "
              f"train={len(tr_idx)}  val={len(val_idx)}")
        print(f"{'='*72}")

        train_ds = KFoldStructuralDataset(
            args.data_dir, tr_idx.tolist(), full_df,
            is_train=True, transform=train_tf,
            img_size=args.img_size, pseudo_df=pseudo_df,
        )
        val_ds = KFoldStructuralDataset(
            args.data_dir, val_idx.tolist(), full_df,
            is_train=False, transform=val_tf,
            img_size=args.img_size,
        )
        train_loader = DataLoader(
            train_ds, args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

        # ── Model ──
        model = TripleStreamEfficientNet(
            num_classes=2,
            pretrained=args.pretrained,
            dropout=args.dropout,
            stoch_depth_p=args.stoch_depth_p,
        ).to(device)
        print(f"   Params: {count_parameters(model):,}")

        criterion = FocalLoss(
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
        pcs_fn = PhysicsConsistencyLoss(temperature=args.pcs_temperature)

        # ── Optimizer ──
        base_opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        if args.use_sam:
            optimizer = SAM(
                model.parameters(), base_opt, rho=args.sam_rho
            )
            print("   Optimizer: SAM(AdamW)")
        else:
            optimizer = base_opt
            print("   Optimizer: AdamW")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_opt, T_0=10, T_mult=1, eta_min=1e-6
        )

        # ── SWA 모델 준비 ──
        swa_model   = AveragedModel(model) if args.swa_epochs > 0 else None
        swa_start   = args.epochs - args.swa_epochs + 1

        # ── Training Loop ──
        PCS_BONUS         = 0.05
        best_score        = float("inf")
        best_epoch        = 0
        best_loss_val     = float("inf")
        best_auc_val      = 0.0
        best_pcs_val      = 0.0
        best_ece_val      = 1.0
        epochs_no_improve = 0
        history           = []

        hdr = (f"{'Ep':>3} | {'TrLoss':>7} | {'PCSRG':>6} | "
               f"{'TrAUC':>6} | {'VLoss':>7} | {'VAUC':>6} | "
               f"{'PCS':>6} | {'ECE':>6} | {'Score':>7} | Time")
        print(hdr)
        print("-" * len(hdr))

        try:
            for epoch in range(1, args.epochs + 1):
                t0 = time.time()

                tr_loss, tr_acc, tr_auc, pcs_reg = train_one_epoch(
                    model, train_loader, criterion, pcs_fn,
                    optimizer, device, args, use_sam=args.use_sam
                )
                v_loss, v_acc, v_auc, v_pcs, v_ece, v_probs, v_lbls = evaluate(
                    model, val_loader, criterion, device
                )
                scheduler.step(epoch - 1)

                # SWA 가중치 수집
                if swa_model is not None and epoch >= swa_start:
                    swa_model.update_parameters(model)

                score   = v_loss - PCS_BONUS * v_pcs
                elapsed = time.time() - t0

                print(
                    f"{epoch:3d} | {tr_loss:7.4f} | {pcs_reg:6.4f} | "
                    f"{tr_auc:6.4f} | {v_loss:7.4f} | {v_auc:6.4f} | "
                    f"{v_pcs:6.4f} | {v_ece:6.4f} | {score:7.4f} | {elapsed:.1f}s"
                )

                history.append(dict(
                    epoch=epoch, train_loss=tr_loss, pcs_reg=pcs_reg,
                    train_auc=tr_auc, dev_loss=v_loss, dev_auc=v_auc,
                    pcs=v_pcs, ece=v_ece,
                ))

                # best ECE 기준 저장
                if score < best_score:
                    best_score        = score
                    best_loss_val     = v_loss
                    best_auc_val      = v_auc
                    best_pcs_val      = v_pcs
                    best_ece_val      = v_ece
                    best_epoch        = epoch
                    epochs_no_improve = 0
                    ckpt_path = os.path.join(
                        args.save_dir, f"best_fold{fold_idx}.pth"
                    )
                    torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        dev_loss=v_loss, dev_auc=v_auc,
                        dev_pcs=v_pcs,  dev_ece=v_ece,
                        composite_score=score,
                        args=vars(args),
                    ), ckpt_path)
                    print(f"  ✅ best  Loss={v_loss:.4f} AUC={v_auc:.4f} "
                          f"PCS={v_pcs:.4f} ECE={v_ece:.4f}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.patience:
                        print(f"  ⏹️  Early stop @ epoch {epoch}")
                        break
        except KeyboardInterrupt:
            print("\n  ⚠️  Training interrupted by user. Saving current SWA if available...")

        # ── SWA BN 업데이트 및 저장 ──────────────────────────────
        swa_ece = float("nan")
        if swa_model is not None:
            print("   📦 Updating SWA BatchNorm statistics...")
            try:
                update_bn(train_loader, swa_model, device=device)
            except Exception as e:
                print(f"   ⚠️  update_bn failed: {e}")

            # SWA 모델 평가
            _, _, _, _, swa_ece, _, _ = evaluate(
                swa_model, val_loader, criterion, device
            )
            swa_path = os.path.join(
                args.save_dir, f"best_fold{fold_idx}_swa.pth"
            )
            torch.save(dict(
                model_state_dict=swa_model.module.state_dict(),
                dev_ece=swa_ece,
                args=vars(args),
            ), swa_path)
            print(f"   💾 SWA saved → {swa_path}  ECE={swa_ece:.4f}")

        # ── Temperature Scaling ───────────────────────────────────
        # [PATCH-3] weights_only=False 로 통일 (predict.py 와 동일).
        #   args=vars(args) 를 포함한 체크포인트는 PyTorch 2.x weights_only=True
        #   모드에서 UnpicklingError 를 유발할 수 있어 폴드가 중단되는 문제 방지.
        ckpt = torch.load(
            os.path.join(args.save_dir, f"best_fold{fold_idx}.pth"),
            map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])

        scaler = TemperatureScaler(model).to(device)
        try:
            T_val = scaler.fit(val_loader, device)
        except Exception as e:
            print(f"   ⚠️  Temperature scaling failed: {e}")
            T_val = 1.0

        # Temperature 값을 체크포인트에 추가 저장
        ckpt["temperature"] = T_val
        torch.save(ckpt, os.path.join(
            args.save_dir, f"best_fold{fold_idx}.pth"
        ))

        # ── GradCAM Consistency ───────────────────────────────────
        try:
            sf, st, sd, _ = val_ds[0]
            gc_score = compute_gradcam_consistency(
                model,
                sf.unsqueeze(0), st.unsqueeze(0), sd.unsqueeze(0),
                device,
            )
        except Exception:
            gc_score = float("nan")
        print(f"   🎨 GradCAM Consistency: {gc_score:.4f}")

        # ── Test 예측 (best-ECE 버전) ─────────────────────────────
        t_ids, t_probs = predict_loader(model, test_loader, device)
        fold_test_probs.append(t_probs)
        if test_ids_ref is None:
            test_ids_ref = t_ids

        # ── Test 예측 (SWA 버전) ──────────────────────────────────
        if swa_model is not None:
            _, swa_probs = predict_loader(swa_model, test_loader, device)
            fold_swa_probs.append(swa_probs)

        fold_results.append(dict(
            fold=fold_idx,
            best_epoch=best_epoch,
            best_loss=best_loss_val,
            best_auc=best_auc_val,
            best_pcs=best_pcs_val,
            best_ece=best_ece_val,
            temperature=T_val,
            swa_ece=swa_ece,
            gradcam_consistency=gc_score,
            history=history,
        ))
        print(
            f"  Fold {fold_idx} done — "
            f"epoch={best_epoch} Loss={best_loss_val:.4f} "
            f"AUC={best_auc_val:.4f} ECE={best_ece_val:.4f} "
            f"T={T_val:.3f}"
        )

    # ── Rank Averaging Ensemble ───────────────────────────────────
    def ensemble_mean(probs_list):
        """Calibrated Mean Ensemble — LogLoss 에 더 적합함.
        """
        mean_probs = np.mean(probs_list, axis=0)
        # LogLoss 방어: 극단적인 확률값 클리핑 (1e-15 ~ 1-1e-15)
        mean_probs = np.clip(mean_probs, 1e-15, 1.0 - 1e-15)
        # 합계 1로 재정규화
        mean_probs = mean_probs / mean_probs.sum(axis=1, keepdims=True)
        return mean_probs

    # best-ECE + SWA 모두 합산
    all_probs_for_ensemble = fold_test_probs.copy()
    if fold_swa_probs:
        all_probs_for_ensemble.extend(fold_swa_probs)
        print(f"   Using {len(fold_test_probs)} best-ECE + "
              f"{len(fold_swa_probs)} SWA = "
              f"{len(all_probs_for_ensemble)} total")

    ensemble_probs = ensemble_mean(all_probs_for_ensemble)

    submission = pd.DataFrame({
        "id":            test_ids_ref,
        "unstable_prob": ensemble_probs[:, 1],
        "stable_prob":   ensemble_probs[:, 0],
    })

    # 원본 sample_submission.csv 순서와 100% 일치하도록 정렬 강제
    sample_sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    submission = submission.set_index("id").reindex(sample_sub["id"]).reset_index()

    out_csv = os.path.join(os.getcwd(), "submission.csv")
    submission.to_csv(out_csv, index=False)
    print(f"💾 Submission saved: {out_csv}")

    u = ensemble_probs[:, 1]
    print(f"   Avg unstable_prob : {u.mean():.4f}")
    print(f"   Unstable (≥0.5)   : {(u >= 0.5).sum()}")
    print(f"   Stable   (<0.5)   : {(u < 0.5).sum()}")

    # ── Report ───────────────────────────────────────────────────
    report_path = os.path.join(os.getcwd(), args.report_file)
    generate_report(fold_results, report_path)
    print(f"📊 Report saved: {report_path}")

    print("\n" + "="*55)
    print("📈 5-Fold CV Summary")
    for k, label in [
        ("best_loss", "Mean Dev Loss"),
        ("best_auc",  "Mean Dev AUC "),
        ("best_pcs",  "Mean PCS     "),
        ("best_ece",  "Mean ECE     "),
        ("temperature","Mean Temp   "),
        ("swa_ece",   "Mean SWA ECE "),
    ]:
        vals = [r.get(k, float('nan')) for r in fold_results]
        print(f"   {label}: {np.nanmean(vals):.4f}")
    print("✨ Done!")


if __name__ == "__main__":
    main()
