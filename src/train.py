"""
src/train.py
============
v9 — Dual-Stream EfficientNet-B0 학습 스크립트 (Enhanced)

핵심 변경 (v8 → v9):
  1. [NEW] 체크포인트 Resume 기능 — 매 에폭 끝에 last_fold{N}.pth 저장,
     --resume 옵션으로 중단 지점부터 재개
  2. [NEW] tqdm 실시간 진행률 표시 — 배치별 Loss/LR/Acc 모니터링
  3. [NEW] KeyboardInterrupt 안전 종료 — Ctrl+C 시 현재 상태 저장 후 종료
  4. [NEW] 시스템 정보 / ETA / 메모리 사용량 로깅
  5. flush=True 적용 — 파이프라인 로그 파일에 실시간 반영
"""

import argparse
import json
import os
import platform
import signal
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

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
#  Logging Helpers
# ──────────────────────────────────────────────────────────────────

def log(msg: str = ""):
    """print with flush and tqdm compatibility."""
    if "tqdm" in sys.modules:
        from tqdm import tqdm
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def log_separator(char="=", width=72):
    log(char * width)


def format_time(seconds: float) -> str:
    """초를 사람이 읽기 쉬운 형태로 변환."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"


def get_system_info(device) -> dict:
    """시스템 정보를 수집."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
    return info


def get_memory_usage() -> str:
    """현재 메모리 사용량 문자열 반환."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1024 / 1024
        return f"{mem_mb:.0f}MB"
    except ImportError:
        return "N/A"


def get_gpu_memory() -> str:
    """GPU 메모리 사용량 문자열 반환."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return f"GPU: {alloc:.0f}MB / {reserved:.0f}MB reserved"
    return ""


# ──────────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Structural Stability — Dual-Stream EfficientNet-B0 v9")

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
    p.add_argument("--model_v",     type=str, default="v9_dualstream_b0")
    p.add_argument("--fold_idx",    type=int, default=None,
                   help="Specific fold to train (1-based). If None, all folds.")

    # [NEW] Resume & Logging
    p.add_argument("--resume",      action="store_true", default=False,
                   help="Resume training from last checkpoint.")
    p.add_argument("--print_freq",  type=int, default=10,
                   help="Batch-level log frequency (every N batches).")
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
#  Checkpoint: Save / Load
# ──────────────────────────────────────────────────────────────────

def save_last_checkpoint(path, model, optimizer, scheduler, epoch,
                         best_score, best_epoch, best_loss_val, best_auc_val,
                         best_pcs_val, best_ece_val, epochs_no_improve, args):
    """매 에폭 끝에 전체 학습 상태를 저장."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_loss_val": best_loss_val,
        "best_auc_val": best_auc_val,
        "best_pcs_val": best_pcs_val,
        "best_ece_val": best_ece_val,
        "epochs_no_improve": epochs_no_improve,
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(state, path)


def load_last_checkpoint(path, model, optimizer, scheduler, device):
    """last checkpoint에서 학습 상태 복원."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    # 모델 불러오기 (v8->v9 등 structure 변화 가능성 대비 strict=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    # Optimizer는 strict하게 불러오기 (파라미터 변동 감지용)
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as e:
        log(f"   ⚠️  Optimizer load failed: {e}. Starting fresh optimizer.")

    # Scheduler는 OneCycleLR처럼 TotalSteps에 민감한 경우 에러 발생 가능
    if scheduler and ckpt.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            log(f"   ⚠️  Scheduler load failed (possibly total_steps changed): {e}.\n"
                f"      Using fresh scheduler for the remaining epochs.")
    return ckpt


# ──────────────────────────────────────────────────────────────────
#  Train / Eval  (with tqdm)
# ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, pcs_fn, optimizer, scheduler,
                    device, args, epoch, total_epochs, use_sam=False):
    model.train()

    run_loss = 0.0
    run_pcs = 0.0
    total = 0
    correct = 0.0
    all_lbl, all_prob = [], []
    batch_times = []

    n_batches = len(loader)

    # TTY 여부에 따라 tqdm 조절
    is_tty = sys.stdout.isatty()
    pbar = tqdm(
        loader,
        desc=f"  Train Ep {epoch}/{total_epochs}",
        leave=False,
        disable=not is_tty,
        bar_format="{l_bar}{bar:20}{r_bar}",
        ncols=100,
    )

    for batch_idx, (front, top, labels) in enumerate(pbar):
        batch_t0 = time.time()

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

        batch_times.append(time.time() - batch_t0)

        # tqdm 진행률 바 업데이트
        current_lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else 0
        avg_loss = run_loss / total
        avg_acc = 100 * correct / total
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.1f}%",
            "lr": f"{current_lr:.2e}",
            "batch": f"{time.time() - batch_t0:.1f}s",
        })

        # print_freq 마다 상세 로그 출력
        if (batch_idx + 1) % args.print_freq == 0:
            eta_batch = np.mean(batch_times[-args.print_freq:]) * (n_batches - batch_idx - 1)
            mem_str = get_memory_usage()
            gpu_str = get_gpu_memory()
            log(
                f"    [{batch_idx+1}/{n_batches}] "
                f"Loss={avg_loss:.4f}  Acc={avg_acc:.1f}%  "
                f"LR={current_lr:.2e}  PCS={run_pcs/total:.4f}  "
                f"ETA={format_time(eta_batch)}  "
                f"RAM={mem_str}  {gpu_str}"
            )

    pbar.close()

    auc = roc_auc_score(all_lbl, all_prob) if len(set(all_lbl)) > 1 else 0.5
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    return run_loss / total, 100 * correct / total, auc, run_pcs / total, avg_batch_time


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=0, total_epochs=0):
    model.eval()
    run_loss = total = correct = pcs_sum = pcs_n = 0.0
    all_lbl, all_prob = [], []

    is_tty = sys.stdout.isatty()
    pbar = tqdm(
        loader,
        desc=f"  Val   Ep {epoch}/{total_epochs}",
        leave=False,
        disable=not is_tty,
        bar_format="{l_bar}{bar:20}{r_bar}",
        ncols=100,
    )

    for front, top, labels in pbar:
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

        # tqdm 업데이트
        pbar.set_postfix({
            "loss": f"{run_loss / total:.4f}",
            "acc": f"{100 * correct / total:.1f}%",
        })

    pbar.close()

    auc = roc_auc_score(all_lbl, all_prob) if len(set(all_lbl)) > 1 else 0.5
    ece = compute_ece(np.array(all_prob), np.array(all_lbl))

    # LogLoss 계산 (제출 메트릭)
    try:
        ll = log_loss(all_lbl, np.clip(all_prob, 1e-15, 1 - 1e-15))
    except Exception:
        ll = float("nan")

    return (
        run_loss / total,
        100 * correct / total,
        auc,
        pcs_sum / pcs_n,
        ece,
        ll,
        np.array(all_prob),
        np.array(all_lbl),
    )


# ──────────────────────────────────────────────────────────────────
#  Report
# ──────────────────────────────────────────────────────────────────

def generate_report(fold_results, report_path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Training Report (v9 — Dual-Stream EfficientNet-B0)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Fold | Best Ep | Dev Loss | LogLoss | Dev AUC | PCS | ECE | T |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in fold_results:
            f.write(
                f"| {r['fold']} | {r['best_epoch']} | {r['best_loss']:.4f} "
                f"| {r.get('best_logloss', float('nan')):.4f} "
                f"| {r['best_auc']:.4f} | {r['best_pcs']:.4f} "
                f"| {r['best_ece']:.4f} | {r['temperature']:.3f} |\n"
            )

        # CV 요약
        f.write("\n## CV Summary\n\n")
        for k, label in [
            ("best_loss", "Mean Dev Loss"),
            ("best_logloss", "Mean LogLoss "),
            ("best_auc", "Mean Dev AUC "),
            ("best_pcs", "Mean PCS     "),
            ("best_ece", "Mean ECE     "),
            ("temperature", "Mean Temp    "),
        ]:
            vals = [r.get(k, float("nan")) for r in fold_results]
            f.write(f"- **{label}**: {np.nanmean(vals):.4f}\n")


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys_info = get_system_info(device)

    log_separator()
    log(f"🚀 Structural Stability Model {args.model_v}")
    log_separator()
    log(f"📅 Started       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"🖥️  Platform      : {sys_info['platform']}")
    log(f"🐍 Python         : {sys_info['python']}")
    log(f"🔥 PyTorch        : {sys_info['torch']}")
    log(f"🖥️  Device         : {device}")
    if device.type == "cuda":
        log(f"🎮 GPU            : {sys_info.get('gpu_name', 'N/A')}")
        log(f"💾 GPU Memory     : {sys_info.get('gpu_memory_total', 'N/A')}")
    log(f"📐 img_size       : {args.img_size}")
    log(f"📦 batch_size     : {args.batch_size}")
    log(f"🔄 epochs         : {args.epochs}")
    log(f"📊 n_folds        : {args.n_folds}")
    log(f"📈 lr             : {args.lr}")
    log(f"⚙️  SAM            : {args.use_sam} (rho={args.sam_rho})")
    log(f"📦 SWA epochs     : {args.swa_epochs}")
    log(f"🎯 FocalLoss      : gamma={args.focal_gamma}  ls={args.label_smoothing}")
    log(f"🔁 Resume         : {args.resume}")
    log(f"📝 Print freq     : every {args.print_freq} batches")
    log(f"💾 Save dir       : {args.save_dir}")
    log_separator()

    full_df = build_full_df(args.data_dir)
    pseudo_df = load_pseudo_v2(args.data_dir) if args.use_pseudo_v2 else None
    if pseudo_df is not None and len(pseudo_df) > 0:
        log(f"🔖 Pseudo v2: {len(pseudo_df)} samples")

    log(f"📊 Dataset: {len(full_df)} samples (train+dev)")

    labels_arr = np.array([0 if l == "stable" else 1 for l in full_df["label"].tolist()])
    n_stable = int((labels_arr == 0).sum())
    n_unstable = int((labels_arr == 1).sum())
    log(f"   Stable: {n_stable} | Unstable: {n_unstable} | Ratio: {n_unstable/len(labels_arr)*100:.1f}%")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    train_tf = get_train_transform(args.img_size)
    val_tf = get_val_transform(args.img_size)

    fold_results = []
    pipeline_start = time.time()

    for fold_idx, (tr_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(full_df)), labels_arr), start=1
    ):
        if args.fold_idx is not None and fold_idx != args.fold_idx:
            continue

        fold_start = time.time()
        log_separator()
        log(f"📂 FOLD {fold_idx}/{args.n_folds}  |  train={len(tr_idx)}  val={len(val_idx)}")
        log_separator()

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

        n_batches_train = len(train_loader)
        n_batches_val = len(val_loader)
        log(f"   Train batches: {n_batches_train} | Val batches: {n_batches_val}")

        model = DualStreamEfficientNet(
            num_classes=2,
            pretrained=args.pretrained,
            dropout=args.dropout,
        ).to(device)
        model = model.to(memory_format=torch.channels_last)
        log(f"   Params: {count_parameters(model):,}")

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
            log("   Optimizer: SAM(AdamW)")
        else:
            optimizer = base_opt
            log("   Optimizer: AdamW")

        total_steps = n_batches_train * args.epochs
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
        best_logloss_val = float("inf")
        epochs_no_improve = 0
        start_epoch = 1

        # ── Resume 로직 ──────────────────────────────────────────
        last_ckpt_path = os.path.join(args.save_dir, f"last_fold{fold_idx}.pth")
        if args.resume and os.path.exists(last_ckpt_path):
            log(f"\n   🔄 Resuming from {last_ckpt_path} ...")
            ckpt = load_last_checkpoint(last_ckpt_path, model, optimizer, scheduler, device)
            start_epoch = ckpt["epoch"] + 1
            best_score = ckpt.get("best_score", float("inf"))
            best_epoch = ckpt.get("best_epoch", 0)
            best_loss_val = ckpt.get("best_loss_val", float("inf"))
            best_auc_val = ckpt.get("best_auc_val", 0.0)
            best_pcs_val = ckpt.get("best_pcs_val", 0.0)
            best_ece_val = ckpt.get("best_ece_val", 1.0)
            best_logloss_val = ckpt.get("best_logloss_val", float("inf"))
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            # 하이퍼파라미터 변경 감지
            prev_args = ckpt.get("args", {})
            params_changed = False
            for k in ["epochs", "batch_size", "lr", "img_size"]:
                if prev_args.get(k) != getattr(args, k):
                    params_changed = True
                    log(f"   🚩 Warning: Parameter '{k}' changed ({prev_args.get(k)} -> {getattr(args, k)})")
            
            if params_changed:
                log("   ⚠️  Hyperparameters changed. Checkpoint metrics might be inconsistent.")

            if start_epoch > args.epochs:
                log(f"   ⏭️  Fold {fold_idx} already completed ({start_epoch-1} >= {args.epochs}). Skipping.")
                # 이미 완료된 fold의 결과를 fold_results에 추가
                best_ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
                if os.path.exists(best_ckpt_path):
                    bc = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                    fold_results.append({
                        "fold": fold_idx,
                        "best_epoch": bc.get("epoch", best_epoch),
                        "best_loss": bc.get("dev_loss", best_loss_val),
                        "best_logloss": bc.get("dev_logloss", best_logloss_val),
                        "best_auc": bc.get("dev_auc", best_auc_val),
                        "best_pcs": bc.get("dev_pcs", best_pcs_val),
                        "best_ece": bc.get("dev_ece", best_ece_val),
                        "temperature": bc.get("temperature", 1.0),
                        "gradcam_consistency": bc.get("gradcam_consistency", float("nan")),
                    })
                continue
        else:
            if args.resume:
                log(f"   ℹ️  No checkpoint found at {last_ckpt_path}. Starting from scratch.")

        log(f"\n   📋 Training epochs {start_epoch} → {args.epochs}")
        log("")

        # Table header
        hdr = (f"| {'Ep':^3} | {'TrLoss':^7} | {'TrAcc':^6} | {'VLoss':^7} | "
               f"{'VLL':^7} | {'VAUC':^6} | {'Score':^7} | {'LR':^8} | {'Time':^7} |")
        log("\n" + hdr)
        log("|-----|---------|--------|---------|---------|--------|---------|----------|---------|")

        epoch_times = []

        # ── 학습 루프 (with graceful interrupt) ───────────────────
        interrupted = False
        current_epoch = start_epoch

        try:
            for epoch in range(start_epoch, args.epochs + 1):
                current_epoch = epoch
                t0 = time.time()

                tr_loss, tr_acc, tr_auc, pcs_reg, avg_bt = train_one_epoch(
                    model, train_loader, criterion, pcs_fn,
                    optimizer, scheduler, device, args,
                    epoch=epoch, total_epochs=args.epochs,
                    use_sam=args.use_sam
                )
                v_loss, v_acc, v_auc, v_pcs, v_ece, v_ll, v_probs, v_lbls = evaluate(
                    model, val_loader, criterion, device,
                    epoch=epoch, total_epochs=args.epochs,
                )

                if swa_model is not None and epoch >= swa_start:
                    swa_model.update_parameters(model)

                score = v_loss - 0.05 * v_pcs
                elapsed = time.time() - t0
                epoch_times.append(elapsed)

                current_lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else 0
                mem_str = get_memory_usage()

                log(
                    f"| {epoch:3d} | {tr_loss:7.4f} | {tr_acc:5.1f}% | {v_loss:7.4f} | "
                    f"{v_ll:7.4f} | {v_auc:6.4f} | {score:7.4f} | {current_lr:8.1e} | {format_time(elapsed):>7s} |"
                )

                # ETA 계산
                remaining_epochs = args.epochs - epoch
                remaining_folds = args.n_folds - fold_idx
                if epoch_times:
                    avg_epoch_time = np.mean(epoch_times)
                    eta_fold = avg_epoch_time * remaining_epochs
                    eta_total = eta_fold + avg_epoch_time * args.epochs * remaining_folds
                    if remaining_epochs > 0:
                        log(f"      ⏱️  ETA this fold: {format_time(eta_fold)} | ETA total: {format_time(eta_total)}")

                if score < best_score:
                    best_score = score
                    best_loss_val = v_loss
                    best_auc_val = v_auc
                    best_pcs_val = v_pcs
                    best_ece_val = v_ece
                    best_logloss_val = v_ll
                    best_epoch = epoch
                    epochs_no_improve = 0

                    ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "dev_loss": v_loss,
                        "dev_logloss": v_ll,
                        "dev_auc": v_auc,
                        "dev_pcs": v_pcs,
                        "dev_ece": v_ece,
                        "composite_score": score,
                        "args": vars(args),
                    }, ckpt_path)
                    log(f"      ✅ BEST updated: Loss={v_loss:.4f} LogLoss={v_ll:.4f} AUC={v_auc:.4f} PCS={v_pcs:.4f} ECE={v_ece:.4f}")
                else:
                    epochs_no_improve += 1
                    log(f"      ⏳ No improvement ({epochs_no_improve}/{args.patience})")
                    if epochs_no_improve >= args.patience:
                        log(f"      ⏹️  Early stop @ epoch {epoch}")
                        break

                # 매 에폭 끝에 last checkpoint 저장
                save_last_checkpoint(
                    last_ckpt_path, model, optimizer, scheduler, epoch,
                    best_score, best_epoch, best_loss_val, best_auc_val,
                    best_pcs_val, best_ece_val, epochs_no_improve, args
                )
                log(f"      💾 Checkpoint saved: {last_ckpt_path}")

        except KeyboardInterrupt:
            interrupted = True
            log(f"\n   ⚠️  KeyboardInterrupt detected at epoch {current_epoch}!")
            log(f"   💾 Saving emergency checkpoint...")
            save_last_checkpoint(
                last_ckpt_path, model, optimizer, scheduler, current_epoch,
                best_score, best_epoch, best_loss_val, best_auc_val,
                best_pcs_val, best_ece_val, epochs_no_improve, args
            )
            log(f"   ✅ Emergency checkpoint saved: {last_ckpt_path}")
            log(f"   💡 Re-run with --resume to continue from epoch {current_epoch + 1}")

        if interrupted:
            # fold_results에 현재까지의 최선 결과 기록
            fold_results.append({
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "best_loss": best_loss_val,
                "best_logloss": best_logloss_val,
                "best_auc": best_auc_val,
                "best_pcs": best_pcs_val,
                "best_ece": best_ece_val,
                "temperature": 1.0,
                "gradcam_consistency": float("nan"),
                "status": "interrupted",
            })
            break  # 전체 fold 루프 탈출

        # ── Fold 완료 후 후처리 ───────────────────────────────────

        # SWA 저장
        if swa_model is not None:
            log("   📦 Updating SWA BatchNorm statistics...")
            try:
                update_bn_custom(train_loader, swa_model, device=device)
                _, _, _, _, swa_ece, _, _, _ = evaluate(swa_model, val_loader, criterion, device)
                swa_path = os.path.join(args.save_dir, f"best_fold{fold_idx}_swa.pth")
                torch.save({
                    "model_state_dict": swa_model.module.state_dict(),
                    "dev_ece": swa_ece,
                    "args": vars(args),
                }, swa_path)
                log(f"   💾 SWA saved: {swa_path}  ECE={swa_ece:.4f}")
            except Exception as e:
                log(f"   ⚠️  SWA failed: {e}")

        # Best ckpt reload
        best_ckpt_path = os.path.join(args.save_dir, f"best_fold{fold_idx}.pth")
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            log(f"   ⚠️  No best checkpoint found for fold {fold_idx}. Using last model state.")
            ckpt = {}

        # Temperature Scaling
        scaler = TemperatureScaler(model).to(device)
        try:
            T_val = scaler.fit(val_loader, device)
        except Exception as e:
            log(f"   ⚠️  Temperature scaling failed: {e}")
            T_val = 1.0

        ckpt["temperature"] = T_val
        if os.path.exists(best_ckpt_path):
            torch.save(ckpt, best_ckpt_path)

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

        log(f"   🎨 GradCAM Consistency: {gc_score:.4f}")

        fold_elapsed = time.time() - fold_start
        fold_results.append({
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "best_loss": best_loss_val,
            "best_logloss": best_logloss_val,
            "best_auc": best_auc_val,
            "best_pcs": best_pcs_val,
            "best_ece": best_ece_val,
            "temperature": T_val,
            "gradcam_consistency": gc_score,
            "fold_time": fold_elapsed,
        })

        log_separator("-")
        log(
            f"  ✅ Fold {fold_idx} done in {format_time(fold_elapsed)} — "
            f"best_epoch={best_epoch}  Loss={best_loss_val:.4f}  LogLoss={best_logloss_val:.4f}  "
            f"AUC={best_auc_val:.4f}  ECE={best_ece_val:.4f}  T={T_val:.3f}"
        )
        log_separator("-")

        # last checkpoint 정리 (fold가 완전히 끝났으므로)
        # → 유지: 재실행 시 skip 로직에 필요

    # ── 전체 요약 ─────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start

    report_path = os.path.join(os.getcwd(), args.report_file)
    generate_report(fold_results, report_path)
    log(f"\n📊 Report saved: {report_path}")

    log_separator()
    log("📈 CV Summary")
    log_separator()
    for k, label in [
        ("best_loss", "Mean Dev Loss  "),
        ("best_logloss", "Mean LogLoss   "),
        ("best_auc", "Mean Dev AUC   "),
        ("best_pcs", "Mean PCS       "),
        ("best_ece", "Mean ECE       "),
        ("temperature", "Mean Temp      "),
    ]:
        vals = [r.get(k, float("nan")) for r in fold_results]
        log(f"   {label}: {np.nanmean(vals):.4f}")

    # Fold별 요약 테이블
    log("")
    log("   Fold Results:")
    for r in fold_results:
        status = r.get("status", "complete")
        ft = format_time(r.get("fold_time", 0))
        log(
            f"   Fold {r['fold']} | Ep {r['best_epoch']:2d} | "
            f"Loss {r['best_loss']:.4f} | LL {r.get('best_logloss', float('nan')):.4f} | "
            f"AUC {r['best_auc']:.4f} | T {r['temperature']:.3f} | "
            f"Time {ft} | {status}"
        )

    log(f"\n⏱️  Total time: {format_time(total_elapsed)}")
    log(f"📅 Finished    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("✨ Done!")


if __name__ == "__main__":
    main()