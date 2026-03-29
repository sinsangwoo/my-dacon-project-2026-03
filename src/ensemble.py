"""
src/ensemble.py
================
다중 모델 예측 결합 → 최종 submission 생성.

결합 대상:
  1. LightGBM (DINOv2 피처) → checkpoints/test_lgbm_dinov2.csv
  2. LightGBM (CLIP 피처)   → checkpoints/test_lgbm_clip.csv
  3. LightGBM (all 피처)    → checkpoints/test_lgbm_all.csv
  4. EfficientNet Fold 1~5  → checkpoints/best_fold{k}.pth (기존 predict.py 출력)

존재하는 예측만 자동 탐지하여 앙상블.

사용법:
    python src/ensemble.py
    python src/ensemble.py --method geometric
    python src/ensemble.py --method weighted --dev_calibrate
    python src/ensemble.py --output_csv final_submission.csv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default="data")
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--output_csv",    default="final_submission.csv")
    p.add_argument("--method",        default="geometric",
                   choices=["arithmetic", "geometric", "weighted", "rank"])
    p.add_argument("--dev_calibrate", action="store_true", default=False,
                   help="Dev set 기반 최적 가중치/T 탐색")
    p.add_argument("--clip_min",      type=float, default=0.005)
    p.add_argument("--clip_max",      type=float, default=0.995)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  예측 파일 수집
# ──────────────────────────────────────────────────────────────────

def collect_predictions(data_dir, save_dir):
    """존재하는 모든 예측 CSV/결과를 탐지하여 {name: DataFrame} 반환."""
    preds = {}

    # LightGBM 예측들
    for feat_name in ["dinov2", "clip", "all"]:
        path = os.path.join(save_dir, f"test_lgbm_{feat_name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "unstable_prob" in df.columns and "id" in df.columns:
                preds[f"lgbm_{feat_name}"] = df[["id", "unstable_prob"]]
                print(f"  ✅ {f'lgbm_{feat_name}':<20s} → {path}")

    # EfficientNet predict.py 출력
    for name, path in [
        ("effnet_ensemble", "submission.csv"),
        ("effnet_ensemble2", os.path.join(save_dir, "submission_effnet.csv")),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "unstable_prob" in df.columns and "id" in df.columns:
                preds[name] = df[["id", "unstable_prob"]]
                print(f"  ✅ {name:<20s} → {path}")

    return preds


# ──────────────────────────────────────────────────────────────────
#  OOF 수집 (dev calibration용)
# ──────────────────────────────────────────────────────────────────

def collect_oof(save_dir):
    """OOF 예측 CSV들을 {name: DataFrame} 반환."""
    oofs = {}
    for feat_name in ["dinov2", "clip", "all"]:
        path = os.path.join(save_dir, f"oof_lgbm_{feat_name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "oof_pred" in df.columns:
                oofs[f"lgbm_{feat_name}"] = df
    return oofs


# ──────────────────────────────────────────────────────────────────
#  앙상블 방법들
# ──────────────────────────────────────────────────────────────────

def arithmetic_mean(probs_list):
    return np.mean(probs_list, axis=0)


def geometric_mean(probs_list):
    log_probs = np.log(np.array(probs_list) + 1e-15)
    mean_log = np.mean(log_probs, axis=0)
    geo = np.exp(mean_log)
    return geo


def rank_average(probs_list):
    """각 모델의 예측을 순위로 변환 후 평균 → 재정규화."""
    from scipy.stats import rankdata
    n = len(probs_list[0])
    ranked = []
    for p in probs_list:
        r = rankdata(p) / n  # 0~1 정규화 순위
        ranked.append(r)
    return np.mean(ranked, axis=0)


def weighted_mean(probs_list, weights):
    weights = np.array(weights) / np.sum(weights)
    result = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        result += w * p
    return result


# ──────────────────────────────────────────────────────────────────
#  Dev 기반 최적 가중치 탐색
# ──────────────────────────────────────────────────────────────────

def find_optimal_weights(oofs, data_dir, n_models):
    """OOF 예측을 기반으로 grid search → 최적 가중치 반환."""
    from sklearn.metrics import log_loss as sk_log_loss

    # dev.csv 라벨 로드
    dev_df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    label_map = {str(row["id"]): (1 if row["label"] == "unstable" else 0)
                 for _, row in dev_df.iterrows()}

    # OOF에서 dev 샘플만 추출
    oof_names = list(oofs.keys())
    if not oof_names:
        print("  ⚠️  OOF 없음. 균등 가중치 사용.")
        return [1.0] * n_models

    # 공통 ID 기반 정렬
    common_ids = None
    oof_arrays = {}
    for name in oof_names:
        df = oofs[name]
        ids = [str(x) for x in df["id"]]
        if common_ids is None:
            common_ids = set(ids)
        else:
            common_ids &= set(ids)

    if not common_ids:
        return [1.0] * n_models

    # dev에 속하는 것만
    dev_ids = [sid for sid in common_ids if sid in label_map]
    if len(dev_ids) < 10:
        print(f"  ⚠️  Dev overlap too small ({len(dev_ids)}). 균등 가중치.")
        return [1.0] * n_models

    dev_ids_sorted = sorted(dev_ids)
    y_dev = np.array([label_map[sid] for sid in dev_ids_sorted])

    oof_preds = []
    for name in oof_names:
        df = oofs[name]
        df["id"] = df["id"].astype(str)
        df = df.set_index("id")
        preds = np.array([df.loc[sid, "oof_pred"] for sid in dev_ids_sorted])
        oof_preds.append(preds)

    # Grid Search (simplex 탐색 — 2~4 모델이면 충분)
    best_ll = float("inf")
    best_w = [1.0] * len(oof_preds)
    steps = 11

    if len(oof_preds) == 1:
        return [1.0]

    elif len(oof_preds) == 2:
        for w1 in np.linspace(0, 1, steps):
            w2 = 1.0 - w1
            blend = w1 * oof_preds[0] + w2 * oof_preds[1]
            blend = np.clip(blend, 1e-15, 1 - 1e-15)
            ll = sk_log_loss(y_dev, blend)
            if ll < best_ll:
                best_ll = ll
                best_w = [w1, w2]

    elif len(oof_preds) == 3:
        for w1 in np.linspace(0, 1, steps):
            for w2 in np.linspace(0, 1 - w1, steps):
                w3 = 1.0 - w1 - w2
                if w3 < 0:
                    continue
                blend = w1 * oof_preds[0] + w2 * oof_preds[1] + w3 * oof_preds[2]
                blend = np.clip(blend, 1e-15, 1 - 1e-15)
                ll = sk_log_loss(y_dev, blend)
                if ll < best_ll:
                    best_ll = ll
                    best_w = [w1, w2, w3]
    else:
        # 4개 이상이면 균등 가중치로 fallback
        best_w = [1.0 / len(oof_preds)] * len(oof_preds)

    print(f"  🏆 Optimal weights (dev LogLoss={best_ll:.5f}):")
    for name, w in zip(oof_names, best_w):
        print(f"     {name}: {w:.3f}")

    # n_models와 oof 수가 다를 수 있으므로 패딩
    while len(best_w) < n_models:
        best_w.append(1.0 / n_models)

    return best_w[:n_models]


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("🚀 Ensemble Pipeline")
    print(f"   method:    {args.method}")
    print(f"   calibrate: {args.dev_calibrate}")
    print(f"   clip:      [{args.clip_min}, {args.clip_max}]")

    # 1. 예측 수집
    print(f"\n📦 Collecting predictions...")
    preds = collect_predictions(args.data_dir, args.save_dir)

    if not preds:
        print("❌ No prediction files found.")
        print("   Run: python src/train_lgbm.py  and/or  python src/predict.py")
        return

    print(f"\n  Total models: {len(preds)}")

    # 2. 공통 ID 정렬
    sample_sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    target_ids = list(sample_sub["id"].astype(str))

    probs_list = []
    model_names = []

    for name, df in preds.items():
        df = df.copy()
        df["id"] = df["id"].astype(str)
        df = df.set_index("id")

        # target_ids 순서로 reindex
        if set(target_ids).issubset(set(df.index)):
            ordered = df.loc[target_ids, "unstable_prob"].values
        else:
            ordered = df.reindex(target_ids)["unstable_prob"].values
            missing = np.isnan(ordered).sum()
            if missing > 0:
                print(f"  ⚠️  {name}: {missing} missing IDs → fill 0.5")
                ordered = np.nan_to_num(ordered, nan=0.5)

        probs_list.append(ordered.astype(np.float64))
        model_names.append(name)

    probs_array = np.array(probs_list)  # (n_models, n_samples)
    print(f"  Probs shape: {probs_array.shape}")

    # 3. 앙상블
    print(f"\n🗳️  Ensembling with method: {args.method}")

    if args.method == "arithmetic":
        unstable_prob = arithmetic_mean(probs_array)

    elif args.method == "geometric":
        unstable_prob = geometric_mean(probs_array)

    elif args.method == "rank":
        try:
            unstable_prob = rank_average(probs_array)
        except ImportError:
            print("  ⚠️  scipy 필요. arithmetic 으로 fallback.")
            unstable_prob = arithmetic_mean(probs_array)

    elif args.method == "weighted":
        if args.dev_calibrate:
            oofs = collect_oof(args.save_dir)
            weights = find_optimal_weights(oofs, args.data_dir, len(probs_list))
        else:
            weights = [1.0] * len(probs_list)
        unstable_prob = weighted_mean(probs_array, weights)

    else:
        unstable_prob = arithmetic_mean(probs_array)

    # 4. 클리핑 + 정규화
    unstable_prob = np.clip(unstable_prob, args.clip_min, args.clip_max)
    stable_prob = 1.0 - unstable_prob

    # 5. Submission 저장
    submission = pd.DataFrame({
        "id": target_ids,
        "unstable_prob": unstable_prob,
        "stable_prob": stable_prob,
    })
    submission.to_csv(args.output_csv, index=False)
    print(f"\n💾 Final submission: {args.output_csv}")

    # 6. 분포 리포트
    u = unstable_prob
    print(f"\n📊 Final Distribution:")
    print(f"   Min: {u.min():.5f}  Max: {u.max():.5f}  Mean: {u.mean():.5f}")
    print(f"   Unstable (≥0.5): {(u >= 0.5).sum()} / {len(u)}")
    print(f"   Stable   (<0.5): {(u < 0.5).sum()}")

    extreme = ((u < 0.1) | (u > 0.9)).mean() * 100
    middle = ((u >= 0.4) & (u <= 0.6)).mean() * 100
    print(f"   Extreme: {extreme:.1f}%  |  Middle: {middle:.1f}%")

    # 7. 모델별 상관관계
    if len(probs_list) > 1:
        print(f"\n📊 Model Correlation Matrix (Pearson):")
        corr = np.corrcoef(probs_array)
        header = "           " + "  ".join(f"{n[:8]:>8s}" for n in model_names)
        print(header)
        for i, name in enumerate(model_names):
            row = f"  {name[:8]:<8s}  " + "  ".join(f"{corr[i,j]:8.4f}" for j in range(len(model_names)))
            print(row)

        # 다양성 점수 (평균 상관이 낮을수록 앙상블 효과↑)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        avg_corr = corr[mask].mean()
        print(f"\n   Avg pairwise correlation: {avg_corr:.4f}")
        if avg_corr < 0.8:
            print("   ✅ Good diversity!")
        elif avg_corr < 0.95:
            print("   ⚠️  Moderate diversity. More diverse models would help.")
        else:
            print("   ❗ High correlation. Ensemble benefit is limited.")

    # 8. Sum check
    sums = unstable_prob + stable_prob
    print(f"\n   Sum==1.0 check: {np.allclose(sums, 1.0)}")

    print("\n✨ Done!")


if __name__ == "__main__":
    main()