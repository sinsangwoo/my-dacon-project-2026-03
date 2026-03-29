"""
src/train_lgbm.py
==================
DINOv2 / CLIP 추출 피처 기반 LightGBM 학습 + Optuna 하이퍼파라미터 최적화.
5-Fold Stratified CV → OOF 예측 + Test 예측 → submission 생성.

사용법:
    python src/train_lgbm.py
    python src/train_lgbm.py --feature dinov2
    python src/train_lgbm.py --feature clip
    python src/train_lgbm.py --feature all --n_trials 100
    python src/train_lgbm.py --no_optuna   # 기본 파라미터로 빠르게

의존성:
    pip install lightgbm optuna
"""

import argparse
import os
import sys
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extract_features import load_features


# ──────────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--feature_dir", default="data/features")
    p.add_argument("--feature",     default="all", choices=["dinov2", "clip", "all"])
    p.add_argument("--output_csv",  default="submission_lgbm.csv")
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--n_folds",     type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--n_trials",    type=int, default=100,
                   help="Optuna 탐색 횟수 (0이면 기본 파라미터)")
    p.add_argument("--no_optuna",   action="store_true", default=False)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Default LightGBM Params
# ──────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_depth": -1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# ──────────────────────────────────────────────────────────────────
#  Optuna Objective
# ──────────────────────────────────────────────────────────────────

def optuna_objective(trial, X_train, y_train, n_folds=5, seed=42):
    """Optuna trial → 5-Fold CV LogLoss."""
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": seed,

        "num_leaves":       trial.suggest_int("num_leaves", 8, 128),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "min_split_gain":   trial.suggest_float("min_split_gain", 0.0, 1.0),
    }

    n_rounds = trial.suggest_int("n_rounds", 100, 2000)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y_train))

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    oof_preds = np.clip(oof_preds, 1e-15, 1.0 - 1e-15)
    score = log_loss(y_train, oof_preds)

    return score


# ──────────────────────────────────────────────────────────────────
#  Train & Predict
# ──────────────────────────────────────────────────────────────────

def train_and_predict(X_train, y_train, X_test, params, n_rounds=1000,
                      n_folds=5, seed=42):
    """5-Fold 학습 → OOF + Test 앙상블 예측 반환."""
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(X_test))
    models = []
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds

        fold_ll = log_loss(y_val, np.clip(val_pred, 1e-15, 1 - 1e-15))
        fold_auc = roc_auc_score(y_val, val_pred) if len(set(y_val)) > 1 else 0.5
        fold_scores.append(fold_ll)
        models.append(model)

        print(f"  Fold {fold_idx}: LogLoss={fold_ll:.5f}  AUC={fold_auc:.4f}  "
              f"best_iter={model.best_iteration}")

    oof_preds = np.clip(oof_preds, 1e-15, 1.0 - 1e-15)
    test_preds = np.clip(test_preds, 1e-15, 1.0 - 1e-15)

    oof_ll = log_loss(y_train, oof_preds)
    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n  📊 OOF LogLoss: {oof_ll:.5f}  |  OOF AUC: {oof_auc:.4f}")
    print(f"     Fold LogLoss: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")

    return oof_preds, test_preds, models


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("🚀 LightGBM Training Pipeline")
    print(f"   feature:     {args.feature}")
    print(f"   n_folds:     {args.n_folds}")
    print(f"   n_trials:    {args.n_trials}")
    print(f"   no_optuna:   {args.no_optuna}")

    # ── 1. 피처 로드 ──────────────────────────────────────────────
    print("\n📦 Loading features...")

    X_traindev, y_traindev, ids_traindev, splits_traindev = load_features(
        args.data_dir, args.feature_dir, args.feature, splits=["train", "dev"]
    )
    X_test, _, ids_test, _ = load_features(
        args.data_dir, args.feature_dir, args.feature, splits=["test"]
    )

    if len(X_traindev) == 0:
        print("❌ No features found. Run extract_features.py first.")
        return
    if len(X_test) == 0:
        print("❌ No test features found. Run extract_features.py first.")
        return

    # 라벨이 -1인 것 제거 (혹시 모를 안전장치)
    valid_mask = y_traindev >= 0
    X_train = X_traindev[valid_mask]
    y_train = y_traindev[valid_mask]
    ids_train = [ids_traindev[i] for i in range(len(ids_traindev)) if valid_mask[i]]

    print(f"   Train+Dev: {X_train.shape}  (stable={int((y_train==0).sum())}, unstable={int((y_train==1).sum())})")
    print(f"   Test:      {X_test.shape}")
    print(f"   Feature dim: {X_train.shape[1]}")

    # ── 2. Optuna 최적화 ──────────────────────────────────────────
    best_params = DEFAULT_PARAMS.copy()
    best_n_rounds = 1000

    if not args.no_optuna and args.n_trials > 0:
        print(f"\n🔍 Optuna Hyperparameter Search ({args.n_trials} trials)...")
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: optuna_objective(
                    trial, X_train, y_train, args.n_folds, args.seed
                ),
                n_trials=args.n_trials,
                show_progress_bar=True,
            )

            print(f"\n  🏆 Best Trial: #{study.best_trial.number}")
            print(f"     LogLoss: {study.best_value:.5f}")
            print(f"     Params:")
            for k, v in study.best_params.items():
                print(f"       {k}: {v}")

            # best params 적용
            best_n_rounds = study.best_params.pop("n_rounds", 1000)
            best_params.update(study.best_params)

            # 결과 저장
            os.makedirs(args.save_dir, exist_ok=True)
            optuna_path = os.path.join(args.save_dir, "optuna_lgbm_best.json")
            with open(optuna_path, "w") as f:
                json.dump({
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "n_rounds": best_n_rounds,
                }, f, indent=2)
            print(f"  💾 Best params saved: {optuna_path}")

        except ImportError:
            print("  ⚠️  optuna 미설치. 기본 파라미터 사용. (pip install optuna)")
    else:
        print("\n  ℹ️  Optuna 비활성. 기본 파라미터 사용.")

    # ── 3. 최종 학습 + 예측 ───────────────────────────────────────
    print(f"\n🏋️ Final Training with best params...")
    import lightgbm as lgb

    oof_preds, test_preds, models = train_and_predict(
        X_train, y_train, X_test,
        best_params, n_rounds=best_n_rounds,
        n_folds=args.n_folds, seed=args.seed,
    )

    # ── 4. OOF 저장 (stacking용) ─────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    oof_df = pd.DataFrame({
        "id": ids_train,
        "oof_pred": oof_preds,
        "label": y_train,
    })
    oof_path = os.path.join(args.save_dir, f"oof_lgbm_{args.feature}.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"  💾 OOF predictions saved: {oof_path}")

    # ── 5. Test 예측 저장 (앙상블용) ──────────────────────────────
    test_pred_df = pd.DataFrame({
        "id": ids_test,
        "unstable_prob": test_preds,
    })
    test_pred_path = os.path.join(args.save_dir, f"test_lgbm_{args.feature}.csv")
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"  💾 Test predictions saved: {test_pred_path}")

    # ── 6. Submission 생성 ────────────────────────────────────────
    print(f"\n📝 Generating submission...")

    sample_sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))

    # 클리핑
    test_preds_clipped = np.clip(test_preds, 0.005, 0.995)

    sub_df = pd.DataFrame({
        "id": ids_test,
        "unstable_prob": test_preds_clipped,
        "stable_prob": 1.0 - test_preds_clipped,
    })

    # sample_submission 순서에 맞춤
    sub_df = sub_df.set_index("id").reindex(sample_sub["id"]).reset_index()

    # NaN 안전장치
    if sub_df["unstable_prob"].isna().any():
        print("  ⚠️  NaN detected in predictions. Filling with 0.5.")
        sub_df["unstable_prob"] = sub_df["unstable_prob"].fillna(0.5)
        sub_df["stable_prob"] = 1.0 - sub_df["unstable_prob"]

    sub_df.to_csv(args.output_csv, index=False)
    print(f"  💾 Submission saved: {args.output_csv}")

    # ── 7. 분포 리포트 ────────────────────────────────────────────
    u = sub_df["unstable_prob"].values
    print(f"\n📊 Prediction Distribution:")
    print(f"   Min: {u.min():.5f}  Max: {u.max():.5f}  Mean: {u.mean():.5f}")
    print(f"   Unstable (≥0.5): {(u >= 0.5).sum()} / {len(u)}")
    print(f"   Stable   (<0.5): {(u < 0.5).sum()}")

    extreme = ((u < 0.1) | (u > 0.9)).mean() * 100
    middle = ((u >= 0.4) & (u <= 0.6)).mean() * 100
    print(f"   Extreme (0-0.1 or 0.9-1.0): {extreme:.1f}%")
    print(f"   Middle  (0.4-0.6):           {middle:.1f}%")

    # Feature Importance (Fold 1)
    if models:
        print(f"\n📊 Top 20 Feature Importance (Fold 1):")
        imp = models[0].feature_importance(importance_type="gain")
        top_idx = np.argsort(imp)[::-1][:20]
        for rank, idx in enumerate(top_idx, 1):
            print(f"   {rank:2d}. feature_{idx:4d}: {imp[idx]:.1f}")

    print("\n✨ Done! Next: python src/ensemble.py (or submit directly)")


if __name__ == "__main__":
    main()