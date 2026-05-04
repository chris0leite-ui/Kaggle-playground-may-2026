"""D3-Optuna — LGBM hyperparameter sweep with two-anchor evaluation.

Optuna search over LGBM hyperparams; objective = OOF AUC on Strat
anchor (5-fold). Hard 1h timeout (Rule 2 single-fold variant: each
trial gets 1 fold first; if first-fold projection >1h skip trial).
Best params re-fit two-anchor 5-fold; emit OOF + submission.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
BASE_S, BASE_G = 0.94075, 0.92059
N_TRIALS = 30
TIMEOUT_S = 3300  # 55 min hard wall


def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    return train, test, y, X, X_test, cat_cols


def fit_5fold(params, X, y, X_test, cat_cols, splits, num_round=2000, es=100):
    oof = np.zeros(len(y), dtype=np.float32)
    tp = np.zeros(len(X_test), dtype=np.float32)
    fold_aucs = []
    for tr, va in splits:
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=num_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(es), lgb.log_evaluation(0)])
        p = m.predict(X.iloc[va])
        oof[va] = p
        tp += m.predict(X_test) / N_FOLDS
        fold_aucs.append(roc_auc_score(y[va], p))
    return oof, tp, float(roc_auc_score(y, oof)), fold_aucs


def main():
    train, test, y, X, X_test, cat_cols = load_data()
    print(f"loaded: train={X.shape}, test={X_test.shape}, cats={cat_cols}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y)), y))

    def objective(trial):
        params = dict(
            objective="binary", metric="auc", verbose=-1, seed=SEED,
            learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_data_in_leaf=trial.suggest_int("min_data_leaf", 50, 500),
            feature_fraction=trial.suggest_float("feature_frac", 0.6, 1.0),
            bagging_fraction=trial.suggest_float("bagging_frac", 0.6, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 0, 10),
            lambda_l1=trial.suggest_float("l1", 1e-8, 10, log=True),
            lambda_l2=trial.suggest_float("l2", 1e-8, 10, log=True),
            max_depth=trial.suggest_int("max_depth", -1, 12),
        )
        # 1-fold quick eval (fold 0)
        tr, va = splits_a[0]
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        p = m.predict(X.iloc[va])
        return roc_auc_score(y[va], p)

    print(f"\n=== Optuna sweep: {N_TRIALS} trials, {TIMEOUT_S}s timeout ===")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_S, show_progress_bar=False)
    print(f"\noptuna done: {len(study.trials)} trials in {time.time()-t0:.0f}s")
    print(f"best 1-fold AUC: {study.best_value:.5f}")
    print(f"best params: {study.best_params}")

    # Refit 5-fold both-anchor with best params
    best_params = dict(
        objective="binary", metric="auc", verbose=-1, seed=SEED,
        learning_rate=study.best_params["lr"],
        num_leaves=study.best_params["num_leaves"],
        min_data_in_leaf=study.best_params["min_data_leaf"],
        feature_fraction=study.best_params["feature_frac"],
        bagging_fraction=study.best_params["bagging_frac"],
        bagging_freq=study.best_params["bagging_freq"],
        lambda_l1=study.best_params["l1"],
        lambda_l2=study.best_params["l2"],
        max_depth=study.best_params["max_depth"],
    )
    print("\n=== 5-fold Strat with best params ===")
    oof_a, test_a, auc_a, fa_a = fit_5fold(best_params, X, y, X_test, cat_cols, splits_a)
    print(f"Strat OOF: {auc_a:.5f}  Δ baseline: {(auc_a-BASE_S)*1e4:+.1f}bp")

    print("\n=== 5-fold GroupKF(Race) ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, train["Race"].values))
    oof_b, test_b, auc_b, fa_b = fit_5fold(best_params, X, y, X_test, cat_cols, splits_b)
    print(f"GroupKF OOF: {auc_b:.5f}  Δ baseline: {(auc_b-BASE_G)*1e4:+.1f}bp")

    save_oof("e5_optuna_lgbm_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_scores=fa_a, cv="StratKF",
                  delta_vs_baseline_bp=(auc_a-BASE_S)*1e4,
                  best_params=study.best_params, n_trials=len(study.trials)))
    save_oof("e5_optuna_lgbm_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_scores=fa_b, cv="GroupKF(Race)",
                  delta_vs_baseline_bp=(auc_b-BASE_G)*1e4,
                  best_params=study.best_params))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_a
    sub.to_csv("submissions/submission_e5_optuna_lgbm.csv", index=False)

    body = (
        f"# E5 — Optuna-tuned LGBM (D3 prep)\n\n"
        f"{len(study.trials)} trials in {time.time()-t0:.0f}s. Best 1-fold AUC: "
        f"{study.best_value:.5f}.\nBest params: {study.best_params}\n\n"
        f"## 5-fold both-anchor\n\n"
        f"| anchor | OOF AUC | Δ baseline |\n|---|---:|---:|\n"
        f"| Strat | **{auc_a:.5f}** | {(auc_a-BASE_S)*1e4:+.1f}bp |\n"
        f"| GroupKF | **{auc_b:.5f}** | {(auc_b-BASE_G)*1e4:+.1f}bp |\n"
    )
    Path("audit/2026-05-04-e5-optuna-lgbm.md").write_text(body)
    print("audit written")


if __name__ == "__main__":
    main()
