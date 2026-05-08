"""Day-12 Part B — AUC-direct base retrain via XGBoost rank:pairwise.

Pick e3_hgbc-equivalent feature set, train an XGBoost with
`objective='rank:pairwise'` (pairwise rank loss, AUC-aligned).

Smoke (1 fold) first; if smoke OOF AUC >= e3_hgbc OOF − 0.001, full 5-fold.

Output: oof_d12_e3_aucpairwise_strat.npy, test_d12_e3_aucpairwise_strat.npy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
E3_BASELINE = 0.94876  # e3_hgbc OOF Strat
SMOKE_DEGRADE_TOLERANCE = 1e-3


def prep_features(train, test):
    """Same prep as e3_hgbc_two_anchor.py: Driver int-encoded; Compound/Race cat."""
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    HIGH_CARD = ["Driver"]; LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True
                             ).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mp).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True
                             ).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mp).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)
    return X.astype(np.float32), X_test.astype(np.float32)


def build_random_groups_xgb(idx, group_size=1000, seed=SEED):
    """Random groups; return (counts, perm)."""
    rng = np.random.default_rng(seed)
    perm_local = rng.permutation(len(idx))
    sorted_idx = idx[perm_local]
    n = len(sorted_idx)
    n_groups = max(1, n // group_size)
    base = n // n_groups; rem = n - base * n_groups
    counts = np.array([base + (1 if g < rem else 0) for g in range(n_groups)],
                      dtype=np.int32)
    return counts, sorted_idx


def fit_xgb_pairwise(X_arr, y, X_test_arr, splits, race_ids,
                     mode="random", group_size=1000, smoke_only=False):
    n = X_arr.shape[0]
    oof = np.zeros(n, dtype=np.float64)
    test_pred = np.zeros(X_test_arr.shape[0], dtype=np.float64)
    biters = []
    for k, (tr, va) in enumerate(splits):
        if mode == "random":
            counts_tr, perm_tr = build_random_groups_xgb(tr, group_size, SEED + k)
            counts_va, perm_va = build_random_groups_xgb(va, group_size, SEED + k)
        elif mode == "race":
            order_tr = np.argsort(race_ids[tr], kind="stable")
            perm_tr = tr[order_tr]
            order_va = np.argsort(race_ids[va], kind="stable")
            perm_va = va[order_va]
            # group counts
            r_tr = race_ids[perm_tr]
            r_va = race_ids[perm_va]
            counts_tr = np.diff(np.concatenate(
                [[0], np.where(np.diff(r_tr) != 0)[0] + 1, [len(r_tr)]]
            )).astype(np.int32)
            counts_va = np.diff(np.concatenate(
                [[0], np.where(np.diff(r_va) != 0)[0] + 1, [len(r_va)]]
            )).astype(np.int32)
        else:
            raise ValueError(mode)

        dtr = xgb.DMatrix(X_arr[perm_tr], label=y[perm_tr])
        dtr.set_group(counts_tr)
        dva = xgb.DMatrix(X_arr[perm_va], label=y[perm_va])
        dva.set_group(counts_va)
        dtest = xgb.DMatrix(X_test_arr)

        params = dict(
            objective="rank:pairwise",
            eval_metric="auc",
            tree_method="hist",
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            reg_lambda=1.0,
            subsample=0.9,
            colsample_bytree=0.9,
            seed=SEED,
            nthread=-1,
        )
        booster = xgb.train(
            params, dtr, num_boost_round=2000,
            evals=[(dva, "val")],
            early_stopping_rounds=50, verbose_eval=0,
        )
        p_va = booster.predict(dva)
        # Need to remap p_va back to original va index ordering
        # (we trained/predicted on perm_va order)
        p_va_full = np.empty_like(p_va)
        p_va_full[np.argsort(perm_va.argsort())] = p_va  # not quite right
        # Cleaner: just store at perm_va indices directly
        oof[perm_va] = p_va
        test_pred += booster.predict(dtest) / N_FOLDS
        biters.append(int(booster.best_iteration))
        fold_auc = float(roc_auc_score(y[perm_va], p_va))
        print(f"  fold {k}: iters={biters[-1]} AUC={fold_auc:.5f}")
        if smoke_only:
            return oof, test_pred, biters, fold_auc
    return oof, test_pred, biters, None


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    race_ids = pd.Categorical(train["Race"]).codes.astype(np.int32)

    X, X_test = prep_features(train, test)
    print(f"Train shape: {X.shape}  Test shape: {X_test.shape}")
    X_arr = X.values
    X_test_arr = X_test.values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # === SMOKE: 1-fold, random groups ===
    print("\n=== SMOKE: XGB rank:pairwise (random groups, 1 fold) ===")
    smoke_split = [splits[0]]
    _, _, _, smoke_auc = fit_xgb_pairwise(
        X_arr, y, X_test_arr, smoke_split, race_ids,
        mode="random", group_size=1000, smoke_only=True)
    print(f"\nSmoke fold-0 AUC: {smoke_auc:.5f}  vs e3 baseline {E3_BASELINE:.5f}")
    print(f"  Δ = {(smoke_auc - E3_BASELINE)*1e4:+.2f}bp")

    if smoke_auc < E3_BASELINE - SMOKE_DEGRADE_TOLERANCE:
        print(f"\n→ SMOKE FAIL (≥ {SMOKE_DEGRADE_TOLERANCE*1e4:.0f}bp regression). "
              f"Skipping full 5-fold.")
        result = dict(smoke_only=True, smoke_fold0_auc=smoke_auc,
                      baseline_e3_auc=E3_BASELINE,
                      delta_bp=(smoke_auc - E3_BASELINE) * 1e4,
                      decision="skip_full_5fold",
                      total_wall_s=time.time() - t0)
        (ART / "d12_aucpairwise_base_results.json").write_text(
            json.dumps(result, indent=2))
        return

    # === FULL 5-fold ===
    print("\n=== FULL 5-fold: XGB rank:pairwise (random groups) ===")
    oof, test_pred, biters, _ = fit_xgb_pairwise(
        X_arr, y, X_test_arr, splits, race_ids,
        mode="random", group_size=1000, smoke_only=False)
    auc = float(roc_auc_score(y, oof))
    print(f"\nFull OOF AUC: {auc:.5f}  vs e3 baseline {E3_BASELINE:.5f}")
    print(f"  Δ = {(auc - E3_BASELINE)*1e4:+.2f}bp  iters={biters}")

    np.save(ART / "oof_d12_e3_aucpairwise_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d12_e3_aucpairwise_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))

    result = dict(smoke_fold0_auc=smoke_auc,
                  full_oof_auc=auc,
                  baseline_e3_auc=E3_BASELINE,
                  full_delta_bp=(auc - E3_BASELINE) * 1e4,
                  best_iters=biters,
                  total_wall_s=time.time() - t0)
    (ART / "d12_aucpairwise_base_results.json").write_text(
        json.dumps(result, indent=2))
    print(f"→ scripts/artifacts/d12_aucpairwise_base_results.json")


if __name__ == "__main__":
    main()
