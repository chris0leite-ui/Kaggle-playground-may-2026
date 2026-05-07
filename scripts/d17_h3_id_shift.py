"""scripts/d17_h3_id_shift.py — Day-17 H3 ID-shift / row-position diagnostic.

Replicates s5e12 2nd-place "Winning based on ID Shift Analysis" precedent.
Granular AV-AUC sweep across id-modular features. If any granularity > 0.510,
trains sparse LR on those features and gates as K=22 base-add candidate.

Usage:
  python scripts/d17_h3_id_shift.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
AV_THRESHOLD = 0.510  # 0.502 (global) + ~3σ

# AV-LGBM params (cheap probe, just need a clean AV-AUC).
AV_LGBM_PARAMS = dict(
    n_estimators=200,
    num_leaves=15,
    min_data_in_leaf=500,
    learning_rate=0.05,
    n_jobs=1,
    verbose=-1,
    objective="binary",
    metric="auc",
)


def _av_auc_single_feature(feature_name: str, train_vals, test_vals):
    """Fit AV LGBM on one feature, return AV-AUC + n unique."""
    n_train = len(train_vals)
    n_test = len(test_vals)
    X = np.concatenate([train_vals, test_vals]).reshape(-1, 1).astype(np.float32)
    y_av = np.concatenate([np.zeros(n_train, dtype=np.int8),
                           np.ones(n_test, dtype=np.int8)])
    n_uniq = int(np.unique(X[:, 0]).size)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    av_oof = np.zeros(len(y_av), dtype=np.float64)
    for tr, va in skf.split(X, y_av):
        clf = lgb.LGBMClassifier(**AV_LGBM_PARAMS)
        clf.fit(X[tr], y_av[tr])
        av_oof[va] = clf.predict_proba(X[va])[:, 1]
    auc = float(roc_auc_score(y_av, av_oof))
    return auc, n_uniq


def main():
    t_global = time.time()

    print("=== Loading train/test ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    n_train, n_test = len(train), len(test)
    print(f"  train={n_train}, test={n_test}")

    y = train[TARGET].astype(int).values
    train_id = train["id"].values
    test_id = test["id"].values

    # Get cardinality of Driver, Race, LapNumber for sweep granularities.
    drv_card = int(pd.concat([train["Driver"], test["Driver"]]).nunique())
    race_card = int(pd.concat([train["Race"], test["Race"]]).nunique())
    lap_card = int(pd.concat([train["LapNumber"], test["LapNumber"]]).nunique())
    print(f"  Driver-card={drv_card}, Race-card={race_card}, LapNumber-card={lap_card}")

    # Define granularities to sweep.
    grids = []
    # Modular
    for N in [7, 11, 100, 1000, drv_card, race_card, lap_card]:
        grids.append(("mod", N))
    # Floor-divide row-position bins
    for D in [10, 100, 1000, 10000]:
        grids.append(("div", D))

    print(f"\n=== AV-AUC sweep over {len(grids)} granularities ===")
    av_results = {}
    significant = []  # list of (key, auc) pairs > threshold

    for kind, N in grids:
        if kind == "mod":
            tr_vals = (train_id % N).astype(np.int64)
            te_vals = (test_id % N).astype(np.int64)
            key = f"id_mod_{N}"
        else:
            tr_vals = (train_id // N).astype(np.int64)
            te_vals = (test_id // N).astype(np.int64)
            key = f"id_div_{N}"

        t0 = time.time()
        auc, n_uniq = _av_auc_single_feature(key, tr_vals, te_vals)
        elapsed = time.time() - t0
        av_results[key] = dict(av_auc=auc, n_unique=n_uniq, elapsed_s=elapsed,
                                kind=kind, N=N)
        flag = " ***" if auc > AV_THRESHOLD else ""
        print(f"  {key:<22s} n_uniq={n_uniq:<6d}  AV-AUC={auc:.5f}  ({elapsed:.1f}s){flag}")
        if auc > AV_THRESHOLD:
            significant.append((key, kind, N, auc))

    # Save AV results.
    out_av = ART / "d17_h3_id_shift_av_results.json"
    out_av.write_text(json.dumps(dict(
        results=av_results,
        threshold=AV_THRESHOLD,
        significant=[dict(key=k, kind=ki, N=n, av_auc=a) for k, ki, n, a in significant],
        n_train=n_train, n_test=n_test,
        cardinalities=dict(Driver=drv_card, Race=race_card, LapNumber=lap_card),
    ), indent=2))
    print(f"\n  → {out_av}")

    if not significant:
        print("\n=== VERDICT: NO granularity exceeded 0.510 threshold ===")
        print("    ID-shift family CLOSED. Producing NULL audit.")
        return dict(verdict="NULL", av_results=av_results, significant=[])

    # === Sparse LR on significant id-modular one-hots ===
    print(f"\n=== Building sparse LR on {len(significant)} significant features ===")
    from scipy.sparse import csr_matrix, hstack as sp_hstack

    def _onehot(vals_train, vals_test):
        """Sparse one-hot for a single integer feature; categories ordered."""
        cats = pd.Index(np.unique(np.concatenate([vals_train, vals_test])))
        cat_to_idx = {c: i for i, c in enumerate(cats.values)}
        ncats = len(cats)
        # Build sparse for train.
        rows_tr = np.arange(len(vals_train))
        cols_tr = np.array([cat_to_idx[v] for v in vals_train])
        data_tr = np.ones(len(vals_train), dtype=np.float32)
        Xtr = csr_matrix((data_tr, (rows_tr, cols_tr)),
                          shape=(len(vals_train), ncats))
        rows_te = np.arange(len(vals_test))
        cols_te = np.array([cat_to_idx[v] for v in vals_test])
        data_te = np.ones(len(vals_test), dtype=np.float32)
        Xte = csr_matrix((data_te, (rows_te, cols_te)),
                          shape=(len(vals_test), ncats))
        return Xtr, Xte, ncats

    feat_blocks_tr = []
    feat_blocks_te = []
    feat_meta = []
    for key, kind, N, auc in significant:
        if kind == "mod":
            tr_vals = (train_id % N).astype(np.int64)
            te_vals = (test_id % N).astype(np.int64)
        else:
            tr_vals = (train_id // N).astype(np.int64)
            te_vals = (test_id // N).astype(np.int64)
        Xtr_block, Xte_block, ncats = _onehot(tr_vals, te_vals)
        feat_blocks_tr.append(Xtr_block)
        feat_blocks_te.append(Xte_block)
        feat_meta.append(dict(key=key, ncats=ncats, av_auc=auc))
        print(f"  {key} → {ncats} one-hot cats")

    Xtr = sp_hstack(feat_blocks_tr, format="csr")
    Xte = sp_hstack(feat_blocks_te, format="csr")
    print(f"  Total feature matrix: train={Xtr.shape}, test={Xte.shape}")

    # 5-fold StratifiedKFold OOF.
    print("\n=== Sparse LR OOF (C=0.1, l2) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(n_train, dtype=np.float64)
    test_pred_folds = np.zeros((N_FOLDS, n_test), dtype=np.float64)
    for fold, (tr, va) in enumerate(skf.split(np.zeros(n_train), y)):
        t0 = time.time()
        clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=200, n_jobs=1)
        clf.fit(Xtr[tr], y[tr])
        oof_proba[va] = clf.predict_proba(Xtr[va])[:, 1]
        test_pred_folds[fold] = clf.predict_proba(Xte)[:, 1]
        fold_auc = roc_auc_score(y[va], oof_proba[va])
        print(f"  fold {fold}: AUC={fold_auc:.5f}  ({time.time()-t0:.1f}s)")
    test_pred = test_pred_folds.mean(axis=0)
    standalone_oof = float(roc_auc_score(y, oof_proba))
    print(f"  Standalone OOF: {standalone_oof:.5f}")

    # Save OOF/test for gate.
    oof2 = np.column_stack([1 - oof_proba, oof_proba])
    test2 = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / "oof_d17_h3_id_shift_strat.npy", oof2)
    np.save(ART / "test_d17_h3_id_shift_strat.npy", test2)
    print(f"  → saved oof_d17_h3_id_shift_strat.npy, test_d17_h3_id_shift_strat.npy")

    # ρ vs PRIMARY (d16 cont_only Path B τ=20k = 0.951208).
    primary_test_path = ART / "test_d16_path_b_K22_continuous_only_tau20000_strat.npy"
    if primary_test_path.exists():
        primary_test = np.load(primary_test_path)
        if primary_test.ndim == 2:
            primary_test = primary_test[:, 1]
        rho, _ = spearmanr(test_pred, primary_test)
        print(f"  ρ vs d16 continuous_only PRIMARY: {rho:.6f}")
    else:
        # fall back to d13e
        primary_test_path = ART / "test_d13e_compound_stint_tau20000_strat.npy"
        primary_test = np.load(primary_test_path)
        if primary_test.ndim == 2:
            primary_test = primary_test[:, 1]
        rho, _ = spearmanr(test_pred, primary_test)
        print(f"  ρ vs d13e PRIMARY (fallback): {rho:.6f}")

    summary = dict(
        verdict="PENDING_GATE",
        av_results=av_results,
        significant=[dict(key=k, kind=ki, N=n, av_auc=a) for k, ki, n, a in significant],
        feat_meta=feat_meta,
        standalone_oof=standalone_oof,
        rho_vs_primary=float(rho),
        elapsed_total_s=time.time() - t_global,
    )
    out_summary = ART / "d17_h3_id_shift_results.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\n  → {out_summary}")
    print(f"\n  Total elapsed: {summary['elapsed_total_s']:.1f}s")
    return summary


if __name__ == "__main__":
    main()
