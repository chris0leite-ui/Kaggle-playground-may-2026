"""scripts/probe_target_reform_strict_oof.py — strict-OOF target reformulation audit.

Audits the suspected target-leakage in probe_target_reform.py and
probe_target_reform_v2.py. The original compute_targets() uses ALL train
labels per (Driver, Race, Year) group, so when LightGBM trains on
(X[tr], target[tr]), the target for tr rows in groups spanning tr+va
INCLUDES information about va-row labels (via total_pits and cumsum).

Strict-OOF fix: for each fold k, compute targets using ONLY y[tr]
labels (with `mask` argument). Train LightGBM(X[tr], target_tr_strict),
predict on X[va] → oof[va]. For test: full-train target + full-train fit.

Re-runs reverse_cum, pit_horizon, inv_laps_until_pit (the prior winners).

Compares:
  Original (leakage): K=21 + cand from previous probes.
  Strict-OOF:         K=21 + cand_strict from this script.

If the OOF lift COLLAPSES under strict-OOF, the original win was
leakage-inflated. If it SURVIVES, the candidate is real.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def compute_targets_masked(df, y, mask):
    """Compute (reverse_cum, pit_horizon, inv_laps) using ONLY y values where
    mask=True. Rows where mask=False contribute 0 to group sums (as if
    they don't exist).

    Returns dict with three target arrays, length len(df). The values for
    mask=False rows are STILL computed but using only mask=True labels in
    their group — this is the strict-OOF semantics."""
    df = df.copy()
    df["_y_masked"] = np.where(mask, y, 0).astype(np.int8)
    df["_idx"] = np.arange(len(df))
    n = len(df)
    reverse_cum = np.zeros(n, dtype=np.float32)
    pit_horizon = np.full(n, 3, dtype=np.int8)
    inv_laps = np.zeros(n, dtype=np.float32)

    for keys, grp in df.groupby(["Driver", "Race", "Year"], sort=False):
        gs = grp.sort_values("LapNumber")
        laps = gs["LapNumber"].values
        ys_masked = gs["_y_masked"].values
        idxs = gs["_idx"].values
        total_pits = int(ys_masked.sum())
        cum = np.cumsum(ys_masked)
        # reverse_cum: # remaining pits using only mask=True labels
        reverse_cum[idxs] = (total_pits - cum).clip(0, 10).astype(np.float32)
        # next_pit_lap and inv_laps
        next_pit_lap_arr = np.full(len(gs), 999, dtype=np.int32)
        last = 999
        for i in range(len(gs) - 1, -1, -1):
            if ys_masked[i] == 1:
                last = laps[i]
                next_pit_lap_arr[i] = 0
            else:
                next_pit_lap_arr[i] = max(0, last - laps[i])
        inv_laps[idxs] = (1.0 / (1.0 + next_pit_lap_arr)).astype(np.float32)
        # pit_horizon (same gap → bucket)
        for i, ix in enumerate(idxs):
            gap = next_pit_lap_arr[i]
            if gap == 0: pit_horizon[ix] = 0
            elif gap <= 2: pit_horizon[ix] = 1
            elif gap <= 5: pit_horizon[ix] = 2
            else: pit_horizon[ix] = 3
    return dict(reverse_cum=reverse_cum, pit_horizon=pit_horizon.astype(np.float32),
                inv_laps=inv_laps)


def lgbm_strict_oof(X, X_test, y, get_targets_fn, splits, cat_cols, name):
    """Train LightGBM 5-fold with STRICT-OOF target construction.

    For fold k: target_k = get_targets_fn(mask=y[tr]_only).
                Train LightGBM(X[tr], target_k[tr]), predict on X[va].
    For test:   target_full = get_targets_fn(mask=all_True).
                Train LightGBM(X_full, target_full), predict on X_test."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    params = dict(objective="regression", metric="rmse",
                  learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  min_data_in_leaf=200, verbose=-1, seed=SEED)
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        # STRICT-OOF: compute target using only tr labels
        tr_mask = np.zeros(len(y), dtype=bool); tr_mask[tr] = True
        target_strict = get_targets_fn(tr_mask)
        dtr = lgb.Dataset(X.iloc[tr], target_strict[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], target_strict[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        print(f"  [{name}] fold {k}: best_iter {m.best_iteration} wall {time.time()-t:.1f}s")
    # Test: full-train target
    target_full = get_targets_fn(np.ones(len(y), dtype=bool))
    dfull = lgb.Dataset(X, target_full, categorical_feature=cat_cols)
    m = lgb.train(params, dfull, num_boost_round=1000,
                  callbacks=[lgb.log_evaluation(0)])
    test_pred = m.predict(X_test)
    return oof, test_pred


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))

    feat_num = ["TyreLife", "RaceProgress", "LapTime_Delta",
                "Cumulative_Degradation", "Position", "LapTime (s)",
                "Stint", "Year", "Position_Change", "LapNumber"]
    feat_cat = ["Driver", "Compound", "Race"]
    X = train[feat_num + feat_cat].copy()
    X_test = test[feat_num + feat_cat].copy()
    for c in feat_cat:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    summary = {}
    for tname in ["reverse_cum", "pit_horizon", "inv_laps"]:
        print(f"\n=== STRICT-OOF: {tname} ===")
        # Closure that re-computes targets given a mask
        def get_target_fn(mask, tname=tname):
            tgts = compute_targets_masked(train, y, mask)
            return tgts[tname]
        oof, test_pred = lgbm_strict_oof(X, X_test, y, get_target_fn,
                                          splits, feat_cat, tname)
        auc = float(roc_auc_score(y, oof))
        rho, _ = spearmanr(test_pred, primary_test)
        d = (auc - auc_primary) * 1e4
        print(f"  std OOF: {auc:.5f}  Δ vs PRIMARY: {d:+.2f} bp  ρ vs PRIMARY: {rho:.6f}")
        np.save(ART / f"oof_target_reform_{tname}_strict_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_target_reform_{tname}_strict_strat.npy",
                np.column_stack([1 - test_pred, test_pred]))
        summary[tname] = dict(std_oof=auc, delta_vs_primary_bp=float(d),
                               rho_vs_primary=float(rho))

    summary["wall_s"] = time.time() - t0
    out = ART / "probe_target_reform_strict_oof.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out}  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
