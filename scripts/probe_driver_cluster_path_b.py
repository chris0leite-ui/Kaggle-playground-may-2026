"""scripts/probe_driver_cluster_path_b.py — Driver-cluster Path B cohort axis.

Round-2 critic-loop finding: cohort axis Year/Race/Year×Stint failed
(d14 sweep), but Driver clustering (k-means on per-Driver stint stats
→ 4 driver styles) was never tested as a cohort axis. Synthetic-data
lens: cluster discovery on Driver stats may surface a synth-gen
latent grouping the host's generator left behind.

Procedure:
  1. Per-Driver stats: mean(Stint), mean(Compound==SOFT proportion),
     mean(TyreLife), mean(LapTime_Delta), pit_rate, n_laps.
  2. k-means k=4 on these stats → driver_cluster.
  3. Path B Compound × driver_cluster (5 × 4 = 20 segments) hier-meta.
  4. τ ∈ {5k, 20k, 100k}, save artifacts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"
MIN_ROWS = 1000
MAX_ITER = 500
TAUS = [5000, 20000, 100000]
N_CLUSTERS = 4

K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _fit_lr_aug(F, y, max_iter=MAX_ITER):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def _predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))

    # Per-Driver stats from train
    train["soft_flag"] = (train["Compound"] == "SOFT").astype(int)
    train_only_pit = train.copy()
    train_only_pit["pit_target"] = y
    drv_stats = train_only_pit.groupby("Driver").agg(
        mean_stint=("Stint", "mean"),
        soft_prop=("soft_flag", "mean"),
        mean_tyre=("TyreLife", "mean"),
        mean_lt_delta=("LapTime_Delta", "mean"),
        pit_rate=("pit_target", "mean"),
        n_laps=("pit_target", "count"),
    ).reset_index()
    print(f"per-Driver stats shape: {drv_stats.shape}")

    # Standardize and cluster
    feat = drv_stats[["mean_stint", "soft_prop", "mean_tyre",
                      "mean_lt_delta", "pit_rate", "n_laps"]].values
    feat_s = StandardScaler().fit_transform(feat)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    drv_stats["cluster"] = km.fit_predict(feat_s)
    cluster_sizes = drv_stats["cluster"].value_counts().sort_index()
    print(f"driver clusters (k={N_CLUSTERS}): {cluster_sizes.tolist()}")

    drv2cluster = dict(zip(drv_stats["Driver"], drv_stats["cluster"]))
    train["drv_cluster"] = train["Driver"].map(drv2cluster).fillna(0).astype(int)
    test["drv_cluster"] = test["Driver"].map(drv2cluster).fillna(0).astype(int)

    # Compound × driver_cluster segmentation (5 × 4 = 20)
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp_map = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp_map).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp_map).astype(int).values
    seg_tr = c_tr * N_CLUSTERS + train["drv_cluster"].values.astype(int)
    seg_te = c_te * N_CLUSTERS + test["drv_cluster"].values.astype(int)
    n_seg = len(cats) * N_CLUSTERS
    counts_full = np.bincount(seg_tr, minlength=n_seg).astype(np.float64)
    print(f"segments: {n_seg}; populated≥{MIN_ROWS}: "
          f"{int(np.sum(counts_full >= MIN_ROWS))}; "
          f"sizes min/med/max {int(counts_full[counts_full>0].min())}/"
          f"{int(np.median(counts_full[counts_full>0]))}/"
          f"{int(counts_full.max())}")

    # Load K=21 pool
    base_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]
    base_tests = [_pos(ART / f"test_{b}_strat.npy") for b in K21_BASES]
    F_oof = _expand(np.column_stack(base_oofs))
    F_test = _expand(np.column_stack(base_tests))
    print(f"K=21 pool F shape: {F_oof.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    oofs = {tau: np.zeros(len(y)) for tau in TAUS}

    # Per-fold hier-meta sweep
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = _fit_lr_aug(F_oof[tr], y[tr])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_tr[tr] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_local[s] = _fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            mask[s] = True
        for tau in TAUS:
            n_local = counts.astype(np.float64)
            alpha = n_local / (n_local + tau)
            W_shrunk = (alpha[:, None] * W_local +
                        (1 - alpha[:, None]) * w_global[None, :])
            for s in np.unique(seg_tr[va]):
                idx = np.where(seg_tr[va] == s)[0]
                w = W_shrunk[s] if mask[s] else w_global
                oofs[tau][va[idx]] = _predict_aug(F_oof[va[idx]], w)
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s "
              f"({int(mask.sum())}/{n_seg} segments fit)")

    # Full-train fit for test
    w_global_full = _fit_lr_aug(F_oof, y)
    W_local_full = np.zeros((n_seg, len(w_global_full)))
    mask_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_tr == s)[0]
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_local_full[s] = _fit_lr_aug(F_oof[idx], y[idx])
        mask_full[s] = True
    test_preds = {}
    for tau in TAUS:
        n_local = counts_full.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local_full +
                    (1 - alpha[:, None]) * w_global_full[None, :])
        tp = np.zeros(len(test))
        for s in np.unique(seg_te):
            idx = np.where(seg_te == s)[0]
            w = W_shrunk[s] if mask_full[s] else w_global_full
            tp[idx] = _predict_aug(F_test[idx], w)
        test_preds[tau] = tp

    print(f"\n=== Compound × driver-cluster Path B sweep ===")
    print(f"  PRIMARY (Compound × Stint τ=20k): {auc_primary:.5f}")
    summary = dict(auc_primary=auc_primary, n_clusters=N_CLUSTERS,
                   cluster_sizes=cluster_sizes.tolist(), taus={})
    for tau in TAUS:
        oof = oofs[tau]
        tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho, _ = spearmanr(tp, primary_test)
        d = (auc - auc_primary) * 1e4
        print(f"  τ={tau}: OOF {auc:.5f}  Δ {d:+.2f} bp  ρ {rho:.6f}")
        np.save(ART / f"oof_drv_cluster_path_b_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_drv_cluster_path_b_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        summary["taus"][str(tau)] = dict(
            oof=auc, delta_vs_primary_bp=float(d), rho_vs_primary=float(rho))
    summary["wall_s"] = time.time() - t0
    out = ART / "probe_driver_cluster_path_b.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out} (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
