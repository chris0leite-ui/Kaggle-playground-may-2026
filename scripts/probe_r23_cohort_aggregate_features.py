"""scripts/probe_r23_cohort_aggregate_features.py — Round 23 inventor:
cohort-aggregate meta-features over (Year, Race, LapNumber).

R17/R20 established cohort-listwise loss as the axis but the Path-B
LR-meta absorbs same-family listwise bases into one direction (R20
lambdarank K=19 +0.01 bp Δ vs R17-K=18 — sub-G2). This is a structurally
distinct attack on the same axis: build COHORT-AGGREGATE FEATURES from
K=17 predictions (not new model bases). These describe each row's
position WITHIN its cohort — direct row→cohort context the LR-meta
hasn't seen.

Features built (5 new bases for Path-B):
  1. cohort_mean_K17avg  — cohort mean of K=17 average prediction
  2. cohort_max_K17avg   — cohort max of K=17 average prediction
  3. cohort_std_K17avg   — cohort std of K=17 average prediction
  4. row_rank_in_cohort_R15  — rank of R15 prediction within its cohort
  5. row_centered_K17avg     — K17avg - cohort_mean (z-score-style)

Leakage: all 5 use already-fold-safe K=17 OOF predictions (no labels).
Cohort grouping is on (Year, Race, LapNumber) — train-side only at OOF
time, test-side computed independently on test rows.

Output: 5 .npy artifacts named oof_R23_<feat>_strat.npy + test_R23_<feat>_strat.npy
Pluggable as Path-B extra-bases for K=18+ pool.

Usage:
  python scripts/probe_r23_cohort_aggregate_features.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from build_K13_pathb_multiseg import K13_FILES
from build_K11_full_pathb import _pos

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

# Same K=17 pool as R17/R20 use
EXTRAS = [
    ("R12_cb_horizon", "oof_R12_cb_horizon_strat.npy",
     "test_R12_cb_horizon_strat.npy"),
    ("R13_cb_stint_completion", "oof_R13_cb_stint_completion_strat.npy",
     "test_R13_cb_stint_completion_strat.npy"),
    ("R14_tabm", "oof_R14_tabm_strat.npy", "test_R14_tabm_strat.npy"),
    ("R15_xendcg_per_seg", "oof_R15_xendcg_per_seg_strat.npy",
     "test_R15_xendcg_per_seg_strat.npy"),
]


def cohort_aggregate(df: pd.DataFrame, values: np.ndarray,
                     cols=("Year", "Race", "LapNumber")):
    """Return per-row {mean, max, std, rank, centered} aggregates within
    each cohort defined by cols."""
    df_w = df[list(cols)].copy()
    df_w["__v__"] = values
    grp = df_w.groupby(list(cols))["__v__"]
    mean_g = grp.transform("mean").values
    max_g = grp.transform("max").values
    std_g = grp.transform("std").fillna(0.0).values
    rank_g = grp.rank(pct=True).values
    centered_g = values - mean_g
    return dict(mean=mean_g, max=max_g, std=std_g, rank=rank_g,
                centered=centered_g)


def main():
    t0 = time.time()
    print("== R23: cohort-aggregate meta-features over (Year, Race, LapNumber) ==",
          flush=True)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    # Build K=17 OOF + test matrices
    oof_cols, test_cols, names = [], [], []
    for name, of, tf in K13_FILES:
        oof_cols.append(_pos(ART / of))
        test_cols.append(_pos(ART / tf))
        names.append(name)
    for nm, of, tf in EXTRAS:
        oof_cols.append(_pos(ART / of))
        test_cols.append(_pos(ART / tf))
        names.append(nm)
    K17_oof = np.column_stack(oof_cols)
    K17_test = np.column_stack(test_cols)
    print(f"  K=17 OOF: {K17_oof.shape}  test: {K17_test.shape}", flush=True)

    K17avg_oof = K17_oof.mean(axis=1)
    K17avg_test = K17_test.mean(axis=1)
    R15_oof = _pos(ART / "oof_R15_xendcg_per_seg_strat.npy")
    R15_test = _pos(ART / "test_R15_xendcg_per_seg_strat.npy")

    # Cohort diagnostics
    cohort_keys = train.groupby(["Year", "Race", "LapNumber"]).size()
    print(f"  cohort (Y,R,L) train: {len(cohort_keys)} median={int(cohort_keys.median())}",
          flush=True)
    test_cohort_keys = test.groupby(["Year", "Race", "LapNumber"]).size()
    print(f"  cohort (Y,R,L) test:  {len(test_cohort_keys)} median={int(test_cohort_keys.median())}",
          flush=True)

    # Cohort aggregates over K17avg
    print("  building cohort aggregates over K17avg ...", flush=True)
    agg_train = cohort_aggregate(train, K17avg_oof)
    agg_test = cohort_aggregate(test, K17avg_test)
    rank_train = cohort_aggregate(train, R15_oof)["rank"]
    rank_test = cohort_aggregate(test, R15_test)["rank"]

    # Build 5 base feature .npy
    feature_set = {
        "R23_cohort_mean_K17avg": (agg_train["mean"], agg_test["mean"]),
        "R23_cohort_max_K17avg":  (agg_train["max"],  agg_test["max"]),
        "R23_cohort_std_K17avg":  (agg_train["std"],  agg_test["std"]),
        "R23_row_rank_in_cohort_R15": (rank_train, rank_test),
        "R23_row_centered_K17avg": (agg_train["centered"], agg_test["centered"]),
    }

    print("\n  Feature diagnostics (vs R15 PRIMARY):", flush=True)
    for fname, (f_tr, f_te) in feature_set.items():
        # Re-scale to [0,1] for compatibility as a "prediction" base in
        # the K-pool (Path-B operates on probability-like columns).
        # Use min-max if non-positive range, else raw.
        f_tr_arr = np.asarray(f_tr, dtype=np.float32)
        f_te_arr = np.asarray(f_te, dtype=np.float32)
        # Min-max normalize using train statistics
        lo, hi = float(f_tr_arr.min()), float(f_tr_arr.max())
        if hi > lo:
            scale = 1.0 / (hi - lo)
            f_tr_arr = ((f_tr_arr - lo) * scale).clip(0, 1)
            f_te_arr = ((f_te_arr - lo) * scale).clip(0, 1)
        np.save(ART / f"oof_{fname}_strat.npy", f_tr_arr)
        np.save(ART / f"test_{fname}_strat.npy", f_te_arr)
        # Diagnostics
        auc = roc_auc_score(y, f_tr_arr) if len(set(f_tr_arr)) > 1 else 0.5
        rho_oof = spearmanr(f_tr_arr, R15_oof).statistic
        rho_test = spearmanr(f_te_arr, R15_test).statistic
        print(f"    {fname:<35s}  AUC={auc:.5f}  ρ_oof_vs_R15={rho_oof:+.4f}  "
              f"ρ_test_vs_R15={rho_test:+.4f}", flush=True)

    print(f"\n  Saved 5 cohort-aggregate feature .npy pairs in {ART}",
          flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
