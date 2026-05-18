"""scripts/probe_c3_race_lap_shrinkage.py — C3 candidate

Per-(Race, LapNumber) Bayesian shrinkage of PRIMARY OOF + test predictions.

Mechanism: for each row, shrink the PRIMARY prediction toward the mean
PRIMARY prediction over the row's (Race, LapNumber) slice with weight w.
This is the soft / partial-trust version of EXP-A3-7 UID-mean smoothing
(which used the TARGET as group statistic and failed -124 bp). Here the
group statistic is the PRIMARY PREDICTION itself, not the target.

Origin: audit/research/2026-05-18-research.md C3.

Pool used as PRIMARY proxy: K=4 forward-greedy LR-meta (OOF 0.95399, LB
0.95351). The K=11 OOFs are not in the 2026-05-08 artifact snapshot.

Usage:
  python scripts/probe_c3_race_lap_shrinkage.py

Output: OOF AUC at w ∈ {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5}.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K4 = ["d17_h1d_yekenot_full", "p1_single_cb_v4_gpu",
      "f1_hgbc_deep", "d16_orig_continuous_only"]


def pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1] if a.ndim == 2 else a.ravel()


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def main():
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    races = train["Race"].values
    laps = train["LapNumber"].astype(int).values
    n_race_lap = train.groupby(["Race", "LapNumber"]).ngroups
    print(f"train {len(train):,} rows; "
          f"Race levels {pd.Series(races).nunique()}; "
          f"LapNumber range {laps.min()}-{laps.max()}; "
          f"unique (Race, LapNumber): {n_race_lap:,}")

    # K=4 LR-meta OOF (reproduce 0.95399 anchor)
    P_oof = np.column_stack([pos(ART / f"oof_{b}_strat.npy") for b in K4])
    F = expand(P_oof)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    primary_oof = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        primary_oof[va] = lr.predict_proba(F[va])[:, 1]
    base_auc = roc_auc_score(y, primary_oof)
    print(f"\nK=4 LR-meta baseline OOF AUC: {base_auc:.5f}")

    # Per-(Race, LapNumber) mean of primary prediction.
    # Compute fold-safe: per fold's validation rows use the per-(Race, Lap)
    # mean computed on training rows ONLY (Rule 24 / Rule 33 inner-CV).
    df = pd.DataFrame({"Race": races, "LapNumber": laps,
                       "pred": primary_oof, "y": y})

    print("\n--- Variant A: shrink toward per-(Race, LapNumber) PRED mean ---")
    print("w     | OOF AUC   | Δ bp vs K=4 baseline")
    print("------|-----------|---------------------")

    # Fit fold-safe: each fold's val rows are shrunk toward the mean computed
    # on the training partition only. We need to redo the fold loop.
    weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = {}
    for w in weights:
        oof_shrunk = np.zeros(len(y))
        for tr, va in skf.split(np.zeros(len(y)), y):
            df_tr = df.iloc[tr]
            grp_mean = (df_tr.groupby(["Race", "LapNumber"])["pred"]
                              .mean().to_dict())
            global_mean = float(df_tr["pred"].mean())
            keys_va = list(zip(df.iloc[va]["Race"].values,
                               df.iloc[va]["LapNumber"].astype(int).values))
            mean_va = np.array([grp_mean.get(k, global_mean) for k in keys_va])
            oof_shrunk[va] = (1 - w) * primary_oof[va] + w * mean_va
        auc_w = roc_auc_score(y, oof_shrunk)
        delta_bp = (auc_w - base_auc) * 1e4
        results[w] = (auc_w, delta_bp)
        print(f"{w:.2f}  | {auc_w:.5f}   | {delta_bp:+.3f} bp")

    print("\n--- Variant B: per-(Race, LapNumber) empirical TARGET rate as prior ---")
    print("(shrinks toward training y-mean per group; Rule 24 fold-safe)")
    print("w     | OOF AUC   | Δ bp vs K=4 baseline")
    print("------|-----------|---------------------")
    for w in weights:
        oof_shrunk = np.zeros(len(y))
        for tr, va in skf.split(np.zeros(len(y)), y):
            df_tr = df.iloc[tr]
            # empirical per-group rate on training rows only
            grp_y = (df_tr.groupby(["Race", "LapNumber"])["y"]
                          .mean().to_dict())
            global_y = float(df_tr["y"].mean())
            keys_va = list(zip(df.iloc[va]["Race"].values,
                               df.iloc[va]["LapNumber"].astype(int).values))
            prior_va = np.array([grp_y.get(k, global_y) for k in keys_va])
            oof_shrunk[va] = (1 - w) * primary_oof[va] + w * prior_va
        auc_w = roc_auc_score(y, oof_shrunk)
        delta_bp = (auc_w - base_auc) * 1e4
        print(f"{w:.2f}  | {auc_w:.5f}   | {delta_bp:+.3f} bp")

    print("\n--- Variant C: per-Race average pred (coarser grouping, only 26 levels) ---")
    print("w     | OOF AUC   | Δ bp vs K=4 baseline")
    print("------|-----------|---------------------")
    for w in weights:
        oof_shrunk = np.zeros(len(y))
        for tr, va in skf.split(np.zeros(len(y)), y):
            df_tr = df.iloc[tr]
            grp_p = df_tr.groupby("Race")["pred"].mean().to_dict()
            global_p = float(df_tr["pred"].mean())
            keys_va = df.iloc[va]["Race"].values
            mean_va = np.array([grp_p.get(k, global_p) for k in keys_va])
            oof_shrunk[va] = (1 - w) * primary_oof[va] + w * mean_va
        auc_w = roc_auc_score(y, oof_shrunk)
        delta_bp = (auc_w - base_auc) * 1e4
        print(f"{w:.2f}  | {auc_w:.5f}   | {delta_bp:+.3f} bp")

    best_w = max(results, key=lambda k: results[k][0])
    best_auc, best_delta = results[best_w]
    print(f"\nBest w={best_w:.2f}: OOF {best_auc:.5f}, Δ {best_delta:+.3f} bp.")

    # Save the best-w shrunk OOF for later K=4+1 gate / Rule-27 check
    # Reconstruct best at the global level for downstream use
    if best_delta > 0:
        # Compute non-fold-safe version (test rows use full-train mean)
        # for downstream test-time application (this is fine for test
        # because test labels are never seen).
        all_tr_means = df.groupby(["Race", "LapNumber"])["pred"].mean().to_dict()
        global_mean = float(df["pred"].mean())
        np.save(ART / "c3_race_lap_shrinkage_best_w.npy",
                np.array([best_w]))
        print(f"  saved best w to {ART}/c3_race_lap_shrinkage_best_w.npy")


if __name__ == "__main__":
    main()
