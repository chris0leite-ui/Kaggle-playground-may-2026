"""Phase B2-cheap — (Race, Compound) pit-lap quantile from INTERNAL
train data. If internal-only doesn't lift, FastF1 external version
would also be null (same key, same join, same axis).

Fold-safe per R24: per-fold, build median pit-lap per (Race, Compound)
from training rows only, apply to validation rows.

Features built per row:
  - median_pit_lap[(Race, Compound)]  fold-safe median LapNumber where PitNextLap=1
  - q1_pit_lap, q3_pit_lap  same, 25th and 75th percentile
  - lap_vs_median  = LapNumber - median
  - lap_vs_median_norm = (LapNumber - median) / race_total_laps
  - in_iqr  = 1 if LapNumber in [q1, q3] else 0

Output: audit/2026-05-20-racecomp-quantile-probe.{log,json}
        scripts/artifacts/oof_R16_racecomp_q_strat.npy (if G2 clears)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def build_race_total(train, test):
    both = pd.concat([train[["Year", "Race", "LapNumber"]],
                      test[["Year", "Race", "LapNumber"]]],
                     ignore_index=True)
    return both.groupby(["Year", "Race"])["LapNumber"].max().rename(
        "race_total_laps").reset_index()


def compute_fold_safe_quantiles(train, y, fold_idx, n_folds):
    """For each fold k:
       - using train rows with fold != k AND PitNextLap==1, group by
         (Race, Compound) and compute the lap-number quantiles.
       - map onto fold-k validation rows.
       Returns (median, q1, q3) arrays indexed by train row.
    """
    n = len(train)
    medians = np.full(n, np.nan, dtype=np.float32)
    q1s = np.full(n, np.nan, dtype=np.float32)
    q3s = np.full(n, np.nan, dtype=np.float32)
    for k in range(n_folds):
        is_val = fold_idx == k
        is_tr = ~is_val
        # use TRAINING positives only (PitNextLap==1)
        pos_mask = is_tr & (y == 1)
        sub = train.loc[pos_mask, ["Race", "Compound", "LapNumber"]]
        if len(sub) == 0:
            continue
        q = sub.groupby(["Race", "Compound"])["LapNumber"].agg(
            median="median", q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)).reset_index()
        merged = train.loc[is_val, ["Race", "Compound"]].merge(
            q, on=["Race", "Compound"], how="left")
        medians[is_val] = merged["median"].values
        q1s[is_val] = merged["q1"].values
        q3s[is_val] = merged["q3"].values
    return medians, q1s, q3s


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    p_primary = np.load(ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy")
    print(f"PRIMARY OOF AUC: {roc_auc_score(y, p_primary):.5f}")

    rtotal = build_race_total(train, test)
    train = train.merge(rtotal, on=["Year", "Race"], how="left")
    test = test.merge(rtotal, on=["Year", "Race"], how="left")
    print(f"Race-total laps: mean={train['race_total_laps'].mean():.1f} "
          f"min={train['race_total_laps'].min()} max={train['race_total_laps'].max()}")

    # Fold layout (matches probe_real_f1_axis.py)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_idx = np.empty(len(y), dtype=int)
    for k, (_, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_idx[vi] = k

    # Fold-safe quantile features
    print("\n=== BUILDING FOLD-SAFE QUANTILES ===")
    med, q1, q3 = compute_fold_safe_quantiles(train, y, fold_idx, N_FOLDS)
    print(f"  median fill rate: {(~np.isnan(med)).mean()*100:.1f}%")
    # Fall back to global median for any (Race,Compound) cells with no
    # positives in train fold
    global_med = float(np.nanmedian(med))
    global_q1 = float(np.nanmedian(q1))
    global_q3 = float(np.nanmedian(q3))
    med = np.where(np.isnan(med), global_med, med)
    q1 = np.where(np.isnan(q1), global_q1, q1)
    q3 = np.where(np.isnan(q3), global_q3, q3)
    lap = train["LapNumber"].values
    rt = train["race_total_laps"].values
    lap_vs_median = lap - med
    lap_vs_median_norm = lap_vs_median / np.clip(rt, 1, None)
    in_iqr = ((lap >= q1) & (lap <= q3)).astype(np.float32)
    # Distance from window
    dist_to_window = np.where(in_iqr == 1, 0.0,
                              np.minimum(np.abs(lap - q1), np.abs(lap - q3)))

    # G1 standalone AUCs
    print("\n=== G1 STANDALONE AUCs ===")
    feats = {
        "lap_vs_median": lap_vs_median,
        "lap_vs_median_norm": lap_vs_median_norm,
        "in_iqr": in_iqr,
        "dist_to_window": dist_to_window,
        "median": med,
    }
    g1 = {}
    for name, arr in feats.items():
        try:
            a = float(roc_auc_score(y, arr))
        except Exception:
            a = float("nan")
        g1[name] = a
        print(f"  {name}: AUC {a:.5f}")

    # G2 LR-meta
    print("\n=== G2 LR-META ===")
    p_clip = np.clip(p_primary, 1e-6, 1 - 1e-6)
    L = logit(p_clip)
    X_base = L.reshape(-1, 1).astype(np.float64)
    X_full = np.column_stack([
        L, lap_vs_median, lap_vs_median_norm, in_iqr, dist_to_window, med,
    ]).astype(np.float64)

    def lr_oof(X):
        oof = np.zeros(len(y))
        for k in range(N_FOLDS):
            iv = fold_idx == k
            it = ~iv
            lr = LogisticRegression(C=1.0, max_iter=300)
            lr.fit(X[it], y[it])
            oof[iv] = lr.predict_proba(X[iv])[:, 1]
        return oof
    oof_base = lr_oof(X_base)
    oof_full = lr_oof(X_full)
    auc_base = roc_auc_score(y, oof_base)
    auc_full = roc_auc_score(y, oof_full)
    delta_bp = (auc_full - auc_base) * 1e4
    print(f"  base (K17-only)    AUC: {auc_base:.5f}")
    print(f"  + racecomp quantile AUC:{auc_full:.5f}")
    print(f"  Δ vs K17-only base: {delta_bp:+.3f} bp")
    print(f"  Δ vs PRIMARY R15: {(auc_full - roc_auc_score(y, p_primary))*1e4:+.3f} bp")

    # Stratum decomposition
    drv = train["Driver"].astype(str)
    is_real = (~drv.str.match(r"^D\d{3}$")).values
    if is_real.sum():
        print(f"\n  Real-F1 Δ: {(roc_auc_score(y[is_real], oof_full[is_real]) - roc_auc_score(y[is_real], oof_base[is_real]))*1e4:+.3f} bp")
        print(f"  Synth Δ:   {(roc_auc_score(y[~is_real], oof_full[~is_real]) - roc_auc_score(y[~is_real], oof_base[~is_real]))*1e4:+.3f} bp")

    # Per-compound Δ — does INTERMEDIATE/WET lift?
    comp = train["Compound"].values
    print(f"\n  WET Δ:           {(roc_auc_score(y[comp=='WET'], oof_full[comp=='WET']) - roc_auc_score(y[comp=='WET'], oof_base[comp=='WET']))*1e4:+.3f} bp (n={int((comp=='WET').sum())})")
    print(f"  INTERMEDIATE Δ:  {(roc_auc_score(y[comp=='INTERMEDIATE'], oof_full[comp=='INTERMEDIATE']) - roc_auc_score(y[comp=='INTERMEDIATE'], oof_base[comp=='INTERMEDIATE']))*1e4:+.3f} bp (n={int((comp=='INTERMEDIATE').sum())})")

    GATE_BP = 0.02
    passed = delta_bp >= GATE_BP
    print(f"\n  G2 gate (+{GATE_BP:.2f} bp): {'PASS' if passed else 'fail'}")

    out = dict(
        date="2026-05-20",
        primary_oof_auc=float(roc_auc_score(y, p_primary)),
        g1_aucs={k: float(v) for k, v in g1.items()},
        auc_lr_base=float(auc_base),
        auc_lr_full=float(auc_full),
        delta_bp=float(delta_bp),
        delta_vs_primary_bp=float((auc_full - roc_auc_score(y, p_primary)) * 1e4),
        g2_passed=bool(passed),
        wall_s=time.time() - t0,
    )
    Path("audit/2026-05-20-racecomp-quantile-probe.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nWall: {out['wall_s']:.1f}s")
    print("Wrote audit/2026-05-20-racecomp-quantile-probe.json")


if __name__ == "__main__":
    main()
