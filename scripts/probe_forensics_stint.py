"""scripts/probe_forensics_stint.py — H7 + H8.

H7 — Confident-error forensics. Identify rows where PRIMARY makes
high-confidence wrong predictions (|p - y| > 0.5). Tag by Compound,
Stint, Year, Race, lap-position-in-stint. The hope: a class of error
has structure suggesting a missing feature.

H8 — Stint-structure features. Build at sequence resolution (combined
train+test, AV-safe at row level per AV-AUC=0.502):
  - stint_length: size of (Race, Driver, Year, Stint) group
  - position_in_stint: rank of LapNumber within group (0-indexed)
  - position_frac: position_in_stint / (stint_length - 1)
  - laps_observed_before: number of laps in this group with LapNumber < this
  - laps_observed_after: number of laps in this group with LapNumber > this
  - lap_gap_to_prev: LapNumber - previous-row LapNumber within group
  - lap_gap_to_next: next-row LapNumber - LapNumber

For each: single-feature OOF AUC (rank). For the four most-promising:
single-LGBM OOF using stint features only.

Cost: <5 min CPU.
Outputs scripts/artifacts/probe_forensics_stint.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def load_primary() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    train = pd.read_csv("data/train.csv")
    prim = np.load(
        ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")[:, 1]
    y = train[TARGET].astype(int).values
    return train, prim, y


def h7_forensics(train: pd.DataFrame, prim: np.ndarray,
                 y: np.ndarray) -> dict:
    """Find confident errors and tag them."""
    print("\n=== H7: confident-error forensics ===")
    err = np.abs(prim - y)
    high_err_mask = err > 0.5
    n_high = int(high_err_mask.sum())
    n_total = len(y)
    print(f"  n confident-error rows (|p-y|>0.5): {n_high:,} / {n_total:,} "
          f"({n_high/n_total*100:.2f}%)")

    # Sub-types: confident-wrong-positive (y=1, p<0.5) vs confident-wrong-neg (y=0, p>0.5)
    cwp = (y == 1) & (prim < 0.5)
    cwn = (y == 0) & (prim > 0.5)
    n_cwp = int(cwp.sum()); n_cwn = int(cwn.sum())
    print(f"  confident-wrong-positives (y=1,p<0.5): {n_cwp:,}")
    print(f"  confident-wrong-negatives (y=0,p>0.5): {n_cwn:,}")

    # By Compound
    breakdown = {}
    for col in ["Compound", "Stint", "Year"]:
        b = {}
        for v in sorted(train[col].astype(str).unique()):
            m = train[col].astype(str).values == v
            n = int(m.sum())
            err_rate = float(high_err_mask[m].mean())
            cwp_rate = float(cwp[m].mean())
            cwn_rate = float(cwn[m].mean())
            b[v] = {"n": n, "err_rate_p>0.5": err_rate,
                    "wrong_pos_rate": cwp_rate,
                    "wrong_neg_rate": cwn_rate}
            print(f"    {col}={v:>15s}  n={n:>6d}  err>0.5: "
                  f"{err_rate*100:5.2f}%  wrong+: {cwp_rate*100:5.2f}%  "
                  f"wrong-: {cwn_rate*100:5.2f}%")
        breakdown[col] = b
    return {"n_high_err": n_high, "n_cwp": n_cwp, "n_cwn": n_cwn,
            "by_segment": breakdown}


def add_stint_features(train: pd.DataFrame,
                       test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Combined-frame stint features."""
    n_tr = len(train)
    df = pd.concat([train.assign(_split="tr"),
                    test.assign(_split="te")],
                   ignore_index=True, sort=False)
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["Race", "Driver", "Year", "Stint", "LapNumber"],
                        kind="stable").reset_index(drop=True)
    g = df.groupby(["Race", "Driver", "Year", "Stint"], sort=False)
    df["stint_length"] = g["LapNumber"].transform("size")
    df["position_in_stint"] = g.cumcount()
    df["position_frac"] = (df["position_in_stint"] /
                            (df["stint_length"] - 1).clip(lower=1))
    df["lap_gap_to_prev"] = g["LapNumber"].diff().fillna(0).astype(float)
    df["lap_gap_to_next"] = (g["LapNumber"].shift(-1) -
                             df["LapNumber"]).fillna(0).astype(float)
    df["laps_obs_before"] = df["position_in_stint"]
    df["laps_obs_after"] = df["stint_length"] - 1 - df["position_in_stint"]
    df["min_lap_in_stint"] = g["LapNumber"].transform("min")
    df["max_lap_in_stint"] = g["LapNumber"].transform("max")
    df["span_in_stint"] = df["max_lap_in_stint"] - df["min_lap_in_stint"]
    df["density_in_stint"] = df["stint_length"] / (df["span_in_stint"] + 1).clip(lower=1)
    feats = ["stint_length", "position_in_stint", "position_frac",
             "lap_gap_to_prev", "lap_gap_to_next",
             "laps_obs_before", "laps_obs_after", "span_in_stint",
             "density_in_stint"]
    df = df.sort_values("row_id").reset_index(drop=True)
    train_out = df.iloc[:n_tr].reset_index(drop=True)
    test_out = df.iloc[n_tr:].reset_index(drop=True)
    return train_out, test_out, feats


def h8_stint_features(train: pd.DataFrame, test: pd.DataFrame,
                      y: np.ndarray, prim: np.ndarray) -> dict:
    print("\n=== H8: stint-structure features ===")
    train_x, test_x, feats = add_stint_features(train, test)
    print(f"  built {len(feats)} stint features")

    # Single-feature OOF AUC (no CV needed, just rank)
    aucs = {}
    for f in feats:
        v = train_x[f].astype(float).values
        # Single-feature AUC; use rank
        try:
            auc = roc_auc_score(y, v)
            auc = max(auc, 1 - auc)  # sign-agnostic
        except Exception:
            auc = float("nan")
        aucs[f] = float(auc)
        print(f"    {f:>20s}  single-feat AUC: {auc:.5f}")

    # Single-LGBM on stint features only, 5-fold strat
    print("\n  Single-LGBM on stint features only:")
    X = train_x[feats].astype(float).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    LGB = dict(objective="binary", metric="auc", learning_rate=0.05,
               num_leaves=63, min_data_in_leaf=200, verbose=-1,
               n_jobs=-1, seed=SEED)
    for tr, va in skf.split(X, y):
        ds_tr = lgb.Dataset(X[tr], label=y[tr])
        ds_va = lgb.Dataset(X[va], label=y[va], reference=ds_tr)
        booster = lgb.train(LGB, ds_tr, num_boost_round=500,
                            valid_sets=[ds_va],
                            callbacks=[lgb.early_stopping(20),
                                       lgb.log_evaluation(0)])
        oof[va] = booster.predict(X[va])
    auc_lgb = roc_auc_score(y, oof)
    print(f"  single-LGBM OOF AUC: {auc_lgb:.5f}")

    # K=2 meta-add gate vs PRIMARY
    Pc = np.clip(np.column_stack([prim, oof]), 1e-9, 1 - 1e-9)
    rk = np.column_stack([rankdata(c) / len(c) for c in Pc.T])
    lg = np.log(Pc / (1 - Pc))
    F = np.hstack([Pc, rk, lg])
    meta_oof = np.zeros(len(y))
    for tr, va in skf.split(F, y):
        lr = LogisticRegression(C=1.0, max_iter=500)
        lr.fit(F[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F[va])[:, 1]
    auc_meta = roc_auc_score(y, meta_oof)
    auc_prim = roc_auc_score(y, prim)
    delta = (auc_meta - auc_prim) * 1e4
    print(f"  K=2 LR-meta [PRIMARY, stint-LGBM] OOF: {auc_meta:.5f}  "
          f"Δ vs PRIMARY: {delta:+.2f} bp")

    np.save(ART / "oof_stint_features_lgbm_strat.npy", oof)
    booster = lgb.train(LGB, lgb.Dataset(X, label=y),
                        num_boost_round=200,
                        callbacks=[lgb.log_evaluation(0)])
    test_pred = booster.predict(test_x[feats].astype(float).values)
    np.save(ART / "test_stint_features_lgbm_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))

    return {"single_feat_aucs": aucs, "stint_lgbm_oof": float(auc_lgb),
            "K2_meta_with_stint_oof": float(auc_meta),
            "delta_K2_meta_vs_primary_bp": float(delta)}


def main() -> None:
    t0 = time.time()
    train, prim, y = load_primary()
    test = pd.read_csv("data/test.csv")
    out_h7 = h7_forensics(train, prim, y)
    out_h8 = h8_stint_features(train, test, y, prim)
    out = {"h7": out_h7, "h8": out_h8, "wall_s": time.time() - t0}
    (ART / "probe_forensics_stint.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote scripts/artifacts/probe_forensics_stint.json. "
          f"Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
