"""scripts/probe_r11_survival_hazard.py — Round 11 mechanism C.

Survival / hazard model on stint-life. Cox-PH-style discrete-time
hazard h(t | x) = h_0(t) * exp(beta'x), with t = TyreLife.

Mechanism-expansion candidate C from HANDOVER.md R10 priority queue.
Lowest-novelty (semi-parametric reduces to logit at single time-point)
but ORTHOGONAL inductive bias: piecewise-constant baseline hazard is
imposed on TyreLife — LGBM does not have that constraint.

Subject: per-(Driver, Race, Stint) tuple.
Duration: max(TyreLife) in TRAIN rows of that stint (per fold).
Event: PitStop=1 observed at the max-TyreLife train-row of that stint.

Per-fold (R24):
  1. Aggregate train rows -> per-stint (duration, event, covariates).
  2. Fit sksurv.linear_model.CoxPHSurvivalAnalysis on train stints.
  3. For each row (both val and test), look up the row's stint covariates
     and TyreLife, compute hazard(t | x) via partial-hazard * baseline.

Gates (R3 4-gate):
  G1 standalone OOF >= 0.90  (semi-parametric, expect lower than LGBM)
  G2 K=14 + Path-B DCS tau=100k vs R7.1 PRIMARY (OOF 0.954471) >= +0.10 bp
  G3 net rare-class-flip >= 0.5
  G4 direction asymmetry

KILL: standalone < 0.85 OR K=14 Delta < -0.10 bp.

Usage:
  python scripts/probe_r11_survival_hazard.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

TARGET = "PitNextLap"
SEED = 42
N_FOLDS = 5
ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]
YEARS = [2022, 2023, 2024, 2025]


def stint_key(df: pd.DataFrame) -> np.ndarray:
    """Stable per-stint integer key from (Driver, Race, Stint)."""
    return (df["Driver"].astype(str) + "|" +
            df["Race"].astype(str) + "|" +
            df["Stint"].astype(str)).values


def aggregate_stints(df: pd.DataFrame, row_mask: np.ndarray) -> pd.DataFrame:
    """Per-stint aggregates over rows where row_mask=True.

    Returns one row per stint with covariates + (optional) duration/event.
    """
    sub = df.loc[row_mask].copy()
    sub["_skey"] = stint_key(sub)
    grp = sub.groupby("_skey", sort=False)

    out = pd.DataFrame({
        "_skey": list(grp.groups.keys()),
    })
    out["duration"] = grp["TyreLife"].max().values.astype(float)
    # event = PitStop at max-TyreLife row of stint
    idxmax = grp["TyreLife"].idxmax().values
    out["event"] = df.loc[idxmax, "PitStop"].astype(int).values
    # covariates (per-stint summary)
    out["compound_idx"] = grp["Compound"].first().map(
        {c: i for i, c in enumerate(COMPOUNDS)}).fillna(0).astype(int).values
    out["stint_num"] = grp["Stint"].first().astype(int).values
    out["year_idx"] = grp["Year"].first().map(
        {y: i for i, y in enumerate(YEARS)}).fillna(0).astype(int).values
    out["lap_start"] = grp["LapNumber"].min().astype(float).values
    out["pos_start"] = grp["Position"].mean().astype(float).values
    out["lapdelta_mean"] = grp["LapTime_Delta"].mean().astype(float).values
    out["cum_deg_max"] = grp["Cumulative_Degradation"].max().astype(float).values
    # named-driver flag: D###-prefixed = anonymous; others (VER, HAM, ...) = named
    drv = grp["Driver"].first().astype(str)
    out["is_named"] = (~drv.str.match(r"^D\d{3}$")).astype(int).values
    return out


def covariate_matrix(stints: pd.DataFrame) -> np.ndarray:
    """Build dense covariate matrix matching the fitted Cox model."""
    rows = []
    # Compound dummies (drop first to avoid collinearity)
    for i in range(1, len(COMPOUNDS)):
        rows.append((stints["compound_idx"] == i).astype(float).values)
    # Year dummies (drop first)
    for i in range(1, len(YEARS)):
        rows.append((stints["year_idx"] == i).astype(float).values)
    rows.append(stints["stint_num"].astype(float).values)
    rows.append(stints["lap_start"].astype(float).values / 50.0)  # scale
    rows.append(stints["pos_start"].astype(float).values / 20.0)  # scale
    rows.append(np.clip(stints["lapdelta_mean"].astype(float).fillna(0).values, -50, 50) / 20.0)
    rows.append(np.clip(stints["cum_deg_max"].astype(float).fillna(0).values, -200, 200) / 100.0)
    rows.append(stints["is_named"].astype(float).values)
    X = np.stack(rows, axis=1).astype(np.float64)
    # Replace any residual NaN/inf
    X[~np.isfinite(X)] = 0.0
    return X


def fit_cox(train_stints: pd.DataFrame) -> CoxPHSurvivalAnalysis:
    """Fit Cox PH on training stints. Returns fitted estimator."""
    X = covariate_matrix(train_stints)
    y = Surv.from_arrays(
        event=train_stints["event"].astype(bool).values,
        time=np.clip(train_stints["duration"].astype(float).values, 1e-3, None),
    )
    est = CoxPHSurvivalAnalysis(alpha=1e-3, n_iter=100, ties="breslow")
    est.fit(X, y)
    return est


def hazard_at_rows(est: CoxPHSurvivalAnalysis, df_rows: pd.DataFrame,
                   stints_for_rows: pd.DataFrame) -> np.ndarray:
    """For each row of df_rows, compute hazard at row's TyreLife.

    Hazard(t | x) ~ H(t | x) - H(max(t-1, eps) | x), the discrete-time
    increment of the cumulative hazard function for the row's stint
    covariate vector at the row's TyreLife.
    """
    # Build row covariate matrix using the row's stint membership
    skey = stint_key(df_rows)
    skey_to_idx = {k: i for i, k in enumerate(stints_for_rows["_skey"].values)}
    row_stint_idx = np.array([skey_to_idx.get(k, -1) for k in skey], dtype=int)

    # For rows whose stint isn't in stints_for_rows (shouldn't happen given
    # aggregation over the row source), default covariates = 0
    X_stints = covariate_matrix(stints_for_rows)
    X_rows = np.zeros((len(df_rows), X_stints.shape[1]), dtype=np.float64)
    valid = row_stint_idx >= 0
    X_rows[valid] = X_stints[row_stint_idx[valid]]

    # Cumulative hazard functions, one per row (callable step-functions)
    chfs = est.predict_cumulative_hazard_function(X_rows)
    # Clip the evaluation point to within the model's trained domain
    # (out-of-domain rows -- e.g. test rows with TyreLife > max train --
    # get the model's last observed hazard.)
    domain_hi = chfs[0].domain[1]
    domain_lo = chfs[0].domain[0]
    t_raw = df_rows["TyreLife"].astype(float).values
    t_eval = np.clip(t_raw, domain_lo, domain_hi)
    h_curr = np.array([chfs[i](t_eval[i]) for i in range(len(df_rows))])
    t_prev = np.clip(t_eval - 1.0, domain_lo, domain_hi)
    h_prev = np.array([chfs[i](t_prev[i]) for i in range(len(df_rows))])
    haz = h_curr - h_prev
    haz = np.clip(haz, 0.0, None)
    return haz.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold on 50k stratified subset")
    args = ap.parse_args()
    t0 = time.time()
    print("== R11-C: Cox-PH hazard at TyreLife as base ==", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}  prior {y_all.mean():.4f}",
          flush=True)

    if args.smoke:
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(len(train), size=50_000, replace=False)
        train = train.iloc[sample_idx].reset_index(drop=True)
        y_all = train[TARGET].astype(int).values
        print(f"  SMOKE: subset to {train.shape}", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_all)), y_all))
    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)

    # Per-fold: build stints over train rows, fit, predict on val and test
    fold_aucs = []
    for k, (tr_idx, va_idx) in enumerate(fold_list):
        tk = time.time()
        tr_mask = np.zeros(len(train), dtype=bool)
        tr_mask[tr_idx] = True
        # train stints: aggregated from train rows of this fold
        tr_stints = aggregate_stints(train, tr_mask)
        print(f"  fold {k}: {len(tr_stints)} train stints "
              f"(events={tr_stints['event'].mean():.3f})", flush=True)
        est = fit_cox(tr_stints)

        # Predict on val rows (need per-row stint covariates; use train-fold
        # stint aggregates when the stint is in train, else fall back to
        # an aggregate over all-train rows of that stint -- but val rows of
        # a stint MAY not have a train-rows entry. Build a second aggregate
        # over the FULL training-set stints to fill those.)
        # For Stratified row folds, the same stint is almost always partly
        # in train -- but to be safe, fall back to all-train stints.
        full_train_stints = aggregate_stints(
            train, np.ones(len(train), dtype=bool))
        # Use tr_stints covariates where present, else full_train_stints
        # (val-only stints get their covariates from the global aggregate;
        # event/duration there will be from val rows too which is fine because
        # we only use covariates, not event/duration, for prediction.)
        merged_skeys = set(tr_stints["_skey"].values)
        fill_mask = ~full_train_stints["_skey"].isin(merged_skeys)
        merged_stints = pd.concat([tr_stints,
                                   full_train_stints.loc[fill_mask]],
                                  ignore_index=True)

        haz_val = hazard_at_rows(est, train.iloc[va_idx], merged_stints)
        oof[va_idx] = haz_val
        try:
            fa = roc_auc_score(y_all[va_idx], haz_val)
            fold_aucs.append(fa)
            print(f"    fold {k} AUC: {fa:.5f}  wall {time.time()-tk:.1f}s",
                  flush=True)
        except ValueError:
            print(f"    fold {k} AUC: ERR  wall {time.time()-tk:.1f}s",
                  flush=True)

        # Test: stint covariates from full-train aggregates union test stints
        # (test rows have a different set of (Driver, Race, Stint); covariates
        # are derived from the test rows themselves, which is fine -- we are
        # only using row covariates, not test labels.)
        test_stints = aggregate_stints(test.assign(PitStop=0), np.ones(len(test), dtype=bool))
        # Set duration/event placeholders -- not used at predict time.
        haz_test = hazard_at_rows(est, test, test_stints)
        test_pred += haz_test / N_FOLDS

    oof_auc = float(roc_auc_score(y_all, oof))
    print(f"\n  Standalone OOF AUC: {oof_auc:.5f}  (per-fold mean "
          f"{np.mean(fold_aucs):.5f})", flush=True)
    print(f"  Total wall: {time.time()-t0:.1f}s", flush=True)

    if args.smoke:
        print("  SMOKE only -- skipping artifact save.")
        return

    # Save artifacts in the K=14 builder's convention
    name = "R11_C_survival_hazard"
    oof_path = ART / f"oof_{name}_strat.npy"
    test_path = ART / f"test_{name}_strat.npy"
    np.save(oof_path, oof)
    np.save(test_path, test_pred)
    print(f"  Saved: {oof_path}  {test_path}", flush=True)

    # G1 gate
    G1_PASS = oof_auc >= 0.90
    print(f"\n  G1 (>= 0.90): {'PASS' if G1_PASS else 'FAIL'} "
          f"(standalone {oof_auc:.5f})", flush=True)

    # rho vs R7.1 PRIMARY OOF (informational; G3 measured in K=14 add)
    r71_oof = np.load(ART / "oof_K13_pathb_driverclass_stint_tau100000.npy")
    rho_oof, _ = spearmanr(oof, r71_oof)
    print(f"  rho_oof vs R7.1 PRIMARY: {rho_oof:.6f}", flush=True)

    summary = dict(
        round="R11_C_survival_hazard",
        oof_auc=oof_auc,
        per_fold_auc=fold_aucs,
        rho_oof_vs_R71=float(rho_oof),
        wall_s=time.time() - t0,
        G1_PASS=G1_PASS,
        oof_path=str(oof_path),
        test_path=str(test_path),
    )
    out_json = Path("audit/2026-05-19-round-11-survival.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
