"""Day-16 H7 — Conformal / per-bin recalibration of PRIMARY OOF.

δ2/3 axis (inference-output normalization). The K=21 LR meta with
[raw,rank,logit] expand is a global calibration; per-segment AUC may
benefit from per-bin recalibration that's INNER-CV-validated to avoid
the posthoc-isotonic-overfits-OOF friction tag.

Inner-CV protocol (5 folds on the OOF rows themselves):
  for fold k:
    fit_isotonic_per_bin(OOF_rows[~k], y[~k])
    apply to OOF_rows[k] -> recalibrated_oof[k]
  recalibrated_oof = stitched
  AUC(recalibrated_oof) vs AUC(PRIMARY_oof)

Bins to test:
  S1. Year × Compound (4 × 5 = 20 bins)
  S2. Year × Compound × Stint (4 × 5 × 6 = ≤120 bins; some empty)
  S3. Compound only (5 bins)
  S4. RaceProgress quintile × Compound (5 × 5 = 25 bins)

For each, fit per-bin sklearn IsotonicRegression on (PRIMARY_oof, y)
within the train side, apply on the held-out side. Bins below
min_samples=200 fall back to global isotonic.

Output:
  oof_d16_h7_conformal_<bin>_strat.npy   if AUC improves
  test_d16_h7_conformal_<bin>_strat.npy
  d16_h7_conformal_results.json   sweep table
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

ART = Path("scripts/artifacts")

PRIMARY_OOF = ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"

SEED = 42
N_FOLDS = 5
MIN_BIN_SAMPLES = 200


def _fit_isotonic_per_bin(p, y, bin_ids, min_samples=MIN_BIN_SAMPLES):
    """Fit one IsotonicRegression per bin id (with global fallback for
    sparse bins). Return dict bin_id -> fitted estimator (or 'global')
    plus a single global estimator."""
    g = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    g.fit(p, y)
    fits = {"__global__": g}
    for bid in np.unique(bin_ids):
        mask = bin_ids == bid
        if mask.sum() < min_samples:
            continue
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(p[mask], y[mask])
        fits[int(bid)] = ir
    return fits


def _apply_isotonic_per_bin(p, bin_ids, fits):
    out = np.empty_like(p, dtype=np.float64)
    g = fits["__global__"]
    for bid in np.unique(bin_ids):
        mask = bin_ids == bid
        if int(bid) in fits:
            out[mask] = fits[int(bid)].predict(p[mask])
        else:
            out[mask] = g.predict(p[mask])
    return out


def encode_bins(train: pd.DataFrame, test: pd.DataFrame, scheme: str):
    """Build integer bin ids for train and test under various schemes."""
    if scheme == "compound":
        comp = pd.concat([train["Compound"], test["Compound"]]).astype(str)
        cats = comp.unique()
        cmap = {v: i for i, v in enumerate(cats)}
        return train["Compound"].astype(str).map(cmap).values, \
               test["Compound"].astype(str).map(cmap).values
    if scheme == "year_compound":
        a = pd.concat([train[["Year", "Compound"]], test[["Year", "Compound"]]])
        keys = a["Year"].astype(str) + "|" + a["Compound"].astype(str)
        cats = keys.unique()
        cmap = {v: i for i, v in enumerate(cats)}
        tr = (train["Year"].astype(str) + "|" + train["Compound"].astype(str)).map(cmap).values
        te = (test["Year"].astype(str) + "|" + test["Compound"].astype(str)).map(cmap).values
        return tr, te
    if scheme == "year_compound_stint":
        a = pd.concat([train[["Year", "Compound", "Stint"]],
                       test[["Year", "Compound", "Stint"]]])
        keys = (a["Year"].astype(str) + "|" + a["Compound"].astype(str) +
                "|" + a["Stint"].astype(str))
        cats = keys.unique()
        cmap = {v: i for i, v in enumerate(cats)}
        tr = (train["Year"].astype(str) + "|" + train["Compound"].astype(str)
              + "|" + train["Stint"].astype(str)).map(cmap).values
        te = (test["Year"].astype(str) + "|" + test["Compound"].astype(str)
              + "|" + test["Stint"].astype(str)).map(cmap).values
        return tr, te
    if scheme == "raceprog_q5_compound":
        # build Race-progress quintile bins on train, apply to test
        rp = train["RaceProgress"].values
        q = np.quantile(rp, [0.2, 0.4, 0.6, 0.8])
        def to_bin(x):
            return np.digitize(x, q)
        tr_q = to_bin(train["RaceProgress"].values)
        te_q = to_bin(test["RaceProgress"].values)
        # cross with compound
        comp = pd.concat([train["Compound"], test["Compound"]]).astype(str)
        cats = comp.unique()
        cmap = {v: i for i, v in enumerate(cats)}
        tr_c = train["Compound"].astype(str).map(cmap).values
        te_c = test["Compound"].astype(str).map(cmap).values
        return tr_q * len(cmap) + tr_c, te_q * len(cmap) + te_c
    raise ValueError(scheme)


def main():
    t0 = time.time()
    print("[h7] loading PRIMARY OOF + train ...", flush=True)
    train = pd.read_csv("data/train.csv",
                        usecols=["PitNextLap", "Year", "Compound", "Stint",
                                 "RaceProgress"])
    test = pd.read_csv("data/test.csv",
                       usecols=["Year", "Compound", "Stint", "RaceProgress"])
    y = train["PitNextLap"].astype(int).values
    n_train, n_test = len(train), len(test)

    p_oof = np.load(PRIMARY_OOF)[:, 1].astype(np.float64)
    p_test = np.load(PRIMARY_TEST)[:, 1].astype(np.float64)
    auc_base = float(roc_auc_score(y, p_oof))
    print(f"[h7] PRIMARY OOF AUC = {auc_base:.6f}", flush=True)

    schemes = ["compound", "year_compound", "year_compound_stint",
               "raceprog_q5_compound"]
    results = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n_train), y))

    for sch in schemes:
        print(f"\n[h7] scheme={sch}", flush=True)
        tr_bin, te_bin = encode_bins(train, test, sch)
        n_bins = int(max(tr_bin.max(), te_bin.max())) + 1
        print(f"  bins: {n_bins}", flush=True)

        recal_oof = np.zeros(n_train, dtype=np.float64)
        for fold, (tr, va) in enumerate(splits):
            fits = _fit_isotonic_per_bin(p_oof[tr], y[tr], tr_bin[tr])
            recal_oof[va] = _apply_isotonic_per_bin(p_oof[va], tr_bin[va], fits)
        auc_recal = float(roc_auc_score(y, recal_oof))
        delta_bp = (auc_recal - auc_base) * 1e4
        print(f"  inner-CV recal OOF AUC = {auc_recal:.6f}  Δ {delta_bp:+.3f}bp",
              flush=True)

        # If positive lift, build full-OOF refit and apply to test
        # (full-train fit, applied to test).
        if delta_bp >= 0.05:
            full_fits = _fit_isotonic_per_bin(p_oof, y, tr_bin)
            recal_test = _apply_isotonic_per_bin(p_test, te_bin, full_fits)
            np.save(ART / f"oof_d16_h7_conformal_{sch}_strat.npy",
                    np.column_stack([1.0 - recal_oof, recal_oof]))
            np.save(ART / f"test_d16_h7_conformal_{sch}_strat.npy",
                    np.column_stack([1.0 - recal_test, recal_test]))
            saved = True
        else:
            saved = False

        results.append(dict(scheme=sch, n_bins=n_bins,
                            inner_cv_oof_auc=auc_recal,
                            delta_bp=delta_bp, saved=saved))

    res = dict(primary_oof_auc=auc_base, sweep=results,
               wall_s=time.time() - t0)
    (ART / "d16_h7_conformal_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n[h7] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
