"""scripts/t2_k10_primary.py — T2: build K=10 LR-meta PRIMARY artifact.

Per E9 forward-selection: K=10 = K=24 in OOF AUC. This script builds
the K=10 LR-meta OOF/test artifact for use as a candidate PRIMARY
replacement (no LB submit; pool-surgery freeze).

K=10 (per E9 pick order):
  1.  d17_h1d_yekenot_full
  2.  p1_single_cb_v3_gpu
  3.  f1_hgbc_deep
  4.  d16_orig_continuous_only
  5.  b_lapsuntilpit
  6.  baseline_two_anchor
  7.  d9_R6_next_compound
  8.  cb_year-cat
  9.  e5_optuna_lgbm
  10. d9f_FM_A
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K10_BASES = [
    "d17_h1d_yekenot_full", "p1_single_cb_v3_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P):
    n = len(P)
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    rk = np.column_stack([np.argsort(np.argsort(c)) / n for c in P.T])
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def main():
    train = pd.read_csv("data/train.csv", usecols=[TARGET])
    y = train[TARGET].astype(int).values

    # Load K=10 OOFs and tests
    P_oof = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in K10_BASES])
    P_test = np.column_stack([_pos(ART / f"test_{b}_strat.npy") for b in K10_BASES])
    F_oof = _expand(P_oof)
    F_test = _expand(P_test)
    print(f"K={len(K10_BASES)} bases; F shape {F_oof.shape}")

    # Reference d18 PRIMARY for ρ comparison
    primary_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    primary_test = _pos(ART / "test_d17_K24_d18pool_h1d_strat.npy")
    primary_oof_auc = roc_auc_score(y, primary_oof)
    print(f"d18 PRIMARY (K=24) OOF AUC: {primary_oof_auc:.5f}")

    # 5-fold OOF
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs = []
    for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                max_iter=2000)
        lr.fit(F_oof[tr], y[tr])
        oof[va] = lr.predict_proba(F_oof[va])[:, 1]
        fa = roc_auc_score(y[va], oof[va])
        fold_aucs.append(round(float(fa), 5))
        print(f"  fold {fi+1}/5 AUC={fa:.5f}", flush=True)
    auc_k10 = roc_auc_score(y, oof)

    # Full-train fit for test
    lr_full = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                 max_iter=2000)
    lr_full.fit(F_oof, y)
    test_pred = lr_full.predict_proba(F_test)[:, 1]

    # Compare to d18 PRIMARY
    rho_oof, _ = spearmanr(oof, primary_oof)
    rho_test, _ = spearmanr(test_pred, primary_test)
    delta_bp = (auc_k10 - primary_oof_auc) * 1e4

    print(f"\n=== T2: K=10 PRIMARY candidate ===")
    print(f"K=10 OOF AUC:        {auc_k10:.5f}")
    print(f"d18 PRIMARY OOF AUC: {primary_oof_auc:.5f}")
    print(f"Δ K=10 − d18:        {delta_bp:+.3f} bp")
    print(f"ρ_oof  (K=10, d18):  {rho_oof:.5f}")
    print(f"ρ_test (K=10, d18):  {rho_test:.5f}")
    print(f"fold AUCs: {fold_aucs}")

    # Save artifacts
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / "oof_K10_lr_meta_strat.npy", oof2)
    np.save(ART / "test_K10_lr_meta_strat.npy", test2)

    out = {
        "k10_bases": K10_BASES,
        "k10_oof_auc": round(float(auc_k10), 6),
        "fold_aucs": fold_aucs,
        "d18_primary_oof_auc": round(float(primary_oof_auc), 6),
        "delta_bp_vs_d18": round(float(delta_bp), 3),
        "rho_oof_vs_d18": round(float(rho_oof), 6),
        "rho_test_vs_d18": round(float(rho_test), 6),
        "lr_meta_coef_l1_norm": round(float(np.abs(lr_full.coef_).sum()), 4),
    }
    json_path = ART / "t2_k10_primary.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n→ artifacts: oof_K10_lr_meta_strat.npy + test_K10_lr_meta_strat.npy")
    print(f"→ JSON:      {json_path}")


if __name__ == "__main__":
    main()
