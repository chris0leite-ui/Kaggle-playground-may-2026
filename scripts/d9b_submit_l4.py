"""Build the L4 K=20 swap submission and write CSV.

The d9b K=N stack winner: drop 2 most-redundant rule_residuals from
PRIMARY, add R6 + R10 + R7 + R14_L4 (4 d9 bases). Rebuild the K=20
LR-meta stack and write submission_d9b_k20_swap_l4.csv.

Per d9b audit: predicted Δ LB = +0.19bp, ρ vs PRIMARY = 0.99978.
Below the 0.9995 tightened tie threshold (HANDOVER §1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

POOL = [
    ("baseline_two_anchor",), ("d2a_te",), ("m2_xgb",),
    ("e1_catboost_sub",), ("e3_hgbc",), ("e5_optuna_lgbm",),
    ("a_horizon",), ("b_lapsuntilpit",),
    ("f1_hgbc_deep",), ("f2_hgbc_shallow",),
    ("cb_year-cat",), ("cb_lossguide",), ("cb_slow-wide-bag",),
    ("realmlp",),
    ("d6_rule_driver_compound",), ("d6_rule_year_race",),
    ("d9_R6_next_compound",), ("d9_R10_driver_eb",),
    ("d9_R7_prev_compound",), ("d9b_R14_L4",),
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1]


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    Xs_oof, Xs_test = [], []
    for (n,) in POOL:
        oo = np.load(ART / f"oof_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te)
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    print(f"K={len(POOL)} swap-L4 stack OOF: {auc:.5f}")
    sub = sample_sub.copy(); sub[TARGET] = tp
    out = "submissions/submission_d9b_k20_swap_l4.csv"
    sub.to_csv(out, index=False)
    print(f"→ wrote {out}")
    np.save(ART / "oof_d9b_k20_swap_l4_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d9b_k20_swap_l4_strat.npy",
            np.column_stack([1 - tp, tp]))
    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy"
                           )[:, 1].astype(np.float64)
    rho, _ = spearmanr(tp, primary_test)
    print(f"ρ vs d6_k18 PRIMARY: {rho:.5f}")


if __name__ == "__main__":
    main()
