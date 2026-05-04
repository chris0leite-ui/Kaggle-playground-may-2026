"""M5j — swap variant: drop d2a_te, add d3a_te_unified (13 bases).

Tests the PI's hypothesis: "would it be better to have one model including
all the TE features and replacing the other one if it performed better?"

If swap-Strat ≥ M5h-Strat (0.95043) → d3a replaces d2a as the canonical
TE base in the pool.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5H_S, M5H_G = 0.95043, 0.93087
SEED, N_FOLDS = 42, 5

POOL_SWAP = [
    ("baseline", "baseline_two_anchor"),
    # ("d2a_te", "d2a_te"),                 # DROPPED in swap
    ("d3a_te_unified", "d3a_te_unified"),   # SWAPPED IN
    ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"),
    ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
]


def load(name, suffix):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rank, logit])


def fit_meta(F_oof, F_test, y, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc, lr_full.coef_.ravel()


def assemble(pool, suffix):
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool:
        oo, te = load(name, suffix)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    P_oof = np.column_stack(Xs_oof)
    P_test = np.column_stack(Xs_test)
    return expand(P_oof), expand(P_test), names


def main():
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    print(f"=== M5j — swap d2a → d3a (pool size {len(POOL_SWAP)}) ===")

    print("=== Strat ===")
    F_oof_s, F_test_s, names = assemble(POOL_SWAP, "strat")
    oof_s, test_s, auc_s, coef_s = fit_meta(F_oof_s, F_test_s, y)
    print(f"  M5j Strat: {auc_s:.5f}  Δ M5h={(auc_s-M5H_S)*1e4:+.1f}bp  K={len(names)}")
    K = len(names)
    l1 = {n: float(abs(coef_s[i]) + abs(coef_s[K+i]) + abs(coef_s[2*K+i]))
          for i, n in enumerate(names)}
    print("  L1 per base (Strat):")
    for n, v in sorted(l1.items(), key=lambda x: -x[1]):
        print(f"    {n:<22s} L1={v:.3f}")

    print("=== GroupKF ===")
    F_oof_g, F_test_g, _ = assemble(POOL_SWAP, "groupkf")
    oof_g, test_g, auc_g, _ = fit_meta(F_oof_g, F_test_g, y)
    print(f"  M5j GroupKF: {auc_g:.5f}  Δ M5h={(auc_g-M5H_G)*1e4:+.1f}bp")

    np.save(ART / "oof_m5j_strat.npy", np.column_stack([1 - oof_s, oof_s]))
    np.save(ART / "test_m5j_strat.npy", np.column_stack([1 - test_s, test_s]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv("submissions/submission_m5j_swap.csv", index=False)

    res = dict(strat=auc_s, groupkf=auc_g,
               delta_m5h_strat_bp=(auc_s - M5H_S) * 1e4,
               delta_m5h_groupkf_bp=(auc_g - M5H_G) * 1e4,
               pool=names, l1_strat=l1)
    (ART / "m5j_swap_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
