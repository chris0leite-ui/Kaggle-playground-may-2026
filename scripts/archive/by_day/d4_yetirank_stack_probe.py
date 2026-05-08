"""d4_yetirank_stack_probe — does YetiRank earn a slot in the M5q pool?

Adds d4_cb_yetirank to the K=14 M5q pool (M5h + RealMLP) → K=15 LR meta.
Compares:
  1. Standalone Strat OOF AUC of YetiRank base.
  2. Spearman ρ vs M5q test (gate: <0.999 to avoid LB tie).
  3. New stack Strat OOF vs M5q's 0.95057.
  4. L1 contribution of YetiRank base in the new LR meta (rank within K).

Decision rule (informal):
  - If ρ < 0.998 AND new_stack_oof >= M5q − 1.0bp AND L1 mid-tier or higher,
    candidate is informative → submit slot.
  - If ρ >= 0.999, expect TIE_EXPECTED at LB; do not submit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5Q_S = 0.95057
SEED, N_FOLDS = 42, 5

POOL_M5Q = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
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
    ("realmlp", "realmlp"),
]
NEW_BASE = ("yetirank", "d4_cb_yetirank")


def load(name, suffix="strat"):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rank, logit])


def fit_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
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


def assemble(pool):
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool:
        oo, te = load(name)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    return (expand(np.column_stack(Xs_oof)),
            expand(np.column_stack(Xs_test)), names)


def l1_per_base(coef, names):
    K = len(names)
    return {n: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2 * K + i]))
            for i, n in enumerate(names)}


def main():
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    # Standalone YetiRank
    oof_yr, test_yr = load(NEW_BASE[1])
    auc_solo = float(roc_auc_score(y, oof_yr))
    m5q_test = np.load(ART / "test_m5q_strat.npy")[:, 1]
    rho_m5q, _ = spearmanr(test_yr, m5q_test)
    pool_consensus = np.load(ART / "test_pool_consensus.npy")
    rho_cons, _ = spearmanr(test_yr, pool_consensus)

    print(f"=== d4_cb_yetirank diversity scorecard ===")
    print(f"M5q baseline: Strat 0.95057, LB 0.95005\n")
    print(f"Standalone YetiRank Strat OOF: {auc_solo:.5f} ({(auc_solo-0.94075)*1e4:+.1f}bp vs base)")
    print(f"ρ vs M5q test:        {rho_m5q:.5f}")
    print(f"ρ vs M5h pool cons.:  {rho_cons:.5f}")

    # Build M5x stack: M5q pool + yetirank
    POOL_M5X = POOL_M5Q + [NEW_BASE]
    F_oof, F_test, names = assemble(POOL_M5X)
    oof_meta, test_meta, auc_stack, coef = fit_meta(F_oof, F_test, y)
    delta_m5q = (auc_stack - M5Q_S) * 1e4
    rho_stack_m5q, _ = spearmanr(test_meta, m5q_test)
    l1 = l1_per_base(coef, names)

    print(f"\n=== M5x stack (M5q pool + yetirank, K=15) ===")
    print(f"Strat OOF: {auc_stack:.5f}  Δ M5q: {delta_m5q:+.1f}bp")
    print(f"ρ stack vs M5q test: {rho_stack_m5q:.5f}  "
          f"[{'TIE_EXPECTED' if rho_stack_m5q >= 0.999 else 'PASS'}]")
    print(f"\nL1 ranking (K={len(names)}):")
    for rank_i, (n, v) in enumerate(sorted(l1.items(), key=lambda x: -x[1]), 1):
        marker = "  ← yetirank" if n == "yetirank" else ""
        print(f"  {rank_i:>2}. {n:<22s} L1={v:.3f}{marker}")

    # Save M5x stack artifact + sub
    np.save(ART / "oof_m5x_strat.npy",
            np.column_stack([1 - oof_meta, oof_meta]))
    np.save(ART / "test_m5x_strat.npy",
            np.column_stack([1 - test_meta, test_meta]))
    sample_sub = pd.read_csv("data/sample_submission.csv")
    sub = sample_sub.copy()
    sub[TARGET] = test_meta
    sub.to_csv("submissions/submission_m5x_yetirank.csv", index=False)

    summary = dict(
        new_base="yetirank",
        standalone_oof=auc_solo,
        rho_vs_m5q_test=float(rho_m5q),
        rho_vs_pool_consensus=float(rho_cons),
        m5x_stack_oof=auc_stack,
        delta_m5q_bp=delta_m5q,
        rho_stack_vs_m5q=float(rho_stack_m5q),
        gate=("TIE_EXPECTED" if rho_stack_m5q >= 0.999 else "PASS"),
        K=len(names),
        l1=l1,
        l1_yetirank_rank=int(sorted(l1.items(), key=lambda x: -x[1]).index(
            ("yetirank", l1["yetirank"])) + 1),
        pool=names,
    )
    (ART / "d4_yetirank_stack_probe.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/d4_yetirank_stack_probe.json")
    print(f"→ submissions/submission_m5x_yetirank.csv")


if __name__ == "__main__":
    main()
