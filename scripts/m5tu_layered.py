"""M5t / M5u — Layered orthogonal bases on M5q anchor.

Day-4 slot 1 won: M5q (M5h + RealMLP, K=14) LB 0.95005 (+14bp).
Slot 2 candidates: layer additional orthogonal bases on top of M5q.

Variants:
  M5t = M5q + H1 (K=15)
  M5u = M5q + H1 + EBM (K=16)

For each: Strat OOF, ρ vs M5q test, L1 of new bases.
Pre-submit-diff vs M5q (the new PRIMARY).
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
M5H_S = 0.95043   # for context
M5Q_S = 0.95057   # current PRIMARY OOF
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
H1 = ("d3f_pseudo_lgbm", "d3f_pseudo_lgbm")
EBM = ("d3e_ebm", "d3e_ebm")


def load(name, suffix="strat"):
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


def assemble(pool, suffix="strat"):
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool:
        oo, te = load(name, suffix)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    return expand(np.column_stack(Xs_oof)), expand(np.column_stack(Xs_test)), names


def l1_per_base(coef, names):
    K = len(names)
    return {n: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2 * K + i]))
            for i, n in enumerate(names)}


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    m5h_test = np.load(ART / "test_m5h_strat.npy")[:, 1]
    m5q_test = np.load(ART / "test_m5q_strat.npy")[:, 1]

    variants = [
        ("M5t",  POOL_M5Q + [H1]),
        ("M5u",  POOL_M5Q + [H1, EBM]),
    ]

    print(f"=== Layered-orthogonal sweep on M5q anchor ===")
    print(f"M5q (current PRIMARY): Strat 0.95057, LB 0.95005\n")
    print(f"{'variant':<6} {'K':>3} {'Strat OOF':>11} {'Δ M5q':>8} "
          f"{'ρ vs M5q':>11} {'ρ vs M5h':>11}")

    results = {}
    for label, pool in variants:
        F_oof, F_test, names = assemble(pool)
        oof, test_p, auc, coef = fit_meta(F_oof, F_test, y)
        rho_q, _ = spearmanr(test_p, m5q_test)
        rho_h, _ = spearmanr(test_p, m5h_test)
        l1 = l1_per_base(coef, names)
        delta = (auc - M5Q_S) * 1e4
        gate_q = "PASS" if rho_q < 0.999 else "TIE_EXPECTED"
        gate_h = "PASS" if rho_h < 0.999 else "TIE_EXPECTED"
        print(f"{label:<6} {len(names):>3} {auc:>11.5f} {delta:>+8.1f} "
              f"{rho_q:>11.5f}[{gate_q[:4]}] {rho_h:>11.5f}[{gate_h[:4]}]")
        results[label] = dict(K=len(names), strat=auc, delta_m5q_bp=delta,
                              spearman_vs_m5q=rho_q, spearman_vs_m5h=rho_h,
                              gate_vs_m5q=gate_q, gate_vs_m5h=gate_h,
                              l1=l1, pool=names, oof=oof, test=test_p)

    # Show full L1 ranking for each variant
    for label, r in results.items():
        print(f"\n=== {label} L1 per base ===")
        for n, v in sorted(r["l1"].items(), key=lambda x: -x[1]):
            marker = ""
            if n == "realmlp": marker = " ← realmlp"
            elif n == "d3f_pseudo_lgbm": marker = " ← H1"
            elif n == "d3e_ebm": marker = " ← EBM"
            print(f"  {n:<22s} L1={v:.3f}{marker}")

    # Save artifacts for each variant
    for label, r in results.items():
        np.save(ART / f"oof_{label.lower()}_strat.npy",
                np.column_stack([1 - r["oof"], r["oof"]]))
        np.save(ART / f"test_{label.lower()}_strat.npy",
                np.column_stack([1 - r["test"], r["test"]]))
        sub = sample_sub.copy()
        sub[TARGET] = r["test"]
        sub.to_csv(f"submissions/submission_{label.lower()}_layered.csv", index=False)

    summary = {
        label: {k: v for k, v in r.items() if k not in ("oof", "test")}
        for label, r in results.items()
    }
    (ART / "m5tu_layered_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/m5tu_layered_results.json")


if __name__ == "__main__":
    main()
