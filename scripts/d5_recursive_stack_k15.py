"""D5 Path C extension — K=15 stack: M5q pool (14) + recursive base.

The 2-base [M5q, recursive] LR stack was null (-0.2bp vs M5q anchor)
but the meta has no degrees of freedom at K=2. Correct test: add
recursive as a 15th base to the M5q pool (M5h's 13 GBDTs + RealMLP +
recursive) and rebuild the LR meta. ρ(recursive_test, m5q_test) =
0.99159 sits in the "real-LB-delta" band per the d4 calibration
(>0.994 / <0.999).

Variants
  M5_K15a: M5q pool + recursive
  M5_K15b: M5q pool + recursive (drop e3_hgbc — recursive is HGBC
           with M5q feature, so e3_hgbc may be the closest analog)
  M5_K15c: M5q pool + recursive (drop both f1/f2 HGBC variants)

Reports per variant: Strat OOF, Δ M5q, ρ vs M5q test, L1 of recursive,
gate verdict.
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

POOL_M5H = [
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
]
REALMLP = ("realmlp", "realmlp")
RECURSIVE = ("recursive", "d5_recursive_m5q")
POOL_M5Q = POOL_M5H + [REALMLP]


def load(name, suffix="strat"):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    eps = 1e-9
    Pc = np.clip(P, eps, 1 - eps)
    logit = np.log(Pc / (1 - Pc))
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
    return (expand(np.column_stack(Xs_oof)),
            expand(np.column_stack(Xs_test)),
            np.column_stack(Xs_test),  # raw test for diversity check
            names)


def l1_per_base(coef, names):
    K = len(names)
    return {n: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2 * K + i]))
            for i, n in enumerate(names)}


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    m5q_test = np.load(ART / "test_m5q_strat.npy")[:, 1]

    POOL_M5Q_minus_e3 = [p for p in POOL_M5Q if p[0] != "e3_hgbc"]
    POOL_M5Q_minus_f12 = [p for p in POOL_M5Q if p[0] not in ("f1_hgbc_deep", "f2_hgbc_shallow")]

    variants = [
        ("M5_K15a", POOL_M5Q + [RECURSIVE]),
        ("M5_K15b", POOL_M5Q_minus_e3 + [RECURSIVE]),
        ("M5_K15c", POOL_M5Q_minus_f12 + [RECURSIVE]),
    ]

    print(f"=== K=15 recursive-augmented stack sweep ===")
    print(f"M5q anchor: Strat {M5Q_S:.5f}, LB 0.95005\n")
    header = f"{'variant':<8} {'K':>3} {'Strat OOF':>11} {'Δ M5q bp':>9} "
    header += f"{'ρ M5q test':>11} {'rec L1':>9} {'gate':<14}"
    print(header)

    results = {}
    for label, pool in variants:
        F_oof, F_test, _raw_test, names = assemble(pool)
        oof, test_p, auc, coef = fit_meta(F_oof, F_test, y)
        rho, _ = spearmanr(test_p, m5q_test)
        l1 = l1_per_base(coef, names)
        delta = (auc - M5Q_S) * 1e4
        rec_l1 = l1.get("recursive", 0.0)
        # Gate: TIE_EXPECTED if rho >= 0.9997 (Kaggle 5-decimal quantization)
        # PASS if rho < 0.999, REAL-LB-DELTA if 0.994 <= rho < 0.999
        if rho >= 0.9997:
            gate = "TIE_EXPECTED"
        elif rho >= 0.999:
            gate = "TIE_LIKELY"
        elif rho >= 0.994:
            gate = "REAL_DELTA"
        else:
            gate = "DIVERGENT"
        print(f"{label:<8} {len(names):>3} {auc:>11.5f} {delta:>+9.2f} "
              f"{rho:>11.5f} {rec_l1:>9.3f}  {gate:<14}")
        results[label] = dict(K=len(names), strat=auc, delta_m5q_bp=delta,
                              spearman_vs_m5q=rho, recursive_l1=rec_l1,
                              gate=gate, l1=l1, pool=names,
                              oof=oof, test=test_p)

    for label, r in results.items():
        print(f"\n=== {label} L1 per base ===")
        for n, v in sorted(r["l1"].items(), key=lambda x: -x[1]):
            marker = " ← RECURSIVE" if n == "recursive" else \
                     (" ← realmlp" if n == "realmlp" else "")
            print(f"  {n:<22s} L1={v:.3f}{marker}")

    for label, r in results.items():
        np.save(ART / f"oof_{label.lower()}_strat.npy",
                np.column_stack([1 - r["oof"], r["oof"]]))
        np.save(ART / f"test_{label.lower()}_strat.npy",
                np.column_stack([1 - r["test"], r["test"]]))
        sub = sample_sub.copy()
        sub[TARGET] = r["test"]
        sub.to_csv(f"submissions/submission_{label.lower()}.csv", index=False)

    summary = {
        label: {k: v for k, v in r.items() if k not in ("oof", "test")}
        for label, r in results.items()
    }
    (ART / "d5_k15_recursive_stacks_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/d5_k15_recursive_stacks_results.json")


if __name__ == "__main__":
    main()
