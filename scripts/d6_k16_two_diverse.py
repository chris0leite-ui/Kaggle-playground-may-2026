"""K=16 stack: M5q pool + recursive + rule_residual.

Both new bases have positive minimal-meta lift over M5q:
  recursive (Day-5):  K=2 minimal LR-meta OOF 0.95055 (-0.2bp; just inside)
  rule_residual:      K=2 minimal LR-meta OOF 0.95061 (+0.4bp PASS)

And both are structurally diverse from M5q:
  ρ(recursive_test, m5q) = 0.99159
  ρ(rule_residual_test, m5q) = 0.92887  (most diverse since RealMLP)

K=15 with each individually:
  +recursive (M5_K15a):    Strat 0.95056  Δ M5q -0.06bp  ρ 0.99991  TIE
  +rule_residual:          Strat 0.95062  Δ M5q +0.51bp  ρ 0.99971  TIE_EXPECTED

Test: K=16 = M5q + recursive + rule_residual. Hypothesis: combining
two structurally-different residual mechanisms gives the LR meta
enough freedom to route corrections that single-base adds cannot.

Strat-only (R1).  Pre-submit-diff vs M5q on result.
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
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
M5Q_S, M5Q_LB = 0.95057, 0.95005
RHO_TIE = 0.999

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
EXTRA = [("recursive", "d5_recursive_m5q"), ("rule_residual", "d6_rule_residual")]


def load(name):
    oo = np.load(ART / f"oof_{name}_strat.npy")[:, 1].astype(np.float64)
    te = np.load(ART / f"test_{name}_strat.npy")[:, 1].astype(np.float64)
    return oo, te


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
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]

    Xs_oof, Xs_test, names = [], [], []
    for label, n in POOL_M5Q + EXTRA:
        oo, te = load(n)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    K = len(names)
    print(f"K={K} pool members")

    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, test_m5q)
    delta = (auc - M5Q_S) * 1e4
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"K=16 LR-meta Strat OOF: {auc:.5f}  Δ M5q {delta:+.2f}bp  "
          f"ρ vs M5q test {rho:.5f}")
    print(f"L1 ranking:")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1]):
        marker = " ← NEW" if n in ("recursive", "rule_residual") else ""
        print(f"  {n:<22s} L1={v:.3f}{marker}")

    pred_lb = M5Q_LB + (auc - M5Q_S) - (
        0 if rho >= RHO_TIE
        else 0.0001 if rho >= 0.995
        else 0.00025 if rho >= 0.99
        else 0.0004)
    print(f"\nPredicted LB: {pred_lb:.5f}  (vs M5q LB {M5Q_LB:.5f})")
    if rho >= RHO_TIE:
        print("  → TIE_EXPECTED (ρ ≥ 0.999)")
    elif auc < M5Q_S + 1.0/1e4:
        print("  → DO NOT SLOT (K=16 OOF below M5q + 1bp threshold)")
    else:
        print("  → SLOT-CANDIDATE (verify pre-submit-diff)")

    np.save(ART / "oof_d6_k16_two_diverse_strat.npy",
            np.column_stack([1 - mo, mo]))
    np.save(ART / "test_d6_k16_two_diverse_strat.npy",
            np.column_stack([1 - tp, tp]))
    sub = sample_sub.copy(); sub[TARGET] = tp
    sub.to_csv("submissions/submission_d6_k16_two_diverse.csv", index=False)
    (ART / "d6_k16_two_diverse_results.json").write_text(json.dumps(dict(
        K=K, strat_oof=auc, delta_m5q_bp=delta,
        rho_vs_m5q_test=float(rho), pred_lb=float(pred_lb),
        l1_ranking=l1,
    ), indent=2))
    print(f"\n→ scripts/artifacts/d6_k16_two_diverse_results.json")


if __name__ == "__main__":
    main()
