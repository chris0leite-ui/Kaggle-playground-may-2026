"""M5o + M5p — orthogonal-family stack candidates for slots 9-10.

M5o = M5h pool + LR-FE (K=14).  Tests: does LR-FE's diversity
  (ρ=0.87 vs M5h, |Δ|@Stint2=0.154) lift LB when added to the full
  GBDT ensemble?

M5p = M5n_3b base set + LR-FE + EBM (K=6). Diverse 3 GBDTs +
  baseline + 2 orthogonal mechanism families. Tests: does a
  small-but-orthogonal-only stack beat M5h?

Both refit LR meta on raw+rank+logit expansion.
Reports Strat OOF + Spearman ρ vs M5h + L1 contribution of new bases.
R1: Strat-only.
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
M5H_S = 0.95043
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

POOL_M5N_3B = [
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    ("baseline", "baseline_two_anchor"),
]

LR_FE = ("d3g_lr_fe", "d3g_lr_fe")
EBM = ("d3e_ebm", "d3e_ebm")
H1 = ("d3f_pseudo_lgbm", "d3f_pseudo_lgbm")


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

    # === M5o = M5h + LR-FE (K=14) ===
    print("=== M5o: M5h + LR-FE (K=14) ===")
    pool_o = POOL_M5H + [LR_FE]
    F_oof, F_test, names = assemble(pool_o)
    oof_o, test_o, auc_o, coef_o = fit_meta(F_oof, F_test, y)
    rho_o, _ = spearmanr(test_o, m5h_test)
    delta_o = (auc_o - M5H_S) * 1e4
    l1_o = l1_per_base(coef_o, names)
    print(f"  Strat OOF: {auc_o:.5f}  Δ M5h: {delta_o:+.1f}bp  ρ vs M5h test: {rho_o:.5f}")
    print(f"  L1 per base (sorted):")
    for n, v in sorted(l1_o.items(), key=lambda x: -x[1]):
        marker = " ← NEW" if n == "d3g_lr_fe" else ""
        print(f"    {n:<22s} L1={v:.3f}{marker}")

    np.save(ART / "oof_m5o_strat.npy", np.column_stack([1 - oof_o, oof_o]))
    np.save(ART / "test_m5o_strat.npy", np.column_stack([1 - test_o, test_o]))
    sub = sample_sub.copy()
    sub[TARGET] = test_o
    sub.to_csv("submissions/submission_m5o_lr_fe_stack.csv", index=False)

    # === M5p = M5n_3b + LR-FE + EBM (K=6) ===
    print("\n=== M5p: M5n_3b + LR-FE + EBM (K=6) ===")
    pool_p = POOL_M5N_3B + [LR_FE, EBM]
    F_oof, F_test, names = assemble(pool_p)
    oof_p, test_p, auc_p, coef_p = fit_meta(F_oof, F_test, y)
    rho_p, _ = spearmanr(test_p, m5h_test)
    delta_p = (auc_p - M5H_S) * 1e4
    l1_p = l1_per_base(coef_p, names)
    print(f"  Strat OOF: {auc_p:.5f}  Δ M5h: {delta_p:+.1f}bp  ρ vs M5h test: {rho_p:.5f}")
    print(f"  L1 per base (sorted):")
    for n, v in sorted(l1_p.items(), key=lambda x: -x[1]):
        marker = " ← NEW" if n in ("d3g_lr_fe", "d3e_ebm") else ""
        print(f"    {n:<22s} L1={v:.3f}{marker}")

    np.save(ART / "oof_m5p_strat.npy", np.column_stack([1 - oof_p, oof_p]))
    np.save(ART / "test_m5p_strat.npy", np.column_stack([1 - test_p, test_p]))
    sub = sample_sub.copy()
    sub[TARGET] = test_p
    sub.to_csv("submissions/submission_m5p_minimal_orthogonal.csv", index=False)

    # === Pre-submit diff vs M5h (mandatory per friction rule) ===
    print("\n=== Pre-submit diff (rho<0.999 gate) ===")
    print(f"  M5o vs M5h: ρ={rho_o:.5f}  {'PASS' if rho_o < 0.999 else 'TIE_EXPECTED'}")
    print(f"  M5p vs M5h: ρ={rho_p:.5f}  {'PASS' if rho_p < 0.999 else 'TIE_EXPECTED'}")

    res = dict(
        M5o=dict(K=len(POOL_M5H) + 1, strat=auc_o, delta_m5h_bp=delta_o,
                 spearman_vs_m5h=rho_o, l1_per_base=l1_o,
                 pool=[p[0] for p in pool_o]),
        M5p=dict(K=len(POOL_M5N_3B) + 2, strat=auc_p, delta_m5h_bp=delta_p,
                 spearman_vs_m5h=rho_p, l1_per_base=l1_p,
                 pool=[p[0] for p in pool_p]),
    )
    (ART / "m5op_orthogonal_stacks_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n→ submissions/submission_m5o_lr_fe_stack.csv")
    print(f"→ submissions/submission_m5p_minimal_orthogonal.csv")


if __name__ == "__main__":
    main()
