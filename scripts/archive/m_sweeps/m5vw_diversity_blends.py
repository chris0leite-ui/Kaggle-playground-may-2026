"""M5v / M5w — Diversity-driven slot-2 candidates.

M5q's LB lift came from RealMLP specifically. Adding more bases on
top of M5q (M5t, M5u) dilutes RealMLP's L1 contribution and ties LB.
Try two different shapes:

  M5v = M5q + LR-FE (K=15). LR-FE is the most-orthogonal base from
        Day-3 (ρ=0.869 vs M5h). Tests whether high-diversity-low-quality
        addition shifts rank when added to a strong anchor.

  M5w = 0.5 * M5q_test + 0.5 * RealMLP_standalone_test.  Simple ensemble
        (not stacking). RealMLP's standalone test is direct prediction,
        not LR-meta-derived. Averaging shifts the rank by RealMLP's
        unique signal.
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
LR_FE = ("d3g_lr_fe", "d3g_lr_fe")


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
    m5q_test = np.load(ART / "test_m5q_strat.npy")[:, 1]
    m5q_oof = np.load(ART / "oof_m5q_strat.npy")[:, 1]
    realmlp_oof = np.load(ART / "oof_realmlp_strat.npy")[:, 1]
    realmlp_test = np.load(ART / "test_realmlp_strat.npy")[:, 1]

    # === M5v = M5q + LR-FE ===
    print("=== M5v: M5q + LR-FE (K=15) ===")
    pool_v = POOL_M5Q + [LR_FE]
    F_oof, F_test, names = assemble(pool_v)
    oof_v, test_v, auc_v, coef_v = fit_meta(F_oof, F_test, y)
    rho_v, _ = spearmanr(test_v, m5q_test)
    delta_v = (auc_v - M5Q_S) * 1e4
    l1_v = l1_per_base(coef_v, names)
    gate_v = "PASS" if rho_v < 0.999 else "TIE_EXPECTED"
    print(f"  Strat OOF: {auc_v:.5f}  Δ M5q: {delta_v:+.1f}bp  "
          f"ρ vs M5q: {rho_v:.5f} [{gate_v}]")
    print(f"  L1 ranking:")
    for n, val in sorted(l1_v.items(), key=lambda x: -x[1]):
        marker = ""
        if n == "realmlp": marker = " ← realmlp"
        elif n == "d3g_lr_fe": marker = " ← LR-FE"
        print(f"    {n:<22s} L1={val:.3f}{marker}")

    # === M5w = simple 0.5 M5q + 0.5 RealMLP_standalone ===
    print("\n=== M5w: 0.5*M5q + 0.5*RealMLP_standalone (simple blend) ===")
    blend_oof = 0.5 * m5q_oof + 0.5 * realmlp_oof
    blend_test = 0.5 * m5q_test + 0.5 * realmlp_test
    auc_w = float(roc_auc_score(y, blend_oof))
    rho_w, _ = spearmanr(blend_test, m5q_test)
    delta_w = (auc_w - M5Q_S) * 1e4
    gate_w = "PASS" if rho_w < 0.999 else "TIE_EXPECTED"
    print(f"  Strat OOF: {auc_w:.5f}  Δ M5q: {delta_w:+.1f}bp  "
          f"ρ vs M5q: {rho_w:.5f} [{gate_w}]")

    # Also: 0.7/0.3 and 0.3/0.7 blends to see weight sensitivity
    print("\n=== Blend weight sensitivity ===")
    for w_q in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        bt = w_q * m5q_test + (1 - w_q) * realmlp_test
        bo = w_q * m5q_oof + (1 - w_q) * realmlp_oof
        a = float(roc_auc_score(y, bo))
        r, _ = spearmanr(bt, m5q_test)
        gate = "PASS" if r < 0.999 else "TIE"
        print(f"  w_M5q={w_q:.1f}  Strat={a:.5f}  Δ M5q={(a-M5Q_S)*1e4:+.1f}bp  "
              f"ρ vs M5q={r:.5f} [{gate}]")

    # Save artifacts for M5v
    np.save(ART / "oof_m5v_strat.npy",
            np.column_stack([1 - oof_v, oof_v]))
    np.save(ART / "test_m5v_strat.npy",
            np.column_stack([1 - test_v, test_v]))
    sub = sample_sub.copy()
    sub[TARGET] = test_v
    sub.to_csv("submissions/submission_m5v_lr_fe_layered.csv", index=False)

    # Save artifacts for M5w (best blend)
    np.save(ART / "oof_m5w_strat.npy",
            np.column_stack([1 - blend_oof, blend_oof]))
    np.save(ART / "test_m5w_strat.npy",
            np.column_stack([1 - blend_test, blend_test]))
    sub = sample_sub.copy()
    sub[TARGET] = blend_test
    sub.to_csv("submissions/submission_m5w_blend_50.csv", index=False)

    res = dict(
        M5v=dict(K=len(pool_v), strat=auc_v, delta_m5q_bp=delta_v,
                 spearman_vs_m5q=rho_v, gate=gate_v, l1=l1_v,
                 pool=names),
        M5w=dict(blend_weights="0.5 M5q + 0.5 RealMLP_standalone",
                 strat=auc_w, delta_m5q_bp=delta_w,
                 spearman_vs_m5q=rho_w, gate=gate_w),
    )
    (ART / "m5vw_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n→ scripts/artifacts/m5vw_results.json")


if __name__ == "__main__":
    main()
