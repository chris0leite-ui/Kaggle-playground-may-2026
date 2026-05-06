"""Variant of d12_groupkf_meta.py: K=20 dropping realmlp (no GroupKF
artifact) for a clean comparison."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_S, PRIMARY_LB = 0.95065, 0.95026
RHO_TIE = 0.999

POOL_KEEP = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"), ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"), ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"), ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"), ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    # realmlp DROPPED — no GroupKF artifact
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]
TOP_3_D9 = [
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
]
EXTRA = [
    ("R14_L4", "d9b_R14_L4"),
    ("FM", "d9c_fm"),
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta_strat(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def fit_lr_meta_groupkf(F_oof, F_test, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in gkf.split(np.zeros(len(y)), y, groups=groups):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def predicted_lb(auc, rho):
    base_lb = PRIMARY_LB + (auc - PRIMARY_S)
    if rho >= RHO_TIE: return base_lb
    if rho >= 0.995: return base_lb - 0.0001
    if rho >= 0.99:  return base_lb - 0.00025
    return base_lb - 0.0004


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    grp = train.groupby(["Race", "Driver", "Year", "Stint"], sort=False).ngroup().values
    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy"
                           )[:, 1].astype(np.float64)

    all_bases = POOL_KEEP + TOP_3_D9 + EXTRA  # K=20
    print(f"K = {len(all_bases)} bases (realmlp dropped)")

    Xs_oof_strat, Xs_test_strat = [], []
    Xs_oof_gkf, Xs_test_gkf = [], []
    names = []
    for label, fname in all_bases:
        s_oof = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        s_test = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        g_oof = np.load(ART / f"oof_{fname}_groupkf.npy")[:, 1].astype(np.float64)
        g_test = np.load(ART / f"test_{fname}_groupkf.npy")[:, 1].astype(np.float64)
        names.append(label)
        Xs_oof_strat.append(s_oof); Xs_test_strat.append(s_test)
        Xs_oof_gkf.append(g_oof);   Xs_test_gkf.append(g_test)

    K = len(names)
    P_oof_s = np.column_stack(Xs_oof_strat); P_test_s = np.column_stack(Xs_test_strat)
    P_oof_g = np.column_stack(Xs_oof_gkf);   P_test_g = np.column_stack(Xs_test_gkf)

    F_oof_s = expand(P_oof_s); F_test_s = expand(P_test_s)
    F_oof_g = expand(P_oof_g); F_test_g = expand(P_test_g)

    print(f"\n=== K={K} STRAT pool, Strat-CV ===")
    mo_s, tp_s, coef_s = fit_lr_meta_strat(F_oof_s, F_test_s, y)
    auc_s = float(roc_auc_score(y, mo_s))
    rho_prim_s, _ = spearmanr(tp_s, primary_test)
    print(f"  Strat OOF: {auc_s:.5f}  ρ vs PRIMARY: {rho_prim_s:.5f}")
    l1_s = {names[i]: float(abs(coef_s[i]) + abs(coef_s[K + i]) + abs(coef_s[2*K + i]))
            for i in range(K)}

    print(f"\n=== K={K} GroupKF pool ===")
    mo_g_strat_cv, tp_g_strat_cv, _ = fit_lr_meta_strat(F_oof_g, F_test_g, y)
    auc_g_strat_cv = float(roc_auc_score(y, mo_g_strat_cv))
    rho_prim_g, _ = spearmanr(tp_g_strat_cv, primary_test)
    print(f"  Strat-CV on GKF pool: AUC {auc_g_strat_cv:.5f}  ρ vs PRIMARY: {rho_prim_g:.5f}")
    mo_g_gkf_cv, tp_g_gkf_cv, coef_g = fit_lr_meta_groupkf(F_oof_g, F_test_g, y, grp)
    auc_g_gkf_cv = float(roc_auc_score(y, mo_g_gkf_cv))
    rho_prim_gcv, _ = spearmanr(tp_g_gkf_cv, primary_test)
    print(f"  GroupKF-CV on GKF pool: AUC {auc_g_gkf_cv:.5f}  ρ vs PRIMARY: {rho_prim_gcv:.5f}")

    l1_g = {names[i]: float(abs(coef_g[i]) + abs(coef_g[K + i]) + abs(coef_g[2*K + i]))
            for i in range(K)}

    print(f"\n=== L1 ranking (sorted by Strat) ===")
    print(f"  {'base':<28s} {'L1_strat':>10s} {'L1_gkf':>10s} {'Δrank':>7s}")
    rank_s = {n: i for i, (n, _) in enumerate(sorted(l1_s.items(), key=lambda kv: -kv[1]))}
    rank_g = {n: i for i, (n, _) in enumerate(sorted(l1_g.items(), key=lambda kv: -kv[1]))}
    for n in sorted(names, key=lambda x: -l1_s[x]):
        print(f"  {n:<28s} {l1_s[n]:>10.4f} {l1_g[n]:>10.4f} {rank_g[n]-rank_s[n]:>+6d}")

    print(f"\n=== Meta agreement ===")
    rho_oof_meta_gcv, _ = spearmanr(mo_s, mo_g_gkf_cv)
    rho_test_meta_gcv, _ = spearmanr(tp_s, tp_g_gkf_cv)
    print(f"  ρ(Strat-meta-OOF, GKF-CV-meta-OOF):  {rho_oof_meta_gcv:.6f}")
    print(f"  ρ(Strat-meta-test, GKF-CV-meta-test): {rho_test_meta_gcv:.6f}")

    pred_lb_strat = predicted_lb(auc_s, rho_prim_s)
    print(f"\n  K={K} STRAT pred-LB: {pred_lb_strat:.5f}  Δ {(pred_lb_strat - PRIMARY_LB)*1e4:+.2f}bp")

    final = dict(K=K,
                 strat_meta=dict(auc=auc_s, rho_vs_primary=float(rho_prim_s), l1=l1_s),
                 gkf_pool_strat_cv=dict(auc=auc_g_strat_cv, rho_vs_primary=float(rho_prim_g)),
                 gkf_pool_gkf_cv=dict(auc=auc_g_gkf_cv, rho_vs_primary=float(rho_prim_gcv), l1=l1_g),
                 meta_agreement=dict(
                     rho_oof_strat_vs_gkf_gkf_cv=float(rho_oof_meta_gcv),
                     rho_test_strat_vs_gkf_gkf_cv=float(rho_test_meta_gcv)),
                 pred_lb_strat=float(pred_lb_strat),
                 total_wall_s=time.time() - t0)
    (ART / "d12_groupkf_meta_no_realmlp_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d12_groupkf_meta_no_realmlp_results.json  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
