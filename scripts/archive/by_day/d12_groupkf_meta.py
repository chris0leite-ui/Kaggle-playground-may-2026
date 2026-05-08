"""Day-12 — K=21 LR-meta on GroupKF OOFs (P6 leakage-blocked).

Mirrors d9c_kn_stack.py Sa (K=21 = POOL_KEEP + TOP_3_D9 + R14_L4 + FM)
but loads `_groupkf` arrays. Computes both Strat-OOF AUC (using the
existing strat OOFs) and GroupKF-OOF AUC (using rebuilt groupkf OOFs)
to compare L1 weight rankings, ρ between meta predictions, and ρ vs
PRIMARY_test.

Diagnostics:
  - Per-base GroupKF–Strat ΔAUC table (leakage-eaters)
  - L1 weight ranking under GroupKF vs Strat
  - Spearman ρ between meta-OOF predictions Strat vs GroupKF
  - ρ between final test predictions Strat-meta vs GroupKF-meta
  - ρ between GroupKF-meta test predictions and current PRIMARY test

If GroupKF-meta produces a test prediction with ρ < 0.998 vs PRIMARY,
also save it as `oof_d12_groupkf_meta_strat.npy` /
`test_d12_groupkf_meta_strat.npy` for downstream use.
"""
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

# Same K=21 pool spec as d9c_kn_stack.py Sa
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
    ("realmlp", "realmlp"),
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
    if rho >= RHO_TIE:
        return base_lb
    if rho >= 0.995:
        return base_lb - 0.0001
    if rho >= 0.99:
        return base_lb - 0.00025
    return base_lb - 0.0004


def safe_load(name, suffix):
    """Load OOF prob (positive class) and return (arr, exists). For
    suffix='strat' or 'groupkf'."""
    p = ART / f"oof_{name}_{suffix}.npy"
    pt = ART / f"test_{name}_{suffix}.npy"
    if p.exists() and pt.exists():
        return (np.load(p)[:, 1].astype(np.float64),
                np.load(pt)[:, 1].astype(np.float64), True)
    return (None, None, False)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    grp = train.groupby(["Race", "Driver", "Year", "Stint"], sort=False).ngroup().values

    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy"
                           )[:, 1].astype(np.float64)
    primary_oof = np.load(ART / "oof_d6_k18_multi_rule_strat.npy"
                          )[:, 1].astype(np.float64)

    all_bases = POOL_KEEP + TOP_3_D9 + EXTRA  # K=21
    print(f"K = {len(all_bases)} bases")

    # Per-base AUC table (Strat vs GroupKF)
    base_table = []
    Xs_oof_strat, Xs_test_strat = [], []
    Xs_oof_gkf, Xs_test_gkf = [], []
    names = []
    skipped_gkf = []
    for label, fname in all_bases:
        s_oof, s_test, s_ok = safe_load(fname, "strat")
        g_oof, g_test, g_ok = safe_load(fname, "groupkf")
        if not s_ok:
            print(f"WARNING: missing strat for {fname}; skipping")
            continue
        s_auc = float(roc_auc_score(y, s_oof))
        if g_ok:
            g_auc = float(roc_auc_score(y, g_oof))
            delta_bp = (g_auc - s_auc) * 1e4
        else:
            g_auc = None
            delta_bp = None
            skipped_gkf.append(fname)
        base_table.append(dict(label=label, fname=fname,
                               strat_auc=s_auc, groupkf_auc=g_auc,
                               delta_bp=delta_bp))
        names.append(label)
        Xs_oof_strat.append(s_oof); Xs_test_strat.append(s_test)
        if g_ok:
            Xs_oof_gkf.append(g_oof); Xs_test_gkf.append(g_test)
        else:
            # Use strat as fallback so K stays equal — but we'll log this
            Xs_oof_gkf.append(s_oof); Xs_test_gkf.append(s_test)

    print(f"\nPer-base GroupKF–Strat ΔAUC table (sorted by Δ):")
    print(f"  {'base':<28s} {'Strat':>8s} {'GKF':>8s} {'Δbp':>8s}")
    print("-" * 60)
    for r in sorted(base_table,
                    key=lambda d: (d["delta_bp"] is None, d["delta_bp"] or 0)):
        ga = f"{r['groupkf_auc']:.5f}" if r["groupkf_auc"] is not None else "(strat)"
        db = f"{r['delta_bp']:+.2f}" if r["delta_bp"] is not None else "n/a"
        print(f"  {r['label']:<28s} {r['strat_auc']:>8.5f} {ga:>8s} {db:>8s}")

    # Build feature matrices
    K = len(names)
    P_oof_s = np.column_stack(Xs_oof_strat); P_test_s = np.column_stack(Xs_test_strat)
    P_oof_g = np.column_stack(Xs_oof_gkf);   P_test_g = np.column_stack(Xs_test_gkf)

    F_oof_s = expand(P_oof_s); F_test_s = expand(P_test_s)
    F_oof_g = expand(P_oof_g); F_test_g = expand(P_test_g)

    print(f"\n=== K={K} LR-meta on STRAT pool (Strat-CV evaluation) ===")
    mo_s, tp_s, coef_s = fit_lr_meta_strat(F_oof_s, F_test_s, y)
    auc_s = float(roc_auc_score(y, mo_s))
    rho_prim_s, _ = spearmanr(tp_s, primary_test)
    print(f"  Strat OOF: {auc_s:.5f}  ρ vs PRIMARY test: {rho_prim_s:.5f}")
    l1_s = {names[i]: float(abs(coef_s[i]) + abs(coef_s[K + i]) + abs(coef_s[2*K + i]))
            for i in range(K)}

    print(f"\n=== K={K} LR-meta on GROUPKF pool (Strat-CV evaluation, then GroupKF-CV) ===")
    # Two-track evaluation:
    # Track A: GroupKF-pool features + Strat-CV → "groupkf-pool, strat-evaluation"
    mo_g_strat_cv, tp_g_strat_cv, coef_g_strat_cv = fit_lr_meta_strat(
        F_oof_g, F_test_g, y)
    auc_g_strat_cv = float(roc_auc_score(y, mo_g_strat_cv))
    rho_prim_g_strat_cv, _ = spearmanr(tp_g_strat_cv, primary_test)
    print(f"  Track A (Strat-CV on GKF pool): Strat OOF {auc_g_strat_cv:.5f}  "
          f"ρ vs PRIMARY test: {rho_prim_g_strat_cv:.5f}")

    # Track B: GroupKF-pool features + GroupKF-CV → "fully leakage-blocked"
    mo_g_gkf_cv, tp_g_gkf_cv, coef_g_gkf_cv = fit_lr_meta_groupkf(
        F_oof_g, F_test_g, y, grp)
    auc_g_gkf_cv = float(roc_auc_score(y, mo_g_gkf_cv))
    rho_prim_g_gkf_cv, _ = spearmanr(tp_g_gkf_cv, primary_test)
    print(f"  Track B (GroupKF-CV on GKF pool): GroupKF OOF {auc_g_gkf_cv:.5f}  "
          f"ρ vs PRIMARY test: {rho_prim_g_gkf_cv:.5f}")

    # L1 ranking comparison: STRAT vs GroupKF (use the GroupKF-CV-trained
    # full-fit coefficients for the GroupKF column)
    l1_g = {names[i]: float(abs(coef_g_gkf_cv[i]) +
                            abs(coef_g_gkf_cv[K + i]) +
                            abs(coef_g_gkf_cv[2*K + i]))
            for i in range(K)}

    print(f"\n=== L1 ranking comparison (Strat vs GroupKF) ===")
    print(f"  {'base':<28s} {'L1_strat':>10s} {'L1_gkf':>10s} {'Δrank':>7s}")
    print("-" * 70)
    rank_s = {n: i for i, (n, _) in enumerate(
        sorted(l1_s.items(), key=lambda kv: -kv[1]))}
    rank_g = {n: i for i, (n, _) in enumerate(
        sorted(l1_g.items(), key=lambda kv: -kv[1]))}
    rows = []
    for n in names:
        rows.append((n, l1_s[n], l1_g[n], rank_g[n] - rank_s[n]))
    for n, ls, lg, dr in sorted(rows, key=lambda r: -r[1]):
        print(f"  {n:<28s} {ls:>10.4f} {lg:>10.4f} {dr:>+6d}")

    print(f"\n=== Meta-prediction agreement ===")
    rho_oof_meta, _ = spearmanr(mo_s, mo_g_strat_cv)
    rho_test_meta, _ = spearmanr(tp_s, tp_g_strat_cv)
    rho_oof_meta_gcv, _ = spearmanr(mo_s, mo_g_gkf_cv)
    rho_test_meta_gcv, _ = spearmanr(tp_s, tp_g_gkf_cv)
    print(f"  ρ(Strat-meta-OOF, GKF-pool/Strat-CV-meta-OOF):  {rho_oof_meta:.6f}")
    print(f"  ρ(Strat-meta-test, GKF-pool/Strat-CV-meta-test): {rho_test_meta:.6f}")
    print(f"  ρ(Strat-meta-OOF, GKF-pool/GKF-CV-meta-OOF):    {rho_oof_meta_gcv:.6f}")
    print(f"  ρ(Strat-meta-test, GKF-pool/GKF-CV-meta-test):  {rho_test_meta_gcv:.6f}")

    # Predicted LB (Strat baseline)
    pred_lb_s = predicted_lb(auc_s, rho_prim_s)
    print(f"\n  Strat K={K} pred-LB:        {pred_lb_s:.5f} "
          f"(Δ {(pred_lb_s - PRIMARY_LB)*1e4:+.2f}bp)")

    # Save K=21 GroupKF-meta if sufficiently divergent
    save_diverged = bool(rho_test_meta_gcv < 0.998)
    if save_diverged:
        np.save(ART / "oof_d12_groupkf_meta_strat.npy",
                np.column_stack([1 - mo_g_gkf_cv, mo_g_gkf_cv]))
        np.save(ART / "test_d12_groupkf_meta_strat.npy",
                np.column_stack([1 - tp_g_gkf_cv, tp_g_gkf_cv]))
        print(f"\n  → SAVED oof_d12_groupkf_meta_strat.npy + test "
              f"(ρ vs PRIMARY meta-test = {rho_test_meta_gcv:.4f} < 0.998)")
    else:
        print(f"\n  ρ vs Strat-meta-test = {rho_test_meta_gcv:.4f} >= 0.998 — "
              f"NOT saving (no meaningful divergence)")

    final = dict(
        K=K,
        n_skipped_gkf=len(skipped_gkf),
        skipped_gkf=skipped_gkf,
        per_base=base_table,
        strat_meta=dict(auc=auc_s, rho_vs_primary=float(rho_prim_s),
                        l1=l1_s),
        gkf_pool_strat_cv=dict(auc=auc_g_strat_cv,
                               rho_vs_primary=float(rho_prim_g_strat_cv)),
        gkf_pool_gkf_cv=dict(auc=auc_g_gkf_cv,
                             rho_vs_primary=float(rho_prim_g_gkf_cv),
                             l1=l1_g),
        meta_agreement=dict(
            rho_oof_strat_vs_gkf_strat_cv=float(rho_oof_meta),
            rho_test_strat_vs_gkf_strat_cv=float(rho_test_meta),
            rho_oof_strat_vs_gkf_gkf_cv=float(rho_oof_meta_gcv),
            rho_test_strat_vs_gkf_gkf_cv=float(rho_test_meta_gcv),
        ),
        pred_lb_strat=float(pred_lb_s),
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        saved_d12=bool(save_diverged),
        total_wall_s=time.time() - t0,
    )
    (ART / "d12_groupkf_meta_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d12_groupkf_meta_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
