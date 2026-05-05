"""K=N stacking experiments over the 10 d9 bases. Builds three stacks:
  S1 K=24: PRIMARY pool (M5q 14 + 4 existing rules) + top-2 most-diverse d9 bases (by ρ vs PRIMARY).
  S2 K=28: PRIMARY pool + ALL 10 d9 bases (let LR meta route via L1).
  S3 K=20 swap: drop the 2 highest-ρ existing rules from PRIMARY, add 2 most-diverse d9 bases.

For each stack: Strat OOF, ρ vs PRIMARY test, predicted-LB heuristic,
L1 ranking. NO submissions written; ranks vs PRIMARY 0.95065 / LB 0.95026.
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
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_S, PRIMARY_LB = 0.95065, 0.95026
RHO_TIE = 0.999

# PRIMARY (d6_k18_multi_rule) base pool: M5q's 14 + 4 rule_residuals
POOL_PRIMARY = [
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
    ("rule_compound_tyre", "d6_rule_residual"),
    ("rule_compound_stint", "d6_rule_compound_stint"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]

D9_NAMES = [
    "R5_weibull_compound", "R6_next_compound", "R7_prev_compound",
    "R8_position_progress", "R9_laptime_delta_z", "R10_driver_eb",
    "R11_stint_overdue", "R12_cumdeg_knee", "R13_race_lapbin",
    "R14_hash_lr_3way",
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


def load_pool(names_files):
    Xs_oof, Xs_test, names = [], [], []
    for label, fname in names_files:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    return Xs_oof, Xs_test, names


def stack_eval(name, Xs_oof, Xs_test, names, y, primary_test, results):
    K = len(names)
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, primary_test)
    pred_lb = predicted_lb(auc, rho)
    delta = (auc - PRIMARY_S) * 1e4
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"\n=== {name} (K={K}) ===")
    print(f"  Strat OOF: {auc:.5f}  Δ PRIMARY {delta:+.2f}bp")
    print(f"  ρ vs PRIMARY test: {rho:.5f}  pred-LB {pred_lb:.5f}  "
          f"(Δ {(pred_lb - PRIMARY_LB)*1e4:+.2f}bp)")
    print(f"  L1 top-15:")
    for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:15]:
        marker = ""
        if n_.startswith("R") and "_" in n_:
            marker = "  ← d9-base"
        elif n_.startswith("rule_"):
            marker = "  ← existing rule"
        print(f"    {n_:<24s} L1={v:.3f}{marker}")
    results[name] = dict(
        K=K, strat_oof=auc, delta_primary_bp=delta,
        rho_vs_primary_test=float(rho), pred_lb=float(pred_lb),
        delta_lb_bp=float((pred_lb - PRIMARY_LB) * 1e4),
        l1_ranking=l1,
    )


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy"
                           )[:, 1].astype(np.float64)

    # Load PRIMARY pool
    Xs_p_oof, Xs_p_test, names_p = load_pool(POOL_PRIMARY)
    # Load d9 bases
    Xs_d_oof, Xs_d_test, names_d = [], [], []
    for n in D9_NAMES:
        oo = np.load(ART / f"oof_d9_{n}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_d9_{n}_strat.npy")[:, 1].astype(np.float64)
        Xs_d_oof.append(oo); Xs_d_test.append(te); names_d.append(n)

    # Compute ρ of each d9 base vs PRIMARY test, to pick top-2 most-diverse
    diversities = []
    for n, te in zip(names_d, Xs_d_test):
        rho, _ = spearmanr(te, primary_test)
        diversities.append((n, float(rho)))
    diversities.sort(key=lambda kv: kv[1])  # ascending ρ → most diverse first
    print("\nd9 bases by ρ vs PRIMARY test (low = diverse):")
    for n, rho in diversities:
        print(f"  {n:<24s} ρ={rho:.5f}")
    top2 = [n for n, _ in diversities[:2]]
    top4 = [n for n, _ in diversities[:4]]
    print(f"\nTop-2 most-diverse d9 bases: {top2}")
    print(f"Top-4 most-diverse d9 bases: {top4}")

    # ρ of existing rule bases vs PRIMARY (for swap stack)
    rule_idx = [i for i, (lbl, _) in enumerate(POOL_PRIMARY)
                if lbl.startswith("rule_")]
    rule_rhos = []
    for i in rule_idx:
        rho, _ = spearmanr(Xs_p_test[i], primary_test)
        rule_rhos.append((names_p[i], float(rho), i))
    rule_rhos.sort(key=lambda kv: -kv[1])  # highest ρ = most redundant first
    print("\nExisting rule bases by ρ vs PRIMARY test (high = redundant):")
    for lbl, rho, _ in rule_rhos:
        print(f"  {lbl:<24s} ρ={rho:.5f}")

    results = {}

    # S1: K=24 — PRIMARY + top-2 most-diverse d9 bases
    add_idx = [names_d.index(n) for n in top2]
    Xs1 = Xs_p_oof + [Xs_d_oof[i] for i in add_idx]
    Ts1 = Xs_p_test + [Xs_d_test[i] for i in add_idx]
    Ns1 = list(names_p) + [names_d[i] for i in add_idx]
    stack_eval("S1_K20_top2_diverse", Xs1, Ts1, Ns1, y, primary_test, results)

    # S2: K=28 — PRIMARY + ALL 10 d9
    Xs2 = Xs_p_oof + Xs_d_oof
    Ts2 = Xs_p_test + Xs_d_test
    Ns2 = list(names_p) + list(names_d)
    stack_eval("S2_K28_all", Xs2, Ts2, Ns2, y, primary_test, results)

    # S3: K=18 swap — drop top-2 most-redundant existing rules,
    # replace with top-2 most-diverse d9 bases
    drop_idxs = {rule_rhos[0][2], rule_rhos[1][2]}
    keep = [(lbl, oo, te) for lbl, oo, te, idx in zip(
        names_p, Xs_p_oof, Xs_p_test, range(len(names_p))) if idx not in drop_idxs]
    Xs3 = [oo for _, oo, _ in keep] + [Xs_d_oof[i] for i in add_idx]
    Ts3 = [te for _, _, te in keep] + [Xs_d_test[i] for i in add_idx]
    Ns3 = [lbl for lbl, _, _ in keep] + [names_d[i] for i in add_idx]
    stack_eval("S3_K18_swap_2rules", Xs3, Ts3, Ns3, y, primary_test, results)

    # S4: K=20 swap — drop top-2 most-redundant existing rules,
    # replace with top-4 most-diverse d9 bases
    add_idx4 = [names_d.index(n) for n in top4]
    Xs4 = [oo for _, oo, _ in keep] + [Xs_d_oof[i] for i in add_idx4]
    Ts4 = [te for _, _, te in keep] + [Xs_d_test[i] for i in add_idx4]
    Ns4 = [lbl for lbl, _, _ in keep] + [names_d[i] for i in add_idx4]
    stack_eval("S4_K20_swap_2rules_add4", Xs4, Ts4, Ns4, y, primary_test, results)

    final = dict(
        diversities=dict(diversities),
        existing_rule_rhos=[(lbl, rho) for lbl, rho, _ in rule_rhos],
        stacks=results,
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        total_wall_s=time.time() - t0,
    )
    (ART / "d9_kn_stack_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9_kn_stack_results.json  "
          f"(total wall {time.time()-t0:.0f}s)")

    print("\n" + "=" * 78)
    print(f"{'stack':<28s} {'OOF':>8s} {'Δprim_bp':>9s} {'ρ_PRIM':>7s} {'predLB':>8s} {'ΔLB_bp':>7s}")
    print("-" * 78)
    for nm, r in results.items():
        print(f"{nm:<28s} {r['strat_oof']:>8.5f} {r['delta_primary_bp']:>+8.2f} "
              f"{r['rho_vs_primary_test']:>7.5f} {r['pred_lb']:>8.5f} "
              f"{r['delta_lb_bp']:>+6.2f}")


if __name__ == "__main__":
    main()
