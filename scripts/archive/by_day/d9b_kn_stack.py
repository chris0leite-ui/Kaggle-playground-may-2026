"""K=N stacks comparing R14 baseline (L0) vs strengthened R14 (L2, L3).

Builds three stacks on top of the PRIMARY pool (M5q 14 + 4 existing
rule_residuals minus the 2 most-redundant rules), then adds 4 d9
bases with R14 substituted at different ladder strengths:

  Sa: S4 baseline (R14 = L0, ρ=0.444, std OOF 0.794)
  Sb: R14 -> L2 (ρ=0.874, std OOF 0.914)
  Sc: R14 -> L3 (ρ=0.875, std OOF 0.916)

Per-stack metrics: Strat OOF, ρ vs PRIMARY, predicted-LB heuristic.
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

# PRIMARY base pool (drop 2 most-redundant rule_residuals -- ρ-ranked
# from the d9 analysis: rule_compound_tyre 0.938, rule_compound_stint 0.937)
POOL_KEEP = [
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
    ("rule_driver_compound", "d6_rule_driver_compound"),  # least-redundant rule
    ("rule_year_race", "d6_rule_year_race"),               # 2nd-least-redundant
]

# Top-3 d9 bases (excluding R14, those will be plugged in below): R6, R10, R7
TOP_3_D9 = [
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
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
    print(f"  L1 top-12:")
    for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:12]:
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

    # Base set: PRIMARY-keep + R6 + R10 + R7 (always)
    base_oof, base_test, base_names = load_pool(POOL_KEEP)
    d9_oof, d9_test, d9_names = load_pool(TOP_3_D9)

    results = {}
    for r14_label, r14_file in [
        ("R14_L0_baseline", "d9_R14_hash_lr_3way"),
        ("R14_L2_with_numerics", "d9b_R14_L2"),
        ("R14_L3_with_compound_2way", "d9b_R14_L3"),
        ("R14_L4_with_driver_numerics", "d9b_R14_L4"),
        ("R14_L5_kitchen_sink", "d9b_R14_L5"),
    ]:
        try:
            r14_oof = np.load(ART / f"oof_{r14_file}_strat.npy")[:, 1].astype(np.float64)
            r14_test = np.load(ART / f"test_{r14_file}_strat.npy")[:, 1].astype(np.float64)
        except FileNotFoundError:
            print(f"  [skipped {r14_label}: artifact not found]")
            continue
        all_oof = base_oof + d9_oof + [r14_oof]
        all_test = base_test + d9_test + [r14_test]
        all_names = base_names + d9_names + [r14_label]
        stack_eval(f"K20_swap_{r14_label}", all_oof, all_test, all_names, y,
                   primary_test, results)

    final = dict(stacks=results,
                 primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
                 total_wall_s=time.time() - t0)
    (ART / "d9b_kn_stack_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9b_kn_stack_results.json  "
          f"(total wall {time.time()-t0:.0f}s)")
    print("\n" + "=" * 78)
    print(f"{'stack':<40s} {'OOF':>8s} {'Δprim':>7s} {'ρ_PRIM':>7s} {'predLB':>8s} {'ΔLB':>6s}")
    print("-" * 78)
    for nm, r in results.items():
        print(f"{nm:<40s} {r['strat_oof']:>8.5f} {r['delta_primary_bp']:>+6.2f} "
              f"{r['rho_vs_primary_test']:>7.5f} {r['pred_lb']:>8.5f} "
              f"{r['delta_lb_bp']:>+5.2f}")


if __name__ == "__main__":
    main()
