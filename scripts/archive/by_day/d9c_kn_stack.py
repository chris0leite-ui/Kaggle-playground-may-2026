"""K=N stacks adding the FM base.

Three configurations on top of the PRIMARY pool (M5q 14 + 2 least-
redundant rules):

  Sa: K=21 = PRIMARY-keep (16) + R6 + R10 + R7 + R14_L4 + FM
  Sb: K=18 = PRIMARY-keep (16) + R7 + FM   (just two diverse adds)
  Sc: K=17 = PRIMARY-keep (16) + FM        (FM solo replaces 2 rules)
  Sd: K=20 swap = drop 2 most-redundant + add R6/R10/R7 + FM (no R14)
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
    if rho >= RHO_TIE: return base_lb
    if rho >= 0.995:   return base_lb - 0.0001
    if rho >= 0.99:    return base_lb - 0.00025
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
        elif n_ == "FM":
            marker = "  ← FM"
        elif n_.startswith("rule_"):
            marker = "  ← existing rule"
        print(f"    {n_:<24s} L1={v:.3f}{marker}")
    results[name] = dict(K=K, strat_oof=auc, delta_primary_bp=delta,
                         rho_vs_primary_test=float(rho), pred_lb=float(pred_lb),
                         delta_lb_bp=float((pred_lb - PRIMARY_LB) * 1e4),
                         l1_ranking=l1)
    return mo, tp


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy"
                           )[:, 1].astype(np.float64)

    base_oof, base_test, base_names = load_pool(POOL_KEEP)
    d9_oof, d9_test, d9_names = load_pool(TOP_3_D9)
    fm_oof = np.load(ART / "oof_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    fm_test = np.load(ART / "test_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    r14_l4_oof = np.load(ART / "oof_d9b_R14_L4_strat.npy"
                         )[:, 1].astype(np.float64)
    r14_l4_test = np.load(ART / "test_d9b_R14_L4_strat.npy"
                          )[:, 1].astype(np.float64)

    results = {}
    # Sa: K=21 PRIMARY-keep + R6/R10/R7 + R14_L4 + FM
    Xs = base_oof + d9_oof + [r14_l4_oof, fm_oof]
    Ts = base_test + d9_test + [r14_l4_test, fm_test]
    Ns = base_names + d9_names + ["R14_L4", "FM"]
    stack_eval("Sa_K21_full", Xs, Ts, Ns, y, primary_test, results)

    # Sb: K=18 PRIMARY-keep + R7 + FM (most diverse pair)
    Xs = base_oof + [d9_oof[2], fm_oof]
    Ts = base_test + [d9_test[2], fm_test]
    Ns = base_names + ["R7_prev_compound", "FM"]
    stack_eval("Sb_K18_R7_FM", Xs, Ts, Ns, y, primary_test, results)

    # Sc: K=17 PRIMARY-keep + FM solo
    Xs = base_oof + [fm_oof]
    Ts = base_test + [fm_test]
    Ns = base_names + ["FM"]
    mo_sc, tp_sc = stack_eval("Sc_K17_FM_solo", Xs, Ts, Ns, y, primary_test, results)

    # Sd: K=20 swap PRIMARY-keep + R6/R10/R7 + FM (no R14)
    Xs = base_oof + d9_oof + [fm_oof]
    Ts = base_test + d9_test + [fm_test]
    Ns = base_names + d9_names + ["FM"]
    mo_sd, tp_sd = stack_eval("Sd_K20_swap_FM", Xs, Ts, Ns, y, primary_test, results)

    # Save best stack predictions for submission building
    np.save(ART / "oof_d9c_Sd_K20_swap_FM_strat.npy",
            np.column_stack([1 - mo_sd, mo_sd]))
    np.save(ART / "test_d9c_Sd_K20_swap_FM_strat.npy",
            np.column_stack([1 - tp_sd, tp_sd]))
    sub = sample_sub.copy(); sub[TARGET] = tp_sd
    sub.to_csv("submissions/submission_d9c_K20_swap_FM.csv", index=False)
    print("→ wrote submissions/submission_d9c_K20_swap_FM.csv")

    np.save(ART / "oof_d9c_Sc_K17_FM_solo_strat.npy",
            np.column_stack([1 - mo_sc, mo_sc]))
    np.save(ART / "test_d9c_Sc_K17_FM_solo_strat.npy",
            np.column_stack([1 - tp_sc, tp_sc]))
    sub = sample_sub.copy(); sub[TARGET] = tp_sc
    sub.to_csv("submissions/submission_d9c_K17_FM_solo.csv", index=False)
    print("→ wrote submissions/submission_d9c_K17_FM_solo.csv")

    final = dict(stacks=results, primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
                 total_wall_s=time.time() - t0)
    (ART / "d9c_kn_stack_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9c_kn_stack_results.json  (wall {time.time()-t0:.0f}s)")
    print("\n" + "=" * 78)
    print(f"{'stack':<30s} {'OOF':>8s} {'Δprim':>7s} {'ρ_PRIM':>7s} {'predLB':>8s} {'ΔLB':>6s}")
    print("-" * 78)
    for nm, r in results.items():
        print(f"{nm:<30s} {r['strat_oof']:>8.5f} {r['delta_primary_bp']:>+6.2f} "
              f"{r['rho_vs_primary_test']:>7.5f} {r['pred_lb']:>8.5f} "
              f"{r['delta_lb_bp']:>+5.2f}")


if __name__ == "__main__":
    main()
