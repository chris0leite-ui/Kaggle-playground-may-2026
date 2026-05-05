"""Day-9g — 3-way multi-FM partition (extends d9f 2-way).

d9f (NEW PRIMARY at LB 0.95031) used a 2-way partition:
  FM_A: D, C, S, T_q5  (driver-dynamics)
  FM_B: R, Y, Rp_q5, P_q5  (race-context)

d9g splits 8 features into 3 disjoint subsets along domain semantics:
  FM_α "driver":  D, C, S        (categorical driver/strategy)
  FM_β "race":    R, Y           (venue + year)
  FM_γ "state":   T_q5, Rp_q5, P_q5  (numeric quintiles)

Two K=N comparisons:
  S1 K=22 swap (drop d9f FM_A + FM_B, add α + β + γ)
  S2 K=24 add  (keep d9f FM_A + FM_B, add α + β + γ as additional bases)

PRIMARY = d9f K=21 swap+multi-FM (LB 0.95031, OOF 0.95073).

Note: FM_β has only 2 features (1 pair) — weak standalone but cheap.
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

# Reuse d9f's machinery
import d9f_multi_fm as d9f

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

# d9f K=21 swap is the new PRIMARY
PRIMARY_OOF_PATH = ART / "test_d9f_K21_swap_strat.npy"  # using test for ρ
PRIMARY_S = 0.95073   # d9f K=21 swap OOF
PRIMARY_LB = 0.95031

# 3-way partitions (Driver / Race / State)
PART_ALPHA = ["D", "C", "S"]       # driver-strategy categoricals
PART_BETA = ["R", "Y"]             # venue + year
PART_GAMMA = ["T", "Rp", "P"]      # numeric-quintile state

POOL_KEEP = d9f.POOL_KEEP
TOP_3_D9 = d9f.TOP_3_D9


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


def stack_eval(label, extra_oof, extra_test, extra_names, y, primary_test,
               base_oof, base_test, base_names, d9_oof, d9_test, d9_names,
               results, include_d9f_AB=False, fma_oof=None, fma_test=None,
               fmb_oof=None, fmb_test=None, include_d9c_FM=False,
               fm_pool_oof=None, fm_pool_test=None):
    Xs = list(base_oof) + list(d9_oof)
    Ts = list(base_test) + list(d9_test)
    Ns = list(base_names) + list(d9_names)
    if include_d9c_FM:
        Xs.append(fm_pool_oof); Ts.append(fm_pool_test); Ns.append("FM_d9c")
    if include_d9f_AB:
        Xs.extend([fma_oof, fmb_oof])
        Ts.extend([fma_test, fmb_test])
        Ns.extend(["FM_A_d9f", "FM_B_d9f"])
    for o, t, n in zip(extra_oof, extra_test, extra_names):
        Xs.append(o); Ts.append(t); Ns.append(n)
    K = len(Ns)
    P_oof = np.column_stack(Xs); P_test = np.column_stack(Ts)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, primary_test)
    delta = (auc - PRIMARY_S) * 1e4
    l1 = {Ns[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"\n=== {label} (K={K}) ===")
    print(f"  Strat OOF: {auc:.5f}  Δ PRIMARY {delta:+.2f}bp  "
          f"ρ vs PRIMARY {rho:.5f}")
    print(f"  L1 top-15:")
    for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:15]:
        marker = ""
        if n_.startswith("FM_α") or n_.startswith("FM_β") or n_.startswith("FM_γ"):
            marker = "  ← d9g-FM"
        elif n_.startswith("FM_A_d9f") or n_.startswith("FM_B_d9f"):
            marker = "  ← d9f-FM"
        elif n_.startswith("FM_d9c"):
            marker = "  ← d9c-FM"
        elif n_.startswith("R") and "_" in n_: marker = "  ← d9-base"
        elif n_.startswith("rule_"): marker = "  ← existing rule"
        print(f"    {n_:<24s} L1={v:.3f}{marker}")
    results[label] = dict(K=K, strat_oof=auc, delta_primary_bp=delta,
                          rho_vs_primary=float(rho), l1_ranking=l1)
    return mo, tp


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_test = np.load(PRIMARY_OOF_PATH)[:, 1].astype(np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Train all 3 partition FMs
    print("\n=== d9g 3-way multi-FM training ===\n")
    oof_alpha, test_alpha = d9f.train_partition_fm(train, test, y, splits,
                                                  PART_ALPHA, "FM_α_driver")
    oof_beta, test_beta = d9f.train_partition_fm(train, test, y, splits,
                                                 PART_BETA, "FM_β_race")
    oof_gamma, test_gamma = d9f.train_partition_fm(train, test, y, splits,
                                                   PART_GAMMA, "FM_γ_state")

    auc_a = float(roc_auc_score(y, oof_alpha))
    auc_b = float(roc_auc_score(y, oof_beta))
    auc_g = float(roc_auc_score(y, oof_gamma))
    rho_ap, _ = spearmanr(test_alpha, primary_test)
    rho_bp, _ = spearmanr(test_beta, primary_test)
    rho_gp, _ = spearmanr(test_gamma, primary_test)
    rho_ab, _ = spearmanr(test_alpha, test_beta)
    rho_ag, _ = spearmanr(test_alpha, test_gamma)
    rho_bg, _ = spearmanr(test_beta, test_gamma)
    print(f"\n  FM_α std OOF: {auc_a:.5f}  ρ vs PRIMARY: {rho_ap:.5f}")
    print(f"  FM_β std OOF: {auc_b:.5f}  ρ vs PRIMARY: {rho_bp:.5f}")
    print(f"  FM_γ std OOF: {auc_g:.5f}  ρ vs PRIMARY: {rho_gp:.5f}")
    print(f"  pairwise ρ: α-β={rho_ab:.4f}  α-γ={rho_ag:.4f}  β-γ={rho_bg:.4f}")

    np.save(ART / "oof_d9g_FM_alpha_strat.npy",
            np.column_stack([1 - oof_alpha, oof_alpha]))
    np.save(ART / "test_d9g_FM_alpha_strat.npy",
            np.column_stack([1 - test_alpha, test_alpha]))
    np.save(ART / "oof_d9g_FM_beta_strat.npy",
            np.column_stack([1 - oof_beta, oof_beta]))
    np.save(ART / "test_d9g_FM_beta_strat.npy",
            np.column_stack([1 - test_beta, test_beta]))
    np.save(ART / "oof_d9g_FM_gamma_strat.npy",
            np.column_stack([1 - oof_gamma, oof_gamma]))
    np.save(ART / "test_d9g_FM_gamma_strat.npy",
            np.column_stack([1 - test_gamma, test_gamma]))

    # Load PRIMARY pool
    base_oof, base_test, base_names = [], [], []
    for label, fname in POOL_KEEP:
        base_oof.append(np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64))
        base_test.append(np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64))
        base_names.append(label)
    d9_oof, d9_test, d9_names = [], [], []
    for label, fname in TOP_3_D9:
        d9_oof.append(np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64))
        d9_test.append(np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64))
        d9_names.append(label)
    fma_oof = np.load(ART / "oof_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    fma_test = np.load(ART / "test_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    fmb_oof = np.load(ART / "oof_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)
    fmb_test = np.load(ART / "test_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)
    fmd9c_oof = np.load(ART / "oof_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    fmd9c_test = np.load(ART / "test_d9c_fm_strat.npy")[:, 1].astype(np.float64)

    results = dict(
        FM_alpha=dict(std_oof=auc_a, rho_vs_primary=float(rho_ap)),
        FM_beta=dict(std_oof=auc_b, rho_vs_primary=float(rho_bp)),
        FM_gamma=dict(std_oof=auc_g, rho_vs_primary=float(rho_gp)),
        pairwise_rho=dict(ab=float(rho_ab), ag=float(rho_ag), bg=float(rho_bg)),
    )

    # S1 K=22 swap (drop d9f FM_A + FM_B, add α + β + γ)
    mo_s1, tp_s1 = stack_eval(
        "S1_K22_swap_3way",
        [oof_alpha, oof_beta, oof_gamma],
        [test_alpha, test_beta, test_gamma],
        ["FM_α", "FM_β", "FM_γ"],
        y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results,
        include_d9f_AB=False)

    # S2 K=24 add (keep d9f FM_A + FM_B, add α + β + γ)
    mo_s2, tp_s2 = stack_eval(
        "S2_K24_add_d9f_plus_3way",
        [oof_alpha, oof_beta, oof_gamma],
        [test_alpha, test_beta, test_gamma],
        ["FM_α", "FM_β", "FM_γ"],
        y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results,
        include_d9f_AB=True, fma_oof=fma_oof, fma_test=fma_test,
        fmb_oof=fmb_oof, fmb_test=fmb_test)

    # S3 K=25 add (everything: d9c FM + d9f FM_A/B + α + β + γ)
    mo_s3, tp_s3 = stack_eval(
        "S3_K25_add_all_FMs",
        [oof_alpha, oof_beta, oof_gamma],
        [test_alpha, test_beta, test_gamma],
        ["FM_α", "FM_β", "FM_γ"],
        y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results,
        include_d9f_AB=True, fma_oof=fma_oof, fma_test=fma_test,
        fmb_oof=fmb_oof, fmb_test=fmb_test,
        include_d9c_FM=True, fm_pool_oof=fmd9c_oof,
        fm_pool_test=fmd9c_test)

    # Save submission CSVs for the three stacks
    for name, mo, tp in [("S1_K22_swap_3way", mo_s1, tp_s1),
                         ("S2_K24_add_d9f_plus_3way", mo_s2, tp_s2),
                         ("S3_K25_add_all_FMs", mo_s3, tp_s3)]:
        np.save(ART / f"test_d9g_{name}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_d9g_{name}.csv", index=False)
        print(f"→ wrote submissions/submission_d9g_{name}.csv")

    final = dict(
        results=results,
        partitions=dict(alpha=PART_ALPHA, beta=PART_BETA, gamma=PART_GAMMA),
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        total_wall_s=time.time() - t0,
    )
    (ART / "d9g_3way_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9g_3way_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
