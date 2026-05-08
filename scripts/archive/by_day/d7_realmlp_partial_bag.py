"""Move F (salvaged): partial RealMLP bag (seed-42 + seed-123-3fold).

Kaggle kernel realmlp-bag-gpu was cancelled mid-fold-3 of seed 123
(after parallel-branch probes downgraded bag EV to Tier-3). Saved
artifacts: 3-fold seed-123 OOF (60% coverage) + partial test (sum/5
divisor, needs 5/3 rescale).

Per-fold AUCs (seed-123, log): f0=0.94724  f1=0.94535  f2=0.94619
Mean = 0.94626 (vs seed-42 5-fold OOF AUC 0.94582; +4.4bp upper-bd).

Two salvage paths tested below:

  Path B (conservative): bagged TEST only.
    - Rescale seed-123 test (× 5/3 → valid 3-fold avg).
    - Rank-average with seed-42 5-fold test → realmlp_bag_test.
    - Use seed-42 OOF unchanged for K=18 meta fit.
    - Variance reduction at INFERENCE (where LB measures), not at fit.

  Path C (hybrid OOF): bagged where coverage exists.
    - Same TEST as Path B.
    - For rows where seed-123 has OOF (60%), rank-average with seed-42.
    - For rows where seed-123 doesn't (40%), use seed-42 OOF unchanged.
    - K=18 meta refit on hybrid OOF.

Decision rule per audit §5:
  - K=18 OOF >= d6_k18 anchor (0.95065) AND ρ < 0.9995 → SLOT
  - Pre-submit-diff vs submission_d6_k18_multi_rule.csv (NEW PRIMARY)

Strat-only (R1).
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
N_FOLDS_PARTIAL = 3
D6_K18_OOF = 0.95065
D6_K18_LB = 0.95026
M5Q_S = 0.95057
RHO_TIE_TIGHT = 0.9995  # tightened per Day-6 LB-result audit

POOL_M5Q_NO_REALMLP = [
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
RULES = ["d6_rule_residual", "d6_rule_compound_stint",
         "d6_rule_driver_compound", "d6_rule_year_race"]


def load_oof_test(name):
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
    n = len(y)

    # Load seed-42 (full) and seed-123 (3-fold partial)
    realmlp42_oof, realmlp42_test = load_oof_test("realmlp")
    seed123_partial = np.load(ART / "oof_realmlp_seed123_partial_3fold_strat.npy")
    realmlp123_oof = seed123_partial[:, 1].astype(np.float64)
    realmlp123_test_raw = np.load(ART / "test_realmlp_seed123_partial_3fold_strat.npy")[:, 1].astype(np.float64)

    # Rescale seed-123 test: divisor was 5, only 3 folds done → multiply by 5/3
    realmlp123_test = realmlp123_test_raw * (N_FOLDS / N_FOLDS_PARTIAL)
    print(f"seed-123 3-fold test mean after rescale: {realmlp123_test.mean():.4f}  "
          f"(class prior {y.mean():.4f})")

    # Coverage mask: rows where seed-123 has a real prediction
    p123 = realmlp123_oof
    cov_mask = (p123 > 0) & (p123 < 1)
    cov_pct = cov_mask.mean() * 100
    print(f"seed-123 OOF coverage: {cov_mask.sum()}/{n} = {cov_pct:.1f}%")

    # Standalone AUCs for sanity
    auc42_oof = float(roc_auc_score(y, realmlp42_oof))
    auc123_partial = float(roc_auc_score(y[cov_mask], p123[cov_mask]))
    print(f"seed-42 5-fold OOF AUC: {auc42_oof:.5f}")
    print(f"seed-123 3-fold OOF AUC (covered rows only): {auc123_partial:.5f}")

    # ---- Build bagged TEST predictions ----
    rk42_test = rankdata(realmlp42_test) / len(realmlp42_test)
    rk123_test = rankdata(realmlp123_test) / len(realmlp123_test)
    realmlp_bag_test = (rk42_test + rk123_test) / 2

    # ---- Path B: bagged TEST only, OOF unchanged (seed-42) ----
    print(f"\n=== Path B: bagged TEST + seed-42 OOF ===")
    Xs_oof, Xs_test, names = [], [], []
    for label, name in POOL_M5Q_NO_REALMLP:
        oo, te = load_oof_test(name)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    Xs_oof.append(realmlp42_oof)
    Xs_test.append(realmlp_bag_test)
    names.append("realmlp_bag_testonly")
    for r in RULES:
        oo, te = load_oof_test(r)
        Xs_oof.append(oo); Xs_test.append(te); names.append(r)

    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo_B, tp_B, _ = fit_lr_meta(F_oof, F_test, y)
    auc_B = float(roc_auc_score(y, mo_B))

    # ρ vs d6_k18 (the new PRIMARY)
    test_d6k18 = np.load(ART / "test_d6_k18_multi_rule_strat.npy")[:, 1]
    rho_B, _ = spearmanr(tp_B, test_d6k18)
    print(f"  K=18 OOF (Path B): {auc_B:.5f}  Δ d6_k18 {(auc_B - D6_K18_OOF)*1e4:+.2f}bp")
    print(f"  ρ vs d6_k18 test:  {rho_B:.5f}  "
          f"({'PASS' if rho_B < RHO_TIE_TIGHT else 'TIE_EXPECTED'} at 0.9995)")

    # Save
    np.save(ART / "oof_d7_realmlp_bag_partB_strat.npy",
            np.column_stack([1 - mo_B, mo_B]))
    np.save(ART / "test_d7_realmlp_bag_partB_strat.npy",
            np.column_stack([1 - tp_B, tp_B]))
    sub = sample_sub.copy(); sub[TARGET] = tp_B
    sub.to_csv("submissions/submission_d7_realmlp_bag_partB.csv", index=False)

    # ---- Path C: hybrid OOF (bagged where covered, seed-42 elsewhere) ----
    print(f"\n=== Path C: hybrid OOF (bag where covered, seed-42 elsewhere) + bagged TEST ===")
    rk42_oof = rankdata(realmlp42_oof) / n
    rk123_oof = np.zeros(n, dtype=np.float64)
    rk123_oof[cov_mask] = rankdata(p123[cov_mask]) / cov_mask.sum()
    # For the 60% covered rows: rank-avg with seed-42
    # For the 40% uncovered rows: keep seed-42 only (in original prob space, not rank)
    realmlp_hybrid_oof = realmlp42_oof.copy()
    # Reconstruct hybrid as rank-averaged probabilities for covered rows
    # We need probs not ranks; so use simple prob-mean for the covered rows
    realmlp_hybrid_oof[cov_mask] = (realmlp42_oof[cov_mask] + p123[cov_mask]) / 2

    Xs_oof_C = list(Xs_oof)
    Xs_oof_C[len(POOL_M5Q_NO_REALMLP)] = realmlp_hybrid_oof
    P_oof_C = np.column_stack(Xs_oof_C)
    F_oof_C = expand(P_oof_C)
    mo_C, tp_C, _ = fit_lr_meta(F_oof_C, F_test, y)
    auc_C = float(roc_auc_score(y, mo_C))
    rho_C, _ = spearmanr(tp_C, test_d6k18)
    print(f"  K=18 OOF (Path C): {auc_C:.5f}  Δ d6_k18 {(auc_C - D6_K18_OOF)*1e4:+.2f}bp")
    print(f"  ρ vs d6_k18 test:  {rho_C:.5f}  "
          f"({'PASS' if rho_C < RHO_TIE_TIGHT else 'TIE_EXPECTED'} at 0.9995)")

    np.save(ART / "oof_d7_realmlp_bag_partC_strat.npy",
            np.column_stack([1 - mo_C, mo_C]))
    np.save(ART / "test_d7_realmlp_bag_partC_strat.npy",
            np.column_stack([1 - tp_C, tp_C]))
    sub = sample_sub.copy(); sub[TARGET] = tp_C
    sub.to_csv("submissions/submission_d7_realmlp_bag_partC.csv", index=False)

    # ---- Verdict + comparison ----
    print(f"\n=== Comparison vs d6_k18_multi_rule (LB 0.95026, OOF 0.95065) ===")
    print(f"{'variant':<12} {'OOF':>9} {'Δ d6':>8} {'ρ vs d6':>10} {'verdict':<25}")
    def verdict(auc, rho):
        if rho >= RHO_TIE_TIGHT: return "tie regime → wasted slot"
        if auc < D6_K18_OOF: return "OOF regression"
        if auc < D6_K18_OOF + 0.5/1e4: return "marginal (sub-0.5bp)"
        return "SLOT-CANDIDATE"
    print(f"{'Path B':<12} {auc_B:>9.5f} {(auc_B-D6_K18_OOF)*1e4:>+8.2f} "
          f"{rho_B:>10.5f}  {verdict(auc_B, rho_B)}")
    print(f"{'Path C':<12} {auc_C:>9.5f} {(auc_C-D6_K18_OOF)*1e4:>+8.2f} "
          f"{rho_C:>10.5f}  {verdict(auc_C, rho_C)}")

    (ART / "d7_realmlp_partial_bag_results.json").write_text(json.dumps(dict(
        seed42_oof=auc42_oof,
        seed123_3fold_oof_covered=auc123_partial,
        seed123_fold_aucs=[0.94724, 0.94535, 0.94619],
        seed123_coverage_pct=float(cov_pct),
        path_B=dict(strat_oof=auc_B,
                    delta_d6_k18_bp=(auc_B - D6_K18_OOF) * 1e4,
                    rho_vs_d6_k18_test=float(rho_B)),
        path_C=dict(strat_oof=auc_C,
                    delta_d6_k18_bp=(auc_C - D6_K18_OOF) * 1e4,
                    rho_vs_d6_k18_test=float(rho_C)),
    ), indent=2))
    print(f"\n→ scripts/artifacts/d7_realmlp_partial_bag_results.json")


if __name__ == "__main__":
    main()
