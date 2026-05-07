"""d17 Phase A — compose CLEAN winners into K=23/K=24 stacks.

Stack-add gates K=21 + N candidates via the canonical 5-fold LR meta
(matches probe_min_meta.py's logic).

Combinations tested:
  C1. K=22  + d16_orig_continuous_only                                  [WINNER baseline]
  C2. K=23  + d16_orig_continuous_only + d16_orig_no_laptime
  C3. K=23  + d16_orig_continuous_only + d16_orig_no_tyrelife_rp
  C4. K=23  + d16_orig_continuous_only + d16_orig_categorical_only
  C5. K=23  + d16_orig_continuous_only + d16_inv_laps_strict (cross-branch, audited)
  C6. K=24  + d16_orig_continuous_only + d16_orig_no_laptime + d16_inv_laps_strict
  C7. K=24  + d16_orig_continuous_only + d16_orig_no_laptime + d16_orig_no_tyrelife_rp
  Optional: C8 + dr_split_v2 (waits for Phase 0)

Output: scripts/artifacts/d17_phase_a_summary.json with K=K_pool+N OOF lift,
        per-base |w|, ρ vs PRIMARY for each combination.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

# K=21 pool (must match d15b PRIMARY pool used in probe_min_meta.py)
K21_POOL = [
    "baseline_two_anchor",
    "d2a_te",
    "m2_xgb",
    "e1_catboost_sub",
    "e3_hgbc",
    "e5_optuna_lgbm",
    "a_horizon",
    "b_lapsuntilpit",
    "f1_hgbc_deep",
    "f2_hgbc_shallow",
    "cb_year-cat",
    "cb_lossguide",
    "cb_slow-wide-bag",
    "realmlp",
    "d6_rule_driver_compound",
    "d6_rule_year_race",
    "d9_R6_next_compound",
    "d9_R10_driver_eb",
    "d9_R7_prev_compound",
    "d9f_FM_A",
    "d9f_FM_B",
]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.astype(np.float64).ravel()


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) / (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def stack_oof(oofs_list, tests_list, y, primary_test):
    P_oof = np.column_stack(oofs_list)
    P_test = np.column_stack(tests_list)
    F_oof = expand(P_oof)
    F_test = expand(P_test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(P_test))
    weights_per_fold = []
    for tri, vai in skf.split(F_oof, y):
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(F_oof[tri], y[tri])
        oof[vai] = lr.predict_proba(F_oof[vai])[:, 1]
        test_pred += lr.predict_proba(F_test)[:, 1] / N_FOLDS
        weights_per_fold.append(lr.coef_[0])
    auc = roc_auc_score(y, oof)
    rho = float(np.corrcoef(test_pred, primary_test)[0, 1])
    w_mean = np.mean(weights_per_fold, axis=0)
    return float(auc), float(rho), w_mean, oof, test_pred


def main():
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("loading data")
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    primary_test = _pos(ART / "test_PRIMARY_K22_strat.npy")
    primary_oof = _pos(ART / "oof_PRIMARY_K22_strat.npy")
    auc_primary = roc_auc_score(y, primary_oof)
    step(f"  PRIMARY OOF AUC {auc_primary:.5f}")

    # Load K=21 base pool
    step("loading K=21 base pool")
    pool_oofs, pool_tests = [], []
    pool_loaded = []
    for name in K21_POOL:
        oof_p = ART / f"oof_{name}_strat.npy"
        test_p = ART / f"test_{name}_strat.npy"
        if not (oof_p.exists() and test_p.exists()):
            step(f"  MISSING: {name}")
            continue
        pool_oofs.append(_pos(oof_p))
        pool_tests.append(_pos(test_p))
        pool_loaded.append(name)
    step(f"  loaded {len(pool_oofs)}/{len(K21_POOL)} bases")
    if len(pool_oofs) < 18:
        raise SystemExit(f"Pool too small: {len(pool_oofs)}")

    # Compute K=21 LR-meta baseline
    auc_k21, rho_k21, w_k21, oof_k21, test_k21 = stack_oof(pool_oofs, pool_tests, y, primary_test)
    step(f"  K={len(pool_oofs)} LR-meta baseline OOF: {auc_k21:.5f}  Δ vs PRIMARY: {(auc_k21 - auc_primary) * 1e4:+.2f} bp")

    # Candidate library (CLEAN only — NO d16_dr_split, NO d16_dr_weighted_orig in v1)
    candidates = {
        "d16_orig_continuous_only":      (_pos(ART / "oof_d16_orig_continuous_only_strat.npy"),
                                           _pos(ART / "test_d16_orig_continuous_only_strat.npy")),
        "d16_orig_no_laptime":           (_pos(ART / "oof_d16_orig_no_laptime_strat.npy"),
                                           _pos(ART / "test_d16_orig_no_laptime_strat.npy")),
        "d16_orig_no_tyrelife_rp":       (_pos(ART / "oof_d16_orig_no_tyrelife_rp_strat.npy"),
                                           _pos(ART / "test_d16_orig_no_tyrelife_rp_strat.npy")),
        "d16_orig_categorical_only":     (_pos(ART / "oof_d16_orig_categorical_only_strat.npy"),
                                           _pos(ART / "test_d16_orig_categorical_only_strat.npy")),
        "d16_inv_laps_strict":           (_pos(ART / "oof_d16_inv_laps_strict_strat.npy"),
                                           _pos(ART / "test_d16_inv_laps_strict_strat.npy")),
    }

    combos = {
        "C1_K22_cont":                ["d16_orig_continuous_only"],
        "C2_K23_cont_nolaptime":      ["d16_orig_continuous_only", "d16_orig_no_laptime"],
        "C3_K23_cont_notyrerp":       ["d16_orig_continuous_only", "d16_orig_no_tyrelife_rp"],
        "C4_K23_cont_catonly":        ["d16_orig_continuous_only", "d16_orig_categorical_only"],
        "C5_K23_cont_invlaps_strict": ["d16_orig_continuous_only", "d16_inv_laps_strict"],
        "C6_K24_cont_nolaptime_invlaps": ["d16_orig_continuous_only", "d16_orig_no_laptime", "d16_inv_laps_strict"],
        "C7_K24_cont_nolaptime_notyrerp": ["d16_orig_continuous_only", "d16_orig_no_laptime", "d16_orig_no_tyrelife_rp"],
    }

    summary = dict(K21_baseline_auc=float(auc_k21), PRIMARY_auc=float(auc_primary),
                    pool_size=len(pool_oofs), combos={})

    best_combo = None
    best_oof = -np.inf

    for cname, candlist in combos.items():
        oofs = pool_oofs + [candidates[c][0] for c in candlist]
        tests = pool_tests + [candidates[c][1] for c in candlist]
        auc, rho, w, oof_, test_ = stack_oof(oofs, tests, y, primary_test)
        delta_baseline = (auc - auc_k21) * 1e4
        delta_primary = (auc - auc_primary) * 1e4
        # per-candidate |w| (sum of 3 cols)
        n_pool = len(pool_oofs)
        cand_weights = {}
        for i, cn in enumerate(candlist):
            idx = n_pool + i
            cand_weights[cn] = dict(
                raw=float(w[idx]),
                rank=float(w[idx + len(oofs)]),
                logit=float(w[idx + 2 * len(oofs)]),
                abs_sum=float(abs(w[idx]) + abs(w[idx + len(oofs)]) + abs(w[idx + 2 * len(oofs)])),
            )
        summary["combos"][cname] = dict(
            auc=float(auc), delta_vs_K21_bp=float(delta_baseline),
            delta_vs_PRIMARY_bp=float(delta_primary),
            rho_vs_PRIMARY=rho, candidates=candlist, weights=cand_weights,
        )
        step(f"  {cname:40s}  K=K+{len(candlist)}  auc {auc:.5f}  ΔK21 {delta_baseline:+.2f}  ΔPRIM {delta_primary:+.2f}  ρ {rho:.5f}")

        # Save winning combos as artifacts (for downstream Path B)
        np.save(ART / f"oof_d17_{cname}_strat.npy", oof_)
        np.save(ART / f"test_d17_{cname}_strat.npy", test_)

        if auc > best_oof:
            best_oof = auc
            best_combo = cname

    summary["best_combo"] = best_combo
    summary["best_oof"] = best_oof
    summary["runtime_s"] = time.time() - t0

    with open(ART / "d17_phase_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step(f"DONE. Best combo: {best_combo} OOF {best_oof:.5f}")


if __name__ == "__main__":
    main()
