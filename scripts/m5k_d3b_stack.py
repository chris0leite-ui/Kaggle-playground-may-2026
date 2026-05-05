"""M5k — drop d3b_seqfe into M5h pool. Test add-only vs swap variants.

Variants (all use raw+rank+logit expansion, LR meta):
  K_add14  : M5h pool + d3b_seqfe (14 bases)
  K_swap_d2a   : M5h with d2a_te → d3b_seqfe (13 bases) — swap TE base
  K_swap_baseline: M5h with baseline → d3b_seqfe (13) — swap baseline
                   (d3b is a baseline-flavor enhanced model; if it dominates
                    baseline in stack contribution, swap is principled)
  K_add14_d3a : M5h + d3a_te_unified + d3b_seqfe (15) — kitchen sink (sanity)

Decision rule:
  - Pick the variant maximizing Strat OOF, ties broken by smaller pool.
  - Within winner, report L1-coef per base (tier-break diagnostic vs
    median rule).

Outputs: oof/test_m5k_*.npy, m5k_lr_meta_results.json, audit, submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5H_S, M5H_G = 0.95043, 0.93087
SEED, N_FOLDS = 42, 5

M5H_BASE = [
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

D3B = ("d3b_seqfe", "d3b_seqfe")
D3A = ("d3a_te_unified", "d3a_te_unified")


def load(name, suffix):
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


def assemble(pool, suffix):
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool:
        oo, te = load(name, suffix)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    P_oof = np.column_stack(Xs_oof)
    P_test = np.column_stack(Xs_test)
    return expand(P_oof), expand(P_test), names


def l1_per_base(coef, names):
    K = len(names)
    return {n: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2 * K + i]))
            for i, n in enumerate(names)}


def run_variant(label, pool, y):
    F_oof_s, F_test_s, names = assemble(pool, "strat")
    oof_s, test_s, auc_s, coef_s = fit_meta(F_oof_s, F_test_s, y)
    F_oof_g, F_test_g, _ = assemble(pool, "groupkf")
    oof_g, test_g, auc_g, coef_g = fit_meta(F_oof_g, F_test_g, y)
    print(f"  {label:<22s} K={len(names):>2d}  Strat={auc_s:.5f} (Δ M5h "
          f"{(auc_s-M5H_S)*1e4:+.1f}bp)  GroupKF={auc_g:.5f} (Δ M5h "
          f"{(auc_g-M5H_G)*1e4:+.1f}bp)")
    return dict(label=label, names=names, strat=auc_s, groupkf=auc_g,
                oof_s=oof_s, test_s=test_s, oof_g=oof_g, test_g=test_g,
                coef_s=coef_s, coef_g=coef_g,
                l1_strat=l1_per_base(coef_s, names))


def main():
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    # Define variants
    variants = []
    variants.append(("K_add14", M5H_BASE + [D3B]))
    variants.append(("K_swap_d2a", [p for p in M5H_BASE if p[0] != "d2a_te"] + [D3B]))
    variants.append(("K_swap_baseline",
                     [p for p in M5H_BASE if p[0] != "baseline"] + [D3B]))
    variants.append(("K_add15_with_d3a", M5H_BASE + [D3A, D3B]))

    print("=== M5k — drop d3b_seqfe into M5h pool ===\n")
    results = []
    for label, pool in variants:
        results.append(run_variant(label, pool, y))

    # Pick winner by Strat OOF, tie-break by smaller pool
    results_sorted = sorted(results, key=lambda r: (-r["strat"], len(r["names"])))
    winner = results_sorted[0]
    print(f"\n=== M5k WINNER: {winner['label']} ===")
    print(f"  Strat={winner['strat']:.5f}  GroupKF={winner['groupkf']:.5f}  "
          f"K={len(winner['names'])}")
    print("  L1 per base (Strat):")
    for n, v in sorted(winner["l1_strat"].items(), key=lambda x: -x[1]):
        print(f"    {n:<22s} L1={v:.3f}")

    # Persist winner
    np.save(ART / "oof_m5k_strat.npy",
            np.column_stack([1 - winner["oof_s"], winner["oof_s"]]))
    np.save(ART / "test_m5k_strat.npy",
            np.column_stack([1 - winner["test_s"], winner["test_s"]]))
    np.save(ART / "oof_m5k_groupkf.npy",
            np.column_stack([1 - winner["oof_g"], winner["oof_g"]]))
    np.save(ART / "test_m5k_groupkf.npy",
            np.column_stack([1 - winner["test_g"], winner["test_g"]]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = winner["test_s"]
    sub.to_csv("submissions/submission_m5k_lr_meta.csv", index=False)

    res = dict(
        winner=winner["label"],
        winner_pool=winner["names"],
        winner_strat=winner["strat"],
        winner_groupkf=winner["groupkf"],
        winner_l1_strat=winner["l1_strat"],
        all_variants={r["label"]: dict(strat=r["strat"], groupkf=r["groupkf"],
                                       K=len(r["names"]), pool=r["names"],
                                       l1_strat=r["l1_strat"])
                      for r in results},
    )
    (ART / "m5k_lr_meta_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n→ scripts/artifacts/m5k_lr_meta_results.json")
    print(f"→ submissions/submission_m5k_lr_meta.csv")


if __name__ == "__main__":
    main()
