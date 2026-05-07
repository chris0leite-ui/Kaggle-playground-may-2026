"""d18 — Solo K=23+v4+h1d+1 marginals across 15 candidate bases.

Faster than the greedy version: just measure each candidate's marginal
lift solo on top of K=23 v4+h1d. Tells us which structurally distinct
bases (DAE, orig-transfer, leak-lookup, Rozen, etc.) survive on top of
the mainline pool.

Output:
  scripts/artifacts/d18_combined_solo_results.json
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
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
    "d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound",
    "d9f_FM_A", "d9f_FM_B",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def _meta_oof(y, F):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def main():
    t0 = time.time()
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values

    candidates = [
        ("d15b_lgbm_dae_only",          "DAE-only (most-diverse base of session)"),
        ("d15b_lgbm_dae_full",          "DAE-full (raw+latent)"),
        ("d15_orig_transfer",           "orig-transfer full-feature LGBM"),
        ("d15_leak_lookup",             "EB leak_lookup univariate+bivariate"),
        ("p1_single_lgbm_v3_feA_te",    "p1 v3 fold-safe Rozen-style LGBM"),
        ("d18_e2_preimage_knn",         "E2 preimage kNN"),
        ("d18b_chain_decomp",           "d18b chain v2 (q10)"),
        ("d18_chain_decomp",            "d18 chain v1 (gauss)"),
        ("d18_f5_class_cond_gmm",       "F5 class-cond GMM"),
        ("d18_j_cond_vector",           "J cond-vector"),
        ("d18_f2_constraint",           "F2 constraint"),
        ("d16_orig_continuous_only",    "d16 orig cont_only"),
    ]

    # Load K=21 + v4 + h1d
    pool_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]
    v4 = _pos(ART / "oof_p1_single_cb_v4_gpu_strat.npy")
    h1d = _pos(ART / "oof_d17_h1d_yekenot_full_strat.npy")
    base_oofs = pool_oofs + [v4, h1d]

    F_K23 = _expand(np.column_stack(base_oofs))
    print(f"Computing K=23 (v4+h1d) baseline ...")
    _, auc_K23 = _meta_oof(y, F_K23)
    print(f"K=23 + v4 + h1d:  OOF {auc_K23:.5f}\n")

    results = []
    for i, (name, label) in enumerate(candidates, 1):
        p = ART / f"oof_{name}_strat.npy"
        if not p.exists():
            print(f"  [skip] {name} missing")
            continue
        cand = _pos(p)
        F = _expand(np.column_stack(base_oofs + [cand]))
        t1 = time.time()
        _, auc = _meta_oof(y, F)
        d = (auc - auc_K23) * 1e4
        results.append(dict(name=name, label=label, k24_oof=auc, delta_bp=d,
                            wall_s=time.time() - t1))
        print(f"  [{i:>2}/{len(candidates)}]  {name:32s}  OOF {auc:.5f}  Δ +{d:+.3f} bp  ({time.time()-t1:.0f}s)")

    results.sort(key=lambda r: -r["delta_bp"])
    print("\n=== ranked solo K=23+1 marginals ===")
    for r in results:
        print(f"  {r['name']:32s}  Δ {r['delta_bp']:+.3f} bp  ({r['label']})")

    summary = dict(
        K23_v4_h1d_baseline=auc_K23,
        candidates=results,
        wall_s=time.time() - t0,
    )
    (ART / "d18_combined_solo_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ d18_combined_solo_results.json  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
