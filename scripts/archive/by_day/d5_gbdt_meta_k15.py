"""d5_gbdt_meta_k15 — GBDT meta over K=15 pool (M5q + recursive).

Day-5 K=15 LR-stack was null (3rd lr-meta-rank-lock confirmation).
But recursive grabbed L1=0.841 in M5_K15a — the LR clearly wants
the signal, just can't extract it as rank. Trees can split on
recursive_proba directly and gate other bases on its value:
exactly the "which-base-best-on-which-row" mechanism d4 GBDT-meta
captured for K=14.

Hypothesis: GBDT-meta over K=15 with recursive provides the
non-linear meta the cross-row residual structure that recursive
encodes. Decision rule: OOF >= 0.95048 (matches d4 lgbm_shallow,
which gave -4bp LB) AND ρ vs M5q < 0.999 → slot 2 candidate.

Strat-only (R1). Mirrors d4_gbdt_meta.py exactly with one extra base.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5Q_S = 0.95057
SEED, N_FOLDS = 42, 5

POOL_K15 = [
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
    ("recursive", "d5_recursive_m5q"),
]


def load(name):
    oof = np.load(ART / f"oof_{name}_strat.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_strat.npy")[:, 1].astype(np.float64)
    return oof, test


def make_features(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    return np.hstack([P, rk]).astype(np.float32)


def lgbm_params(profile: str) -> dict:
    if profile == "shallow":
        return dict(num_leaves=8, max_depth=3, learning_rate=0.05,
                    n_estimators=2000, min_child_samples=200, reg_lambda=1.0,
                    subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                    verbose=-1)
    if profile == "medium":
        return dict(num_leaves=32, max_depth=5, learning_rate=0.03,
                    n_estimators=2000, min_child_samples=100, reg_lambda=2.0,
                    subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                    verbose=-1)
    raise ValueError(profile)


def fit_lgbm_meta(F_oof, F_test, y, profile: str):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_p = np.zeros(len(F_test), dtype=np.float64)
    biters = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = lgb.LGBMClassifier(**lgbm_params(profile))
        m.fit(F_oof[tr], y[tr], eval_set=[(F_oof[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        meta_oof[va] = m.predict_proba(F_oof[va])[:, 1]
        test_p += m.predict_proba(F_test)[:, 1] / N_FOLDS
        biters.append(int(m.best_iteration_))
    return meta_oof, test_p, biters


def fit_hgbc_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_p = np.zeros(len(F_test), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=4, max_leaf_nodes=15,
            min_samples_leaf=200, l2_regularization=1.0,
            max_iter=2000, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=50, random_state=SEED)
        m.fit(F_oof[tr], y[tr])
        meta_oof[va] = m.predict_proba(F_oof[va])[:, 1]
        test_p += m.predict_proba(F_test)[:, 1] / N_FOLDS
    return meta_oof, test_p


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]

    Xs_oof, Xs_test, names = [], [], []
    for label, name in POOL_K15:
        oo, te = load(name)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    F_oof = make_features(np.column_stack(Xs_oof))
    F_test = make_features(np.column_stack(Xs_test))
    print(f"F_oof shape: {F_oof.shape}  F_test shape: {F_test.shape}")
    print(f"K=15 pool (M5q 14 bases + recursive)")
    print(f"Anchor: M5q LR-meta Strat 0.95057, LB 0.95005")
    print(f"d4 K=14 GBDT-meta benchmarks: lgbm_sh 0.95048 / lgbm_md 0.95047 / hgbc 0.95042\n")

    results = {}
    for variant in ("lgbm_shallow", "lgbm_medium", "hgbc"):
        t0 = time.time()
        if variant.startswith("lgbm"):
            profile = variant.split("_")[1]
            mo, tp, biters = fit_lgbm_meta(F_oof, F_test, y, profile)
        else:
            mo, tp = fit_hgbc_meta(F_oof, F_test, y)
            biters = None
        auc = float(roc_auc_score(y, mo))
        rho_m5q, _ = spearmanr(tp, test_m5q)
        delta = (auc - M5Q_S) * 1e4
        # Gate matches d4 + d5 K=15 LR rules
        if rho_m5q >= 0.9997: gate = "TIE_EXPECTED"
        elif rho_m5q >= 0.999: gate = "TIE_LIKELY"
        elif rho_m5q >= 0.994: gate = "REAL_DELTA"
        else: gate = "DIVERGENT"
        wall = time.time() - t0
        print(f"[{variant}] Strat OOF: {auc:.5f}  Δ M5q: {delta:+.2f}bp  "
              f"ρ vs M5q test: {rho_m5q:.5f}  [{gate}]  wall={wall:.1f}s")
        if biters: print(f"  best_iters: {biters}")
        results[variant] = dict(strat_oof=auc, delta_m5q_bp=delta,
                                rho_vs_m5q_test=float(rho_m5q), gate=gate,
                                wall=wall, best_iters=biters)
        slug = f"d5_meta_k15_{variant}"
        np.save(ART / f"oof_{slug}_strat.npy", np.column_stack([1 - mo, mo]))
        np.save(ART / f"test_{slug}_strat.npy", np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_{slug}.csv", index=False)

    (ART / "d5_gbdt_meta_k15_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n→ scripts/artifacts/d5_gbdt_meta_k15_results.json")


if __name__ == "__main__":
    main()
