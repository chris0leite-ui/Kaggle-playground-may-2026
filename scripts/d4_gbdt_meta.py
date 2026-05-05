"""d4_gbdt_meta — GBDT meta-learner over the M5q K=14 base pool.

Hypothesis: LR-meta-rank-lock on M5q is an LR-on-GBDT-pool artifact.
LR can only build linear combinations + their rank/logit transforms;
non-linear interactions among bases are invisible. Today's evidence
(YetiRank ρ=0.666 + NB ρ=0.853, both add 0+0.24bp via LR meta) is
consistent with this.

A non-linear meta (GBDT or HGBC) can capture: which-base-best on
which-row patterns, base-base interactions, conditional weights.

Design:
  Features: 14 raw base probs + 14 rank-normalized cols = 28 features.
            (Logit is a monotone transform of raw → redundant for GBDT.)
  Inner-CV: outer 5-fold Strat → meta_oof produced fold-out-of-fold.
            Test pred = average of 5 fold-meta-models on F_test.
  Variants: shallow (depth=3), medium (depth=5), HGBC. Conservative
            settings — meta overfits on K=28×440k easily.

Strat-only (R1).
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

POOL_M5Q = [
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
    for label, name in POOL_M5Q:
        oo, te = load(name)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    F_oof = make_features(np.column_stack(Xs_oof))
    F_test = make_features(np.column_stack(Xs_test))
    print(f"F_oof shape: {F_oof.shape}  F_test shape: {F_test.shape}")
    print(f"Anchor: M5q LR-meta Strat 0.95057, LB 0.95005\n")

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
        gate = "PASS" if rho_m5q < 0.999 else "TIE_EXPECTED"
        wall = time.time() - t0
        print(f"[{variant}] Strat OOF: {auc:.5f}  Δ M5q: {delta:+.2f}bp  "
              f"ρ vs M5q test: {rho_m5q:.5f}  [{gate}]  wall={wall:.1f}s")
        if biters: print(f"  best_iters: {biters}")
        results[variant] = dict(strat_oof=auc, delta_m5q_bp=delta,
                                rho_vs_m5q_test=float(rho_m5q), gate=gate,
                                wall=wall, best_iters=biters)
        # Save artifact + sub
        slug = f"m5_meta_{variant}"
        np.save(ART / f"oof_{slug}_strat.npy", np.column_stack([1 - mo, mo]))
        np.save(ART / f"test_{slug}_strat.npy", np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_{slug}.csv", index=False)

    (ART / "d4_gbdt_meta_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n→ scripts/artifacts/d4_gbdt_meta_results.json")


if __name__ == "__main__":
    main()
