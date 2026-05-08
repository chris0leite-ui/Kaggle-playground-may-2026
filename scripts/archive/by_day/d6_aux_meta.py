"""F5 / d6_aux_meta — auxiliary disagreement features + GBDT meta.

Critic-loop §3 F5 mechanism: replace LR-meta over (raw + rank) with
LGBM-shallow meta over (raw + rank + per-row aux disagreement
features).  LR can only fit a linear function of base outputs;
disagreement (std, range, count-above) is an LB-relevant signal
the LR meta cannot use.

Anchors:
  M5q (LR meta over expand)                  Strat 0.95057  LB 0.95005
  m5_meta_lgbm_shallow (raw+rank, no aux)    Strat 0.95048  LB 0.95001

Test: does adding per-row aux features lift OOF above M5q? If
positive AND ρ vs M5q < 0.999, slot it for Day-7. If null, F5 is
falsified and we move directly to F1 (hazard-rate reformulation).

Strat-only (R1).  Pre-submit-diff vs M5q before any submission.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5Q_S = 0.95057
M5_META_LGBM_SHALLOW_S = 0.95048
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


def make_aux(P):
    """Per-row disagreement features computed across the K base preds."""
    mean = P.mean(axis=1, keepdims=True)
    std = P.std(axis=1, keepdims=True)
    pmax = P.max(axis=1, keepdims=True)
    pmin = P.min(axis=1, keepdims=True)
    rng = pmax - pmin
    median = np.median(P, axis=1, keepdims=True)
    skew = mean - median
    p75 = np.percentile(P, 75, axis=1, keepdims=True)
    p25 = np.percentile(P, 25, axis=1, keepdims=True)
    iqr = p75 - p25
    cnt_hi = (P > 0.5).sum(axis=1, keepdims=True).astype(np.float64) / P.shape[1]
    cnt_lo = (P < 0.05).sum(axis=1, keepdims=True).astype(np.float64) / P.shape[1]
    return np.hstack([mean, std, pmax, pmin, rng, median, skew,
                      iqr, cnt_hi, cnt_lo])


def make_features(P, with_aux: bool):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    parts = [P, rk]
    if with_aux:
        parts.append(make_aux(P))
    return np.hstack(parts).astype(np.float32)


def lgbm_params() -> dict:
    return dict(num_leaves=8, max_depth=3, learning_rate=0.05,
                n_estimators=2000, min_child_samples=200, reg_lambda=1.0,
                subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                verbose=-1)


def fit_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_p = np.zeros(len(F_test), dtype=np.float64)
    biters, importances = [], None
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = lgb.LGBMClassifier(**lgbm_params())
        m.fit(F_oof[tr], y[tr], eval_set=[(F_oof[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        meta_oof[va] = m.predict_proba(F_oof[va])[:, 1]
        test_p += m.predict_proba(F_test)[:, 1] / N_FOLDS
        biters.append(int(m.best_iteration_))
        imp = m.booster_.feature_importance(importance_type="gain")
        importances = imp.astype(np.float64) if importances is None \
            else importances + imp.astype(np.float64)
    return meta_oof, test_p, biters, importances / N_FOLDS


def feature_names(with_aux: bool, K: int):
    raw = [f"raw__{n}" for n, _ in POOL_M5Q]
    rk = [f"rank__{n}" for n, _ in POOL_M5Q]
    names = raw + rk
    if with_aux:
        names += ["aux__mean", "aux__std", "aux__max", "aux__min",
                  "aux__range", "aux__median", "aux__skew_mean_minus_med",
                  "aux__iqr", "aux__cnt_above_0.5", "aux__cnt_below_0.05"]
    return names


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]

    Xs_oof, Xs_test = [], []
    for _, name in POOL_M5Q:
        oo, te = load(name)
        Xs_oof.append(oo); Xs_test.append(te)
    P_oof = np.column_stack(Xs_oof)
    P_test = np.column_stack(Xs_test)
    K = P_oof.shape[1]

    print(f"Pool K={K}; train rows={len(y)}; test rows={len(P_test)}")
    print(f"Anchors: M5q LR-meta {M5Q_S:.5f}  |  "
          f"m5_meta_lgbm_shallow no-aux {M5_META_LGBM_SHALLOW_S:.5f}\n")

    results = {}
    # Re-run no-aux baseline so deltas use the same code path / seeds.
    for label, with_aux in (("noaux_replicate", False), ("with_aux", True)):
        t0 = time.time()
        F_oof = make_features(P_oof, with_aux)
        F_test = make_features(P_test, with_aux)
        names = feature_names(with_aux, K)
        mo, tp, biters, imp = fit_meta(F_oof, F_test, y)
        auc = float(roc_auc_score(y, mo))
        rho_m5q, _ = spearmanr(tp, test_m5q)
        d_m5q = (auc - M5Q_S) * 1e4
        d_noaux = (auc - M5_META_LGBM_SHALLOW_S) * 1e4
        gate = "PASS" if rho_m5q < 0.999 else "TIE_EXPECTED"
        wall = time.time() - t0
        print(f"[{label}] F={F_oof.shape[1]:>3}  Strat {auc:.5f}  "
              f"Δ M5q {d_m5q:+.2f}bp  Δ no-aux {d_noaux:+.2f}bp  "
              f"ρ vs M5q test {rho_m5q:.5f}  [{gate}]  wall={wall:.1f}s")
        print(f"  best_iters: {biters}")
        # Top-15 features
        order = np.argsort(-imp)[:15]
        print(f"  Top-15 feature importance (gain):")
        for i in order:
            print(f"    {names[i]:<35s} {imp[i]:>10.0f}")
        # Save artifacts
        slug = f"d6_aux_meta_{label}"
        np.save(ART / f"oof_{slug}_strat.npy",
                np.column_stack([1 - mo, mo]))
        np.save(ART / f"test_{slug}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_{slug}.csv", index=False)
        results[label] = dict(
            n_features=int(F_oof.shape[1]), strat_oof=auc,
            delta_m5q_bp=d_m5q, delta_noaux_bp=d_noaux,
            rho_vs_m5q_test=float(rho_m5q), gate=gate,
            wall=wall, best_iters=biters,
            top_features=[(names[i], float(imp[i])) for i in order],
        )

    (ART / "d6_aux_meta_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n→ scripts/artifacts/d6_aux_meta_results.json")


if __name__ == "__main__":
    main()
