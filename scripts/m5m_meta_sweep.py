"""M5m — meta-stacker sweep on M5h pool with inner-CV gating.

Tests whether the LR meta (raw+rank+logit, C=1.0) is leaving signal on
the table relative to alternative metas:

  A. Caruana hill-climb (forward-greedy convex weights, AUC-direct)
  B. LightGBM meta-stacker (non-linear interactions across bases)
  C. Equal-weight floor: geomean, mean-rank
  D. L1-penalty LR meta (sparse; effectively a prune)

All evaluated via 5-fold inner CV on the OOF rows themselves (per
the lesson from per-group isotonic: in-sample lift on the OOF is
unreliable; honest gen requires held-out inner-fold).

R1: Strat-only.

Output: best variant's OOF + test artifacts; submission for the winner.
Reference: M5h Strat 0.95043, LB 0.94991 (gap −5.2bp).
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

import lightgbm as lgb

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5H_AGG_S = 0.95043
SEED, N_FOLDS = 42, 5

POOL = [
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


def load(name, suffix="strat"):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rank, logit])


# ============================================================
# Variant A: Caruana hill-climb (forward greedy, AUC-direct)
# ============================================================

def hillclimb_weights(P_oof, y, n_iter=50):
    """Caruana 2004 ensemble selection. Start empty, add one base at a time
    (with replacement) maximizing AUC of the running average. Returns
    integer counts; normalize to weights. AUC-direct optimization."""
    K = P_oof.shape[1]
    counts = np.zeros(K, dtype=np.int64)
    cum = np.zeros(P_oof.shape[0])
    best_auc = -np.inf
    for step in range(n_iter):
        best_b, best_b_auc = -1, -np.inf
        for b in range(K):
            trial = (cum + P_oof[:, b]) / (counts.sum() + 1)
            a = roc_auc_score(y, trial)
            if a > best_b_auc:
                best_b_auc = a
                best_b = b
        counts[best_b] += 1
        cum = cum + P_oof[:, best_b]
        best_auc = best_b_auc
    return counts / counts.sum(), best_auc


def hc_apply(P, weights):
    return P @ weights


# ============================================================
# Variant B: LightGBM meta-stacker
# ============================================================

LGBM_META_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=8, max_depth=3,
    feature_fraction=1.0, bagging_fraction=1.0, min_data_in_leaf=200,
    reg_lambda=1.0, verbose=-1, seed=SEED,
)


def lgbm_meta_oof(F_oof, F_test, y, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_proba = np.zeros(F_test.shape[0], dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        dtr = lgb.Dataset(F_oof[tr], y[tr])
        dva = lgb.Dataset(F_oof[va], y[va])
        m = lgb.train(LGBM_META_PARAMS, dtr, num_boost_round=500,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        meta_oof[va] = m.predict(F_oof[va])
        test_proba += m.predict(F_test) / n_folds
    return meta_oof, test_proba


# ============================================================
# Variant C: equal-weight floor (geomean, mean-rank)
# ============================================================

def geomean(P):
    return np.exp(np.mean(np.log(np.clip(P, 1e-12, 1.0)), axis=1))


def mean_rank(P):
    n = P.shape[0]
    return np.mean(np.column_stack([rankdata(c) / n for c in P.T]), axis=1)


# ============================================================
# Variant D: L1-LR meta (sparse weights, no raw+rank+logit expansion)
# ============================================================

def l1_lr_meta(F_oof, F_test, y, C=1.0, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_proba = np.zeros(F_test.shape[0], dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=C, max_iter=2000, solver="liblinear",
                                penalty="l1")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
        test_proba += lr.predict_proba(F_test)[:, 1] / n_folds
    lr_full = LogisticRegression(C=C, max_iter=2000, solver="liblinear",
                                  penalty="l1")
    lr_full.fit(F_oof, y)
    return meta_oof, test_proba, lr_full.coef_.ravel()


# ============================================================
# Inner-CV protocol for hill-climb (A) — fits weights on inner-train,
# evals on inner-val
# ============================================================

def hc_inner_cv(P_oof, y, n_folds=N_FOLDS, seed=SEED, n_iter=50):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    inner_oof = np.zeros(len(y), dtype=np.float64)
    weights_per_fold = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        w, _ = hillclimb_weights(P_oof[tr], y[tr], n_iter=n_iter)
        inner_oof[va] = hc_apply(P_oof[va], w)
        weights_per_fold.append(w)
    return inner_oof, np.mean(weights_per_fold, axis=0)


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    Xs_oof, Xs_test, names = [], [], []
    for label, name in POOL:
        oo, te = load(name, "strat")
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    P_oof = np.column_stack(Xs_oof)
    P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof)   # [raw, rank, logit] expansion (M5h-style)
    F_test = expand(P_test)
    print(f"Pool size: {len(names)}; expanded features: {F_oof.shape[1]}\n")

    results = {}

    # === C: equal-weight floor ===
    print("=== C — equal-weight floor ===")
    geo_oof = geomean(P_oof)
    rank_oof = mean_rank(P_oof)
    auc_geo = float(roc_auc_score(y, geo_oof))
    auc_rank = float(roc_auc_score(y, rank_oof))
    print(f"  geomean        OOF AUC = {auc_geo:.5f}  Δ M5h = {(auc_geo-M5H_AGG_S)*1e4:+.1f}bp")
    print(f"  mean-rank      OOF AUC = {auc_rank:.5f}  Δ M5h = {(auc_rank-M5H_AGG_S)*1e4:+.1f}bp")
    results["C_geomean"] = dict(strat=auc_geo, oof=geo_oof,
                                test=geomean(P_test))
    results["C_meanrank"] = dict(strat=auc_rank, oof=rank_oof,
                                 test=mean_rank(P_test))

    # === A: hill-climb (in-sample + inner-CV) ===
    print("\n=== A — Caruana hill-climb (AUC-direct) ===")
    w_full, auc_hc_full = hillclimb_weights(P_oof, y, n_iter=50)
    print(f"  in-sample  OOF AUC = {auc_hc_full:.5f}  Δ M5h = {(auc_hc_full-M5H_AGG_S)*1e4:+.1f}bp")
    print(f"  weights:")
    for n, w in sorted(zip(names, w_full), key=lambda x: -x[1]):
        if w > 0:
            print(f"    {n:<22s} {w:.3f}")

    inner_oof_hc, w_avg = hc_inner_cv(P_oof, y, n_iter=50)
    auc_hc_inner = float(roc_auc_score(y, inner_oof_hc))
    print(f"  inner-CV    OOF AUC = {auc_hc_inner:.5f}  Δ M5h = {(auc_hc_inner-M5H_AGG_S)*1e4:+.1f}bp")
    results["A_hillclimb"] = dict(strat=auc_hc_inner, in_sample=auc_hc_full,
                                  oof=inner_oof_hc,
                                  test=hc_apply(P_test, w_full),
                                  weights=w_full.tolist())

    # === B: LGBM meta-stacker (uses raw [raw,rank,logit] expansion,
    # already inner-CV via 5-fold) ===
    print("\n=== B — LightGBM meta-stacker (non-linear) ===")
    print("  features: [raw, rank, logit] expansion (39 feats)")
    lgbm_oof, lgbm_test = lgbm_meta_oof(F_oof, F_test, y)
    auc_lgbm = float(roc_auc_score(y, lgbm_oof))
    print(f"  inner-CV    OOF AUC = {auc_lgbm:.5f}  Δ M5h = {(auc_lgbm-M5H_AGG_S)*1e4:+.1f}bp")
    results["B_lgbm_meta"] = dict(strat=auc_lgbm, oof=lgbm_oof, test=lgbm_test)

    # === D: L1-LR meta sweep on raw-only features (13 features) ===
    print("\n=== D — L1-LR meta (raw probs only, no rank/logit) ===")
    for C in [0.1, 1.0, 10.0]:
        oof_d, test_d, coef_d = l1_lr_meta(P_oof, P_test, y, C=C)
        auc_d = float(roc_auc_score(y, oof_d))
        n_nonzero = int((np.abs(coef_d) > 1e-6).sum())
        print(f"  L1 C={C:>5.1f}  OOF AUC = {auc_d:.5f}  Δ M5h = {(auc_d-M5H_AGG_S)*1e4:+.1f}bp  "
              f"nonzero={n_nonzero}/{len(coef_d)}")
        results[f"D_L1_LR_C{C}"] = dict(strat=auc_d, oof=oof_d, test=test_d,
                                         coefs=coef_d.tolist(),
                                         n_nonzero=n_nonzero)

    # === D2: L1-LR meta on full [raw,rank,logit] expansion ===
    print("\n=== D2 — L1-LR meta on raw+rank+logit (39 feats) ===")
    for C in [0.1, 1.0, 10.0]:
        oof_d, test_d, coef_d = l1_lr_meta(F_oof, F_test, y, C=C)
        auc_d = float(roc_auc_score(y, oof_d))
        n_nonzero = int((np.abs(coef_d) > 1e-6).sum())
        print(f"  L1+expand C={C:>5.1f}  OOF AUC = {auc_d:.5f}  Δ M5h = {(auc_d-M5H_AGG_S)*1e4:+.1f}bp  "
              f"nonzero={n_nonzero}/{len(coef_d)}")
        results[f"D2_L1expand_C{C}"] = dict(strat=auc_d, oof=oof_d, test=test_d,
                                            coefs=coef_d.tolist(),
                                            n_nonzero=n_nonzero)

    # === Pick winner ===
    sorted_r = sorted(results.items(), key=lambda kv: -kv[1]["strat"])
    print("\n=== RANKING ===")
    for label, res in sorted_r:
        print(f"  {label:<22s} Strat={res['strat']:.5f}  Δ M5h={(res['strat']-M5H_AGG_S)*1e4:+.1f}bp")

    winner_label, winner = sorted_r[0]
    print(f"\n=== WINNER: {winner_label} ({winner['strat']:.5f}; "
          f"Δ M5h {(winner['strat']-M5H_AGG_S)*1e4:+.1f}bp) ===")

    # Save winner artifacts
    np.save(ART / "oof_m5m_strat.npy",
            np.column_stack([1 - winner["oof"], winner["oof"]]))
    np.save(ART / "test_m5m_strat.npy",
            np.column_stack([1 - winner["test"], winner["test"]]))

    sub = sample_sub.copy()
    sub[TARGET] = winner["test"]
    sub.to_csv("submissions/submission_m5m_lr_meta.csv", index=False)

    summary = {
        label: {k: v for k, v in res.items() if k not in ("oof", "test")}
        for label, res in results.items()
    }
    summary["winner"] = winner_label
    summary["pool"] = names
    (ART / "m5m_meta_sweep_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/m5m_meta_sweep_results.json")
    print(f"→ submissions/submission_m5m_lr_meta.csv (held)")


if __name__ == "__main__":
    main()
