"""d17 Phase B — extend feature-restriction transfer beyond LGBM/7-feat.

B1. Multi-arch on the WIN subset (7 features): CatBoost + XGBoost + LGBM-tuned.
    Rationale: friction `external-data-arch-bag-redundant-when-shared-training-data`
    was on FULL features. With FEATURE-SUBSET fixed, multi-arch may diversify.

B2. Subset-size sweep N ∈ {3, 5, 9, 11}. Current is 7. Tests whether the lift
    survives at smaller (more restrictive) or larger (less restrictive) feature sets.

B3. Physics-specialist subsets (4 variants):
    - lap_timing       : LapTime + LapTime_Delta + Compound
    - tyre_lifecycle   : TyreLife + Compound + Stint
    - race_progress    : RaceProgress + LapNumber + Position
    - degradation      : CumDeg + LapTime_Delta + TyreLife

B4. Synth-trained restricted to marginal-aligned features (DROP heavily-corrupted
    LapNumber + Stint + RaceProgress per Phase 1 KS=0.18 each). NEW pool class:
    "leakage-robust GBDT" trained on synth.

All variants saved as oof_d17_*_strat.npy / test_*_strat.npy. K=21+1 gates run
in Phase E wrap-up.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

CAT_ALL = ["Driver", "Compound", "Race"]
NUM_ALL = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
           "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
           "RaceProgress", "Position_Change"]

# Marginal-aligned (low KS) ranking from Phase 1 (P1.2):
# Position_Change 0.015, TyreLife 0.017, Position 0.019, LapTime 0.056,
# Year 0.060, Cumulative_Degradation 0.071, PitStop 0.117 →
# corruption tier: low (<0.10): Position_Change, TyreLife, Position, LapTime, Year, CumDeg, PitStop
#                  high (>0.10): Stint 0.175, LapTime_Delta 0.179, RaceProgress 0.186, LapNumber 0.188
LOW_KS_FEATURES = ["TyreLife", "Position", "LapTime (s)", "Cumulative_Degradation",
                    "Position_Change", "Year"]
HIGH_KS_FEATURES = ["Stint", "LapTime_Delta", "RaceProgress", "LapNumber"]
WIN_SUBSET = ["LapTime (s)", "LapTime_Delta", "TyreLife", "RaceProgress",
              "Cumulative_Degradation", "Position", "LapNumber"]  # the d16 winner


def align_cats(dfs, cat_cols):
    for c in cat_cols:
        union = pd.concat([d[c].astype(str) for d in dfs], axis=0)
        cats = sorted(union.dropna().unique())
        for d in dfs:
            d[c] = pd.Categorical(d[c].astype(str), categories=cats)


def fit_predict_orig_lgbm(orig, tr, te, feats, name, params=None):
    """Train LGBM on orig (80/20), predict on synth_train and synth_test."""
    cat_in = [c for c in CAT_ALL if c in feats]
    Xo = orig[feats].copy()
    y_orig = orig[TARGET].astype(int).values
    Xtr, Xva, ytr, yva = train_test_split(Xo, y_orig, test_size=0.2, random_state=SEED, stratify=y_orig)
    p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
              verbose=-1, n_jobs=4, seed=SEED)
    if params:
        p.update(params)
    m = lgb.train(p, lgb.Dataset(Xtr, ytr, categorical_feature=cat_in),
                   num_boost_round=800,
                   valid_sets=[lgb.Dataset(Xva, yva, categorical_feature=cat_in)],
                   callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    held_auc = roc_auc_score(yva, m.predict(Xva))
    oof_synth = m.predict(tr[feats])
    pred_te = m.predict(te[feats])
    synth_auc = roc_auc_score(tr[TARGET].astype(int).values, oof_synth)
    np.save(ART / f"oof_d17_{name}_strat.npy", oof_synth)
    np.save(ART / f"test_d17_{name}_strat.npy", pred_te)
    return dict(name=name, n_features=len(feats), feats=feats,
                 orig_held_auc=float(held_auc), synth_auc=float(synth_auc),
                 best_iter=m.best_iteration)


def fit_predict_orig_xgb(orig, tr, te, feats, name):
    import xgboost as xgb
    cat_in = [c for c in CAT_ALL if c in feats]
    y_orig = orig[TARGET].astype(int).values

    # XGB-friendly encoding
    Xo = orig[feats].copy()
    Xs = tr[feats].copy()
    Xt = te[feats].copy()
    for c in cat_in:
        codes_o = Xo[c].astype(str)
        codes_s = Xs[c].astype(str)
        codes_t = Xt[c].astype(str)
        union = pd.concat([codes_o, codes_s, codes_t]).unique()
        m_ = {v: i for i, v in enumerate(union)}
        Xo[c] = codes_o.map(m_)
        Xs[c] = codes_s.map(m_)
        Xt[c] = codes_t.map(m_)
    Xtr, Xva, ytr, yva = train_test_split(Xo, y_orig, test_size=0.2, random_state=SEED, stratify=y_orig)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dsy = xgb.DMatrix(Xs)
    dte = xgb.DMatrix(Xt)
    params = dict(objective="binary:logistic", eval_metric="auc", eta=0.05,
                   max_depth=8, subsample=0.9, colsample_bytree=0.9,
                   min_child_weight=20, tree_method="hist", verbosity=0, seed=SEED)
    m = xgb.train(params, dtr, num_boost_round=800,
                   evals=[(dva, "va")], early_stopping_rounds=40, verbose_eval=False)
    held_auc = roc_auc_score(yva, m.predict(dva, iteration_range=(0, m.best_iteration + 1)))
    oof_synth = m.predict(dsy, iteration_range=(0, m.best_iteration + 1))
    pred_te = m.predict(dte, iteration_range=(0, m.best_iteration + 1))
    synth_auc = roc_auc_score(tr[TARGET].astype(int).values, oof_synth)
    np.save(ART / f"oof_d17_{name}_strat.npy", oof_synth)
    np.save(ART / f"test_d17_{name}_strat.npy", pred_te)
    return dict(name=name, arch="xgb", n_features=len(feats),
                 orig_held_auc=float(held_auc), synth_auc=float(synth_auc),
                 best_iter=int(m.best_iteration))


def fit_predict_orig_cb(orig, tr, te, feats, name):
    from catboost import CatBoostClassifier
    cat_in = [c for c in CAT_ALL if c in feats]
    cat_pos = [feats.index(c) for c in cat_in]
    y_orig = orig[TARGET].astype(int).values
    Xo = orig[feats].copy()
    for c in cat_in:
        Xo[c] = Xo[c].astype(str).fillna("nan")
    Xtr, Xva, ytr, yva = train_test_split(Xo, y_orig, test_size=0.2, random_state=SEED, stratify=y_orig)
    Xs = tr[feats].copy()
    Xt = te[feats].copy()
    for c in cat_in:
        Xs[c] = Xs[c].astype(str).fillna("nan")
        Xt[c] = Xt[c].astype(str).fillna("nan")
    m = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6,
                            cat_features=cat_pos if cat_pos else None,
                            verbose=False, early_stopping_rounds=40,
                            random_seed=SEED, thread_count=4)
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    held_auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    oof_synth = m.predict_proba(Xs)[:, 1]
    pred_te = m.predict_proba(Xt)[:, 1]
    synth_auc = roc_auc_score(tr[TARGET].astype(int).values, oof_synth)
    np.save(ART / f"oof_d17_{name}_strat.npy", oof_synth)
    np.save(ART / f"test_d17_{name}_strat.npy", pred_te)
    return dict(name=name, arch="cb", n_features=len(feats),
                 orig_held_auc=float(held_auc), synth_auc=float(synth_auc),
                 best_iter=int(m.tree_count_))


def fit_predict_synth_lgbm(tr, te, feats, name):
    """5-fold synth-trained LGBM on a feature subset. Used for B4."""
    cat_in = [c for c in CAT_ALL if c in feats]
    Xs = tr[feats].copy()
    Xt = te[feats].copy()
    y = tr[TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    pred_te = np.zeros(len(te))
    p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
              verbose=-1, n_jobs=4, seed=SEED)
    for fi, (tri, vai) in enumerate(skf.split(Xs, y)):
        m = lgb.train(p, lgb.Dataset(Xs.iloc[tri], y[tri], categorical_feature=cat_in),
                       num_boost_round=600,
                       valid_sets=[lgb.Dataset(Xs.iloc[vai], y[vai], categorical_feature=cat_in)],
                       callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        oof[vai] = m.predict(Xs.iloc[vai])
        pred_te += m.predict(Xt) / N_FOLDS
    auc = roc_auc_score(y, oof)
    np.save(ART / f"oof_d17_{name}_strat.npy", oof)
    np.save(ART / f"test_d17_{name}_strat.npy", pred_te)
    return dict(name=name, arch="synth-restricted-lgbm", n_features=len(feats),
                 oof_auc=float(auc))


def main():
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    align_cats([tr, te, orig], CAT_ALL)

    summary = {}

    # ==================================================================
    # B1. Multi-arch on WIN subset (7 features)
    # ==================================================================
    step("B1  multi-arch on WIN subset (7 feats)")
    summary["B1"] = []
    summary["B1"].append(fit_predict_orig_xgb(orig, tr, te, WIN_SUBSET, "orig_continuous_xgb"))
    step(f"  XGB done: {summary['B1'][-1]}")
    summary["B1"].append(fit_predict_orig_cb(orig, tr, te, WIN_SUBSET, "orig_continuous_cb"))
    step(f"  CB done:  {summary['B1'][-1]}")
    summary["B1"].append(fit_predict_orig_lgbm(orig, tr, te, WIN_SUBSET, "orig_continuous_lgbm_deep",
                                                 dict(num_leaves=255, max_depth=10, min_data_in_leaf=50)))
    step(f"  LGBM-deep done: {summary['B1'][-1]}")

    # ==================================================================
    # B2. Subset-size sweep N ∈ {3, 5, 9, 11}
    # ==================================================================
    step("B2  subset-size sweep")
    summary["B2"] = []
    # Use the WIN_SUBSET ordering as feature priority (most-physics first)
    for n in [3, 5, 9]:
        sub = WIN_SUBSET[:n] if n <= 7 else WIN_SUBSET + LOW_KS_FEATURES[:n - 7]
        sub = list(dict.fromkeys(sub))[:n]  # uniq
        name = f"orig_continuous_n{n}"
        summary["B2"].append(fit_predict_orig_lgbm(orig, tr, te, sub, name))
        step(f"  n={n}: {summary['B2'][-1]}")
    # n=11 = all numerics (corresponds to continuous_only's parent set)
    sub = list(NUM_ALL[:])
    name = "orig_continuous_n11"
    summary["B2"].append(fit_predict_orig_lgbm(orig, tr, te, sub, name))
    step(f"  n=11: {summary['B2'][-1]}")

    # ==================================================================
    # B3. Physics-specialist subsets
    # ==================================================================
    step("B3  physics-specialist subsets")
    PHYSICS_SUBSETS = {
        "phys_lap_timing":      ["LapTime (s)", "LapTime_Delta", "Compound"],
        "phys_tyre_lifecycle":  ["TyreLife", "Compound", "Stint"],
        "phys_race_progress":   ["RaceProgress", "LapNumber", "Position"],
        "phys_degradation":     ["Cumulative_Degradation", "LapTime_Delta", "TyreLife"],
    }
    summary["B3"] = []
    for name, sub in PHYSICS_SUBSETS.items():
        summary["B3"].append(fit_predict_orig_lgbm(orig, tr, te, sub, name))
        step(f"  {name}: {summary['B3'][-1]}")

    # ==================================================================
    # B4. Synth-trained RESTRICTED to marginal-aligned features
    # ==================================================================
    step("B4  synth-trained restricted (drop high-KS features)")
    # WIN_SUBSET minus high-KS features (keep only TyreLife, Position, LapTime, CumDeg, Position_Change)
    # Then add Compound (cat, useful)
    SYNTH_RESTRICTED = LOW_KS_FEATURES + ["Compound"]  # 7 features, all low-KS
    summary["B4"] = []
    summary["B4"].append(fit_predict_synth_lgbm(tr, te, SYNTH_RESTRICTED, "synth_restricted_lowKS"))
    step(f"  {summary['B4'][-1]}")
    # Also: WIN_SUBSET (the d16 features) but synth-trained
    summary["B4"].append(fit_predict_synth_lgbm(tr, te, WIN_SUBSET, "synth_restricted_winsubset"))
    step(f"  {summary['B4'][-1]}")

    summary["runtime_s"] = time.time() - t0
    with open(ART / "d17_phase_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step("DONE")


if __name__ == "__main__":
    main()
