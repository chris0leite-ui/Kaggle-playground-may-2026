"""scripts/lr_leverage_phaseB.py — Probes 3 and 6.

Probe 3: GBDT-meta-stacker on K=24+lr_mega feature matrix.
         Tests whether rank-lock is linear-meta-specific.
Probe 6: lr_mega_oof_prob as a single GBDT input feature.
         Tests whether mega's compressed signal is GBDT-recoverable.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

warnings.filterwarnings("ignore")

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K24_GBDT_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
    "d16_orig_continuous_only", "p1_single_cb_v3_gpu",
    "d17_h1d_yekenot_full",
]

NUM_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
]
CAT_COLS = ["Driver", "Compound", "Race"]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def _lr_meta_oof(y, F):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def _gbdt_meta_oof(y, F, params=None):
    if params is None:
        params = dict(
            objective="binary", metric="auc",
            num_leaves=15, max_depth=4, learning_rate=0.05,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
            min_data_in_leaf=200, lambda_l2=0.1, verbose=-1,
        )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        d_tr = lgb.Dataset(F[tr], y[tr])
        d_va = lgb.Dataset(F[va], y[va], reference=d_tr)
        m = lgb.train(params, d_tr, num_boost_round=2000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(F[va], num_iteration=m.best_iteration)
    return oof, float(roc_auc_score(y, oof))


def probe3_gbdt_meta(y):
    print("\n=== Probe 3: GBDT-meta-stacker on K=24+mega ===", flush=True)
    P_gbdt = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in K24_GBDT_BASES])
    mega = _pos(ART / "oof_lr_mega_strat.npy")
    F = _expand(np.column_stack([P_gbdt, mega]))

    print(f"  feature matrix: {F.shape}", flush=True)
    print("  LR-meta baseline...", flush=True)
    t0 = time.time()
    _, auc_lr = _lr_meta_oof(y, F)
    print(f"    OOF {auc_lr:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    print("  GBDT-meta (LightGBM, depth=4, num_leaves=15)...", flush=True)
    t0 = time.time()
    _, auc_gbdt = _gbdt_meta_oof(y, F)
    delta_bp = (auc_gbdt - auc_lr) * 1e4
    print(f"    OOF {auc_gbdt:.5f}  ({time.time()-t0:.1f}s)  Δ vs LR-meta {delta_bp:+.3f} bp",
          flush=True)

    return dict(lr_meta_auc=auc_lr, gbdt_meta_auc=auc_gbdt,
                delta_vs_lr_meta_bp=float(delta_bp), n_features=int(F.shape[1]))


def probe6_mega_as_gbdt_feature(train, test, y):
    print("\n=== Probe 6: mega_oof_prob as a single GBDT feature ===", flush=True)
    mega_oof = _pos(ART / "oof_lr_mega_strat.npy")
    mega_test = _pos(ART / "test_lr_mega_strat.npy")

    # Build raw + cat-encoded baseline (LGBM-style)
    drv_map = pd.concat([train["Driver"], test["Driver"]]).astype(str)
    drv_codes = drv_map.factorize()[0]
    train_drv = drv_codes[:len(train)]
    test_drv = drv_codes[len(train):]
    cmp_map = pd.concat([train["Compound"], test["Compound"]]).astype(str)
    cmp_codes = cmp_map.factorize()[0]
    train_cmp = cmp_codes[:len(train)]
    test_cmp = cmp_codes[len(train):]
    rac_map = pd.concat([train["Race"], test["Race"]]).astype(str)
    rac_codes = rac_map.factorize()[0]
    train_rac = rac_codes[:len(train)]
    test_rac = rac_codes[len(train):]

    Xtr_raw = np.column_stack([
        train[NUM_COLS].fillna(0).values.astype(np.float32),
        train_drv, train_cmp, train_rac,
    ])
    Xte_raw = np.column_stack([
        test[NUM_COLS].fillna(0).values.astype(np.float32),
        test_drv, test_cmp, test_rac,
    ])
    cat_idx = [Xtr_raw.shape[1] - 3, Xtr_raw.shape[1] - 2, Xtr_raw.shape[1] - 1]

    params = dict(
        objective="binary", metric="auc",
        num_leaves=63, max_depth=6, learning_rate=0.05,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        min_data_in_leaf=200, lambda_l2=0.1, verbose=-1,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print(f"  baseline: LGBM on raw + 3 cat-codes (14 features)...", flush=True)
    t0 = time.time()
    oof_base = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        d_tr = lgb.Dataset(Xtr_raw[tr], y[tr], categorical_feature=cat_idx)
        d_va = lgb.Dataset(Xtr_raw[va], y[va], reference=d_tr,
                           categorical_feature=cat_idx)
        m = lgb.train(params, d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_base[va] = m.predict(Xtr_raw[va], num_iteration=m.best_iteration)
    auc_base = float(roc_auc_score(y, oof_base))
    print(f"    OOF {auc_base:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    Xtr_with = np.column_stack([Xtr_raw, mega_oof])
    Xte_with = np.column_stack([Xte_raw, mega_test])
    print(f"  with mega: LGBM on raw + 3 cat + mega_oof (15 features)...", flush=True)
    t0 = time.time()
    oof_with = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        d_tr = lgb.Dataset(Xtr_with[tr], y[tr], categorical_feature=cat_idx)
        d_va = lgb.Dataset(Xtr_with[va], y[va], reference=d_tr,
                           categorical_feature=cat_idx)
        m = lgb.train(params, d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_with[va] = m.predict(Xtr_with[va], num_iteration=m.best_iteration)
    auc_with = float(roc_auc_score(y, oof_with))
    delta_bp = (auc_with - auc_base) * 1e4
    print(f"    OOF {auc_with:.5f}  ({time.time()-t0:.1f}s)  Δ vs raw-only {delta_bp:+.3f} bp",
          flush=True)

    return dict(baseline_auc=auc_base, with_mega_feat_auc=auc_with,
                delta_bp=float(delta_bp))


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"data: train {train.shape}, prior {y.mean():.4f}", flush=True)

    out = {}
    out["probe3"] = probe3_gbdt_meta(y)
    out["probe6"] = probe6_mega_as_gbdt_feature(train, test, y)

    out_json = ART / "lr_leverage_phaseB.json"
    out_json.write_text(json.dumps(out, indent=2, default=lambda o: float(o)))
    print(f"\n→ {out_json}", flush=True)


if __name__ == "__main__":
    main()
