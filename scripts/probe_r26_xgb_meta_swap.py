"""scripts/probe_r26_xgb_meta_swap.py — Stacker swap: replace Path-B
LR-meta with global XGBoost on K=20 pool to test whether LR-meta is the
saturation bottleneck.

Today: 4 cohort-listwise nulls under Path-B (R20 lambdarank K=19 sub-G2,
R21 YetiRank K=19 regression, R23 cohort-agg K=24 regression, R20+R21
combo K=20 regression). All 4 share the same per-segment LR-meta +
shrinkage stacker. Hypothesis: the LR-meta cannot model non-linear
interactions between cohort-listwise bases and other bases — XGBoost
can.

Compares 3 stackers on K=20 pool (R17+R20+R21):
  1. K=20 Path-B DCS τ=100k  (already computed, baseline)
  2. Global XGBoost meta (5-fold strat, depth=4)
  3. Global LightGBM meta (5-fold strat, num_leaves=15)

Inputs: K=20 expanded matrix (60 cols = 20 raw + 20 rank + 20 logit).

Output: oof_R26_xgb_meta_strat.npy + test_R26_xgb_meta_strat.npy +
       oof_R26_lgb_meta_strat.npy + test_R26_lgb_meta_strat.npy
       JSON summary with all 3 OOF AUCs.

Usage:
  python scripts/probe_r26_xgb_meta_swap.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from build_K13_pathb_multiseg import K13_FILES
from build_K11_full_pathb import _pos

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def build_k20_matrix(train, test):
    """K=20 = K=13 + R12 + R13 + R14 + R15 + R17 + R20 + R21."""
    oof_cols, test_cols, names = [], [], []
    for name, of, tf in K13_FILES:
        oof_cols.append(_pos(ART / of))
        test_cols.append(_pos(ART / tf))
        names.append(name)
    extras = [
        ("R12_cb_horizon", "oof_R12_cb_horizon_strat.npy",
         "test_R12_cb_horizon_strat.npy"),
        ("R13_cb_stint_completion", "oof_R13_cb_stint_completion_strat.npy",
         "test_R13_cb_stint_completion_strat.npy"),
        ("R14_tabm", "oof_R14_tabm_strat.npy", "test_R14_tabm_strat.npy"),
        ("R15_xendcg_per_seg", "oof_R15_xendcg_per_seg_strat.npy",
         "test_R15_xendcg_per_seg_strat.npy"),
        ("R17_listwise_race_lap", "oof_R17_listwise_race_lap_strat.npy",
         "test_R17_listwise_race_lap_strat.npy"),
        ("R20_listwise_yrl_lambdarank", "oof_R20_listwise_yrl_lambdarank_strat.npy",
         "test_R20_listwise_yrl_lambdarank_strat.npy"),
        ("R21_yetirank_yrl", "oof_R21_yetirank_yrl_strat.npy",
         "test_R21_yetirank_yrl_strat.npy"),
    ]
    for nm, of, tf in extras:
        oof_cols.append(_pos(ART / of))
        test_cols.append(_pos(ART / tf))
        names.append(nm)
    return (np.column_stack(oof_cols), np.column_stack(test_cols), names)


def expand(M):
    n, k = M.shape
    R = np.column_stack([rankdata(c) / n for c in M.T])
    eps = 1e-9
    P = np.clip(M, eps, 1 - eps)
    L = np.log(P / (1 - P))
    return np.column_stack([M, R, L])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("== R26 stacker swap: XGB / LGB meta on K=20 pool ==", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    K20_oof, K20_test, names = build_k20_matrix(train, test)
    F_oof = expand(K20_oof)
    F_test = expand(K20_test)
    print(f"  K=20 expanded: {F_oof.shape}  test: {F_test.shape}", flush=True)

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(y), 50_000, replace=False)
        F_oof = F_oof[idx]
        y_sub = y[idx]
    else:
        y_sub = y

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_sub)), y_sub))
    n_eff = 1 if args.smoke else N_FOLDS

    # Path-B K=20 OOF reference (already computed)
    pathb_K20 = np.load(ART / "oof_K20_pathb_driverclass_stint_tau100000.npy")
    pathb_auc = roc_auc_score(y, pathb_K20)
    print(f"  Path-B K=20 OOF (reference): {pathb_auc:.6f}", flush=True)
    R15_oof = _pos(ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy")
    R15_auc = roc_auc_score(y, R15_oof)
    print(f"  R15 PRIMARY OOF:            {R15_auc:.6f}\n", flush=True)

    results = {}

    for meta_name in ("xgb", "lgb"):
        print(f"\n--- {meta_name.upper()} meta ---", flush=True)
        oof_meta = np.zeros(len(y_sub), dtype=np.float64)
        test_meta = np.zeros(len(F_test), dtype=np.float64)
        for k, (ti, vi) in enumerate(fold_list[:n_eff], 1):
            t_f = time.time()
            X_tr, X_va = F_oof[ti], F_oof[vi]
            y_tr, y_va = y_sub[ti], y_sub[vi]

            if meta_name == "xgb":
                clf = xgb.XGBClassifier(
                    n_estimators=2000, learning_rate=0.03,
                    max_depth=4, min_child_weight=10,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.0, reg_lambda=1.0,
                    objective="binary:logistic", eval_metric="auc",
                    tree_method="hist", random_state=SEED,
                    early_stopping_rounds=100, verbosity=0,
                )
                clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                pred_va = clf.predict_proba(X_va)[:, 1]
                pred_te = clf.predict_proba(F_test)[:, 1]
            else:
                ds_tr = lgb.Dataset(X_tr, label=y_tr)
                ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
                params = dict(objective="binary", metric="auc",
                              learning_rate=0.03, num_leaves=15,
                              min_data_in_leaf=200, feature_fraction=0.8,
                              bagging_fraction=0.8, bagging_freq=5,
                              lambda_l2=1.0, verbose=-1, random_state=SEED)
                booster = lgb.train(params, ds_tr, num_boost_round=2000,
                                    valid_sets=[ds_va],
                                    callbacks=[lgb.early_stopping(100, verbose=False),
                                               lgb.log_evaluation(0)])
                pred_va = booster.predict(X_va)
                pred_te = booster.predict(F_test)

            oof_meta[vi] = pred_va
            if not args.smoke:
                test_meta += pred_te / n_eff
            auc_va = roc_auc_score(y_va, pred_va)
            print(f"  fold {k}: AUC={auc_va:.5f}  wall={time.time()-t_f:.0f}s",
                  flush=True)

        if args.smoke:
            continue
        auc_full = float(roc_auc_score(y_sub, oof_meta))
        delta_pathb = (auc_full - pathb_auc) * 1e4
        delta_R15 = (auc_full - R15_auc) * 1e4
        rho_pathb_oof = spearmanr(oof_meta, pathb_K20).statistic
        rho_R15_oof = spearmanr(oof_meta, R15_oof).statistic
        print(f"\n  {meta_name.upper()} meta OOF AUC: {auc_full:.6f}", flush=True)
        print(f"    Δ vs Path-B K=20: {delta_pathb:+.4f} bp", flush=True)
        print(f"    Δ vs R15:         {delta_R15:+.4f} bp", flush=True)
        print(f"    ρ_oof vs Path-B:  {rho_pathb_oof:.6f}", flush=True)
        print(f"    ρ_oof vs R15:     {rho_R15_oof:.6f}", flush=True)

        np.save(ART / f"oof_R26_{meta_name}_meta_strat.npy",
                oof_meta.astype(np.float32))
        np.save(ART / f"test_R26_{meta_name}_meta_strat.npy",
                test_meta.astype(np.float32))
        results[meta_name] = dict(oof_auc=auc_full, delta_pathb_bp=delta_pathb,
                                  delta_R15_bp=delta_R15,
                                  rho_pathb=float(rho_pathb_oof),
                                  rho_R15=float(rho_R15_oof))

    summary = dict(pathb_K20_auc=pathb_auc, R15_auc=R15_auc,
                   stacker_results=results, K20_bases=names)
    out_json = Path("audit/2026-05-21-round-26-stacker-swap.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote {out_json}  wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
