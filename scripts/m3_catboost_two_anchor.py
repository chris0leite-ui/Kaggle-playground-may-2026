"""M3 CatBoost — two-anchor 5-fold (SHRUNK config).

Strat (LB proxy) + GroupKF(Race) (leakage-honest).
Shrunk: iters=800, lr=0.08, depth=6, l2=3.0, od_wait=50.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof
from m3_catboost_audit import write_audit

TARGET, ID_COL = "PitNextLap", "id"
CAT_COLS = ["Driver", "Race", "Compound"]
BASE_S, BASE_G = 0.94075, 0.92059
P = dict(iterations=800, learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
         random_seed=42, eval_metric="AUC", od_type="Iter", od_wait=50,
         verbose=0, thread_count=-1, allow_writing_files=False)


def run_anchor(name, splits, X, y, X_test, want_imp=False):
    oof = np.zeros(len(y), dtype=np.float32)
    test_p = np.zeros(len(X_test), dtype=np.float32)
    scores, biters, walls, fi0 = [], [], [], None
    pool_test = Pool(X_test, cat_features=CAT_COLS)
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        ptr = Pool(X.iloc[tr], y[tr], cat_features=CAT_COLS)
        pva = Pool(X.iloc[va], y[va], cat_features=CAT_COLS)
        m = CatBoostClassifier(**P); m.fit(ptr, eval_set=pva)
        p_va = m.predict_proba(pva)[:, 1]
        oof[va] = p_va
        test_p += m.predict_proba(pool_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        bi = int(m.get_best_iteration()); wall = time.time() - t0
        scores.append(s); biters.append(bi); walls.append(wall)
        if want_imp and k == 0:
            fi = m.get_feature_importance(ptr, type="PredictionValuesChange")
            fi0 = sorted(zip(X.columns.tolist(), [float(v) for v in fi]),
                         key=lambda t: -t[1])
        print(f"  [{name}] f{k}: AUC={s:.5f} bi={bi} wall={wall:.1f}s")
    return (oof, test_p, float(roc_auc_score(y, oof)), scores,
            float(np.std(scores)), biters, walls, fi0)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in CAT_COLS:
        X[c] = X[c].astype(str); X_test[c] = X_test[c].astype(str)

    print("=== Anchor A: StratifiedKFold(5, seed=42) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y)), y))
    oof_a, test_a, auc_a, fs_a, sd_a, bi_a, w_a, fi0 = run_anchor(
        "STRAT", splits_a, X, y, X_test, want_imp=True)

    print("=== Anchor B: GroupKFold(5) on Race ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, train["Race"].values))
    oof_b, test_b, auc_b, fs_b, sd_b, bi_b, w_b, _ = run_anchor(
        "GROUP", splits_b, X, y, X_test)
    total = time.time() - t0
    d_a = (auc_a - BASE_S) * 1e4; d_b = (auc_b - BASE_G) * 1e4
    print(f"\nStrat OOF: {auc_a:.5f}  Δ={d_a:+.1f}bp")
    print(f"GroupKF OOF: {auc_b:.5f}  Δ={d_b:+.1f}bp")

    save_oof("m3_catboost_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=sd_a, fold_scores=fs_a,
                  cv="StratifiedKFold(5, seed=42)", metric="roc_auc",
                  delta_vs_baseline_bp=d_a, params_used=P,
                  best_iters=bi_a, fold_walls=w_a))
    save_oof("m3_catboost_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=sd_b, fold_scores=fs_b,
                  cv="GroupKFold(5) on Race", metric="roc_auc",
                  delta_vs_baseline_bp=d_b, params_used=P,
                  best_iters=bi_b, fold_walls=w_b))
    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_m3_catboost.csv", index=False)
    write_audit(dict(auc_a=auc_a, sd_a=sd_a, fs_a=fs_a, d_a=d_a, bi_a=bi_a,
                     w_a=w_a, auc_b=auc_b, sd_b=sd_b, fs_b=fs_b, d_b=d_b,
                     bi_b=bi_b, w_b=w_b, total=total, fi0=fi0,
                     P=P, BASE_S=BASE_S, BASE_G=BASE_G))
    Path("scripts/artifacts/m3_catboost_summary.json").write_text(json.dumps(
        dict(auc_strat=auc_a, auc_groupkf=auc_b, std_strat=sd_a, std_groupkf=sd_b,
             total_wall_s=total, best_iters_strat=bi_a, best_iters_groupkf=bi_b,
             g1_pass_strat=auc_a >= BASE_S - 5e-4,
             g1_pass_groupkf=auc_b >= BASE_G - 5e-4,
             delta_strat_bp=d_a, delta_groupkf_bp=d_b,
             top_features=(fi0 or [])[:10]), indent=2, default=str))


if __name__ == "__main__":
    main()
