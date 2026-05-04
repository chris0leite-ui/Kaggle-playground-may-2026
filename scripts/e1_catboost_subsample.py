"""E1 — Row-subsample CatBoost (80% bagging) two-anchor 5-fold.

Tests whether row-subsampling bounds CatBoost's Race-overfit (M3 was
-41.4bp on GroupKF). If E1 closes that gap meaningfully without
losing Strat lift, it's a better stack base than M3.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
CAT_COLS = ["Driver", "Race", "Compound"]
BASE_S, BASE_G = 0.94075, 0.92059
M3_S, M3_G = 0.94612, 0.91645  # M3 CatBoost reference


def make_cb():
    return CatBoostClassifier(
        iterations=800, learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
        random_seed=SEED, eval_metric="AUC", od_type="Iter", od_wait=50,
        verbose=0, thread_count=-1, allow_writing_files=False,
        bootstrap_type="Bernoulli", subsample=0.8,
    )


def run_anchor(name, splits, X, y, X_test):
    oof = np.zeros(len(y), dtype=np.float32)
    tp = np.zeros(len(X_test), dtype=np.float32)
    pool_te = Pool(X_test, cat_features=CAT_COLS)
    fs, walls, biters = [], [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        ptr = Pool(X.iloc[tr], y[tr], cat_features=CAT_COLS)
        pva = Pool(X.iloc[va], y[va], cat_features=CAT_COLS)
        m = make_cb(); m.fit(ptr, eval_set=pva)
        p = m.predict_proba(pva)[:, 1]
        oof[va] = p
        tp += m.predict_proba(pool_te)[:, 1] / N_FOLDS
        wall = time.time() - t0
        s = float(roc_auc_score(y[va], p))
        bi = int(m.get_best_iteration())
        fs.append(s); walls.append(wall); biters.append(bi)
        print(f"  [{name}] f{k}: AUC={s:.5f} bi={bi} wall={wall:.1f}s")
    return oof, tp, float(roc_auc_score(y, oof)), fs, float(np.std(fs)), walls, biters


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
    oof_a, test_a, auc_a, fs_a, sd_a, w_a, bi_a = run_anchor("STRAT", splits_a, X, y, X_test)

    print("=== Anchor B: GroupKFold(5) on Race ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, train["Race"].values))
    oof_b, test_b, auc_b, fs_b, sd_b, w_b, bi_b = run_anchor("GROUP", splits_b, X, y, X_test)

    da = (auc_a - BASE_S) * 1e4; db = (auc_b - BASE_G) * 1e4
    da_m3 = (auc_a - M3_S) * 1e4; db_m3 = (auc_b - M3_G) * 1e4
    total = time.time() - t0
    print(f"\nStrat: {auc_a:.5f}  Δ baseline={da:+.1f}bp  Δ M3={da_m3:+.1f}bp")
    print(f"GroupKF: {auc_b:.5f}  Δ baseline={db:+.1f}bp  Δ M3={db_m3:+.1f}bp")
    print(f"total wall: {total:.0f}s")

    save_oof("e1_catboost_sub_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=sd_a, fold_scores=fs_a,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=da, delta_vs_m3_bp=da_m3,
                  best_iters=bi_a, fold_walls_s=w_a))
    save_oof("e1_catboost_sub_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=sd_b, fold_scores=fs_b,
                  cv="GroupKFold(Race)", metric="roc_auc",
                  delta_vs_baseline_bp=db, delta_vs_m3_bp=db_m3,
                  best_iters=bi_b, fold_walls_s=w_b))

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_e1_catboost_sub.csv", index=False)

    body = (
        f"# E1 — Row-subsample CatBoost (subsample=0.8) two-anchor (2026-05-04)\n\n"
        f"Tests whether row-subsampling bounds M3's Race-overfit. M3 baseline:\n"
        f"Strat 0.94612, GroupKF 0.91645 (-41.4bp Race-overfit).\n\n"
        f"## Two-anchor results\n\n"
        f"| anchor | OOF AUC | fold_std | per-fold | Δ baseline | Δ vs M3 |\n"
        f"|---|---:|---:|---|---:|---:|\n"
        f"| Strat | **{auc_a:.5f}** | {sd_a:.5f} | {[f'{x:.4f}' for x in fs_a]} | "
        f"{da:+.1f}bp | {da_m3:+.1f}bp |\n"
        f"| GroupKF | **{auc_b:.5f}** | {sd_b:.5f} | {[f'{x:.4f}' for x in fs_b]} | "
        f"{db:+.1f}bp | {db_m3:+.1f}bp |\n\n"
        f"## Wall: {total:.0f}s. Best-iter mean: Strat {np.mean(bi_a):.0f}, GroupKF {np.mean(bi_b):.0f}.\n"
    )
    Path("audit/2026-05-04-e1-catboost-sub.md").write_text(body)
    print(f"audit written")


if __name__ == "__main__":
    main()
