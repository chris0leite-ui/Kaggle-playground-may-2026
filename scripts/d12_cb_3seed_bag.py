"""D12 CatBoost lossguide 3-seed bag (Strat anchor only, CPU).

Lossguide CPU variant (no GPU). 3 seeds × 5-fold StratKF(seed=42).
Probability-mean bag across seeds.

Seeds: 42, 7, 123 (subset of {42,7,123,456,789} per 1.5h CPU cap;
fold0 smoke at 277s → 5×5 = 115 min > 90 min cap).

Output:
  scripts/artifacts/oof_d12_cb_5seed_bag_strat.npy  (named "5seed" for
    contract; actually 3-seed per cap)
  scripts/artifacts/test_d12_cb_5seed_bag_strat.npy
  scripts/artifacts/d12_cb_3seed_bag_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
CAT_COLS = ["Driver", "Race", "Compound", "Year"]
N_FOLDS = 5
CV_SEED = 42
SEEDS = [42, 7, 123]   # 3-seed bag per CPU cap (115 min > 90 cap → drop to 3)

P_BASE = dict(iterations=800, learning_rate=0.08, l2_leaf_reg=3.0,
              eval_metric="AUC", od_type="Iter", od_wait=50,
              verbose=0, thread_count=-1, allow_writing_files=False,
              grow_policy="Lossguide", num_leaves=64, max_depth=8)


def run_seed(seed, splits, X, y, X_test):
    oof = np.zeros(len(y), dtype=np.float64)
    tp = np.zeros(len(X_test), dtype=np.float64)
    fs, fw, biters = [], [], []
    pool_test = Pool(X_test, cat_features=CAT_COLS)
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        ptr = Pool(X.iloc[tr], y[tr], cat_features=CAT_COLS)
        pva = Pool(X.iloc[va], y[va], cat_features=CAT_COLS)
        params = dict(P_BASE); params["random_seed"] = seed
        m = CatBoostClassifier(**params)
        m.fit(ptr, eval_set=pva)
        p_va = m.predict_proba(pva)[:, 1]
        oof[va] = p_va
        tp += m.predict_proba(pool_test)[:, 1] / N_FOLDS
        wall = time.time() - t0
        s = float(roc_auc_score(y[va], p_va))
        bi = int(m.get_best_iteration())
        fs.append(s); fw.append(wall); biters.append(bi)
        print(f"  [seed{seed}/f{k}] AUC={s:.5f} bi={bi} wall={wall:.1f}s")
    auc = float(roc_auc_score(y, oof))
    print(f"[seed{seed}] OOF={auc:.5f} fold_std={np.std(fs):.5f} "
          f"total={sum(fw):.0f}s")
    return oof, tp, auc, fs, fw, biters


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    print(f"[setup] data load {time.time()-t0:.1f}s, n_train={len(y)}, n_test={len(X_test)}")

    seed_oofs = []
    seed_tests = []
    seed_results = {}
    for seed in SEEDS:
        oof, tp, auc, fs, fw, biters = run_seed(seed, splits, X, y, X_test)
        seed_oofs.append(oof)
        seed_tests.append(tp)
        seed_results[f"seed{seed}"] = dict(
            oof_auc=auc, fold_scores=fs, fold_walls=fw,
            fold_std=float(np.std(fs)), best_iters=biters)
        np.save(ART / f"oof_d12_cb_seed{seed}_strat.npy",
                np.column_stack([1 - oof, oof]).astype(np.float32))
        np.save(ART / f"test_d12_cb_seed{seed}_strat.npy",
                np.column_stack([1 - tp, tp]).astype(np.float32))

    bag_oof = np.mean(seed_oofs, axis=0)
    bag_test = np.mean(seed_tests, axis=0)
    bag_auc = float(roc_auc_score(y, bag_oof))
    print(f"\n=== BAG ({len(SEEDS)} seeds, prob-mean) OOF AUC = {bag_auc:.5f} ===")
    for s, o in zip(SEEDS, seed_oofs):
        print(f"  seed{s}: OOF AUC = {roc_auc_score(y, o):.5f}")

    # contract-named arrays (task asked for "5seed_bag" naming for both)
    np.save(ART / "oof_d12_cb_5seed_bag_strat.npy",
            np.column_stack([1 - bag_oof, bag_oof]).astype(np.float32))
    np.save(ART / "test_d12_cb_5seed_bag_strat.npy",
            np.column_stack([1 - bag_test, bag_test]).astype(np.float32))

    results = dict(
        seeds=SEEDS,
        cv_seed=CV_SEED,
        per_seed=seed_results,
        bag_oof_auc=bag_auc,
        total_wall_s=time.time() - t0,
        seeds_dropped_per_cap=[456, 789],
        cap_note="CPU 1.5h cap; fold0 smoke 277s → 5x5 proj 115min > 90min cap",
    )
    (ART / "d12_cb_3seed_bag_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
