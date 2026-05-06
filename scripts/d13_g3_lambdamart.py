"""G3 — stint-grouped LambdaMART base.

LightGBM with `objective=lambdarank, group=stint_id`. Different loss
than every base in the pool. Probe Q1 confirmed binary-target with
0.199 prior and 18.7% of stints having 2+ positives — pairwise rank
within tiny groups (mean 3.87 laps) still defines a valid order.

Day-12 LambdaRank-meta failed at -86bp, but that was Race-grouped
(thousands of rows per group, wrong granularity). Stint-grouped is
~4 rows per group — the right unit.

Run: python scripts/d13_g3_lambdamart.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from d13_g_common import (  # noqa: E402
    N_FOLDS, SEED, load_data, load_primary, make_splits,
    report_candidate, save_base,
)

NAME = "d13_g3_stintgrouped_lambdamart"


def make_stint_group(df: pd.DataFrame) -> np.ndarray:
    """Stable per-row stint group id (Race, Driver, Year, Stint)."""
    return df.groupby(["Race", "Driver", "Year", "Stint"],
                      sort=False).ngroup().values


def fold_groups(stint_id: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-fold group sizes for LightGBM ranking — must reflect the
    contiguous order of `idx`. We sort idx by stint_id first to keep
    each stint contiguous, then return run-lengths."""
    sub = stint_id[idx]
    order = np.argsort(sub, kind="stable")
    sub_sorted = sub[order]
    _, counts = np.unique(sub_sorted, return_counts=True)
    return order, counts


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    print(f"=== G3 stint-grouped LambdaMART ===  smoke={smoke}")
    train, test, sub, y = load_data()
    primary_oof, primary_test = load_primary()

    drop_cols = ["id", "PitNextLap"]
    cat_cols = ["Driver", "Compound", "Race"]
    feat_cols = [c for c in train.columns if c not in drop_cols]
    Xtr = train[feat_cols].copy()
    Xte = test[feat_cols].copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    print(f"feature_count={len(feat_cols)}")

    stint_id = make_stint_group(train)
    n_groups = int(stint_id.max() + 1)
    print(f"stint_id: n_groups={n_groups}, mean group size={len(y)/n_groups:.2f}")

    splits = make_splits(y, train, kind="strat")
    if smoke:
        splits = splits[:1]
        sub_idx = np.random.RandomState(0).choice(
            len(y), size=min(50000, len(y)), replace=False)
        n_rounds = 200
    else:
        n_rounds = 2000

    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10],
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=200,
        label_gain=[0, 1],     # binary labels: 0 → gain 0, 1 → gain 1
        verbose=-1,
        seed=SEED,
    )

    oof = np.zeros(len(y), dtype=np.float64)
    test_avg = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        if smoke:
            tr = np.intersect1d(tr, sub_idx)
        t_fold = time.time()
        # build tr/va with rows reordered by stint_id (groups must be contig.)
        tr_order, tr_groups = fold_groups(stint_id, tr)
        va_order, va_groups = fold_groups(stint_id, va)
        tr_sorted = tr[tr_order]
        va_sorted = va[va_order]

        dtr = lgb.Dataset(Xtr.iloc[tr_sorted], y[tr_sorted],
                          group=tr_groups, categorical_feature=cat_cols)
        dva = lgb.Dataset(Xtr.iloc[va_sorted], y[va_sorted],
                          group=va_groups, categorical_feature=cat_cols,
                          reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=n_rounds,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100),
                                 lgb.log_evaluation(0)])
        # LambdaMART produces unbounded scores; rank to [0,1] for stacking
        from scipy.stats import rankdata
        raw_va = m.predict(Xtr.iloc[va])
        p_va = rankdata(raw_va) / len(raw_va)
        oof[va] = p_va
        raw_te = m.predict(Xte)
        p_te = rankdata(raw_te) / len(raw_te)
        test_avg += p_te / len(splits)
        a = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(a)
        print(f"  f{k}: AUC={a:.5f}  wall={time.time()-t_fold:.1f}s  "
              f"best_iter={m.best_iteration}")

    if smoke:
        std_auc = fold_aucs[0]
        print(f"\nSmoke fold-0 AUC: {std_auc:.5f}  wall={time.time()-t0:.1f}s")
        return

    info = report_candidate(NAME, oof, test_avg, y, primary_oof, primary_test,
                            splits)
    info["fold_aucs"] = fold_aucs
    info["wall_seconds"] = time.time() - t0
    save_base(NAME, oof, test_avg, info)
    print(f"\nTotal wall: {info['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
