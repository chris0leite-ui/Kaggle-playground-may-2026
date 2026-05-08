"""G5 — G1 (within-stint) + G2' (cross-driver) features combined.

If G1 and G2' both clear gates independently, the union may yield
the strongest single base. Tests whether the two feature packs are
complementary or redundant under one LGBM.

Run after G1 / G2' have completed (uses same imports + helpers).
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
    SEED, load_data, load_primary, make_splits,
    report_candidate, save_base,
)
from d13_g1_within_stint_fe import add_within_stint_features  # noqa: E402
from d13_g2_cross_driver_fe import add_cross_driver_features  # noqa: E402

NAME = "d13_g5_combined"


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    print(f"=== G5 G1+G2 combined FE ===  smoke={smoke}")
    train, test, sub, y = load_data()
    primary_oof, primary_test = load_primary()

    print("Adding γ1 within-stint pack ...")
    train = add_within_stint_features(train)
    test = add_within_stint_features(test)
    print("Adding γ4 cross-driver pack ...")
    train, test = add_cross_driver_features(train, test)

    drop_cols = ["id", "PitNextLap"]
    cat_cols = ["Driver", "Compound", "Race"]
    feat_cols = [c for c in train.columns if c not in drop_cols]
    Xtr = train[feat_cols].copy()
    Xte = test[feat_cols].copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    g_added = [c for c in feat_cols if c.startswith("g")]
    print(f"feature_count={len(feat_cols)} (was 14, +{len(g_added)} γ); "
          f"added: {g_added}")

    splits = make_splits(y, train, kind="strat")
    if smoke:
        splits = splits[:1]
        sub_idx = np.random.RandomState(0).choice(
            len(y), size=min(50000, len(y)), replace=False)
        n_rounds = 200
    else:
        n_rounds = 2000

    params = dict(
        objective="binary", learning_rate=0.05, num_leaves=63,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        min_data_in_leaf=200, verbose=-1, seed=SEED,
    )

    oof = np.zeros(len(y), dtype=np.float64)
    test_avg = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        if smoke:
            tr = np.intersect1d(tr, sub_idx)
        t_fold = time.time()
        dtr = lgb.Dataset(Xtr.iloc[tr], y[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(Xtr.iloc[va], y[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=n_rounds,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100),
                                 lgb.log_evaluation(0)])
        p_va = m.predict(Xtr.iloc[va])
        oof[va] = p_va
        test_avg += m.predict(Xte) / len(splits)
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
