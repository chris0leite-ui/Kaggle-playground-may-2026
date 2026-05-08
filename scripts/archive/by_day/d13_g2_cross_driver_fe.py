"""G2' — cross-driver intra-race feature pack (γ4).

Probe Q5 finding: at the (Race, Year, LapNumber) block level
(mean 77 drivers/block), block_tyrelife_std has +0.29 row-corr with
target and block_hard_frac +0.25. No base in the pool consumes these.

Build 8 closed-form cross-driver features computed on train+test
combined (no target leakage; only feature aggregations).

Run: python scripts/d13_g2_cross_driver_fe.py [--smoke]
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

NAME = "d13_g2_cross_driver"


def add_cross_driver_features(train: pd.DataFrame, test: pd.DataFrame
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """8 block-level features per (Race, Year, LapNumber).

    Computed on train+test combined to maximise block coverage. Uses
    no target information — pure feature aggregation.
    """
    full = pd.concat([
        train.assign(__src="tr"),
        test.assign(__src="te"),
    ], axis=0, ignore_index=True)

    g = full.groupby(["Race", "Year", "LapNumber"], sort=False)
    full["g2_block_tyrelife_std"] = g["TyreLife"].transform("std").fillna(0.0)
    full["g2_block_tyrelife_mean"] = g["TyreLife"].transform("mean")
    full["g2_block_pos_std"] = g["Position"].transform("std").fillna(0.0)
    full["g2_block_n"] = g["LapNumber"].transform("size").astype(np.int32)
    full["g2_tyrelife_minus_block"] = full["TyreLife"] - full["g2_block_tyrelife_mean"]
    full["g2_tyrelife_rank_in_block"] = (
        g["TyreLife"].rank(method="average", pct=True))

    # compound fractions per block
    for comp in ["HARD", "MEDIUM", "SOFT"]:
        flag = (full["Compound"] == comp).astype(np.float32)
        full[f"g2_block_{comp.lower()}_frac"] = (
            flag.groupby([full["Race"], full["Year"], full["LapNumber"]])
                .transform("mean"))

    tr_out = full[full["__src"] == "tr"].drop(columns="__src").reset_index(drop=True)
    te_out = full[full["__src"] == "te"].drop(columns="__src").reset_index(drop=True)
    return tr_out, te_out


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    print(f"=== G2' cross-driver intra-race FE ===  smoke={smoke}")
    train, test, sub, y = load_data()
    primary_oof, primary_test = load_primary()

    print("Building γ4 features (8 block-level features) on train+test ...")
    train, test = add_cross_driver_features(train, test)

    drop_cols = ["id", "PitNextLap"]
    cat_cols = ["Driver", "Compound", "Race"]
    feat_cols = [c for c in train.columns if c not in drop_cols]
    Xtr = train[feat_cols].copy()
    Xte = test[feat_cols].copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    g2_added = [c for c in feat_cols if c.startswith("g2_")]
    print(f"feature_count={len(feat_cols)} (was 14, +{len(g2_added)} γ4); "
          f"added: {g2_added}")

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
