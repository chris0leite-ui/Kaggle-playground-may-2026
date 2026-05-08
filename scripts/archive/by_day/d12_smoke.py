"""Smoke for d12 LambdaRank meta — 1 fold each variant, ~50k row subsample.

Validates: groups build correctly, lambdarank trains, yetirank trains.
"""
from __future__ import annotations
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from d12_lambdarank_meta import (
    POOL_KEEP, TOP_3_D9, PARTITION_FMS, expand, load_pool, build_groups,
)

ART = Path("scripts/artifacts")
SEED = 42


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train["PitNextLap"].astype(int).values
    race_ids = pd.Categorical(train["Race"]).codes.astype(np.int32)

    base_oof, base_test, base_names = load_pool(POOL_KEEP)
    d9_oof, d9_test, _ = load_pool(TOP_3_D9)
    fm_oof, fm_test, _ = load_pool(PARTITION_FMS)
    Xs_oof = base_oof + d9_oof + fm_oof
    Xs_test = base_test + d9_test + fm_test

    P_oof = np.column_stack(Xs_oof)
    P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof)
    F_test = expand(P_test)
    print(f"K={len(Xs_oof)}  F_oof={F_oof.shape}")

    # Use 1 fold split (80/20 stratified by y)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    pos = idx[y == 1]; neg = idx[y == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos_va = int(0.2 * len(pos)); n_neg_va = int(0.2 * len(neg))
    va = np.concatenate([pos[:n_pos_va], neg[:n_neg_va]])
    tr = np.concatenate([pos[n_pos_va:], neg[n_neg_va:]])
    rng.shuffle(va); rng.shuffle(tr)

    print(f"split: tr={len(tr)} va={len(va)}")

    # -- LR meta smoke --
    t = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(F_oof[tr], y[tr])
    p_va_lr = lr.predict_proba(F_oof[va])[:, 1]
    auc_lr = roc_auc_score(y[va], p_va_lr)
    print(f"[lr] val AUC {auc_lr:.5f}  wall {time.time()-t:.1f}s")

    # -- LambdaMART smoke --
    t = time.time()
    tr_counts, tr_perm = build_groups(race_ids, tr)
    va_counts, va_perm = build_groups(race_ids, va)
    print(f"  tr groups={len(tr_counts)} va groups={len(va_counts)}")
    Xtr = F_oof[tr_perm]; ytr = y[tr_perm]
    Xva = F_oof[va_perm]; yva = y[va_perm]
    dtr = lgb.Dataset(Xtr, label=ytr, group=tr_counts)
    dva = lgb.Dataset(Xva, label=yva, group=va_counts, reference=dtr)
    booster = lgb.train(
        dict(objective="lambdarank", metric=["auc"], eval_at=[100],
             learning_rate=0.1, num_leaves=15, min_data_in_leaf=200,
             lambda_l2=1.0, verbose=-1, seed=SEED),
        dtr, num_boost_round=500,
        valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False,
                                                       first_metric_only=True),
                                     lgb.log_evaluation(50)],
    )
    p_va_lm = booster.predict(Xva, num_iteration=booster.best_iteration)
    auc_lm = roc_auc_score(yva, p_va_lm)
    print(f"[lambdamart] val AUC {auc_lm:.5f}  iters={booster.best_iteration}"
          f"  wall {time.time()-t:.1f}s")

    # -- YetiRank smoke --
    t = time.time()
    from catboost import CatBoost, Pool
    F_oof32 = F_oof.astype(np.float32)
    # CatBoost requires queries to be contiguous; reuse build_groups to sort.
    _, tr_perm_yt = build_groups(race_ids, tr)
    _, va_perm_yt = build_groups(race_ids, va)
    tr_pool = Pool(F_oof32[tr_perm_yt], y[tr_perm_yt],
                   group_id=race_ids[tr_perm_yt])
    va_pool = Pool(F_oof32[va_perm_yt], y[va_perm_yt],
                   group_id=race_ids[va_perm_yt])
    m = CatBoost({
        "loss_function": "YetiRank",
        "iterations": 200,
        "learning_rate": 0.1,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "od_type": "Iter",
        "od_wait": 20,
        "random_seed": SEED,
        "verbose": False,
        "thread_count": -1,
    })
    m.fit(tr_pool, eval_set=va_pool, verbose=False)
    p_va_yt = m.predict(F_oof32[va])
    auc_yt = roc_auc_score(y[va], p_va_yt)
    print(f"[yetirank] val AUC {auc_yt:.5f}  iters={m.tree_count_}"
          f"  wall {time.time()-t:.1f}s")

    print(f"\nTotal smoke wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
