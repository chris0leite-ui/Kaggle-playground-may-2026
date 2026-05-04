"""M3 CatBoost 1-fold time-probe — full data, full hyperparams.

Measures wall time of one StratKFold fold fit + predict on the full
439140-row train + full 188165-row test. Projects 5-fold both-anchor
wall (10 fits + 10 test predicts).

HARD STOP: if projection >= 1h, write PROBE-FAIL audit and stop.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import SEED

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"loaded: train={train.shape} test={test.shape} "
          f"in {time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.zeros(len(y)), y)))

    t1 = time.time()
    pool_tr = Pool(X.iloc[tr], y[tr], cat_features=CAT_COLS)
    pool_va = Pool(X.iloc[va], y[va], cat_features=CAT_COLS)
    pool_test = Pool(X_test, cat_features=CAT_COLS)
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=100,
        verbose=0,
        thread_count=-1,
        allow_writing_files=False,
    )
    model.fit(pool_tr, eval_set=pool_va)
    fit_t = time.time() - t1

    t2 = time.time()
    p_va = model.predict_proba(pool_va)[:, 1]
    p_te = model.predict_proba(pool_test)[:, 1]
    pred_t = time.time() - t2

    auc = float(roc_auc_score(y[va], p_va))
    fold_t = fit_t + pred_t
    proj_5fold = fold_t * 10  # 10 fits = 5 strat + 5 group
    print(f"fold AUC = {auc:.5f}  best_iter={model.get_best_iteration()}")
    print(f"fit={fit_t:.1f}s  predict={pred_t:.1f}s  fold_total={fold_t:.1f}s")
    print(f"projection 5-fold both-anchor (10 fits): {proj_5fold:.0f}s "
          f"= {proj_5fold/60:.1f} min")

    if proj_5fold >= 3600:
        out = Path(f"audit/{dt.date.today().isoformat()}-m3-catboost-PROBE-FAIL.md")
        out.write_text(
            f"# M3 CatBoost — PROBE FAIL ({dt.date.today()})\n\n"
            f"Single-fold full-data wall: {fold_t:.0f}s "
            f"(fit {fit_t:.0f}s + predict {pred_t:.0f}s)\n"
            f"Projected 5-fold both-anchor (10 fits): "
            f"{proj_5fold:.0f}s = {proj_5fold/60:.1f} min\n\n"
            f"Threshold: >= 3600s. STOPPED. "
            f"best_iter={model.get_best_iteration()}, fold AUC={auc:.5f}.\n"
        )
        print(f"PROBE FAIL → {out}")
        return

    print("PROBE OK — proceed to full 5-fold both-anchor.")


if __name__ == "__main__":
    main()
