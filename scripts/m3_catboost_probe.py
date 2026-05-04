"""M3 CatBoost 1-fold time-probe — SHRUNK config.

Re-probe after 2026-05-04 PROBE-FAIL (96.4 min projected w/ depth=8,
iters=2000). Shrunk: depth=6, iters=800, lr=0.08, od_wait=50.
Expected wall ~150-250s/fold; 10 fits ≈ 25-40 min — under 60-min cap.

HARD STOP: if projection >= 3600s, write PROBE-FAIL-2 audit and stop.
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


def shrunk_params() -> dict:
    return dict(
        iterations=800,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=50,
        verbose=0,
        thread_count=-1,
        allow_writing_files=False,
    )


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
    model = CatBoostClassifier(**shrunk_params())
    model.fit(pool_tr, eval_set=pool_va)
    fit_t = time.time() - t1

    t2 = time.time()
    p_va = model.predict_proba(pool_va)[:, 1]
    p_te = model.predict_proba(pool_test)[:, 1]
    pred_t = time.time() - t2

    auc = float(roc_auc_score(y[va], p_va))
    fold_t = fit_t + pred_t
    proj_5fold = fold_t * 10  # 10 fits = 5 strat + 5 group
    best_iter = model.get_best_iteration()
    es_fired = best_iter < 800 - 1
    print(f"fold AUC = {auc:.5f}  best_iter={best_iter}  ES fired={es_fired}")
    print(f"fit={fit_t:.1f}s  predict={pred_t:.1f}s  fold_total={fold_t:.1f}s")
    print(f"projection 5-fold both-anchor (10 fits): {proj_5fold:.0f}s "
          f"= {proj_5fold/60:.1f} min")

    if proj_5fold >= 3600:
        out = Path(f"audit/{dt.date.today().isoformat()}-m3-catboost-PROBE-FAIL-2.md")
        out.write_text(
            f"# M3 CatBoost — PROBE FAIL 2 (shrunk) ({dt.date.today()})\n\n"
            f"Shrunk config (depth=6, iters=800, lr=0.08, od_wait=50) STILL fails:\n"
            f"Single-fold full-data wall: {fold_t:.0f}s "
            f"(fit {fit_t:.0f}s + predict {pred_t:.0f}s)\n"
            f"Projected 5-fold both-anchor (10 fits): "
            f"{proj_5fold:.0f}s = {proj_5fold/60:.1f} min\n\n"
            f"Threshold: >= 3600s. STOPPED. "
            f"best_iter={best_iter}, fold AUC={auc:.5f}.\n"
        )
        print(f"PROBE FAIL 2 → {out}")
        return

    # signal-survival check
    baseline_auc = 0.94075
    threshold = baseline_auc + 0.0030
    if auc < threshold:
        print(f"SIGNAL DESTROYED — fold AUC {auc:.5f} < threshold {threshold:.5f} "
              f"(baseline + 30bp). Do NOT proceed to full 5-fold.")
        return

    print(f"PROBE OK — fold AUC {auc:.5f} >= {threshold:.5f}; proceed to full.")


if __name__ == "__main__":
    main()
