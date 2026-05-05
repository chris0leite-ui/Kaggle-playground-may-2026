"""M2 XGB 1-fold full-data time probe.

Measures wall time of one full-data StratKFold fold with target hyperparams.
Projects 5-fold both-anchor (10 fits) wall.
HARD STOP: if projection >= 1h (3600s), do not proceed to full run.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import SEED

TARGET = "PitNextLap"
ID_COL = "id"


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    print(f"loaded train {len(train)} rows; cats={cat_cols}; "
          f"load_secs={time.time()-t0:.1f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.zeros(len(y)), y)))

    t1 = time.time()
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=20,
        n_estimators=2000,
        early_stopping_rounds=100,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    p = model.predict_proba(X.iloc[va])[:, 1]
    auc = float(roc_auc_score(y[va], p))
    fit_secs = time.time() - t1
    proj = fit_secs * 10  # 5 folds * 2 anchors
    print(f"probe fold0 AUC={auc:.5f} best_iter={model.best_iteration} "
          f"fit_secs={fit_secs:.1f}")
    print(f"projection 5-fold both-anchor (10 fits) = {proj:.0f}s "
          f"= {proj/60:.1f}min")
    if proj >= 3600:
        print(f"HARD STOP — projection >= 1h. Writing PROBE-FAIL audit.")
        from pathlib import Path
        Path("audit/2026-05-04-m2-xgb-PROBE-FAIL.md").write_text(
            f"# M2 XGB PROBE FAIL — 2026-05-04\n\n"
            f"Single full-data StratKFold fold fit: {fit_secs:.0f}s\n"
            f"Projected 5-fold both-anchor (10 fits) wall: {proj:.0f}s = {proj/60:.1f}min\n"
            f"Threshold: 3600s (1h). Aborting full 5-fold run.\n"
            f"Best iter: {model.best_iteration}, fold0 AUC: {auc:.5f}\n"
        )
        raise SystemExit(2)
    print(f"PASS — projection < 1h, OK to proceed to full run.")


if __name__ == "__main__":
    main()
