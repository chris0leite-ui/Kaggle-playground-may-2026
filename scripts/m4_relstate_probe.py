"""M4 1-fold time-probe — full data, full hyperparams. Project 5-fold both-anchor cost."""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import SEED
from m4_relstate_smoke import LAPTIME_COL, add_relstate_features, make_lgb_params

TARGET = "PitNextLap"
ID_COL = "id"


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"loaded: train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    # Concat for FE so we can compute deltas using all available laps; keep separation
    train["__src"] = 0
    test["__src"] = 1
    test[TARGET] = -1  # sentinel
    full = pd.concat([train, test], axis=0, ignore_index=True)
    full_id_order = full[ID_COL].values.copy()
    fe_full, added, skipped = add_relstate_features(full)
    fe_full = fe_full.set_index(ID_COL).loc[full_id_order].reset_index()

    train_fe = fe_full[fe_full["__src"] == 0].drop(columns=["__src"]).reset_index(drop=True)
    test_fe = fe_full[fe_full["__src"] == 1].drop(columns=["__src", TARGET]).reset_index(drop=True)

    # Restore original train/test row order
    train_fe = train_fe.set_index(ID_COL).loc[train[ID_COL].values].reset_index()
    test_fe = test_fe.set_index(ID_COL).loc[test[ID_COL].values].reset_index()

    print(f"FE done. added={added} skipped={skipped}  t={time.time()-t0:.1f}s")

    y = train_fe[TARGET].astype(int).values
    X = train_fe.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test_fe.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    print(f"features: {list(X.columns)}")
    print(f"NaN train: {X.isna().sum().sum()}  NaN test: {X_test.isna().sum().sum()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.zeros(len(y)), y)))

    t1 = time.time()
    dtrain = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
    dval = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
    model = lgb.train(make_lgb_params(), dtrain, num_boost_round=2000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p_va = model.predict(X.iloc[va])
    _ = model.predict(X_test)  # include test predict cost
    fold_secs = time.time() - t1
    auc = roc_auc_score(y[va], p_va)
    proj = fold_secs * 10  # 5-fold * 2 anchors
    print(f"probe fold AUC={auc:.5f}  best_iter={model.best_iteration}")
    print(f"fold wall: {fold_secs:.1f}s")
    print(f"projection 5-fold both-anchor (10 fits): {proj:.0f}s = {proj/60:.1f}min")
    print(f"total probe wall: {time.time()-t0:.1f}s")

    if proj >= 3600:
        msg = (f"# M4 relstate PROBE FAIL\n\n"
               f"projected 5-fold both-anchor wall: {proj:.0f}s = {proj/60:.1f}min "
               f"(threshold = 3600s).\nfold_secs={fold_secs:.1f}, "
               f"best_iter={model.best_iteration}, fold AUC={auc:.5f}.\n"
               f"FE features: {added} (skipped: {skipped})\n")
        Path("audit/2026-05-04-m4-relstate-PROBE-FAIL.md").write_text(msg)
        print("HARD STOP — projection ≥1h, see audit file.")
        return False
    return True


if __name__ == "__main__":
    main()
