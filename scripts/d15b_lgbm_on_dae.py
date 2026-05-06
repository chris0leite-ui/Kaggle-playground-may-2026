"""Stage-2B: 5-fold StratifiedKFold LGBM with DAE latent features.

Variants:
    (a) full = raw features (numerics + label-encoded cats) + 768d latent
        -> oof_d15b_lgbm_dae_full_strat.npy
        -> test_d15b_lgbm_dae_full_strat.npy
    (b) only = 768d latent only
        -> oof_d15b_lgbm_dae_only_strat.npy
        -> test_d15b_lgbm_dae_only_strat.npy

LGBM hparams: num_leaves=63, lr=0.05, min_child_samples=200,
n_estimators=2000, early_stopping_rounds=100 on val fold.

Saves 2-column [P0, P1] arrays.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
HIGH_CARD = ["Driver", "Race"]
LOW_CARD = ["Compound", "Year"]
ALL_CAT = HIGH_CARD + LOW_CARD


def build_raw(train, test):
    raw_train = train[NUMERICS].astype(np.float32).copy()
    raw_test = test[NUMERICS].astype(np.float32).copy()
    for c in ALL_CAT:
        all_vals = pd.concat([train[c], test[c]], ignore_index=True
                             ).astype(str).unique()
        mp = {v: i for i, v in enumerate(sorted(all_vals))}
        raw_train[c] = train[c].astype(str).map(mp).astype(np.int32).values
        raw_test[c] = test[c].astype(str).map(mp).astype(np.int32).values
    return raw_train, raw_test


def cv_lgbm(X_train, X_test, y, name, cat_pos):
    print(f"\n=== CV LGBM: {name}  X_train {X_train.shape}  cat_pos={cat_pos} ===",
          flush=True)
    params = dict(
        objective="binary", metric="auc",
        num_leaves=63, learning_rate=0.05,
        min_child_samples=200,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
        verbose=-1, seed=SEED,
        num_threads=4,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(X_test.shape[0], dtype=np.float64)
    fold_aucs, walls, iters = [], [], []
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        lgb_train = lgb.Dataset(X_train[tr], y[tr],
                                categorical_feature=cat_pos,
                                free_raw_data=False)
        lgb_val = lgb.Dataset(X_train[va], y[va],
                              categorical_feature=cat_pos,
                              reference=lgb_train, free_raw_data=False)
        bst = lgb.train(
            params, lgb_train, num_boost_round=2000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        pv = bst.predict(X_train[va], num_iteration=bst.best_iteration)
        pt = bst.predict(X_test, num_iteration=bst.best_iteration)
        oof[va] = pv
        test_pred += pt / N_FOLDS
        s = float(roc_auc_score(y[va], pv))
        fold_aucs.append(s)
        iters.append(int(bst.best_iteration))
        walls.append(time.time() - t0)
        print(f"  f{k}: AUC={s:.5f}  iters={bst.best_iteration}  "
              f"wall={time.time()-t0:.1f}s", flush=True)
    auc = float(roc_auc_score(y, oof))
    print(f"  OOF AUC: {auc:.5f}  std={np.std(fold_aucs):.5f}  "
          f"sum_wall={sum(walls):.1f}s", flush=True)

    np.save(ART / f"oof_{name}_strat.npy", np.column_stack([1 - oof, oof]))
    np.save(ART / f"test_{name}_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))
    return dict(oof_auc=auc, fold_aucs=fold_aucs, iters=iters,
                walls=walls, sum_wall=sum(walls))


def main():
    t_total = time.time()
    print("Loading data + latent...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    latent_train = np.load(ART / "d15b_dae_X_train_latent.npy")
    latent_test = np.load(ART / "d15b_dae_X_test_latent.npy")
    print(f"  Train {train.shape}  Test {test.shape}", flush=True)
    print(f"  Latent train {latent_train.shape}  test {latent_test.shape}", flush=True)

    raw_train, raw_test = build_raw(train, test)
    raw_train_arr = raw_train.values.astype(np.float32)
    raw_test_arr = raw_test.values.astype(np.float32)
    cat_pos_full = [list(raw_train.columns).index(c) for c in ALL_CAT]

    Xfull_train = np.hstack([raw_train_arr, latent_train])
    Xfull_test = np.hstack([raw_test_arr, latent_test])

    results = {}

    # Variant (a): raw + latent
    results["d15b_lgbm_dae_full"] = cv_lgbm(
        Xfull_train, Xfull_test, y, "d15b_lgbm_dae_full",
        cat_pos=cat_pos_full,
    )

    # Variant (b): latent only -- no categoricals
    results["d15b_lgbm_dae_only"] = cv_lgbm(
        latent_train.astype(np.float32), latent_test.astype(np.float32), y,
        "d15b_lgbm_dae_only", cat_pos=[],
    )

    print(f"\n=== TOTAL WALL: {time.time()-t_total:.1f}s ===", flush=True)
    (ART / "d15b_lgbm_results.json").write_text(json.dumps(results, indent=2))
    print(f"  -> {ART}/d15b_lgbm_results.json", flush=True)


if __name__ == "__main__":
    main()
