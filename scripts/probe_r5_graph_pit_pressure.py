"""scripts/probe_r5_graph_pit_pressure.py — Round 5 Phase C

Graph-class per-(Race, Lap) pit-pressure features. The Round-4
plateau break (mechanism-orthogonal stacking) suggests cross-class
combinations work. Phase C adds graph-class as a third orthogonal
axis on top of row-class FE (r4_segment_fe) and sequence-class HMM
(r4_hmm_seq).

The premise: at lap t in race R, the SOCIAL CONTEXT — which other
drivers in the same race are pitting at the same / nearby laps —
is predictive of this driver's pit decision at lap t+1. This is
information NOT in the row's own 14 features.

Features added (4):
- pit_pressure_lap: training-only fraction of OTHER drivers at
  same (Y, R, L) who pit next lap. Captures "everyone is pitting
  this lap → I might too."
- pit_pressure_lag3: fraction of training rows at (Y, R, L-1, L-2,
  L-3) who pit. Captures "pit window started 1-3 laps ago."
- compound_pit_pressure: training-only fraction of OTHER drivers
  in same (Y, R, Compound) at same L who pit. Captures compound-
  specific pit timing.
- race_pit_rate: per-(Y, R) overall pit rate (training-only).
  Captures race-level baseline.

Fold-safe per Rule 24: each fold's training partition is used to
compute the aggregate; val + test rows look up the value.

Smoke: --smoke runs 1 fold, 50k rows.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)

LGB_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    lambda_l1=0.0, lambda_l2=1.0, max_depth=-1, n_jobs=-1,
    verbose=-1, random_state=SEED,
)

NEW_COLS = ["pit_pressure_lap", "pit_pressure_lag3",
            "compound_pit_pressure", "race_pit_rate"]


def compute_pit_pressure_features(train_part: pd.DataFrame,
                                   target_df: pd.DataFrame,
                                   global_pit_rate: float) -> pd.DataFrame:
    """Compute 4 pit-pressure features for rows in target_df, aggregating
    over train_part only. Vectorized via pandas merge for speed.

    Args:
        train_part: training partition with PitNextLap column.
        target_df: rows to compute features for (val or test).
        global_pit_rate: fallback when keys are missing.
    """
    target = target_df.reset_index(drop=True).copy()

    # 1. per-(Year, Race, LapNumber) pit rate
    g_lap = (train_part.groupby(["Year", "Race", "LapNumber"])[TARGET]
             .mean().reset_index().rename(columns={TARGET: "pit_pressure_lap"}))
    target = target.merge(g_lap, on=["Year", "Race", "LapNumber"], how="left")

    # 2. per-(Year, Race) overall pit rate
    g_race = (train_part.groupby(["Year", "Race"])[TARGET].mean()
              .reset_index().rename(columns={TARGET: "race_pit_rate"}))
    target = target.merge(g_race, on=["Year", "Race"], how="left")

    # 3. per-(Year, Race, Compound, LapNumber)
    g_comp = (train_part.groupby(["Year", "Race", "Compound", "LapNumber"])[TARGET]
              .mean().reset_index()
              .rename(columns={TARGET: "compound_pit_pressure"}))
    target = target.merge(g_comp,
                          on=["Year", "Race", "Compound", "LapNumber"],
                          how="left")

    # 4. lagged pit pressure: mean of pit_pressure at (Y, R, L-1), (L-2), (L-3)
    # Build via 3 separate joins.
    g_lap_renamed = g_lap.rename(columns={"LapNumber": "_lap_orig"})
    lag_means = np.zeros(len(target), dtype=np.float64)
    lag_counts = np.zeros(len(target), dtype=np.int32)
    for dl in (1, 2, 3):
        join_key = target[["Year", "Race", "LapNumber"]].copy()
        join_key["_lap_orig"] = join_key["LapNumber"] - dl
        m = join_key.merge(g_lap_renamed, on=["Year", "Race", "_lap_orig"], how="left")
        vals = m["pit_pressure_lap"].values
        good = ~np.isnan(vals)
        lag_means[good] += vals[good]
        lag_counts[good] += 1
    pit_pressure_lag3 = np.where(lag_counts > 0,
                                  lag_means / np.maximum(lag_counts, 1),
                                  global_pit_rate).astype(np.float32)

    # Fill NaNs with sensible fallbacks
    target["pit_pressure_lap"] = target["pit_pressure_lap"].fillna(global_pit_rate).astype(np.float32)
    target["race_pit_rate"]    = target["race_pit_rate"].fillna(global_pit_rate).astype(np.float32)
    # compound_pit_pressure falls back to lap-level, then global
    target["compound_pit_pressure"] = target["compound_pit_pressure"].fillna(
        target["pit_pressure_lap"]).fillna(global_pit_rate).astype(np.float32)
    target["pit_pressure_lag3"] = pit_pressure_lag3

    return target[NEW_COLS].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="r5_graph_pit_pressure")
    ap.add_argument("--max-rounds", type=int, default=2000)
    args = ap.parse_args()

    print(f"=== R5 Phase C: graph-class pit-pressure | smoke={args.smoke} ===")
    t0 = time.time()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}")

    if args.smoke:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train), size=50_000, replace=False)
        train = train.iloc[np.sort(idx)].reset_index(drop=True)
        y_all = train[TARGET].astype(int).values
        args.max_rounds = min(args.max_rounds, 500)
        print(f"  SMOKE -> train {train.shape}")

    global_pit_rate = float(y_all.mean())
    print(f"  global pit rate: {global_pit_rate:.4f}")

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_iter = list(skf.split(np.zeros(len(y_all)), y_all))
    if args.smoke:
        fold_iter = fold_iter[:1]

    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []

    cat_cols = ["Driver", "Compound", "Race"]

    for fold, (ti, vi) in enumerate(fold_iter, 1):
        t_fold = time.time()
        print(f"\n  --- Fold {fold}: ti={len(ti):,} va={len(vi):,} ---")

        train_ti = train.iloc[ti].copy()
        train_va = train.iloc[vi].copy()
        test_fold = test.copy()

        # Compute graph features per fold
        t_g = time.time()
        feats_ti = compute_pit_pressure_features(train_ti, train_ti, global_pit_rate)
        feats_va = compute_pit_pressure_features(train_ti, train_va, global_pit_rate)
        feats_te = compute_pit_pressure_features(train_ti, test_fold, global_pit_rate)
        train_ti = pd.concat([train_ti.reset_index(drop=True),
                              feats_ti.reset_index(drop=True)], axis=1)
        train_va = pd.concat([train_va.reset_index(drop=True),
                              feats_va.reset_index(drop=True)], axis=1)
        test_fold = pd.concat([test_fold.reset_index(drop=True),
                               feats_te.reset_index(drop=True)], axis=1)
        print(f"    graph features built: {time.time()-t_g:.1f}s")

        # Label-encode categoricals (fold-local, consistent across train/val/test)
        for c in cat_cols:
            u = pd.concat([train_ti[c], train_va[c], test_fold[c]],
                          ignore_index=True).unique()
            m = {v: i for i, v in enumerate(u)}
            train_ti[c] = train_ti[c].map(m).astype(np.int32)
            train_va[c] = train_va[c].map(m).astype(np.int32)
            test_fold[c] = test_fold[c].map(m).astype(np.int32)

        feat_cols = [c for c in train_ti.columns if c not in {"id", TARGET}]
        X_tr = train_ti[feat_cols].fillna(0).values
        X_va = train_va[feat_cols].fillna(0).values
        X_te = test_fold[feat_cols].fillna(0).values
        y_tr = train_ti[TARGET].astype(int).values
        y_va = train_va[TARGET].astype(int).values

        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=args.max_rounds)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

        oof_va = m.predict_proba(X_va)[:, 1]
        oof[vi] = oof_va
        if not args.smoke:
            test_pred += m.predict_proba(X_te)[:, 1] / N_FOLDS

        auc_va = roc_auc_score(y_va, oof_va)
        fold_aucs.append(float(auc_va))
        print(f"    Fold {fold}: AUC={auc_va:.5f} iters={m.best_iteration_} "
              f"wall={time.time()-t_fold:.1f}s")

    auc_full = (fold_aucs[0] if args.smoke
                else float(roc_auc_score(y_all, oof)))
    print(f"\n  OOF AUC{'  (smoke fold-1)' if args.smoke else ' (full)'}: "
          f"{auc_full:.5f}  fold-std={np.std(fold_aucs):.5f}  "
          f"total wall={time.time()-t0:.1f}s")

    if not args.smoke:
        np.save(ART / f"oof_{args.name}_strat.npy",
                np.column_stack([1 - oof, oof]).astype(np.float64))
        np.save(ART / f"test_{args.name}_strat.npy",
                np.column_stack([1 - test_pred, test_pred]).astype(np.float64))
        (ART / f"{args.name}_results.json").write_text(json.dumps(dict(
            name=args.name, oof_auc=auc_full, fold_aucs=fold_aucs,
            n_new_cols=len(NEW_COLS), new_cols=NEW_COLS,
        ), indent=2))
        print(f"\n  -> oof_{args.name}_strat.npy   test_{args.name}_strat.npy")


if __name__ == "__main__":
    main()
