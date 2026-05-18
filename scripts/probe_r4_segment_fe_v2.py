"""scripts/probe_r4_segment_fe_v2.py — Round 4 Phase A variant 2

Phase A v1 (`probe_r4_segment_fe.py`) produced +0.263 bp at K=4+1
(strongest in 15+ rounds but G2-fails 0.30 threshold). Per-segment
AUCs on WET+S1 went DOWN (0.81 → 0.77) — the WET features added
noise without signal.

v2 drops the WET features and adds TyreLife × Compound + Position
× Stint, exposing the tire-degradation interaction the failure-
analysis agent flagged but v1 didn't capture explicitly.

Feature delta vs v1:
- DROP: cumdeg_wet, is_wet_s1 (degenerate WET segment).
- KEEP: cumdeg_inter, cumdeg_hard, cumdeg_medium, cumdeg_soft,
       poschg_named, is_named, is_inter_s2.
- ADD:  tyrelife_inter, tyrelife_hard, tyrelife_medium, tyrelife_soft
       (tire-life × compound interactions; degradation grows with
       lap count, so TyreLife is the structural correlate).
       poschg_s2, poschg_s3 (position-change × mid-stint context).

Same harness as v1 (5-fold Stratified, standard LGBM_PARAMS).
"""
from __future__ import annotations
import argparse
import json
import re
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

NEW_COLS = [
    "cumdeg_inter", "cumdeg_hard", "cumdeg_medium", "cumdeg_soft",
    "poschg_named", "is_named", "is_inter_s2",
    "tyrelife_inter", "tyrelife_hard", "tyrelife_medium", "tyrelife_soft",
    "poschg_s2", "poschg_s3",
]


def _is_named(driver_series: pd.Series) -> pd.Series:
    return ~driver_series.astype(str).str.match(r"^D\d{3}$").fillna(False)


def add_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_inter = (df["Compound"] == "INTERMEDIATE").astype(np.float32)
    is_hard  = (df["Compound"] == "HARD").astype(np.float32)
    is_med   = (df["Compound"] == "MEDIUM").astype(np.float32)
    is_soft  = (df["Compound"] == "SOFT").astype(np.float32)
    named    = _is_named(df["Driver"]).astype(np.float32)
    stint2   = (df["Stint"] == 2).astype(np.float32)
    stint3   = (df["Stint"] == 3).astype(np.float32)

    df["cumdeg_inter"]   = df["Cumulative_Degradation"] * is_inter
    df["cumdeg_hard"]    = df["Cumulative_Degradation"] * is_hard
    df["cumdeg_medium"]  = df["Cumulative_Degradation"] * is_med
    df["cumdeg_soft"]    = df["Cumulative_Degradation"] * is_soft
    df["poschg_named"]   = df["Position_Change"] * named
    df["is_named"]       = named
    df["is_inter_s2"]    = is_inter * stint2
    df["tyrelife_inter"] = df["TyreLife"] * is_inter
    df["tyrelife_hard"]  = df["TyreLife"] * is_hard
    df["tyrelife_medium"] = df["TyreLife"] * is_med
    df["tyrelife_soft"]  = df["TyreLife"] * is_soft
    df["poschg_s2"]      = df["Position_Change"] * stint2
    df["poschg_s3"]      = df["Position_Change"] * stint3
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="r4_segment_fe_v2")
    ap.add_argument("--max-rounds", type=int, default=2000)
    args = ap.parse_args()

    print(f"=== R4 Phase A v2 segment FE | smoke={args.smoke} ===")
    t0 = time.time()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}")

    train = add_segment_features(train)
    test = add_segment_features(test)
    print(f"  added {len(NEW_COLS)} interaction features")

    if args.smoke:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train), size=50_000, replace=False)
        train = train.iloc[np.sort(idx)].reset_index(drop=True)
        y_all = train[TARGET].astype(int).values
        args.max_rounds = min(args.max_rounds, 500)
        print(f"  SMOKE -> train {train.shape}")

    compound_raw = train["Compound"].values.copy()
    stint_raw    = train["Stint"].values.copy()
    named_raw    = train["is_named"].values.astype(bool).copy()

    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        u = pd.concat([train[c], test[c]], ignore_index=True).unique()
        m = {v: i for i, v in enumerate(u)}
        train[c] = train[c].map(m).astype(np.int32)
        test[c] = test[c].map(m).astype(np.int32)

    feat_cols = [c for c in train.columns if c not in {"id", TARGET}]
    print(f"  total features: {len(feat_cols)} "
          f"({len(feat_cols) - len(NEW_COLS)} raw + {len(NEW_COLS)} new)")

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_iter = list(skf.split(np.zeros(len(y_all)), y_all))
    if args.smoke:
        fold_iter = fold_iter[:1]

    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []

    for fold, (ti, vi) in enumerate(fold_iter, 1):
        t_fold = time.time()
        X_tr = train.iloc[ti][feat_cols].fillna(0).values
        X_va = train.iloc[vi][feat_cols].fillna(0).values
        X_te = test[feat_cols].fillna(0).values

        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=args.max_rounds)
        m.fit(X_tr, y_all[ti],
              eval_set=[(X_va, y_all[vi])],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

        oof_va = m.predict_proba(X_va)[:, 1]
        oof[vi] = oof_va
        if not args.smoke:
            test_pred += m.predict_proba(X_te)[:, 1] / N_FOLDS

        auc_va = roc_auc_score(y_all[vi], oof_va)
        fold_aucs.append(float(auc_va))
        print(f"  Fold {fold}: AUC={auc_va:.5f} iters={m.best_iteration_} "
              f"wall={time.time()-t_fold:.1f}s")

    auc_full = (fold_aucs[0] if args.smoke
                else float(roc_auc_score(y_all, oof)))
    print(f"\n  OOF AUC{'  (smoke fold-1)' if args.smoke else ' (full)'}: "
          f"{auc_full:.5f}  fold-std={np.std(fold_aucs):.5f}  "
          f"total wall={time.time()-t0:.1f}s")

    if not args.smoke:
        masks = {
            "INTER + Stint==2": (compound_raw == "INTERMEDIATE") & (stint_raw == 2),
            "INTER + Stint==3": (compound_raw == "INTERMEDIATE") & (stint_raw == 3),
            "named-driver":     named_raw,
            "anonymous D0XX":   ~named_raw,
        }
        print("\n  Per-segment OOF AUC (R4 v2):")
        for name, mk in masks.items():
            n = int(mk.sum())
            p = int(y_all[mk].sum())
            if n < 50 or p < 5 or p > n - 5:
                print(f"    {name:<22s}  n={n:>6d}  pos={p:>5d}  (skip)")
                continue
            a = roc_auc_score(y_all[mk], oof[mk])
            print(f"    {name:<22s}  n={n:>6d}  pos={p:>5d}  AUC={a:.4f}")

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
