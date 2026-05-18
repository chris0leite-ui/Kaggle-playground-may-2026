"""scripts/probe_c4_uid_magic.py — C4 candidate

UID magic-features as a single LightGBM base. IEEE-CIS Fraud 1st-place
pattern (Deotte/NVIDIA): UID = group1 + '_' + group2 + '_' +
floor(time / W), then 20-40 per-UID aggregates of continuous columns.

Distinct from:
- EXP-A3-7 UID smoothing on TARGET → -124 bp FAIL (target-leak).
- EXP-A2-3 nested-fold TE on bigram/trigram → different mechanism (encodes
  per-group target mean, not raw feature aggregates).

Implemented as a single LightGBM trained on (raw 14 features) + UID
features, 5-fold OOF Stratified, fold-safe by Rule 24: UID-aggregates
are computed on each fold's training partition only.

Origin: audit/research/2026-05-18-research.md C4.

Usage:
  python scripts/probe_c4_uid_magic.py [--smoke]
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

# UID definitions: (name, key columns, time bucket col, time-bucket width)
# Each row gets UID = key1__key2__floor(time / W). One UID family per W.
UIDS = [
    ("UID_DR_lap5",  ["Driver", "Race"],  "LapNumber", 5),
    ("UID_DR_lap10", ["Driver", "Race"],  "LapNumber", 10),
    ("UID_DR_lap20", ["Driver", "Race"],  "LapNumber", 20),
    ("UID_DRY",      ["Driver", "Race", "Year"], None,  None),
    ("UID_DCS",      ["Driver", "Compound", "Stint"],  None,  None),
]

# Aggregate columns. For each UID, we compute mean/std/min/max/count of these.
AGG_COLS = ["LapTime (s)", "TyreLife", "Position", "Cumulative_Degradation",
            "LapTime_Delta", "Position_Change", "RaceProgress"]
AGG_FUNCS = ["mean", "std", "min", "max"]


def build_uid_features(train_fold_df, val_df, test_df):
    """For each UID, compute training-only groupby aggregates and merge onto
    train/val/test. Returns three DataFrames augmented with new columns.
    """
    train_aug = train_fold_df.copy()
    val_aug = val_df.copy()
    test_aug = test_df.copy()

    new_cols = []
    for uid_name, key_cols, time_col, time_W in UIDS:
        # Construct UID strings
        def make_uid(df):
            if time_col is None:
                s = df[key_cols[0]].astype(str)
                for k in key_cols[1:]:
                    s = s + "__" + df[k].astype(str)
            else:
                t = (df[time_col].astype(int) // time_W).astype(int)
                s = df[key_cols[0]].astype(str)
                for k in key_cols[1:]:
                    s = s + "__" + df[k].astype(str)
                s = s + "__t" + t.astype(str)
            return s

        train_uid = make_uid(train_fold_df)
        val_uid = make_uid(val_df)
        test_uid = make_uid(test_df)

        # Per-UID count on training only
        cnt_map = train_uid.value_counts().to_dict()
        c_col = f"{uid_name}_count"
        train_aug[c_col] = train_uid.map(cnt_map).fillna(0).astype(np.float32)
        val_aug[c_col] = val_uid.map(cnt_map).fillna(0).astype(np.float32)
        test_aug[c_col] = test_uid.map(cnt_map).fillna(0).astype(np.float32)
        new_cols.append(c_col)

        # Per-UID aggregates on training only
        df_tmp = train_fold_df[AGG_COLS].copy()
        df_tmp["_uid"] = train_uid.values
        agg = df_tmp.groupby("_uid")[AGG_COLS].agg(AGG_FUNCS)
        agg.columns = [f"{uid_name}_{c}_{f}".replace(" ", "_").replace("(", "").replace(")", "")
                       for c, f in agg.columns]

        for col in agg.columns:
            m = agg[col].to_dict()
            train_aug[col] = train_uid.map(m).astype(np.float32)
            val_aug[col] = val_uid.map(m).astype(np.float32)
            test_aug[col] = test_uid.map(m).astype(np.float32)
            new_cols.append(col)

    return train_aug, val_aug, test_aug, new_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 50k rows, 200 max-rounds")
    ap.add_argument("--name", default="probe_c4_uid_magic")
    ap.add_argument("--max-rounds", type=int, default=2000)
    args = ap.parse_args()

    print(f"=== C4 UID magic-features probe | smoke={args.smoke} ===")
    t0 = time.time()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}")

    if args.smoke:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train), size=50_000, replace=False)
        train = train.iloc[np.sort(idx)].reset_index(drop=True)
        y = train[TARGET].astype(int).values
        args.max_rounds = min(args.max_rounds, 500)
        print(f"  SMOKE → train {train.shape}")

    # Categorical encoding (simple label-encode for LGBM categorical_feature)
    cat_cols = ["Driver", "Compound", "Race"]
    cat_maps = {}
    for c in cat_cols:
        u = pd.concat([train[c], test[c]], ignore_index=True).unique()
        m = {v: i for i, v in enumerate(u)}
        cat_maps[c] = m
        train[c] = train[c].map(m).astype(np.int32)
        test[c] = test[c].map(m).astype(np.int32)

    feat_cols_raw = [c for c in train.columns
                     if c not in {"id", TARGET}]

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_iter = list(skf.split(np.zeros(len(y)), y))
    if args.smoke:
        fold_iter = fold_iter[:1]
        print(f"  SMOKE 1 fold")

    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []

    for fold, (ti, vi) in enumerate(fold_iter, 1):
        t_fold = time.time()
        print(f"\n  --- Fold {fold}: ti={len(ti):,} va={len(vi):,} ---")

        train_ti = train.iloc[ti].reset_index(drop=True)
        train_va = train.iloc[vi].reset_index(drop=True)
        test_fold = test.copy()

        t_uid = time.time()
        train_ti, train_va, test_fold, new_cols = build_uid_features(
            train_ti, train_va, test_fold)
        print(f"    UID FE: +{len(new_cols)} cols, wall {time.time()-t_uid:.1f}s")

        feats = feat_cols_raw + new_cols
        X_tr = train_ti[feats].fillna(0).values
        X_va = train_va[feats].fillna(0).values
        X_te = test_fold[feats].fillna(0).values
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

    auc_full = float(roc_auc_score(y, oof)) if not args.smoke else fold_aucs[0]
    print(f"\n  OOF AUC{'  (smoke fold-1)' if args.smoke else ' (full)'}: {auc_full:.5f}  "
          f"fold-std={np.std(fold_aucs):.5f}  total wall={time.time()-t0:.1f}s")

    if not args.smoke:
        # Save OOF + test for downstream K=4+1 gate
        oof_2c = np.column_stack([1 - oof, oof]).astype(np.float64)
        test_2c = np.column_stack([1 - test_pred, test_pred]).astype(np.float64)
        np.save(ART / f"oof_{args.name}_strat.npy", oof_2c)
        np.save(ART / f"test_{args.name}_strat.npy", test_2c)
        (ART / f"{args.name}_results.json").write_text(json.dumps(dict(
            name=args.name, oof_auc=auc_full, fold_aucs=fold_aucs,
            n_uid_families=len(UIDS), n_new_cols=len(new_cols),
        ), indent=2))
        print(f"  → oof_{args.name}_strat.npy   test_{args.name}_strat.npy")


if __name__ == "__main__":
    main()
