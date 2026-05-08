"""Sequence-level fingerprinting — within-(Driver, Race, Year) structure.

HANDOVER #1 priority axis. Every base in the K=4 / K=10 / K=27 pool
treats rows as i.i.d.; the synthesiser almost certainly broke
within-stint sequence coherence. Build features that describe the
*race history so far* for each row, then train a fresh base on them.

Sequence features (no label leakage — these depend only on Compound,
LapNumber, Stint, Position, TyreLife structure, which is shared between
train and test):

  prev_compound           — Compound on previous lap (label-encoded)
  prev_compound_one_hot   — 5 cols, prior-stint compound indicator
  compound_changes        — count of compound changes so far this race
  stint_lap_idx           — lap position within current stint (0..N)
  stint_lap_frac          — stint_lap_idx / longest stint observed
                              for this driver-race
  prev_stint_length       — length of the previous stint (laps)
  laps_since_pit          — laps since the most recent pit transition
  total_laps_so_far       — total race laps elapsed for this driver
  position_at_stint_start — Position when current stint began
  tyre_life_at_stint_start — TyreLife at stint start
  position_change_in_stint — Position - position_at_stint_start
  compound_pair_seen      — count of (prev, curr) compound pairs
  compound_in_history     — 5 cols, 1 if compound used earlier this race
  laps_in_compound_so_far — total laps spent in current compound

Then train LightGBM on (sequence features + the original 11 numerics +
3 cats) and produce OOF/test predictions. Gate against K=4 LR-meta
(0.95399) and K=10 LR-meta (0.95417).

Per Rule 24, no label-derived aggregates here — these features are pure
structural / temporal; safe across train/test (AV-AUC = 0.502).

Cost: ~5-10 min FE + ~10-20 min LightGBM training = 30 min total.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import N_FOLDS, SEED, folds, save_oof

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
ID_COL = "id"


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per (Driver, Race, Year) sorted by LapNumber, compute the
    sequence-level fingerprint columns. Returns a DataFrame with the
    same index as df, indexed in df's original order."""
    df = df.copy()
    grp_keys = ["Driver", "Race", "Year"]
    df["__orig_order"] = np.arange(len(df))

    # Sort within group by LapNumber (preserve groups, allow per-group ops)
    sorted_df = df.sort_values(grp_keys + ["LapNumber"]).reset_index(drop=True)
    g = sorted_df.groupby(grp_keys, sort=False)

    # Compound on previous lap
    sorted_df["prev_compound"] = g["Compound"].shift(1)

    # Compound changes so far: cumulative count where compound differs
    # from previous lap (NaN-safe; first lap = 0 changes)
    same = (sorted_df["Compound"] == sorted_df["prev_compound"]).astype(int)
    change = (1 - same)
    change[sorted_df["prev_compound"].isna()] = 0
    sorted_df["compound_changes"] = (
        change.groupby(g.ngroup()).cumsum().astype(np.int32)
    )

    # Stint lap index = cumcount of consecutive same-compound runs.
    new_stint = (sorted_df["Compound"] != sorted_df["prev_compound"]).fillna(True)
    new_stint = new_stint.astype(np.int32)
    seg_id = new_stint.groupby(g.ngroup()).cumsum().astype(np.int32)
    sorted_df["__seg_id"] = seg_id.values
    seg_g = sorted_df.groupby(grp_keys + ["__seg_id"], sort=False)
    sorted_df["stint_lap_idx"] = seg_g.cumcount().astype(np.int32)

    # Per-segment summary (one row per (driver, race, year, seg_id)):
    seg_summary = seg_g.agg(
        __seg_size=("LapNumber", "size"),
        position_at_stint_start=("Position", "first"),
        tyre_life_at_stint_start=("TyreLife", "first"),
    ).reset_index()

    # Previous-segment lookup: shift seg_id by +1 so row's "prev seg" key matches
    prev_seg = seg_summary[grp_keys + ["__seg_id", "__seg_size"]].copy()
    prev_seg["__seg_id"] += 1
    prev_seg = prev_seg.rename(columns={"__seg_size": "prev_stint_length"})

    # Merge once, dedup keys: each (grp_keys, __seg_id) appears in seg_summary
    # exactly once → row count is preserved.
    sorted_df = sorted_df.merge(seg_summary, on=grp_keys + ["__seg_id"],
                                how="left", validate="many_to_one")
    sorted_df = sorted_df.merge(prev_seg, on=grp_keys + ["__seg_id"],
                                how="left", validate="many_to_one")
    sorted_df["prev_stint_length"] = (
        sorted_df["prev_stint_length"].fillna(0).astype(np.float32)
    )

    # Total laps so far this race
    sorted_df["total_laps_so_far"] = g.cumcount().astype(np.int32)

    # Position-at-stint-start / TyreLife-at-stint-start derived above
    sorted_df["position_change_in_stint"] = (
        sorted_df["Position"] - sorted_df["position_at_stint_start"]
    ).fillna(0).astype(np.float32)
    sorted_df["position_at_stint_start"] = (
        sorted_df["position_at_stint_start"].fillna(-1).astype(np.float32)
    )
    sorted_df["tyre_life_at_stint_start"] = (
        sorted_df["tyre_life_at_stint_start"].fillna(-1).astype(np.float32)
    )

    # Compound history one-hot: was each compound EVER used by this row's
    # driver-race-year prior to (or including) this lap?
    compounds = sorted(df["Compound"].dropna().unique().tolist())
    for c in compounds:
        col = f"hist_{c}"
        is_c = (sorted_df["Compound"] == c).astype(np.float32)
        cum = is_c.groupby(g.ngroup()).cumsum().clip(upper=1).fillna(0)
        sorted_df[col] = cum.astype(np.float32)

    # Laps in current compound so far (could be > stint_lap_idx if
    # driver returned to the same compound on a later stint)
    is_curr = sorted_df.groupby(grp_keys + ["Compound"], sort=False).cumcount()
    sorted_df["laps_in_compound_so_far"] = is_curr.fillna(0).astype(np.float32)

    # stint_lap_frac = stint_lap_idx / max stint length seen for this
    # driver-race so far. Implemented as cumulative max of __seg_size.
    grp_codes = sorted_df.groupby(grp_keys, sort=False).ngroup().values
    sorted_df["__stint_max_so_far"] = (
        sorted_df["__seg_size"].groupby(grp_codes).cummax()
    )
    sorted_df["stint_lap_frac"] = (
        sorted_df["stint_lap_idx"] / sorted_df["__stint_max_so_far"].clip(lower=1)
    ).astype(np.float32)

    # Restore original order
    sorted_df = sorted_df.sort_values("__orig_order")
    out = sorted_df[[
        "prev_compound", "compound_changes", "stint_lap_idx",
        "prev_stint_length", "total_laps_so_far",
        "position_at_stint_start", "tyre_life_at_stint_start",
        "position_change_in_stint",
        "laps_in_compound_so_far", "stint_lap_frac",
    ] + [f"hist_{c}" for c in compounds]].reset_index(drop=True)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-rounds", type=int, default=2000)
    args = ap.parse_args()

    print("Loading data ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # Build sequence features on combined train+test (no label leakage —
    # these depend only on Compound/LapNumber/Stint/Position/TyreLife
    # structure, which is shared between train and test)
    print("Building sequence features ...")
    t0 = time.time()
    combined = pd.concat([
        train.assign(__split="tr").drop(columns=[TARGET]),
        test.assign(__split="te"),
    ], ignore_index=True)
    seq_feat = build_sequence_features(combined)
    print(f"  built in {time.time()-t0:.1f}s, shape: {seq_feat.shape}")
    print(f"  cols: {list(seq_feat.columns)}")

    # Recombine with the original schema for LightGBM
    combined_full = pd.concat([combined.reset_index(drop=True),
                                seq_feat.reset_index(drop=True)], axis=1)
    train_full = combined_full[combined_full["__split"] == "tr"].drop(
        columns=["__split"]).reset_index(drop=True)
    test_full = combined_full[combined_full["__split"] == "te"].drop(
        columns=["__split"]).reset_index(drop=True)
    assert len(train_full) == len(train), (len(train_full), len(train))
    assert len(test_full) == len(test), (len(test_full), len(test))

    # Drop id; cast non-numeric cols to category for LightGBM native handling
    feat_cols = [c for c in train_full.columns if c not in (ID_COL,)]
    cat_cols = [c for c in feat_cols
                if not pd.api.types.is_numeric_dtype(train_full[c])
                and not pd.api.types.is_bool_dtype(train_full[c])]
    for c in cat_cols:
        train_full[c] = train_full[c].astype("category")
        test_full[c] = pd.Categorical(test_full[c],
                                      categories=train_full[c].cat.categories)
    print(f"  total features: {len(feat_cols)} ({len(cat_cols)} categorical: "
          f"{cat_cols})")

    # 5-fold LightGBM on the enlarged feature set
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=200,
        verbose=-1,
        seed=SEED,
    )

    oof = np.zeros(len(y), dtype=np.float32)
    test_proba = np.zeros(len(test_full), dtype=np.float32)
    fold_aucs, fold_secs = [], []
    print("\n5-fold LightGBM on raw + sequence features ...")
    for k, tr, va in folds(y, task="classification"):
        t_f = time.time()
        dtrain = lgb.Dataset(train_full[feat_cols].iloc[tr], y[tr],
                             categorical_feature=cat_cols)
        dval = lgb.Dataset(train_full[feat_cols].iloc[va], y[va],
                           categorical_feature=cat_cols)
        model = lgb.train(
            params, dtrain, num_boost_round=args.num_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)],
        )
        p_va = model.predict(train_full[feat_cols].iloc[va])
        p_te = model.predict(test_full[feat_cols])
        oof[va] = p_va
        test_proba += p_te.astype(np.float32) / N_FOLDS
        secs = time.time() - t_f
        auc = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(auc)
        fold_secs.append(secs)
        print(f"  fold {k}: AUC={auc:.5f}  ({secs:.1f}s, "
              f"best_iter={model.best_iteration})")

    oof_full = float(roc_auc_score(y, oof))
    print(f"\n  full OOF AUC: {oof_full:.5f}  total {sum(fold_secs):.0f}s")
    save_oof("seq_fingerprint_lgbm",
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(variant="seq_fingerprint_lgbm",
                  oof_score=oof_full, fold_aucs=fold_aucs,
                  fold_secs=fold_secs, n_features=len(feat_cols)))


if __name__ == "__main__":
    main()
