"""Phase 2 — orig-stint recovery base on synth alone (no public CSV).

Mechanism: P1 found that synth's `(Driver, Race, Year, Stint)` is a
fabricated label; only 15% of synth groups agree on stint_start. The
correct per-row preimage is `stint_start_imputed = LapNumber - TyreLife
+ 1`, and rows sharing `(Race, Year, Compound, stint_start_imputed)`
are likely from the same orig stint (different orig drivers possibly).

This script builds DGP-recovery features:

  1. stint_start_imputed: LN - TL + 1 per row.
  2. orig_stint_te: target-encoded mean PitNextLap per
     (Race, Year, Compound, stint_start_imputed) cell, per-fold refit
     (Rule 24 fold-safe).
  3. orig_stint_size: count of synth rows in the cell.
  4. orig_stint_tl_max: max TyreLife observed in cell (orig stint length).
  5. orig_stint_consistency: 1 if row's stint_start agrees with mode of
     its synth (Driver, Race, Year, Stint) group.

Then trains a 5-fold LightGBM with these + standard 14 features.
Per-fold refit of TE per Rule 24.

Outputs OOF + test artifacts so the K=4+1 gate can run downstream.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
SEED = 42
N_FOLDS = 5

NAME = "p2_orig_stint_recovery"


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stint_start_imputed"] = (df["LapNumber"] - df["TyreLife"] + 1
                                 ).astype("int32")
    # Synth-label group mode of stint_start (per row)
    df["_grp"] = (df["Driver"] + "|" + df["Race"] + "|"
                  + df["Year"].astype(str) + "|" + df["Stint"].astype(str))
    grp_mode = df.groupby("_grp")["stint_start_imputed"].transform(
        lambda x: x.mode().iloc[0])
    df["stint_consistent_with_synth_label"] = (
        df["stint_start_imputed"] == grp_mode).astype("int8")
    df.drop(columns=["_grp"], inplace=True)
    # Cell features per (Race, Year, Compound, stint_start_imputed)
    cell_key = ["Race", "Year", "Compound", "stint_start_imputed"]
    g = df.groupby(cell_key)
    df["cell_size"] = g["LapNumber"].transform("size").astype("int32")
    df["cell_tl_max"] = g["TyreLife"].transform("max").astype("float32")
    df["cell_tl_min"] = g["TyreLife"].transform("min").astype("float32")
    df["cell_lap_max"] = g["LapNumber"].transform("max").astype("int32")
    df["cell_lap_min"] = g["LapNumber"].transform("min").astype("int32")
    df["cell_pos_spread"] = (g["Position"].transform("max")
                             - g["Position"].transform("min")).astype("int32")
    df["cell_tl_range"] = df["cell_tl_max"] - df["cell_tl_min"]
    df["cell_implied_stint_len"] = df["cell_tl_max"]  # max TL = stint length
    return df


def fold_safe_te(train_df: pd.DataFrame, val_df: pd.DataFrame,
                 test_df: pd.DataFrame, key: list, target: str,
                 smooth: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-fold target encoding with smoothing. Fit on train_df only."""
    global_mean = float(train_df[target].mean())
    g = train_df.groupby(key)[target].agg(["mean", "size"])
    smoothed = (g["mean"] * g["size"] + global_mean * smooth) / (g["size"] + smooth)
    smoothed_d = smoothed.to_dict()
    def lookup(df):
        keys = list(zip(*[df[k] for k in key]))
        return np.array([smoothed_d.get(k, global_mean) for k in keys],
                        dtype=np.float32)
    return lookup(train_df), lookup(val_df), lookup(test_df)


def main():
    ts = time.time()
    print("Loading data...", flush=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["PitNextLap"].astype("int8").to_numpy()
    print(f"  train {train.shape} | test {test.shape}  [{time.time()-ts:.1f}s]",
          flush=True)

    # Build base features (computed on full data, no labels needed)
    print("Building DGP-recovery features...", flush=True)
    full = pd.concat([train.drop(columns=["PitNextLap"]), test],
                     ignore_index=True)
    full = build_base_features(full)
    train_feat = full.iloc[:len(train)].copy()
    train_feat["PitNextLap"] = train["PitNextLap"].to_numpy()
    test_feat = full.iloc[len(train):].copy()
    print(f"  full features built  [{time.time()-ts:.1f}s]", flush=True)

    # Encode categoricals (label encoding, treated as cat by LGBM)
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        cats = pd.Categorical(full[c])
        train_feat[c] = pd.Categorical(train_feat[c], categories=cats.categories
                                       ).codes.astype("int32")
        test_feat[c] = pd.Categorical(test_feat[c], categories=cats.categories
                                      ).codes.astype("int32")

    feat_cols = [
        # standard 14 (excl id, target)
        "Driver", "Compound", "Race", "Year", "PitStop",
        "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
        # DGP-recovery additions
        "stint_start_imputed",
        "stint_consistent_with_synth_label",
        "cell_size", "cell_tl_max", "cell_tl_min",
        "cell_lap_max", "cell_lap_min", "cell_pos_spread",
        "cell_tl_range", "cell_implied_stint_len",
        # TE filled below per fold
    ]

    # 5-fold OOF training
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float32)
    test_preds = np.zeros(len(test), dtype=np.float32)
    fold_aucs = []

    LGB_PARAMS = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=4,
        seed=SEED,
        verbose=-1,
    )

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(train)),
                                              train_feat["PitNextLap"])):
        fts = time.time()
        tr_df = train_feat.iloc[tr]
        va_df = train_feat.iloc[va]
        # per-fold TE on (Race, Year, Compound, stint_start_imputed)
        cell_key = ["Race", "Year", "Compound", "stint_start_imputed"]
        tr_te, va_te, te_te = fold_safe_te(tr_df, va_df, test_feat,
                                            cell_key, "PitNextLap", smooth=30.0)
        # also (Compound, stint_start_imputed) coarser cell
        cell_key2 = ["Compound", "stint_start_imputed"]
        tr_te2, va_te2, te_te2 = fold_safe_te(tr_df, va_df, test_feat,
                                                cell_key2, "PitNextLap", smooth=50.0)
        # also (Race, Year, stint_start_imputed) coarser
        cell_key3 = ["Race", "Year", "stint_start_imputed"]
        tr_te3, va_te3, te_te3 = fold_safe_te(tr_df, va_df, test_feat,
                                                cell_key3, "PitNextLap", smooth=30.0)

        Xtr = tr_df[feat_cols].copy()
        Xva = va_df[feat_cols].copy()
        Xte = test_feat[feat_cols].copy()
        Xtr["te_cell_RYCS"] = tr_te
        Xva["te_cell_RYCS"] = va_te
        Xte["te_cell_RYCS"] = te_te
        Xtr["te_cell_CS"] = tr_te2
        Xva["te_cell_CS"] = va_te2
        Xte["te_cell_CS"] = te_te2
        Xtr["te_cell_RYS"] = tr_te3
        Xva["te_cell_RYS"] = va_te3
        Xte["te_cell_RYS"] = te_te3

        ytr = tr_df["PitNextLap"].to_numpy()
        yva = va_df["PitNextLap"].to_numpy()

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            categorical_feature=["Driver", "Compound", "Race", "Year"],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(Xva)[:, 1]
        test_preds += model.predict_proba(Xte)[:, 1] / N_FOLDS
        fold_auc = roc_auc_score(yva, oof[va])
        fold_aucs.append(fold_auc)
        print(f"  fold {fold} AUC {fold_auc:.5f} [{time.time()-fts:.0f}s, "
              f"total {time.time()-ts:.0f}s]", flush=True)

    overall = roc_auc_score(y, oof)
    fold_std = float(np.std(fold_aucs))
    print(f"\nOverall OOF AUC {overall:.5f} (fold std {fold_std:.5f})",
          flush=True)

    # Save in (n, 2) format expected by probe.py (col 1 = positive class)
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_preds, test_preds])
    np.save(ART / f"oof_{NAME}_strat.npy", oof2)
    np.save(ART / f"test_{NAME}_strat.npy", test2)
    summary = {
        "name": NAME,
        "overall_oof_auc": float(overall),
        "fold_aucs": fold_aucs,
        "fold_std": fold_std,
        "n_features": len(feat_cols) + 3,
        "feat_cols": feat_cols + ["te_cell_RYCS", "te_cell_CS", "te_cell_RYS"],
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved oof_{NAME}_strat.npy / test_{NAME}_strat.npy /"
          f" {NAME}_results.json", flush=True)


if __name__ == "__main__":
    main()
