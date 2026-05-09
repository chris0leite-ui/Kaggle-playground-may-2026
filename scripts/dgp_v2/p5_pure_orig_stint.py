"""Phase 5 — pure orig-stint base (no standard 14 features).

Tests whether the orig-stint recovery features alone (no raw 14)
capture standalone signal and whether they're more orthogonal to
PRIMARY than P2 was. P2 OOF 0.93971 had ρ=0.953 — relatively low
but K=4+1 lift only +0.09 bp because DGP feats redundant with
LGBM's splits on raw 14.

Pure-recovery features:
  - stint_start_imputed (LN - TL + 1)
  - cell_size, cell_tl_max, cell_tl_min, cell_lap_max, cell_lap_min
  - cell_pos_spread, cell_tl_range, cell_implied_stint_len
  - per-fold TE on (Race, Year, Compound, stint_start_imputed)
  - per-fold TE on (Compound, stint_start_imputed)
  - per-fold TE on (Race, Year, stint_start_imputed)
  - per-fold TE on (Compound, stint_start_imputed, cell_implied_stint_len_bin)

Standard 14 features ARE EXCLUDED. The model has only DGP-recovery
information.

Predicted: standalone OOF ~0.85-0.90; ρ vs PRIMARY ~0.6-0.8 (most-
diverse positive); K=4+1 lift +0.5-2 bp if orthogonal.
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

NAME = "p5_pure_orig_stint"


def build_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stint_start_imputed"] = (df["LapNumber"] - df["TyreLife"] + 1).astype("int32")
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
    df["cell_implied_stint_len"] = df["cell_tl_max"]
    df["cell_implied_stint_len_bin"] = pd.cut(
        df["cell_implied_stint_len"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 80],
        labels=False,
    ).astype("int8")
    # TyreLife relative to cell stint length (lap_in_stint relative position)
    df["tl_frac_of_cell"] = (df["TyreLife"] / df["cell_implied_stint_len"].clip(lower=1)
                              ).astype("float32")
    return df


def fold_safe_te(train_df, val_df, test_df, key, target, smooth=30.0):
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
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"Loaded train {train.shape} test {test.shape}", flush=True)

    full = pd.concat([train.drop(columns=["PitNextLap"]), test], ignore_index=True)
    full = build_recovery_features(full)
    train_feat = full.iloc[:len(train)].copy()
    train_feat["PitNextLap"] = train["PitNextLap"].to_numpy()
    test_feat = full.iloc[len(train):].copy()

    # Pure recovery feature columns (NO standard 14)
    feat_cols = [
        "stint_start_imputed",
        "cell_size", "cell_tl_max", "cell_tl_min",
        "cell_lap_max", "cell_lap_min", "cell_pos_spread",
        "cell_tl_range", "cell_implied_stint_len",
        "cell_implied_stint_len_bin",
        "tl_frac_of_cell",
    ]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y = train_feat["PitNextLap"].to_numpy()
    oof = np.zeros(len(train), dtype=np.float32)
    test_preds = np.zeros(len(test), dtype=np.float32)
    fold_aucs = []

    PARAMS = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, n_estimators=2000,
        num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=4, seed=SEED, verbose=-1,
    )

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(train)), y)):
        fts = time.time()
        tr_df = train_feat.iloc[tr]
        va_df = train_feat.iloc[va]
        # 4 TE features
        te_specs = [
            (["Race", "Year", "Compound", "stint_start_imputed"], 30.0, "te_RYCS"),
            (["Compound", "stint_start_imputed"], 50.0, "te_CS"),
            (["Race", "Year", "stint_start_imputed"], 30.0, "te_RYS"),
            (["Compound", "stint_start_imputed", "cell_implied_stint_len_bin"], 30.0, "te_CSL"),
            (["Compound", "Race", "Year"], 50.0, "te_CRY"),
        ]
        Xtr = tr_df[feat_cols].copy()
        Xva = va_df[feat_cols].copy()
        Xte = test_feat[feat_cols].copy()
        for key, smooth, te_name in te_specs:
            tr_te, va_te, te_te = fold_safe_te(tr_df, va_df, test_feat,
                                                 key, "PitNextLap", smooth=smooth)
            Xtr[te_name] = tr_te
            Xva[te_name] = va_te
            Xte[te_name] = te_te

        ytr = tr_df["PitNextLap"].to_numpy()
        yva = va_df["PitNextLap"].to_numpy()

        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va] = m.predict_proba(Xva)[:, 1]
        test_preds += m.predict_proba(Xte)[:, 1] / N_FOLDS
        fold_auc = roc_auc_score(yva, oof[va])
        fold_aucs.append(fold_auc)
        print(f"  fold {fold} AUC {fold_auc:.5f} [{time.time()-fts:.0f}s]",
              flush=True)

    overall = roc_auc_score(y, oof)
    print(f"Overall OOF AUC {overall:.5f}", flush=True)

    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_preds, test_preds])
    np.save(ART / f"oof_{NAME}_strat.npy", oof2)
    np.save(ART / f"test_{NAME}_strat.npy", test2)
    summary = {
        "name": NAME,
        "overall_oof_auc": float(overall),
        "fold_aucs": [float(x) for x in fold_aucs],
        "fold_std": float(np.std(fold_aucs)),
        "n_features": len(feat_cols) + 5,
        "feat_cols": feat_cols + [s[2] for s in te_specs],
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
