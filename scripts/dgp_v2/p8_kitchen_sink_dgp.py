"""Phase 8 — kitchen-sink DGP-aware base.

Combines all P1-P7 DGP-finding features into one LGBM:

  - Standard 14 features (Driver, Compound, Race, Year, PitStop,
    LapNumber, Stint, TyreLife, Position, LapTime, LapTime_Delta,
    Cumulative_Degradation, RaceProgress, Position_Change).

  - P2/P5 stint-recovery: stint_start_imputed, cell_size, cell_tl_max,
    cell_tl_min, cell_pos_spread, cell_implied_stint_len, tl_frac_of_cell,
    cell_implied_stint_len_bin.

  - P7 driver atypicality: driver_year_count, driver_year_cv,
    driver_total_count, is_active_in_year, is_d_prefix.

  - P6 memorization: vlog_count for LapTime/LapTime_Delta/RaceProgress/
    CumDeg, plus tuple counts for (LT, TL), (R, Y, C, stint_start_imputed),
    (Compound, Stint, LapNumber).

  - P3 CTGAN-replay disc output (if available): host-specific bias score.

  - Per-fold TE on (Race, Year, Compound, stint_start_imputed),
    (Compound, stint_start_imputed), (Compound, Race, Year).

This is the comprehensive DGP-aware base. K=4+1 gate is the goal.
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

NAME = "p8_kitchen_sink_dgp"


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
    full = pd.concat([train.drop(columns=["PitNextLap"]), test], ignore_index=True)
    n_train = len(train)
    print(f"train {train.shape} test {test.shape} full {full.shape}",
          flush=True)

    # ---- Stint recovery (P2/P5) ----
    full["stint_start_imputed"] = (full["LapNumber"] - full["TyreLife"] + 1
                                    ).astype("int32")
    cell_key = ["Race", "Year", "Compound", "stint_start_imputed"]
    g = full.groupby(cell_key)
    full["cell_size"] = g["LapNumber"].transform("size").astype("int32")
    full["cell_tl_max"] = g["TyreLife"].transform("max").astype("float32")
    full["cell_tl_min"] = g["TyreLife"].transform("min").astype("float32")
    full["cell_pos_spread"] = (g["Position"].transform("max")
                               - g["Position"].transform("min")).astype("int32")
    full["cell_tl_range"] = full["cell_tl_max"] - full["cell_tl_min"]
    full["cell_implied_stint_len"] = full["cell_tl_max"]
    full["cell_implied_stint_len_bin"] = pd.cut(
        full["cell_implied_stint_len"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 80],
        labels=False,
    ).astype("int8")
    full["tl_frac_of_cell"] = (full["TyreLife"]
                               / full["cell_implied_stint_len"].clip(lower=1)
                               ).astype("float32")

    # ---- Driver atypicality (P7) ----
    print(f"Driver feats... [{time.time()-ts:.0f}s]", flush=True)
    dy_count = full.groupby(["Driver", "Year"]).size()
    dy_dict = dy_count.to_dict()
    full["driver_year_count"] = [
        dy_dict.get((d, y), 0)
        for d, y in zip(full["Driver"], full["Year"])
    ]
    dr_year_pivot = full.groupby(["Driver", "Year"]).size().unstack(fill_value=0)
    dr_cv = (dr_year_pivot.std(axis=1) / dr_year_pivot.mean(axis=1).clip(lower=1)
             ).to_dict()
    full["driver_year_cv"] = full["Driver"].map(dr_cv).fillna(0).astype("float32")
    dr_total = full["Driver"].value_counts().to_dict()
    full["driver_total_count"] = full["Driver"].map(dr_total).astype("int32")
    full["is_active_in_year"] = (full["driver_year_count"] > 100).astype("int8")
    full["is_d_prefix"] = full["Driver"].str.match(r"^D\d{3}$").astype("int8")

    # ---- Memorization signature (P6) ----
    print(f"Memorization feats... [{time.time()-ts:.0f}s]", flush=True)
    cont_cols = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                 "RaceProgress"]
    for c in cont_cols:
        counts = full[c].value_counts()
        full[f"vlog_count__{c}"] = np.log1p(full[c].map(counts)
                                            ).astype("float32")
    pair_specs = [
        ("LapTime (s)", "TyreLife"),
        ("Cumulative_Degradation", "Stint"),
    ]
    for a, b in pair_specs:
        k = full[a].astype(str) + "|" + full[b].astype(str)
        full[f"plog_count__{a}_{b}"] = np.log1p(k.map(k.value_counts())
                                                 ).astype("float32")

    # 4-tuple count (orig-stint identifier)
    k = (full["Race"] + "|" + full["Year"].astype(str) + "|"
         + full["Compound"] + "|" + full["stint_start_imputed"].astype(str))
    full["log_tcnt_R_Y_C_SS"] = np.log1p(k.map(k.value_counts())
                                           ).astype("float32")

    # ---- Optional: P3 CTGAN replay disc output ----
    p3_path = ART / "test_p3_ctgan_replay_disc_strat.npy"
    if p3_path.exists():
        p3_oof = np.load(ART / "oof_p3_ctgan_replay_disc_strat.npy")[:, 1]
        p3_test = np.load(p3_path)[:, 1]
        full["ctgan_disc"] = np.concatenate([p3_oof, p3_test]).astype("float32")
        has_p3 = True
        print(f"P3 disc feature loaded.", flush=True)
    else:
        has_p3 = False
        print(f"P3 disc not yet available; building without.", flush=True)

    # ---- Encode + train ----
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        cats = pd.Categorical(full[c]).categories
        full[c] = pd.Categorical(full[c], categories=cats).codes.astype("int32")

    feat_cols = [
        # standard 14
        "Driver", "Compound", "Race", "Year", "PitStop",
        "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
        # stint recovery
        "stint_start_imputed", "cell_size", "cell_tl_max", "cell_tl_min",
        "cell_pos_spread", "cell_tl_range", "cell_implied_stint_len",
        "cell_implied_stint_len_bin", "tl_frac_of_cell",
        # driver atypicality
        "driver_year_count", "driver_year_cv", "driver_total_count",
        "is_active_in_year", "is_d_prefix",
        # memorization
        "vlog_count__LapTime (s)", "vlog_count__LapTime_Delta",
        "vlog_count__Cumulative_Degradation", "vlog_count__RaceProgress",
        "plog_count__LapTime (s)_TyreLife", "plog_count__Cumulative_Degradation_Stint",
        "log_tcnt_R_Y_C_SS",
    ]
    if has_p3:
        feat_cols.append("ctgan_disc")

    train_feat = full.iloc[:n_train].copy()
    train_feat["PitNextLap"] = train["PitNextLap"].to_numpy()
    test_feat = full.iloc[n_train:].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = train_feat["PitNextLap"].to_numpy()
    oof = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(len(test_feat), dtype=np.float32)
    fold_aucs = []
    PARAMS = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, n_estimators=2500,
        num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=4, seed=SEED, verbose=-1,
        n_jobs=2,
    )
    for fold, (tr, va) in enumerate(skf.split(np.zeros(n_train), y)):
        fts = time.time()
        tr_df = train_feat.iloc[tr]
        va_df = train_feat.iloc[va]
        # per-fold TEs
        te_specs = [
            (["Race", "Year", "Compound", "stint_start_imputed"], 30.0, "te_RYCS"),
            (["Compound", "stint_start_imputed"], 50.0, "te_CS"),
            (["Compound", "Race", "Year"], 50.0, "te_CRY"),
        ]
        Xtr = tr_df[feat_cols].copy()
        Xva = va_df[feat_cols].copy()
        Xte = test_feat[feat_cols].copy()
        for key, smooth, te_name in te_specs:
            tr_te, va_te, te_te = fold_safe_te(tr_df, va_df, test_feat,
                                                 key, "PitNextLap",
                                                 smooth=smooth)
            Xtr[te_name] = tr_te
            Xva[te_name] = va_te
            Xte[te_name] = te_te
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(
            Xtr, y[tr], eval_set=[(Xva, y[va])],
            categorical_feature=cat_cols + ["Year"],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va] = m.predict_proba(Xva)[:, 1]
        test_preds += m.predict_proba(Xte)[:, 1] / 5
        fold_aucs.append(float(roc_auc_score(y[va], oof[va])))
        print(f"  fold {fold} AUC {fold_aucs[-1]:.5f} [{time.time()-fts:.0f}s, "
              f"total {time.time()-ts:.0f}s]", flush=True)

    overall = float(roc_auc_score(y, oof))
    print(f"Overall OOF {overall:.5f}", flush=True)
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_preds, test_preds])
    np.save(ART / f"oof_{NAME}_strat.npy", oof2)
    np.save(ART / f"test_{NAME}_strat.npy", test2)
    summary = {
        "name": NAME,
        "overall_oof_auc": overall,
        "fold_aucs": fold_aucs,
        "fold_std": float(np.std(fold_aucs)),
        "n_features": len(feat_cols) + 3,
        "feat_cols": feat_cols + [s[2] for s in te_specs],
        "p3_disc_included": has_p3,
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
