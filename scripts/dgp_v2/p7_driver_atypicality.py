"""Phase 7 — Driver temporal consistency + tuple atypicality features.

DGP findings (P1, P1b, P6 prep):

  - 131 abbrev driver codes track real career timelines for ACTIVE
    drivers; LONG-RETIRED codes (BAR, MAS, BUT) appear uniformly in
    2022-2025 (fabricated). 756 D-prefix codes are pure ghosts.
  - 97.55% of LapTime values are literal copies; 2.45% are CTGAN
    interpolations.
  - Joint tuples have variable rarity; rare tuples are CTGAN-tail.

Features:

  1. driver_year_count: rows for this driver in this year (synth-only).
  2. driver_year_zscore: standardised across all (driver, year) pairs.
  3. driver_year_cv: per-driver coefficient of variation across years
     (low for active drivers, high for rookies/retirees).
  4. is_likely_active_in_year: driver_year_count > 100 (heuristic).
  5. tuple_count_lt_tl_cmp_race_year: rare tuples = CTGAN-tail.
  6. tuple_count_lt_tl: pair count.
  7. tuple_count_compound_stint_lap: high-mode count (Compound,
     stint_start_imputed, LapNumber).

Train LGBM 5-fold and gate K=4+1.
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

NAME = "p7_driver_atypicality"


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    full = pd.concat([train.drop(columns=["PitNextLap"]), test], ignore_index=True)
    n_train = len(train)
    print(f"train {train.shape} test {test.shape}", flush=True)

    full["stint_start_imputed"] = (full["LapNumber"] - full["TyreLife"] + 1
                                    ).astype("int32")

    # ---- Driver-year features ----
    print("Building driver-year features...", flush=True)
    dy_count = full.groupby(["Driver", "Year"]).size()
    dy_dict = dy_count.to_dict()
    full["driver_year_count"] = [
        dy_dict.get((d, y), 0)
        for d, y in zip(full["Driver"], full["Year"])
    ]
    # Per-driver CV across years
    dr_year_pivot = full.groupby(["Driver", "Year"]).size().unstack(fill_value=0)
    dr_cv = (dr_year_pivot.std(axis=1) / dr_year_pivot.mean(axis=1).clip(lower=1)
             ).to_dict()
    full["driver_year_cv"] = full["Driver"].map(dr_cv).fillna(0).astype("float32")
    # Per-driver TOTAL count
    dr_total = full["Driver"].value_counts().to_dict()
    full["driver_total_count"] = full["Driver"].map(dr_total).astype("int32")
    # is_active heuristic
    full["is_active_in_year"] = (full["driver_year_count"] > 100).astype("int8")
    full["is_d_prefix"] = full["Driver"].str.match(r"^D\d{3}$").astype("int8")

    print(f"  driver feats done [{time.time()-ts:.0f}s]", flush=True)

    # ---- Tuple counts ----
    print("Building tuple-count features...", flush=True)
    # 2-tuple
    k = full["LapTime (s)"].astype(str) + "|" + full["TyreLife"].astype(str)
    counts = k.value_counts()
    full["tcnt_lt_tl"] = k.map(counts).astype("int32")

    # 4-tuple (likely orig-stint identifier)
    k = (full["Race"] + "|" + full["Year"].astype(str) + "|"
         + full["Compound"] + "|" + full["stint_start_imputed"].astype(str))
    counts = k.value_counts()
    full["tcnt_R_Y_C_SS"] = k.map(counts).astype("int32")
    full["log_tcnt_R_Y_C_SS"] = np.log1p(full["tcnt_R_Y_C_SS"]).astype("float32")

    # Compound-Stint-LapNumber
    k = (full["Compound"] + "|" + full["Stint"].astype(str) + "|"
         + full["LapNumber"].astype(str))
    counts = k.value_counts()
    full["tcnt_C_S_L"] = k.map(counts).astype("int32")
    print(f"  tuple feats done [{time.time()-ts:.0f}s]", flush=True)

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
        # P7 additions
        "driver_year_count", "driver_year_cv", "driver_total_count",
        "is_active_in_year", "is_d_prefix",
        "stint_start_imputed",
        "tcnt_lt_tl", "tcnt_R_Y_C_SS", "log_tcnt_R_Y_C_SS", "tcnt_C_S_L",
    ]
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
        learning_rate=0.05, n_estimators=2000,
        num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=4, seed=SEED, verbose=-1,
        n_jobs=2,
    )
    for fold, (tr, va) in enumerate(skf.split(np.zeros(n_train), y)):
        fts = time.time()
        Xtr = train_feat.iloc[tr][feat_cols]
        Xva = train_feat.iloc[va][feat_cols]
        Xte = test_feat[feat_cols]
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
        print(f"  fold {fold} AUC {fold_aucs[-1]:.5f} [{time.time()-fts:.0f}s]",
              flush=True)

    overall = float(roc_auc_score(y, oof))
    print(f"Overall OOF AUC {overall:.5f}", flush=True)
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_preds, test_preds])
    np.save(ART / f"oof_{NAME}_strat.npy", oof2)
    np.save(ART / f"test_{NAME}_strat.npy", test2)
    summary = {
        "name": NAME,
        "overall_oof_auc": overall,
        "fold_aucs": fold_aucs,
        "fold_std": float(np.std(fold_aucs)),
        "feat_cols": feat_cols,
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
