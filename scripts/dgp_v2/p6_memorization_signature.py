"""Phase 6 — memorization signature features (synth-only, no public CSV).

Hypothesis (from d15 prior audit): 97.55% of synth LapTime values are
literal copies from the orig empirical marginal; 94.98% LapTime_Delta;
99.95% RaceProgress; 87.38% Cumulative_Degradation.

If a synth row has continuous values that appear MULTIPLE times in
synth, those values are likely literal copies from a SINGLE orig row,
re-used by CTGAN. If a value appears rarely (count=1 or 2), CTGAN
either interpolated it or sampled a rare orig value.

This script builds memorization features:

  1. value_count_<col> per row: how many synth rows share this value?
  2. value_density_<col>: log(count) per col
  3. row_uniqueness: count of distinct singleton-frequency continuous
     values in the row
  4. per-pair value-pair counts: how many rows share (LapTime, TyreLife)
     as a tuple? (TyreLife, RaceProgress)? etc.

These are all UNCONDITIONAL on label, so fold-safe by construction.

Predicted: standalone OOF ~0.93-0.94; ρ vs PRIMARY ~0.95-0.97;
K=4+1 lift +0.2-1 bp. Mechanism: CTGAN-modal rows (high count)
have cleaner labels; CTGAN-tail rows (low count) have noisier labels.
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

NAME = "p6_memorization"


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    full = pd.concat([train.drop(columns=["PitNextLap"]), test], ignore_index=True)
    n_train = len(train)
    print(f"train {train.shape} test {test.shape} full {full.shape}", flush=True)

    # ---- Memorization features ----
    print("Building per-value count features...", flush=True)
    cont_cols = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                 "RaceProgress"]
    for c in cont_cols:
        # count occurrences of each value in synth
        counts = full[c].value_counts()
        full[f"vcount__{c}"] = full[c].map(counts).astype("int32")
        full[f"vlog_count__{c}"] = np.log1p(full[f"vcount__{c}"]).astype("float32")
        n_singleton = int((full[f"vcount__{c}"] == 1).sum())
        print(f"  {c}: n_unique={len(counts)} singletons={n_singleton} "
              f"max_count={counts.max()}", flush=True)

    # Pairwise tuple counts
    print("Building tuple-count features...", flush=True)
    pair_specs = [
        ("LapTime (s)", "TyreLife"),
        ("LapTime (s)", "Cumulative_Degradation"),
        ("LapTime_Delta", "TyreLife"),
        ("Cumulative_Degradation", "Stint"),
        ("RaceProgress", "TyreLife"),
    ]
    for a, b in pair_specs:
        key = full[a].astype(str) + "|" + full[b].astype(str)
        counts = key.value_counts()
        full[f"pcount__{a}_{b}"] = key.map(counts).astype("int32")
        full[f"plog_count__{a}_{b}"] = np.log1p(full[f"pcount__{a}_{b}"]
                                                ).astype("float32")
        print(f"  ({a}, {b}): n_unique_pairs={len(counts)} "
              f"max_count={counts.max()}", flush=True)

    # Tripleton: (LapTime, TyreLife, Compound) key
    key3 = (full["LapTime (s)"].astype(str) + "|"
            + full["TyreLife"].astype(str) + "|"
            + full["Compound"].astype(str))
    counts3 = key3.value_counts()
    full["tcount__lt_tl_cmp"] = key3.map(counts3).astype("int32")
    full["tlog_count__lt_tl_cmp"] = np.log1p(full["tcount__lt_tl_cmp"]
                                              ).astype("float32")
    print(f"  (LapTime, TyreLife, Compound): n_unique={len(counts3)} "
          f"max_count={counts3.max()}", flush=True)

    # Singleton row indicator: 1 if any singleton value exists in continuous cols
    full["row_has_singleton_cont"] = (
        sum(full[f"vcount__{c}"] == 1 for c in cont_cols)
    ).astype("int8")

    print(f"Feature build done [{time.time()-ts:.0f}s]", flush=True)

    # ---- Train base ----
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        cats = pd.Categorical(full[c]).categories
        full[c] = pd.Categorical(full[c], categories=cats).codes.astype("int32")

    feat_cols = [
        "Driver", "Compound", "Race", "Year", "PitStop",
        "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
    ]
    feat_cols += [f"vlog_count__{c}" for c in cont_cols]
    feat_cols += [f"plog_count__{a}_{b}" for a, b in pair_specs]
    feat_cols += ["tlog_count__lt_tl_cmp", "row_has_singleton_cont"]

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
        print(f"  fold {fold} AUC {fold_aucs[-1]:.5f} "
              f"[{time.time()-fts:.0f}s, {time.time()-ts:.0f}s]", flush=True)

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
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
