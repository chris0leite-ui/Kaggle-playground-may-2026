"""scripts/d15_decode_ntl.py — Lens 1: decode-via-original base.

Discovery: aadigupta1601/f1-strategy-dataset-pit-stop-prediction is the
verified source dataset. 31 of 887 synth Drivers match originals; ~5.5%
of synth rows match on (Driver, Year, Race, LapNumber). For matched
rows we can directly look up the host-removed Normalized_TyreLife and
the original TyreLife / Stint / Compound / LapTime values (which the
synthesizer corrupted with noise).

This script builds a decoded-feature LGBM base:
  - NTL_decoded     : Normalized_TyreLife from original where matched, NaN else
  - NTL_estimate    : TyreLife / max(TyreLife) within (Driver, Race, Year, Stint) for synth
  - matched_flag    : 1 if (Driver,Year,Race,LapNumber) ∈ original
  - tyrelife_orig   : original TyreLife (denoised) where matched, synth TyreLife else
  - laptime_orig    : original LapTime where matched, synth LapTime else
  - delta_tl_synth  : synth_TyreLife - tyrelife_orig (synthesizer noise estimate, 0 if not matched)
  - delta_lt_synth  : synth_LapTime - laptime_orig
  - laptime_delta_orig
  - cumulative_degradation_orig

Trains a 5-fold StratifiedKFold LGBM with these decoded features added
to the standard feature set. Saves oof_/test_d15_decode_ntl_strat.npy
in the harness's expected format (n,2) for downstream gate + min-meta.
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"

KEY = ["Driver", "Year", "Race", "LapNumber"]

ORIG_COLS = ["Normalized_TyreLife", "TyreLife", "Stint", "Compound",
             "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation"]


def main():
    print("Loading data...")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    y = tr[TARGET].astype(int).values

    print(f"  train: {tr.shape}, test: {te.shape}, orig: {orig.shape}")

    # === Decode: lookup original by (Driver, Year, Race, LapNumber) ===
    orig_lookup = orig[KEY + ORIG_COLS].copy()
    # Rename to disambiguate
    rename = {c: f"{c}_orig" for c in ORIG_COLS}
    orig_lookup.rename(columns=rename, inplace=True)

    tr = tr.merge(orig_lookup, on=KEY, how="left")
    te = te.merge(orig_lookup, on=KEY, how="left")

    tr["matched_flag"] = tr["Normalized_TyreLife_orig"].notna().astype(np.int8)
    te["matched_flag"] = te["Normalized_TyreLife_orig"].notna().astype(np.int8)
    print(f"  train matched: {tr['matched_flag'].sum():,} ({tr['matched_flag'].mean():.3%})")
    print(f"  test  matched: {te['matched_flag'].sum():,} ({te['matched_flag'].mean():.3%})")

    # === NTL_estimate: TyreLife / max(TyreLife) within stint of synth ===
    for df in [tr, te]:
        g = df.groupby(["Driver", "Race", "Year", "Stint"])["TyreLife"].transform("max")
        df["NTL_estimate"] = (df["TyreLife"] / g.clip(lower=1)).astype(np.float32)

    # === Hybrid NTL: orig where matched, estimate else ===
    for df in [tr, te]:
        df["NTL_hybrid"] = df["Normalized_TyreLife_orig"].fillna(df["NTL_estimate"]).astype(np.float32)

    # === Synthesizer-noise estimates (only meaningful for matched rows) ===
    for df in [tr, te]:
        df["delta_tl_synth"] = (df["TyreLife"] - df["TyreLife_orig"]).fillna(0).astype(np.float32)
        df["delta_lt_synth"] = (df["LapTime (s)"] - df["LapTime (s)_orig"]).fillna(0).astype(np.float32)
        # When not matched, fill orig features with synth values (denoising = no-op)
        df["TyreLife_denoised"] = df["TyreLife_orig"].fillna(df["TyreLife"]).astype(np.float32)
        df["LapTime_denoised"] = df["LapTime (s)_orig"].fillna(df["LapTime (s)"]).astype(np.float32)

    # === Feature set ===
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
        # Align categories
        te[c] = te[c].cat.set_categories(tr[c].cat.categories)

    feature_cols = [
        "Driver", "Compound", "Race", "Year",
        "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
        # Decoded features
        "NTL_estimate", "NTL_hybrid", "matched_flag",
        "TyreLife_denoised", "LapTime_denoised",
        "delta_tl_synth", "delta_lt_synth",
    ]
    print(f"\nFeatures ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")

    X = tr[feature_cols]
    Xte = te[feature_cols]

    # === 5-fold StratifiedKFold LGBM ===
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr), dtype=np.float64)
    test_pred = np.zeros(len(te), dtype=np.float64)

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        lambda_l1=0.0,
        lambda_l2=0.0,
        verbose=-1,
        n_jobs=-1,
        seed=SEED,
    )

    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                             categorical_feature=cat_cols)
        dval = lgb.Dataset(X.iloc[va_idx], label=y[va_idx],
                           categorical_feature=cat_cols, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=2000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(0)])
        oof[va_idx] = model.predict(X.iloc[va_idx])
        test_pred += model.predict(Xte) / N_FOLDS
        fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"  fold {k}: AUC={fold_auc:.5f}  best_iter={model.best_iteration}  ({time.time()-t0:.0f}s)")

    overall_auc = roc_auc_score(y, oof)
    print(f"\n=== d15_decode_ntl OOF AUC: {overall_auc:.5f} ===")

    # Save in harness format (n, 2)
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / "oof_d15_decode_ntl_strat.npy", oof2)
    np.save(ART / "test_d15_decode_ntl_strat.npy", test2)
    print(f"  → saved oof/test_d15_decode_ntl_strat.npy")

    # Quick comparison vs e3 baseline
    e3_oof = np.load(ART / "oof_e3_hgbc_strat.npy")
    e3_oof_pos = e3_oof[:, 1] if e3_oof.ndim == 2 else e3_oof
    e3_auc = roc_auc_score(y, e3_oof_pos)
    print(f"\nReference e3_hgbc OOF AUC: {e3_auc:.5f}")
    print(f"Δ vs e3:                    {(overall_auc - e3_auc) * 1e4:+.2f} bp")


if __name__ == "__main__":
    main()
