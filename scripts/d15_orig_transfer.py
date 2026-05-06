"""scripts/d15_orig_transfer.py — Lens 2: train on original, predict synth.

Discovery: synth `LapTime (s)` matches the original's empirical
distribution at 97.55% (synth) / 97.59% (test); LapTime_Delta at 95%;
Cumulative_Degradation at 87%; RaceProgress at 99.95%. The
synthesizer is sampling near-marginally from original.

This base trains an LGBM purely on the original 101k-row dataset and
predicts on synth train+test. By construction:
  - It cannot leak across synth folds (the model never saw synth labels)
  - It carries the *true DGP signal* from the original
  - It is orthogonal to every existing synth-trained base

Saves oof_/test_d15_orig_transfer_strat.npy in harness format. The
"OOF" here is just the model's prediction on each synth-train row
(no folding needed — the model is fit ONCE on the original, since the
training set is disjoint from synth).
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET = "PitNextLap"


def main():
    print("Loading data...")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    y_synth = tr[TARGET].astype(int).values
    y_orig = orig[TARGET].astype(int).values

    print(f"  synth train: {tr.shape}, synth test: {te.shape}, orig: {orig.shape}")
    print(f"  orig pos rate: {y_orig.mean():.4f}, synth pos rate: {y_synth.mean():.4f}")

    # === Filter original to rows with valid Compound (drop nan, drop Pre-Season) ===
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    print(f"  orig after filter: {orig.shape}")

    # === Feature set (must be present in both orig and synth) ===
    cat_cols = ["Driver", "Compound", "Race"]
    num_cols = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
                "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                "RaceProgress", "Position_Change",
                # Original-only features:
                "Normalized_TyreLife"]

    # For synth, we don't have Normalized_TyreLife. Compute it via the discovered
    # formula: TyreLife / max(TyreLife) within (Driver, Race, Year, Stint).
    for df in [tr, te]:
        g = df.groupby(["Driver", "Race", "Year", "Stint"])["TyreLife"].transform("max")
        df["Normalized_TyreLife"] = df["TyreLife"] / g.clip(lower=1)

    feat_cols = cat_cols + num_cols
    print(f"  features ({len(feat_cols)}): {feat_cols}")

    # === Categorical alignment: union of categories from synth+orig (synth dominates Driver universe) ===
    # Train cat dtypes use union for proper code mapping.
    for c in cat_cols:
        union = pd.concat([tr[c], te[c], orig[c]], axis=0).astype(str)
        cats = sorted(union.unique())
        for df in [tr, te, orig]:
            df[c] = pd.Categorical(df[c].astype(str), categories=cats)

    X_orig = orig[feat_cols]
    X_tr = tr[feat_cols]
    X_te = te[feat_cols]
    y_orig = orig[TARGET].astype(int).values

    print(f"  X_orig: {X_orig.shape}, X_tr: {X_tr.shape}, X_te: {X_te.shape}")
    print(f"  y_orig pos rate: {y_orig.mean():.4f}")

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.04,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=100,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        lambda_l1=0.0,
        lambda_l2=0.0,
        verbose=-1,
        n_jobs=-1,
        seed=SEED,
    )

    # === Train on original with internal early-stop using a held-out 15% of original ===
    from sklearn.model_selection import train_test_split
    Xo_tr, Xo_va, yo_tr, yo_va = train_test_split(
        X_orig, y_orig, test_size=0.15, random_state=SEED, stratify=y_orig)

    dtrain = lgb.Dataset(Xo_tr, label=yo_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(Xo_va, label=yo_va, categorical_feature=cat_cols, reference=dtrain)

    t0 = time.time()
    model = lgb.train(params, dtrain, num_boost_round=3000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(0)])
    print(f"  train wall: {time.time() - t0:.0f}s, best_iter: {model.best_iteration}")
    print(f"  orig held-out AUC: {model.best_score['valid_0']['auc']:.5f}")

    # === Predict on synth train + test ===
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    # The "OOF" here is just predict-on-train (no folding because train data is orig).
    # This is leakage-free w.r.t. synth folds.
    auc_synth = roc_auc_score(y_synth, pred_tr)
    print(f"\n=== ORIG-trained LGBM, synth-train AUC: {auc_synth:.5f} ===")

    # Save in harness format
    oof2 = np.column_stack([1 - pred_tr, pred_tr])
    test2 = np.column_stack([1 - pred_te, pred_te])
    np.save(ART / "oof_d15_orig_transfer_strat.npy", oof2)
    np.save(ART / "test_d15_orig_transfer_strat.npy", test2)
    print(f"  → saved oof/test_d15_orig_transfer_strat.npy")

    # Reference baselines
    e3_oof = np.load(ART / "oof_e3_hgbc_strat.npy")
    e3_oof_pos = e3_oof[:, 1] if e3_oof.ndim == 2 else e3_oof
    e3_auc = roc_auc_score(y_synth, e3_oof_pos)
    print(f"\n  Reference e3_hgbc OOF AUC: {e3_auc:.5f}")
    print(f"  Δ orig_transfer vs e3:     {(auc_synth - e3_auc) * 1e4:+.2f} bp")

    # ρ vs PRIMARY
    primary = np.load(ART / "test_d13e_compound_stint_tau20000_strat.npy")
    primary_pos = primary[:, 1] if primary.ndim == 2 else primary
    from scipy.stats import spearmanr
    rho, _ = spearmanr(pred_te, primary_pos)
    print(f"  ρ(test) vs PRIMARY: {rho:.5f}")

    # Feature importance
    imp = pd.DataFrame({
        "feat": X_orig.columns,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    print(f"\n  Feature importance (gain):")
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
