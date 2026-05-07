"""scripts/d15_leak_lookup.py — leaked-posterior lookup features.

Different mechanism from d15_orig_transfer: instead of training a model
on the original and transferring to synth, we use the original DIRECTLY
as a per-row lookup table for `P(PitNextLap | feature value)`.

Key observation (from d15 fingerprint analysis):
  - 97.55% of synth `LapTime` values exist in original (drawn from
    its empirical marginal — CTGAN/CopulaGAN signature)
  - 94.98% of synth `LapTime_Delta` values exist
  - 99.95% of synth `RaceProgress` values exist
  - 87.38% of synth `Cumulative_Degradation` values exist
  - 100%   of synth `TyreLife`, `Compound`, `Race`, `Year` overlap

The synth literally drew per-feature values from the original. So
for each synth row we can compute `mean(PitNextLap | feature == v)`
on the original — a leaked posterior conditional on what the
synthesizer kept of the marginal structure.

Features built (all from the original, applied to synth):
  Univariate (smoothed empirical-Bayes with prior = global pos rate):
    leak_lt       : mean PitNextLap | LapTime
    leak_ld       : mean PitNextLap | LapTime_Delta (binned to 0.1s)
    leak_rp       : mean PitNextLap | RaceProgress (binned to 0.005)
    leak_cd       : mean PitNextLap | Cumulative_Degradation (binned)
    leak_tl       : mean PitNextLap | TyreLife (integer)
    leak_pos      : mean PitNextLap | Position (integer)
  Bivariate (sparser but more specific; key existence ≈ 5-15%):
    leak_lt_tl    : mean PitNextLap | (LapTime_round, TyreLife)
    leak_tl_cmp   : mean PitNextLap | (TyreLife, Compound)
    leak_rp_cmp   : mean PitNextLap | (RaceProgress_bin, Compound)

Smoothing: EB with α = n / (n + tau), tau=10. For sparse joint keys we
use tau=20.

Train LGBM with these 9 leak features + the standard synth feature set.
Save oof/test as d15_leak_lookup_strat.npy. Then probe via min-meta
and hier-meta(K=22 add).
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


def make_lookup(orig_df, key_cols, prior, tau=10):
    """EB-smoothed mean target per key-tuple from original."""
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    g = orig_df.groupby(key_cols)[TARGET].agg(['sum', 'count'])
    g['mean_smoothed'] = (g['sum'] + tau * prior) / (g['count'] + tau)
    return g['mean_smoothed']


def apply_lookup(synth_df, lookup, key_cols, prior):
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    if len(key_cols) == 1:
        return synth_df[key_cols[0]].map(lookup).fillna(prior).astype(np.float32)
    else:
        idx = pd.MultiIndex.from_arrays([synth_df[c] for c in key_cols])
        return pd.Series(lookup.reindex(idx).fillna(prior).values,
                         index=synth_df.index, dtype=np.float32)


def main():
    print("Loading data...")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    y = tr[TARGET].astype(int).values

    prior = float(orig[TARGET].mean())
    print(f"  orig prior P(PitNextLap=1) = {prior:.4f}")

    # === Build binned/rounded keys on both orig and synth (must match) ===
    for df in [tr, te, orig]:
        df["lt_r"] = df["LapTime (s)"].round(2).astype(np.float32)
        df["ld_b"] = (df["LapTime_Delta"] / 0.1).round().astype(np.int32)
        df["rp_b"] = (df["RaceProgress"] / 0.005).round().astype(np.int32)
        df["cd_b"] = (df["Cumulative_Degradation"] / 0.5).round().astype(np.int32)
        df["tl"] = df["TyreLife"].astype(np.int32)

    # === Build lookup tables from ORIGINAL ===
    print("\nBuilding leak-lookup tables on original...")
    lookups = {}
    # Univariate
    lookups["leak_lt"]      = make_lookup(orig, "lt_r", prior, tau=10)
    lookups["leak_ld"]      = make_lookup(orig, "ld_b", prior, tau=10)
    lookups["leak_rp"]      = make_lookup(orig, "rp_b", prior, tau=10)
    lookups["leak_cd"]      = make_lookup(orig, "cd_b", prior, tau=10)
    lookups["leak_tl"]      = make_lookup(orig, "tl", prior, tau=10)
    lookups["leak_pos"]     = make_lookup(orig, "Position", prior, tau=10)
    lookups["leak_lap"]     = make_lookup(orig, "LapNumber", prior, tau=10)
    lookups["leak_stint"]   = make_lookup(orig, "Stint", prior, tau=10)
    lookups["leak_compound"]= make_lookup(orig, "Compound", prior, tau=10)
    lookups["leak_race"]    = make_lookup(orig, "Race", prior, tau=10)
    # Bivariate
    lookups["leak_lt_tl"]   = make_lookup(orig, ["lt_r", "tl"], prior, tau=20)
    lookups["leak_tl_cmp"]  = make_lookup(orig, ["tl", "Compound"], prior, tau=20)
    lookups["leak_rp_cmp"]  = make_lookup(orig, ["rp_b", "Compound"], prior, tau=20)
    lookups["leak_lt_cmp"]  = make_lookup(orig, ["lt_r", "Compound"], prior, tau=20)
    lookups["leak_rp_stint"]= make_lookup(orig, ["rp_b", "Stint"], prior, tau=20)
    # Trivariate (most specific)
    lookups["leak_tl_cmp_stint"] = make_lookup(orig, ["tl", "Compound", "Stint"], prior, tau=30)

    print("  lookup table sizes:")
    for name, lk in lookups.items():
        print(f"    {name:<22s}  n_keys={len(lk):>7d}  mean={lk.mean():.4f}")

    # === Apply lookups to synth train + test ===
    print("\nApplying lookups to synth...")
    keymap = {
        "leak_lt":  "lt_r", "leak_ld": "ld_b", "leak_rp": "rp_b",
        "leak_cd":  "cd_b", "leak_tl": "tl", "leak_pos": "Position",
        "leak_lap": "LapNumber", "leak_stint": "Stint",
        "leak_compound": "Compound", "leak_race": "Race",
        "leak_lt_tl": ["lt_r", "tl"], "leak_tl_cmp": ["tl", "Compound"],
        "leak_rp_cmp": ["rp_b", "Compound"], "leak_lt_cmp": ["lt_r", "Compound"],
        "leak_rp_stint": ["rp_b", "Stint"],
        "leak_tl_cmp_stint": ["tl", "Compound", "Stint"],
    }
    leak_cols = list(lookups.keys())
    for name in leak_cols:
        tr[name] = apply_lookup(tr, lookups[name], keymap[name], prior)
        te[name] = apply_lookup(te, lookups[name], keymap[name], prior)

    # Sanity: leak feature AUC against synth target
    print("\nLeak feature standalone AUC (synth train):")
    for name in leak_cols:
        auc = roc_auc_score(y, tr[name].values)
        print(f"  {name:<22s} AUC={auc:.4f}")

    # === LGBM with leak features + standard synth features ===
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
        te[c] = te[c].cat.set_categories(tr[c].cat.categories)

    feature_cols = [
        # Standard synth features
        "Driver", "Compound", "Race", "Year",
        "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
    ] + leak_cols

    print(f"\nTotal features: {len(feature_cols)} ({len(leak_cols)} leak)")

    X = tr[feature_cols]; Xte = te[feature_cols]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr), dtype=np.float64)
    test_pred = np.zeros(len(te), dtype=np.float64)

    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, num_leaves=127, max_depth=-1,
        min_data_in_leaf=200, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=5,
        verbose=-1, n_jobs=-1, seed=SEED,
    )

    t0 = time.time()
    feat_imp = None
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                          categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx],
                          categorical_feature=cat_cols, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        test_pred += m.predict(Xte) / N_FOLDS
        if feat_imp is None:
            feat_imp = pd.DataFrame({"feat": X.columns,
                                     "gain": m.feature_importance("gain")})
        else:
            feat_imp["gain"] += m.feature_importance("gain")
        fa = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"  fold {k}: AUC={fa:.5f}  best_iter={m.best_iteration}  ({time.time()-t0:.0f}s)")

    auc = roc_auc_score(y, oof)
    print(f"\n=== d15_leak_lookup OOF AUC: {auc:.5f} ===")

    # Save
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / "oof_d15_leak_lookup_strat.npy", oof2)
    np.save(ART / "test_d15_leak_lookup_strat.npy", test2)
    print(f"  → saved oof/test_d15_leak_lookup_strat.npy")

    # ρ vs PRIMARY
    primary = np.load(ART / "test_d13e_compound_stint_tau20000_strat.npy")[:, 1]
    from scipy.stats import spearmanr
    rho, _ = spearmanr(test_pred, primary)
    print(f"  ρ vs PRIMARY: {rho:.5f}")

    # Reference baselines
    e3 = np.load(ART / "oof_e3_hgbc_strat.npy")
    e3_pos = e3[:, 1] if e3.ndim == 2 else e3
    print(f"\n  e3_hgbc OOF: {roc_auc_score(y, e3_pos):.5f}")
    print(f"  Δ vs e3:     {(auc - roc_auc_score(y, e3_pos))*1e4:+.2f}bp")

    # Top features
    feat_imp = feat_imp.sort_values("gain", ascending=False).head(20)
    print(f"\n  Top 20 features by gain:")
    print(feat_imp.to_string(index=False))


if __name__ == "__main__":
    main()
