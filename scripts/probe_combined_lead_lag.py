"""scripts/probe_combined_lead_lag.py — Probe #1 hiding-in-plain-sight.

Hypothesis: U3 says train/test split is i.i.d. row-level within (Race, Driver,
Year) sequences (97.4% of test rows have a same-(Race, Driver) successor).
AV-AUC=0.502 (Day-12 + d12 confirmation; Rule 25 PASS) ⇒ combined-set FE is
SAFE.

The team's >100 probes never compute lead/lag features over train+test
COMBINED. d13_data_probe.py:77-87 computed Compound_next, Stint_next as
diagnostics on train alone. p1_features.py:200-219 computes within-(Driver,
Race, Year) lags from train alone. d9h, d12_groupkf_rebuild, d14_h1 also
compute next_compound but only on the FM-base side, never on the K-meta
winner.

Probe 2 (target structure) found pos_rate decays monotonically from observed
stint end (27.2% at lap 0 → 7.9% at lap 7). Combined-frame stint-end
identification should sharpen `laps_until_stint_end`.

Tests:
  L1. Single-feature OOF AUC for each combined-frame lead/lag feature.
  L2. Combined-frame lap_from_actual_stint_end (uses train+test to find
      the TRUE last-observed lap per stint).
  L3. Reference baseline: small LGBM on raw 11 features alone.
  L4. Treatment: same LGBM + L1+L2 features (combined-frame).
  L5. Diagnostic: compute the same features TRAIN-ONLY and compare. The lift
      between L4 and L5 isolates the train+test combined-frame value.

Output: scripts/artifacts/probe_combined_lead_lag.json + console summary.
Cost: ~30 min CPU (single LGBM 5-fold × 2 = 2 trainings).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
OUT = ART / "probe_combined_lead_lag.json"
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbose=-1,
    n_jobs=-1,
    seed=SEED,
)

SOURCE_CAT_COLS = ["Driver", "Compound", "Race"]
CAT_COLS = ["Driver_cat", "Compound_cat", "Race_cat"]
RAW_FEATS = [
    "Driver_cat", "Compound_cat", "Race_cat", "Year",
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]


def encode_cats(*dfs: pd.DataFrame) -> None:
    """Joint factorize Driver, Compound, Race across all dfs."""
    for c in SOURCE_CAT_COLS:
        all_vals = pd.concat([d[c].astype(str) for d in dfs])
        codes, _ = all_vals.factorize()
        cuts = np.cumsum([0] + [len(d) for d in dfs])
        for i, d in enumerate(dfs):
            d[f"{c}_cat"] = codes[cuts[i]:cuts[i + 1]].astype("int32")


def add_lead_lag_features(df: pd.DataFrame, kind: str) -> list[str]:
    """Compute lead/lag features within (Race, Driver, Year) sequences.

    Mutates df in place. Returns list of feature column names added.
    `kind` is just a label for logging.
    """
    df.sort_values(["Race", "Driver", "Year", "LapNumber"], inplace=True)
    g = df.groupby(["Race", "Driver", "Year"], sort=False)

    feats = []
    # leads
    for col in ["TyreLife", "Stint", "LapNumber", "LapTime (s)",
                "LapTime_Delta", "Cumulative_Degradation", "Position",
                "Position_Change", "RaceProgress", "PitStop"]:
        nm = f"lead_{col.replace(' (s)', '')}"
        df[nm] = g[col].shift(-1)
        feats.append(nm)

    # categorical leads as "changed" indicators (encoded as 0/1/-1)
    df["lead_Compound_str"] = g["Compound"].shift(-1)
    df["lead_compound_changed"] = (
        (df["lead_Compound_str"] != df["Compound"]).astype("float32")
    )
    df.loc[df["lead_Compound_str"].isna(), "lead_compound_changed"] = -1.0
    feats.append("lead_compound_changed")
    df.drop(columns=["lead_Compound_str"], inplace=True)

    # lags
    for col in ["TyreLife", "Stint", "LapNumber", "LapTime (s)",
                "LapTime_Delta", "Cumulative_Degradation", "Position",
                "Position_Change", "RaceProgress", "PitStop"]:
        nm = f"lag_{col.replace(' (s)', '')}"
        df[nm] = g[col].shift(1)
        feats.append(nm)

    df["lag_Compound_str"] = g["Compound"].shift(1)
    df["lag_compound_changed"] = (
        (df["lag_Compound_str"] != df["Compound"]).astype("float32")
    )
    df.loc[df["lag_Compound_str"].isna(), "lag_compound_changed"] = -1.0
    feats.append("lag_compound_changed")
    df.drop(columns=["lag_Compound_str"], inplace=True)

    # derived diffs
    df["lead_TyreLife_diff"] = df["lead_TyreLife"] - df["TyreLife"]
    df["lead_Stint_diff"] = df["lead_Stint"] - df["Stint"]
    df["lead_LapNumber_diff"] = df["lead_LapNumber"] - df["LapNumber"]
    df["lag_TyreLife_diff"] = df["TyreLife"] - df["lag_TyreLife"]
    df["lag_Stint_diff"] = df["Stint"] - df["lag_Stint"]
    df["has_lead"] = df["lead_TyreLife"].notna().astype("float32")
    df["has_lag"] = df["lag_TyreLife"].notna().astype("float32")
    feats += ["lead_TyreLife_diff", "lead_Stint_diff", "lead_LapNumber_diff",
              "lag_TyreLife_diff", "lag_Stint_diff", "has_lead", "has_lag"]

    # combined-frame stint-end identification (works whether kind is combined
    # or train-only; semantics differ — that's the point of L5)
    g_stint = df.groupby(["Race", "Driver", "Year", "Stint"], sort=False)
    df["lap_in_stint_obs"] = g_stint.cumcount()
    df["stint_size_obs"] = g_stint["Stint"].transform("size")
    df["lap_from_stint_end_obs"] = (df["stint_size_obs"] - 1
                                    - df["lap_in_stint_obs"])
    feats += ["lap_in_stint_obs", "stint_size_obs", "lap_from_stint_end_obs"]

    print(f"  [{kind}] added {len(feats)} lead/lag features")
    return feats


def single_feat_auc(y: np.ndarray, x: np.ndarray, name: str) -> float:
    """Single-feature full-data AUC (replace NaN with median)."""
    med = float(np.nanmedian(x))
    xx = np.where(np.isfinite(x), x, med)
    try:
        return float(roc_auc_score(y, xx))
    except Exception:
        return float("nan")


def lgbm_oof(X: pd.DataFrame, y: np.ndarray, feats: list[str],
             cat_feats: list[str], label: str) -> dict:
    """5-fold StratKF OOF AUC."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs = []
    fold_iters = []
    Xf = X[feats]
    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        dtrain = lgb.Dataset(Xf.iloc[tr_idx], label=y[tr_idx],
                             categorical_feature=cat_feats)
        dval = lgb.Dataset(Xf.iloc[va_idx], label=y[va_idx],
                           categorical_feature=cat_feats, reference=dtrain)
        m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(Xf.iloc[va_idx])
        fa = roc_auc_score(y[va_idx], oof[va_idx])
        fold_aucs.append(float(fa))
        fold_iters.append(int(m.best_iteration or 0))
        print(f"  [{label}] fold {k}: AUC={fa:.5f}  best_iter={m.best_iteration}  ({time.time() - t0:.0f}s)")
    overall = float(roc_auc_score(y, oof))
    print(f"  [{label}] OOF AUC = {overall:.5f}  fold_std = {np.std(fold_aucs):.5f}")
    return {
        "oof_auc": overall,
        "fold_aucs": fold_aucs,
        "fold_std": float(np.std(fold_aucs)),
        "best_iters": fold_iters,
        "feature_count": len(feats),
        "wall_sec": float(time.time() - t0),
    }


def main() -> None:
    print("Loading train + test...")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    print(f"  train: {tr.shape}  test: {te.shape}")
    y = tr[TARGET].astype(int).to_numpy()

    out: dict = {}
    out["row_count_train"] = int(len(tr))
    out["row_count_test"] = int(len(te))

    # ===== Encode categoricals jointly across train+test ===============
    encode_cats(tr, te)

    # ===== Combined-frame lead/lag (the point of this probe) ===========
    print("\n[L1+L2] Computing COMBINED-FRAME lead/lag...")
    tr_c = tr.copy()
    te_c = te.copy()
    combined = pd.concat([tr_c.assign(_split="tr"),
                          te_c.assign(_split="te")], ignore_index=True)
    feats_c = add_lead_lag_features(combined, "combined")
    # Sort by id to recover original row order, then split back
    tr_combined = (combined[combined["_split"] == "tr"]
                   .sort_values("id").reset_index(drop=True))
    # The original tr is sorted by id (Kaggle convention); rejoin so y matches.
    tr_sorted_orig = tr.sort_values("id").reset_index(drop=True)
    assert (tr_combined["id"].values == tr_sorted_orig["id"].values).all()
    y_sorted = tr_sorted_orig[TARGET].astype(int).to_numpy()

    # ===== Train-only lead/lag (for L5 comparison) =====================
    print("\n[L5] Computing TRAIN-ONLY lead/lag for comparison...")
    tr_only = tr.copy()
    feats_t = add_lead_lag_features(tr_only, "train_only")
    tr_only = tr_only.sort_values("id").reset_index(drop=True)

    # ===== L1 single-feature AUC table (combined frame) ================
    print("\n[L1] Single-feature AUC (combined frame)...")
    saucs = {}
    for f in feats_c:
        x = tr_combined[f].to_numpy(dtype=np.float64, na_value=np.nan)
        saucs[f] = single_feat_auc(y_sorted, x, f)
    saucs_train_only = {}
    for f in feats_t:
        x = tr_only[f].to_numpy(dtype=np.float64, na_value=np.nan)
        saucs_train_only[f] = single_feat_auc(y_sorted, x, f)
    out["L1_single_feat_auc_combined"] = saucs
    out["L1_single_feat_auc_train_only"] = saucs_train_only

    print(f"\n  Top 10 combined single-feat AUCs:")
    for f, a in sorted(saucs.items(), key=lambda kv: -abs(kv[1] - 0.5))[:10]:
        a_t = saucs_train_only.get(f, float("nan"))
        print(f"    {f:35} combined={a:.4f}  train_only={a_t:.4f}  Δ={a - a_t:+.4f}")

    # ===== L3 reference LGBM on raw features ===========================
    print("\n[L3] LGBM on raw 14 features (reference)...")
    out["L3_lgbm_raw"] = lgbm_oof(tr_sorted_orig, y_sorted,
                                  RAW_FEATS, CAT_COLS, "L3_raw")

    # ===== L4 LGBM with combined-frame lead/lag features ===============
    print("\n[L4] LGBM raw + combined-frame lead/lag features...")
    feats_L4 = RAW_FEATS + feats_c
    # Use tr_combined as the X frame (already aligned to id-sorted order)
    out["L4_lgbm_with_combined_lead_lag"] = lgbm_oof(
        tr_combined, y_sorted, feats_L4, CAT_COLS, "L4_combined")

    # ===== L5 LGBM with train-only lead/lag features ===================
    print("\n[L5] LGBM raw + TRAIN-ONLY lead/lag (control)...")
    feats_L5 = RAW_FEATS + feats_t
    out["L5_lgbm_with_train_only_lead_lag"] = lgbm_oof(
        tr_only, y_sorted, feats_L5, CAT_COLS, "L5_train_only")

    # ===== Summary ====================================================
    raw_auc = out["L3_lgbm_raw"]["oof_auc"]
    cb_auc = out["L4_lgbm_with_combined_lead_lag"]["oof_auc"]
    to_auc = out["L5_lgbm_with_train_only_lead_lag"]["oof_auc"]
    out["summary"] = {
        "L4_minus_L3_bp": (cb_auc - raw_auc) * 1e4,    # combined - raw
        "L5_minus_L3_bp": (to_auc - raw_auc) * 1e4,    # train_only - raw
        "L4_minus_L5_bp": (cb_auc - to_auc) * 1e4,     # combined - train_only
        "interpretation": (
            "L4 - L5 isolates the combined-frame value (Rule 25 PASS via AV "
            "0.502). L4 - L3 is the full lead/lag feature lift over raw."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print(f"\n=== HEADLINE ===")
    print(f"  L3 raw OOF AUC          : {raw_auc:.5f}")
    print(f"  L4 raw + combined L/L  : {cb_auc:.5f}  Δ vs L3 = {(cb_auc - raw_auc) * 1e4:+.2f} bp")
    print(f"  L5 raw + train-only L/L: {to_auc:.5f}  Δ vs L3 = {(to_auc - raw_auc) * 1e4:+.2f} bp")
    print(f"  L4 - L5 (combined value): {(cb_auc - to_auc) * 1e4:+.2f} bp")


if __name__ == "__main__":
    main()
