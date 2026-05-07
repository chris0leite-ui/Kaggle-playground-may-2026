"""scripts/probe_field_state_strict.py — Strict fold-safe re-run of probe 4.

Purpose: defend against the Day-17 `target-construction-layer-leakage`
family. Probe 4's `fs_cum_pits` etc. are aggregates over `PitStop`, a
FEATURE column (single-feat AUC 0.521 ≈ chance vs PitNextLap, per U2).
NOT aggregates over the label. So Rule 24 doesn't strictly apply.

But the user flag is appropriate: even feature-derived aggregates can
encode val-fold information when computed on the full train. This
script does the strict per-fold re-fit:

  For each CV fold:
    Compute field-state aggregates from training rows of that fold ONLY
    (not full train, not train+test). Merge into both train and val
    rows. Train LGBM on tr-fold features. Predict val-fold.

  F3-strict vs F3:
    If similar (within fold_std 0.00058), no fold-safety leak.
    If F3-strict << F3 by >5 bp, same family as Day-17 leakage —
    discard.

Cost: ~5-7 min CPU (5 folds × 1 LGBM each).
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
OUT = ART / "probe_field_state_strict.json"
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"

LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    verbose=-1, n_jobs=-1, seed=SEED,
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
    for c in SOURCE_CAT_COLS:
        all_vals = pd.concat([d[c].astype(str) for d in dfs])
        codes, _ = all_vals.factorize()
        cuts = np.cumsum([0] + [len(d) for d in dfs])
        for i, d in enumerate(dfs):
            d[f"{c}_cat"] = codes[cuts[i]:cuts[i + 1]].astype("int32")


FS_FEATS = [
    "fs_field_size", "fs_n_pitting_now", "fs_pit_rate_now",
    "fs_mean_TyreLife", "fs_max_TyreLife", "fs_min_TyreLife",
    "fs_std_TyreLife", "fs_mean_Stint", "fs_max_Stint",
    "fs_mean_Position", "fs_mean_LapTime", "fs_mean_RaceProgress",
    "fs_cum_pits", "fs_cum_pit_lap_count", "fs_cum_pit_rate",
    "fs_compound_n", "fs_compound_n_pitting", "fs_compound_pit_rate",
    "fs_compound_mean_TyreLife", "fs_compound_max_TyreLife",
    "fs_TyreLife_vs_field_mean", "fs_TyreLife_vs_field_max",
    "fs_Position_vs_field_mean", "fs_Stint_vs_field_mean",
]


def build_field_state(source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-(R,Y,L) and per-(R,Y,L,Compound) lookup tables."""
    g = source_df.groupby(["Race", "Year", "LapNumber"])
    a = g.agg(
        fs_field_size=("id", "size"),
        fs_n_pitting_now=("PitStop", "sum"),
        fs_pit_rate_now=("PitStop", "mean"),
        fs_mean_TyreLife=("TyreLife", "mean"),
        fs_max_TyreLife=("TyreLife", "max"),
        fs_min_TyreLife=("TyreLife", "min"),
        fs_std_TyreLife=("TyreLife", "std"),
        fs_mean_Stint=("Stint", "mean"),
        fs_max_Stint=("Stint", "max"),
        fs_mean_Position=("Position", "mean"),
        fs_mean_LapTime=("LapTime (s)", "mean"),
        fs_mean_RaceProgress=("RaceProgress", "mean"),
    ).reset_index()
    rs = (source_df.sort_values(["Race", "Year", "LapNumber"])
                  .groupby(["Race", "Year", "LapNumber"])["PitStop"]
                  .sum().reset_index())
    rs["fs_cum_pits"] = rs.groupby(["Race", "Year"])["PitStop"].cumsum()
    rs["fs_cum_pit_lap_count"] = (rs.groupby(["Race", "Year"])["PitStop"]
                                    .cumcount() + 1)
    rs["fs_cum_pit_rate"] = rs["fs_cum_pits"] / rs["fs_cum_pit_lap_count"]
    a = a.merge(
        rs[["Race", "Year", "LapNumber",
            "fs_cum_pits", "fs_cum_pit_lap_count", "fs_cum_pit_rate"]],
        on=["Race", "Year", "LapNumber"], how="left"
    )
    gc = source_df.groupby(["Race", "Year", "LapNumber", "Compound"])
    ac = gc.agg(
        fs_compound_n=("id", "size"),
        fs_compound_n_pitting=("PitStop", "sum"),
        fs_compound_pit_rate=("PitStop", "mean"),
        fs_compound_mean_TyreLife=("TyreLife", "mean"),
        fs_compound_max_TyreLife=("TyreLife", "max"),
    ).reset_index()
    return a, ac


def merge_field_state(df: pd.DataFrame, a: pd.DataFrame,
                      ac: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(a, on=["Race", "Year", "LapNumber"], how="left")
    out = out.merge(ac, on=["Race", "Year", "LapNumber", "Compound"],
                    how="left")
    out["fs_TyreLife_vs_field_mean"] = (out["TyreLife"]
                                        - out["fs_mean_TyreLife"])
    out["fs_TyreLife_vs_field_max"] = (out["TyreLife"]
                                       - out["fs_max_TyreLife"])
    out["fs_Position_vs_field_mean"] = (out["Position"]
                                        - out["fs_mean_Position"])
    out["fs_Stint_vs_field_mean"] = (out["Stint"]
                                     - out["fs_mean_Stint"])
    return out


def lgbm_5fold_strict(tr: pd.DataFrame, te: pd.DataFrame,
                      y: np.ndarray, label: str,
                      use_test_in_aggregate: bool) -> dict:
    """Strict per-fold field-state aggregate computation."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    feats_full = RAW_FEATS + FS_FEATS
    oof = np.zeros(len(y))
    fold_aucs = []
    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr_in_fold = tr.iloc[tr_idx].copy()
        va_rows = tr.iloc[va_idx].copy()

        # Build aggregates from THIS FOLD'S training rows only
        # (optionally also include test rows since they have PitStop column)
        if use_test_in_aggregate:
            source = pd.concat([tr_in_fold, te], ignore_index=True)
        else:
            source = tr_in_fold
        a, ac = build_field_state(source)

        tr_in_fold_fs = merge_field_state(tr_in_fold, a, ac)
        va_rows_fs = merge_field_state(va_rows, a, ac)

        dtrain = lgb.Dataset(tr_in_fold_fs[feats_full],
                             label=tr_in_fold_fs[TARGET].astype(int).values,
                             categorical_feature=CAT_COLS)
        dval = lgb.Dataset(va_rows_fs[feats_full],
                           label=va_rows_fs[TARGET].astype(int).values,
                           categorical_feature=CAT_COLS, reference=dtrain)
        m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(va_rows_fs[feats_full])
        fa = roc_auc_score(y[va_idx], oof[va_idx])
        fold_aucs.append(float(fa))
        print(f"  [{label}] fold {k}: AUC={fa:.5f}  best_iter={m.best_iteration}  ({time.time() - t0:.0f}s)")

    overall = float(roc_auc_score(y, oof))
    print(f"  [{label}] OOF AUC = {overall:.5f}  fold_std = {np.std(fold_aucs):.5f}")
    return {
        "oof_auc": overall,
        "fold_aucs": fold_aucs,
        "fold_std": float(np.std(fold_aucs)),
        "wall_sec": float(time.time() - t0),
    }


def main() -> None:
    print("Loading data...")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    encode_cats(tr, te)
    y = tr[TARGET].astype(int).to_numpy()

    out: dict = {
        "F3_full_train_OOF": 0.94230,  # from probe 4 reference
        "F4_full_train_only_OOF": 0.94241,
        "F2_raw_OOF": 0.94074,
    }

    # ===== F3-strict: per-fold aggregates from tr_fold + test ===========
    print("\n[F3-strict] Per-fold field-state from tr_fold + test...")
    out["F3_strict_combined"] = lgbm_5fold_strict(
        tr, te, y, "F3_strict_combined", use_test_in_aggregate=True)

    # ===== F4-strict: per-fold aggregates from tr_fold only ===========
    print("\n[F4-strict] Per-fold field-state from tr_fold ONLY...")
    out["F4_strict_train_only"] = lgbm_5fold_strict(
        tr, te, y, "F4_strict_train_only", use_test_in_aggregate=False)

    f3s = out["F3_strict_combined"]["oof_auc"]
    f4s = out["F4_strict_train_only"]["oof_auc"]
    out["summary"] = {
        "F3_strict_minus_F2_raw_bp": (f3s - 0.94074) * 1e4,
        "F4_strict_minus_F2_raw_bp": (f4s - 0.94074) * 1e4,
        "F3_strict_minus_F3_full_bp": (f3s - 0.94230) * 1e4,
        "F4_strict_minus_F4_full_bp": (f4s - 0.94241) * 1e4,
        "F3_strict_minus_F4_strict_bp": (f3s - f4s) * 1e4,
        "verdict": (
            "If |F3_strict - F3_full| < 5 bp AND F3_strict - F2_raw > 5 bp, "
            "field-state lift is fold-safe-real. If F3_strict << F3_full or "
            "F3_strict ≈ F2_raw, same family as Day-17 leakage — discard."
        ),
    }

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print(f"\n=== HEADLINE ===")
    print(f"  F2 raw (reference)          : 0.94074")
    print(f"  F3 full-train combined      : 0.94230  (probe 4)")
    print(f"  F4 full-train train-only    : 0.94241  (probe 4)")
    print(f"  F3 strict per-fold combined : {f3s:.5f}  Δ vs full = {(f3s - 0.94230) * 1e4:+.2f} bp")
    print(f"  F4 strict per-fold tr-only  : {f4s:.5f}  Δ vs full = {(f4s - 0.94241) * 1e4:+.2f} bp")
    print(f"  F3 strict - F2 raw          : {(f3s - 0.94074) * 1e4:+.2f} bp")
    print(f"  F4 strict - F2 raw          : {(f4s - 0.94074) * 1e4:+.2f} bp")
    if abs((f3s - 0.94230) * 1e4) < 5 and (f3s - 0.94074) * 1e4 > 5:
        print(f"\n  VERDICT: FOLD-SAFE-REAL ✓")
    elif (f3s - 0.94074) * 1e4 < 3:
        print(f"\n  VERDICT: LEAKY ✗ — same family as Day-17 target-reform")
    else:
        print(f"\n  VERDICT: PARTIAL — {(f3s - 0.94074) * 1e4:+.2f} bp survives strict")


if __name__ == "__main__":
    main()
