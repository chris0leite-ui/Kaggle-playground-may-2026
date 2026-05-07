"""scripts/probe_field_state.py — Probe #4: field-state aggregates.

Hypothesis: pit decisions in F1 are partly herd-conditional (undercut /
overcut strategy). Currently NO base in K=23 has access to "how many
*other* drivers in this race-lap are pitting". This is computable from
train+test combined as an aggregate over OTHER rows (PitStop is a
column, not the label; AV-AUC=0.502 ⇒ Rule 25 PASS).

The 3-probe pass earlier today closed own-row sequence/feature axes:
single-row lead/lag premium evaporates at GBDT (-0.36 bp), NTL caps at
0.687 single-rule, target structure noisy. What survives: aggregates
over OTHER rows in the same race-lap context that the GBDT cannot
reconstruct from a single row's features.

Tests:
  F1.  Single-feature OOF AUC for each field-state aggregate (no model).
  F2.  L3 reference: LGBM on raw 14 features.
  F3.  L4: LGBM on raw + field-state aggregates.
  F4.  Diagnostic: build the same aggregates train-only and compare to
       isolate the combined-frame value.

Output: scripts/artifacts/probe_field_state.json + console summary.
Cost: ~10 min CPU (2 LGBM 5-fold runs + aggregate construction).
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
OUT = ART / "probe_field_state.json"
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
    for c in SOURCE_CAT_COLS:
        all_vals = pd.concat([d[c].astype(str) for d in dfs])
        codes, _ = all_vals.factorize()
        cuts = np.cumsum([0] + [len(d) for d in dfs])
        for i, d in enumerate(dfs):
            d[f"{c}_cat"] = codes[cuts[i]:cuts[i + 1]].astype("int32")


def add_field_state(df: pd.DataFrame, source_df: pd.DataFrame,
                    label: str) -> list[str]:
    """Compute field-state aggregates from `source_df` and merge into `df`.

    Aggregates are GROUP-conditional, NOT row-derived. They use other rows'
    feature values (PitStop, TyreLife, ...) so are AV-safe.

    Returns the list of feature columns added.
    """
    feats: list[str] = []

    # ===== per (Race, Year, LapNumber) =====================================
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
    df_out = df.merge(a, on=["Race", "Year", "LapNumber"], how="left")
    feats.extend([c for c in a.columns if c.startswith("fs_")])

    # ===== per (Race, Year) — cumulative pitting up to and including lap ===
    rs = (source_df.sort_values(["Race", "Year", "LapNumber"])
                  .groupby(["Race", "Year", "LapNumber"])["PitStop"]
                  .sum().reset_index())
    rs["fs_cum_pits"] = rs.groupby(["Race", "Year"])["PitStop"].cumsum()
    rs["fs_cum_pit_lap_count"] = (rs.groupby(["Race", "Year"])["PitStop"]
                                    .cumcount() + 1)
    rs["fs_cum_pit_rate"] = rs["fs_cum_pits"] / rs["fs_cum_pit_lap_count"]
    df_out = df_out.merge(
        rs[["Race", "Year", "LapNumber",
            "fs_cum_pits", "fs_cum_pit_lap_count", "fs_cum_pit_rate"]],
        on=["Race", "Year", "LapNumber"], how="left",
    )
    feats.extend(["fs_cum_pits", "fs_cum_pit_lap_count", "fs_cum_pit_rate"])

    # ===== per (Race, Year, LapNumber, Compound) ===========================
    gc = source_df.groupby(["Race", "Year", "LapNumber", "Compound"])
    ac = gc.agg(
        fs_compound_n=("id", "size"),
        fs_compound_n_pitting=("PitStop", "sum"),
        fs_compound_pit_rate=("PitStop", "mean"),
        fs_compound_mean_TyreLife=("TyreLife", "mean"),
        fs_compound_max_TyreLife=("TyreLife", "max"),
    ).reset_index()
    df_out = df_out.merge(
        ac, on=["Race", "Year", "LapNumber", "Compound"], how="left"
    )
    feats.extend([c for c in ac.columns if c.startswith("fs_compound_")])

    # ===== relative-to-field row features ================================
    # row's own value vs field
    df_out["fs_TyreLife_vs_field_mean"] = (df_out["TyreLife"]
                                           - df_out["fs_mean_TyreLife"])
    df_out["fs_TyreLife_vs_field_max"] = (df_out["TyreLife"]
                                          - df_out["fs_max_TyreLife"])
    df_out["fs_Position_vs_field_mean"] = (df_out["Position"]
                                           - df_out["fs_mean_Position"])
    df_out["fs_Stint_vs_field_mean"] = (df_out["Stint"]
                                        - df_out["fs_mean_Stint"])
    feats.extend(["fs_TyreLife_vs_field_mean", "fs_TyreLife_vs_field_max",
                  "fs_Position_vs_field_mean", "fs_Stint_vs_field_mean"])

    print(f"  [{label}] added {len(feats)} field-state features")
    return df_out, feats


def single_feat_auc(y: np.ndarray, x: np.ndarray) -> float:
    med = float(np.nanmedian(x))
    xx = np.where(np.isfinite(x), x, med)
    try:
        return float(roc_auc_score(y, xx))
    except Exception:
        return float("nan")


def lgbm_oof(X: pd.DataFrame, y: np.ndarray, feats: list[str],
             cat_feats: list[str], label: str) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs, fold_iters = [], []
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

    encode_cats(tr, te)
    y = tr.sort_values("id")[TARGET].astype(int).to_numpy()
    tr_id_sorted = tr.sort_values("id").reset_index(drop=True)

    out: dict = {}

    # ===== Combined-frame field-state (the candidate) ==================
    print("\n[F1] Computing COMBINED-FRAME field-state aggregates...")
    combined = pd.concat([tr.assign(_split="tr"),
                          te.assign(_split="te")], ignore_index=True)
    tr_with_fs_combined, feats_c = add_field_state(
        tr_id_sorted, combined, "combined")
    # Sort tr to match y order (id-sorted)
    tr_with_fs_combined = (tr_with_fs_combined.sort_values("id")
                           .reset_index(drop=True))
    assert (tr_with_fs_combined["id"].values
            == tr_id_sorted["id"].values).all()

    # ===== Train-only field-state (control) ============================
    print("\n[F4] Computing TRAIN-ONLY field-state aggregates (control)...")
    tr_with_fs_trainonly, feats_t = add_field_state(
        tr_id_sorted, tr, "train_only")
    tr_with_fs_trainonly = (tr_with_fs_trainonly.sort_values("id")
                            .reset_index(drop=True))

    # ===== F1 single-feature AUC ======================================
    print("\n[F1] Single-feature AUC (combined frame)...")
    saucs_c = {f: single_feat_auc(
        y, tr_with_fs_combined[f].to_numpy(np.float64, na_value=np.nan)
    ) for f in feats_c}
    saucs_t = {f: single_feat_auc(
        y, tr_with_fs_trainonly[f].to_numpy(np.float64, na_value=np.nan)
    ) for f in feats_t}
    out["F1_single_feat_auc_combined"] = saucs_c
    out["F1_single_feat_auc_train_only"] = saucs_t

    print(f"\n  Top 12 combined single-feat AUCs:")
    for f, a in sorted(saucs_c.items(),
                       key=lambda kv: -abs(kv[1] - 0.5))[:12]:
        a_t = saucs_t.get(f, float("nan"))
        print(f"    {f:38} combined={a:.4f}  train_only={a_t:.4f}  Δ={a - a_t:+.4f}")

    # ===== F2 reference LGBM on raw ====================================
    print("\n[F2] LGBM on raw 14 features (reference)...")
    out["F2_lgbm_raw"] = lgbm_oof(
        tr_id_sorted, y, RAW_FEATS, CAT_COLS, "F2_raw")

    # ===== F3 LGBM with combined-frame field-state =====================
    print("\n[F3] LGBM raw + combined-frame field-state...")
    feats_F3 = RAW_FEATS + feats_c
    out["F3_lgbm_with_combined_fs"] = lgbm_oof(
        tr_with_fs_combined, y, feats_F3, CAT_COLS, "F3_combined")

    # ===== F4 LGBM with train-only field-state (control) ===============
    print("\n[F4] LGBM raw + TRAIN-ONLY field-state (control)...")
    feats_F4 = RAW_FEATS + feats_t
    out["F4_lgbm_with_train_only_fs"] = lgbm_oof(
        tr_with_fs_trainonly, y, feats_F4, CAT_COLS, "F4_train_only")

    # ===== summary =====================================================
    raw_auc = out["F2_lgbm_raw"]["oof_auc"]
    cb_auc = out["F3_lgbm_with_combined_fs"]["oof_auc"]
    to_auc = out["F4_lgbm_with_train_only_fs"]["oof_auc"]
    out["summary"] = {
        "F3_minus_F2_bp": (cb_auc - raw_auc) * 1e4,
        "F4_minus_F2_bp": (to_auc - raw_auc) * 1e4,
        "F3_minus_F4_bp": (cb_auc - to_auc) * 1e4,
        "interpretation": (
            "F3 - F4 isolates the combined-frame value (Rule 25 PASS via "
            "AV 0.502). F3 - F2 is the full field-state lift over raw."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print(f"\n=== HEADLINE ===")
    print(f"  F2 raw OOF AUC          : {raw_auc:.5f}")
    print(f"  F3 raw + combined fs    : {cb_auc:.5f}  Δ vs F2 = {(cb_auc - raw_auc) * 1e4:+.2f} bp")
    print(f"  F4 raw + train-only fs  : {to_auc:.5f}  Δ vs F2 = {(to_auc - raw_auc) * 1e4:+.2f} bp")
    print(f"  F3 - F4 (combined value): {(cb_auc - to_auc) * 1e4:+.2f} bp")


if __name__ == "__main__":
    main()
