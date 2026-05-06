"""P1 single-model feature factory — Rozen 0.95354 recipe.

Implements `make_features_A` (~118 features) from
romanrozen/f1-pit-driver-race-year-encoding-0-95354 plus the CV
target encoding helper. Intent: replicate single-LGBM OOF AUC ~0.952
to test PI hypothesis (single model can close the gap to top-5%).

Friction-tagged differences from Rozen:
- We use StratifiedKFold(seed=42, n_splits=5) per project convention
  (matches all OOF artifacts in scripts/artifacts/).
- We do NOT prepend external aadigupta_orig data by default; the
  Driver-code overlap is only 31/887 so its primary value is generic
  per-row signal, which we test as an ablation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPOUND_MAX_LIFE_MAP = {
    "SOFT": 15, "MEDIUM": 30, "HARD": 50,
    "INTERMEDIATE": 25, "WET": 20,
}

COMBO_COLS = [("Race", "Compound"), ("Race", "Year"), ("Driver", "Compound")]


def make_features_A(df_in: pd.DataFrame, fit: bool = False,
                    state: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Engineer ~118 features. Sorted by (Driver, Race, Year, LapNumber)
    so lag/rolling features carry within-group context.

    Returns (df_with_features, state). On fit=True populates state with
    factorize maps so transform-time produces the same int codes.
    """
    if state is None:
        state = {}
    df = (df_in.copy()
          .sort_values(["Driver", "Race", "Year", "LapNumber"])
          .reset_index(drop=True))
    g = df.groupby(["Driver", "Race", "Year"])

    # tyre / compound family
    df["tyre_life_sq"] = df["TyreLife"] ** 2
    df["tyre_life_log"] = np.log1p(df["TyreLife"])
    df["tyre_life_sqrt"] = np.sqrt(df["TyreLife"])
    df["deg_per_lap"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)
    df["compound_life_ratio"] = df["TyreLife"] / (
        df.groupby("Compound")["TyreLife"].transform("max") + 1e-9)
    df["compound_max_life"] = df["Compound"].map(COMPOUND_MAX_LIFE_MAP).fillna(30)
    df["compound_tyre_norm"] = (df["TyreLife"] / df["compound_max_life"]).clip(0, 2)
    df["tyre_overdue_norm"] = (df["compound_tyre_norm"] > 0.85).astype(int)

    # race-progress family
    df["est_total_laps"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-9)
                            ).round().clip(30, 80)
    df["laps_remaining"] = (df["est_total_laps"] - df["LapNumber"]).clip(lower=0)
    df["tyre_pct_remaining"] = df["TyreLife"] / (df["laps_remaining"] + 1)
    df["is_pit_window"] = ((df["RaceProgress"] >= 0.28)
                           & (df["RaceProgress"] <= 0.62)).astype(int)
    df["is_late_race"] = (df["RaceProgress"] > 0.75).astype(int)
    df["position_pressure"] = df["Position"] * (1 - df["RaceProgress"])
    df["urgency_score"] = df["Cumulative_Degradation"].abs() * (1 - df["RaceProgress"])
    df["race_phase"] = pd.cut(
        df["RaceProgress"], bins=[0, .25, .5, .75, 1.01],
        labels=[0, 1, 2, 3]).astype(float)
    df["norm_position"] = 1 - (df["Position"] - 1) / 19.0
    df["life_x_progress"] = df["TyreLife"] * df["RaceProgress"]

    # lag / rolling within (Driver, Race, Year). Vectorised: shift once
    # over the whole sorted frame, then mask boundary rows where group
    # changed via a group-id fingerprint.
    grp_id = (df["Driver"].astype(str) + "|"
              + df["Race"].astype(str) + "|"
              + df["Year"].astype(str)).factorize()[0]
    df["_grp_id"] = grp_id
    df["delta_lag1"] = df["LapTime_Delta"].shift(1).where(
        df["_grp_id"] == pd.Series(grp_id).shift(1).values)
    df["delta_lag2"] = df["LapTime_Delta"].shift(2).where(
        df["_grp_id"] == pd.Series(grp_id).shift(2).values)
    df["prev_pit"] = df["PitStop"].shift(1).where(
        df["_grp_id"] == pd.Series(grp_id).shift(1).values).fillna(0)
    df["delta_accel"] = df["LapTime_Delta"] - df["delta_lag1"]

    # For rolling, use groupby+rolling natively (faster than transform+lambda).
    rolled_lt = df.groupby("_grp_id")["LapTime (s)"].rolling(
        15, min_periods=1)
    # Compute means for windows 3,5,7,10,15 by setting each window separately
    # (groupby+rolling.mean is C-optimized).
    for w in [3, 5, 7, 10, 15]:
        df[f"roll{w}_lt"] = (df.groupby("_grp_id")["LapTime (s)"]
                             .rolling(w, min_periods=1).mean()
                             .reset_index(level=0, drop=True).values)
    for w in [3, 7]:
        df[f"roll{w}_d"] = (df.groupby("_grp_id")["LapTime_Delta"]
                            .rolling(w, min_periods=1).mean()
                            .reset_index(level=0, drop=True).values)
    df["roll3_std"] = (df.groupby("_grp_id")["LapTime (s)"]
                       .rolling(3, min_periods=1).std()
                       .reset_index(level=0, drop=True).fillna(0).values)
    df["lap_vs_r3"] = df["LapTime (s)"] - df["roll3_lt"]
    df["lap_vs_r7"] = df["LapTime (s)"] - df["roll7_lt"]
    df["lap_vs_r15"] = df["LapTime (s)"] - df["roll15_lt"]
    df = df.drop(columns=["_grp_id"])

    # within-stint
    df["lap_in_stint"] = g.cumcount() + 1
    g_stint = df.groupby(["Driver", "Race", "Year", "Stint"])
    df["stint_lap_idx"] = g_stint.cumcount() + 1
    df["stint_size_far"] = g_stint["LapNumber"].transform("count")
    df["stint_pct"] = df["stint_lap_idx"] / df["stint_size_far"].clip(lower=1)
    df["pit_imminent"] = (df["stint_pct"] >= 0.85).astype(int)
    df["pit_in_5"] = (df["laps_remaining"] <= 5).astype(int)

    # additional ratios
    df["tyre_life_pct"] = df["TyreLife"] / (df["compound_max_life"] + 1e-9)
    df["laps_until_stop"] = (df["compound_max_life"] - df["TyreLife"]).clip(lower=0)
    df["compound_ord"] = df["Compound"].map(
        {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}).fillna(1)

    # combo categoricals -> int codes
    for c1, c2 in COMBO_COLS:
        combo_str = df[c1].astype(str) + "_" + df[c2].astype(str)
        key = f"{c1}_{c2}_"
        if fit:
            codes, uniques = combo_str.factorize()
            state[f"combo_{key}"] = {v: i for i, v in enumerate(uniques)}
        df[key] = combo_str.map(state[f"combo_{key}"]).fillna(-1).astype("int32")

    # raw cats -> int codes (for native LGBM categorical handling)
    for c in ["Driver", "Race", "Compound"]:
        if fit:
            codes, uniques = df[c].astype(str).factorize()
            state[f"cat_{c}"] = {v: i for i, v in enumerate(uniques)}
        df[f"{c}_cat"] = df[c].astype(str).map(state[f"cat_{c}"]).fillna(-1).astype("int32")

    return df, state


def cv_target_encode(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     group_cols: list[str], target: pd.Series,
                     fold_list: list, smoothing: int = 30
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold target encoding with global-mean smoothing.

    For training: each row's encoding uses ONLY training-fold rows of
    the same group (no leakage from the row's own fold).
    For test: encoding uses ALL training rows.
    """
    global_mean = float(target.mean())
    n = len(train_df)
    oof_enc = np.full(n, global_mean, dtype=np.float32)

    # Vectorised string concat (10× faster than agg("__".join, axis=1))
    def _key(df):
        s = df[group_cols[0]].astype(str)
        for c in group_cols[1:]:
            s = s + "__" + df[c].astype(str)
        return s.reset_index(drop=True)
    key = _key(train_df)
    key_test = _key(test_df)
    target_arr = target.reset_index(drop=True)

    for ti, vi in fold_list:
        stats = (pd.DataFrame({"key": key.iloc[ti].values,
                               "target": target_arr.iloc[ti].values})
                 .groupby("key")["target"].agg(["sum", "count"]))
        stats["enc"] = ((stats["sum"] + smoothing * global_mean)
                        / (stats["count"] + smoothing))
        oof_enc[vi] = key.iloc[vi].map(stats["enc"].to_dict()).fillna(global_mean).values

    stats_full = (pd.DataFrame({"key": key.values, "target": target_arr.values})
                  .groupby("key")["target"].agg(["sum", "count"]))
    stats_full["enc"] = ((stats_full["sum"] + smoothing * global_mean)
                         / (stats_full["count"] + smoothing))
    test_enc = key_test.map(stats_full["enc"].to_dict()).fillna(global_mean).values
    return oof_enc.astype(np.float32), test_enc.astype(np.float32)


TE_CONFIGS = [
    (["Driver", "Race", "Year"], 20, "te_drv_race_yr"),
    (["Driver", "Race"], 30, "te_drv_race"),
    (["Race", "Compound"], 25, "te_race_comp"),
    (["Driver", "Compound"], 25, "te_drv_comp"),
    (["Race", "Year"], 20, "te_race_yr"),
    (["Driver", "Race", "Compound"], 15, "te_drv_race_comp"),
]


def feature_columns_for_lgbm(train_A: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (feature_cols, categorical_cols) for the LGBM model."""
    drop = {"id", "PitNextLap", "Driver", "Race", "Compound", "split"}
    feats = [c for c in train_A.columns if c not in drop]
    cat_cols = [c for c in feats
                if c.endswith("_cat") or c.endswith("_") or c == "race_phase"]
    return feats, cat_cols
