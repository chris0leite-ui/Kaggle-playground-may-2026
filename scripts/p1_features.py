"""P1 v2 single-model feature factory — Rozen-aligned.

Fixes the OOF→LB inversion seen in v1 (LB 0.94107 vs OOF 0.94970, gap
−863 bp) by removing the leaky cluster of per-split-count features
(stint_size_far / stint_pct / count-based pit_imminent / pit_in_5) and
replacing them with Rozen's fit-on-train aggregate merges + 1950-2022
historical pit priors.

Key changes vs v1:
1. REMOVED: `stint_size_far` (.transform('count')), `stint_pct`,
   `lap_in_stint`-cumcount cascade. These computed counts on each
   split (train OR test) separately, producing different feature values
   for the same physical stint and causing severe distribution shift.
2. ADDED: train-only FS_A aggregates merged via lookup tables
   (race_avg_pit_lap, race_total_laps, compound_avg_life, race_max_stint,
   compound_race_lt, race_compound_max_life, dc_avg_stint_life). All
   computed once on TRAIN with PitNextLap labels, applied identically
   to test.
3. ADDED: pit_imminent / pit_in_5 routed through compound_avg_life
   (physics-based) instead of split-dependent counts.
4. ADDED: External 1950-2022 driver/circuit historical pit priors from
   external/f1_official_1950_2022/pitstops.csv.
5. ADDED: Numeric-to-categorical via np.floor().factorize() for raw
   numerics (LGBM categorical-handling lever).
6. KEPT: lag/rolling within (Driver, Race, Year) (Rozen has these too,
   despite same-split limitation; impact is small per Rozen's wins).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

COMPOUND_MAX_LIFE_MAP = {
    "SOFT": 15, "MEDIUM": 30, "HARD": 50,
    "INTERMEDIATE": 25, "WET": 20,
}
COMBO_COLS = [("Race", "Compound"), ("Race", "Year"), ("Driver", "Compound")]
EXT_PIT_PATH = Path("external/f1_official_1950_2022/pitstops.csv")


def _ndrv(s):
    return str(s).strip().split()[-1].lower()


def _nrace(s):
    s = str(s).strip().lower()
    return re.sub(r"grand\s+prix|\bgp\b", "", s).strip()


def _load_hist_priors():
    if not EXT_PIT_PATH.exists():
        return None
    df = pd.read_csv(EXT_PIT_PATH)
    df.columns = df.columns.str.strip()
    drv_col = next((c for c in df.columns if "driver" in c.lower()), None)
    race_col = next((c for c in df.columns if "grand" in c.lower() or "race" in c.lower()), None)
    lap_col = next((c for c in df.columns if "lap" in c.lower() and "time" not in c.lower()), None)
    if not (drv_col and lap_col):
        return None
    df["_dk"] = df[drv_col].map(_ndrv)
    df["_lap"] = pd.to_numeric(df[lap_col], errors="coerce")
    if race_col:
        df["_rk"] = df[race_col].map(_nrace)
    out = {"driver": {}, "circuit": {}}
    drv = (df.dropna(subset=["_lap"]).groupby("_dk")
           .agg(pit_hist_avg_lap=("_lap", "mean"),
                pit_hist_std_lap=("_lap", "std")))
    drv["pit_hist_std_lap"] = drv["pit_hist_std_lap"].fillna(8.0)
    out["driver"] = drv.to_dict(orient="index")
    if race_col:
        ckt = (df.dropna(subset=["_lap"]).groupby("_rk")
               .agg(pit_ckt_avg_lap=("_lap", "mean"),
                    pit_ckt_std_lap=("_lap", "std")))
        ckt["pit_ckt_std_lap"] = ckt["pit_ckt_std_lap"].fillna(8.0)
        out["circuit"] = ckt.to_dict(orient="index")
    return out


def fit_fs_a(df_with_labels: pd.DataFrame) -> dict:
    """Compute the LABEL-CONDITIONAL aggregates from a labelled subset.

    Strict rule: this function uses ONLY rows in df_with_labels (which
    must carry PitNextLap). Call once per CV fold using ti rows only;
    call once on full train for the final test prediction (5-fold
    averaging instead). Never include val/holdout labels in this fit.
    """
    if "PitNextLap" not in df_with_labels.columns:
        raise ValueError("fit_fs_a requires PitNextLap column")
    fs_a = {}
    fs_a["pit_laps"] = (df_with_labels[df_with_labels["PitNextLap"] == 1]
        .groupby(["Race", "Year"])["LapNumber"].mean()
        .rename("race_avg_pit_lap"))
    fs_a["total_laps"] = (df_with_labels.groupby(["Race", "Year"])["LapNumber"]
        .max().rename("race_total_laps"))
    fs_a["comp_life"] = (df_with_labels[df_with_labels["PitNextLap"] == 1]
        .groupby("Compound")["TyreLife"].mean()
        .rename("compound_avg_life"))
    fs_a["race_stints"] = (df_with_labels.groupby(["Race", "Year"])["Stint"]
        .max().rename("race_max_stint"))
    fs_a["compound_race_lt"] = (df_with_labels.groupby(
        ["Race", "Year", "Compound"])["LapTime (s)"].median()
        .rename("compound_race_median_lt"))
    fs_a["race_compound_max"] = (df_with_labels.groupby(
        ["Race", "Year", "Compound"])["TyreLife"].max()
        .rename("race_compound_max_life"))
    fs_a["dc_avg_stint_life"] = (df_with_labels[df_with_labels["PitNextLap"] == 1]
        .groupby(["Driver", "Compound"])["TyreLife"].mean()
        .rename("dc_avg_stint_life"))
    return fs_a


def apply_fs_a(df: pd.DataFrame, fs_a: dict) -> pd.DataFrame:
    """Merge an FS_A dict into df and compute the derived features.

    df must already carry the static columns produced by
    make_features_static (notably stint_start_lap and the raw cols).
    """
    df = df.copy()
    df = df.merge(fs_a["pit_laps"].reset_index(),    on=["Race", "Year"], how="left")
    df = df.merge(fs_a["total_laps"].reset_index(),  on=["Race", "Year"], how="left")
    df = df.merge(fs_a["comp_life"].reset_index(),   on="Compound",      how="left")
    df = df.merge(fs_a["race_stints"].reset_index(), on=["Race", "Year"], how="left")
    df = df.merge(fs_a["compound_race_lt"].reset_index(),
                  on=["Race", "Year", "Compound"], how="left")
    df = df.merge(fs_a["race_compound_max"].reset_index(),
                  on=["Race", "Year", "Compound"], how="left")
    df = df.merge(fs_a["dc_avg_stint_life"].reset_index(),
                  on=["Driver", "Compound"], how="left")

    # Derived features (all from consistent lookups)
    df["pit_window_flag"] = (np.abs(df["LapNumber"]
        - df["race_avg_pit_lap"].fillna(35)) <= 3).astype(int)
    df["tyre_vs_comp_avg"] = df["TyreLife"] - df["compound_avg_life"].fillna(25)
    df["overdue_pit"] = (df["TyreLife"] > df["compound_avg_life"].fillna(25)).astype(int)
    df["laps_remaining_race"] = df["race_total_laps"].fillna(60) - df["LapNumber"]
    df["tyre_age_pct_race"] = df["TyreLife"] / (df["race_total_laps"].fillna(60) + 1)
    df["stint_progress"] = df["Stint"] / (df["race_max_stint"].fillna(3) + 1)
    df["tyre_life_pct"] = df["TyreLife"] / df["compound_avg_life"].fillna(25).clip(lower=1)
    df["stint_end_est"] = df["stint_start_lap"] + df["compound_avg_life"].fillna(25)
    df["laps_until_stop"] = (df["stint_end_est"] - df["LapNumber"]).clip(lower=-20)
    df["pit_imminent"] = (df["laps_until_stop"] <= 2).astype(int)
    df["pit_in_5"] = (df["laps_until_stop"] <= 5).astype(int)
    df["lap_vs_compound_baseline"] = (df["LapTime (s)"]
        - df["compound_race_median_lt"].fillna(df["LapTime (s)"])).clip(-5, 15)
    df["tyre_freshness_pct"] = (1 - df["TyreLife"] / (df["race_compound_max_life"].fillna(40) + 1)).clip(0, 1)
    df["driver_vs_avg_life"] = (df["TyreLife"] - df["dc_avg_stint_life"].fillna(25)).clip(-20, 20)
    df["driver_overdue_personal"] = (df["TyreLife"] > df["dc_avg_stint_life"].fillna(25)).astype(int)

    df["deg_x_win"] = df["Cumulative_Degradation"] * df["is_pit_window"]
    df["over_x_win"] = df["overdue_pit"] * df["is_pit_window"]
    df["tyre_x_pres"] = df["TyreLife"] * df["position_pressure"]
    return df


def make_features_static(df_in: pd.DataFrame, fit: bool = False,
                         state: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Engineer LABEL-INDEPENDENT features only.

    These features depend ONLY on the row's own values (or aggregates
    over rows in df_in WITHOUT using PitNextLap). Safe to compute on
    full train+test or per-fold.

    Pair with fit_fs_a + apply_fs_a for the label-conditional cluster.
    """
    if state is None:
        state = {}
    df = (df_in.copy()
          .sort_values(["Driver", "Race", "Year", "LapNumber"])
          .reset_index(drop=True))

    # === 1. Tyre / compound algebra ===
    df["tyre_life_sq"] = df["TyreLife"] ** 2
    df["tyre_life_log"] = np.log1p(df["TyreLife"])
    df["tyre_life_sqrt"] = np.sqrt(df["TyreLife"])
    df["deg_per_lap"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)
    df["compound_max_life"] = df["Compound"].map(COMPOUND_MAX_LIFE_MAP).fillna(30)
    df["compound_tyre_norm"] = (df["TyreLife"] / df["compound_max_life"]).clip(0, 2)
    df["tyre_overdue_norm"] = (df["compound_tyre_norm"] > 0.85).astype(int)

    # === 2. Race-progress family ===
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

    # === 3. Lag / rolling within (Driver, Race, Year) ===
    grp_id = (df["Driver"].astype(str) + "|"
              + df["Race"].astype(str) + "|"
              + df["Year"].astype(str)).factorize()[0]
    df["_grp_id"] = grp_id
    grp_id_s = pd.Series(grp_id)
    df["delta_lag1"] = df["LapTime_Delta"].shift(1).where(
        df["_grp_id"] == grp_id_s.shift(1).values)
    df["delta_lag2"] = df["LapTime_Delta"].shift(2).where(
        df["_grp_id"] == grp_id_s.shift(2).values)
    df["prev_pit"] = df["PitStop"].shift(1).where(
        df["_grp_id"] == grp_id_s.shift(1).values).fillna(0)
    df["delta_accel"] = df["LapTime_Delta"] - df["delta_lag1"]
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

    g_stint = df.groupby(["Driver", "Race", "Year", "Stint"])
    df["lap_in_stint"] = g_stint.cumcount()
    df["stint_start_lap"] = g_stint["LapNumber"].transform("min")
    df = df.drop(columns=["_grp_id"])

    df["lap_div_rp"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype(np.float32)
    df["tl_div_ln"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype(np.float32)
    df["compound_ord"] = df["Compound"].map(
        {"SOFT": 2, "MEDIUM": 1, "HARD": 0, "INTERMEDIATE": 3, "WET": 4}).fillna(1)

    # === 4. Historical priors (1950-2022, no train labels involved) ===
    if fit:
        state["hist"] = _load_hist_priors()
    hist = state.get("hist")
    df["_dk"] = df["Driver"].map(_ndrv)
    df["_rk"] = df["Race"].map(_nrace)
    if hist is not None:
        drv = hist.get("driver", {})
        ckt = hist.get("circuit", {})
        df["pit_hist_avg_lap"] = df["_dk"].map(
            lambda k: drv.get(k, {}).get("pit_hist_avg_lap", 30.0))
        df["pit_hist_std_lap"] = df["_dk"].map(
            lambda k: drv.get(k, {}).get("pit_hist_std_lap", 8.0))
        df["drv_laps_vs_hist"] = (df["TyreLife"] - df["pit_hist_avg_lap"]).clip(-25, 25)
        df["drv_hist_overdue"] = (df["TyreLife"] > df["pit_hist_avg_lap"]).astype(int)
        df["ckt_hist_avg_lap"] = df["_rk"].map(
            lambda k: ckt.get(k, {}).get("pit_ckt_avg_lap", 28.0))
        df["ckt_hist_std_lap"] = df["_rk"].map(
            lambda k: ckt.get(k, {}).get("pit_ckt_std_lap", 8.0))
        df["laps_vs_ckt_hist"] = (df["TyreLife"] - df["ckt_hist_avg_lap"]).clip(-25, 25)
        df["in_ckt_pit_window"] = (df["laps_vs_ckt_hist"].abs()
            <= df["ckt_hist_std_lap"]).astype(int)
    else:
        for c in ["pit_hist_avg_lap", "pit_hist_std_lap", "drv_laps_vs_hist",
                  "drv_hist_overdue", "ckt_hist_avg_lap", "ckt_hist_std_lap",
                  "laps_vs_ckt_hist", "in_ckt_pit_window"]:
            df[c] = 0.0
    df = df.drop(columns=["_dk", "_rk"], errors="ignore")

    # === 5. Combo categoricals (factorize maps in state) ===
    for c1, c2 in COMBO_COLS:
        combo_str = df[c1].astype(str) + "_" + df[c2].astype(str)
        key = f"{c1}_{c2}_"
        if fit:
            codes, uniques = combo_str.factorize()
            state[f"combo_{key}"] = {v: i for i, v in enumerate(uniques)}
        df[key] = combo_str.map(state[f"combo_{key}"]).fillna(-1).astype("int32")

    for c in ["Driver", "Race", "Compound"]:
        if fit:
            codes, uniques = df[c].astype(str).factorize()
            state[f"cat_{c}"] = {v: i for i, v in enumerate(uniques)}
        df[f"{c}_cat"] = df[c].astype(str).map(state[f"cat_{c}"]).fillna(-1).astype("int32")

    return df, state


def make_features_A(df_in: pd.DataFrame, fit: bool = False,
                    state: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """LEGACY (LEAKY) v2 path — kept for reference; do not use for new probes.

    The FS_A merge here uses df_in's own labels for label-conditional
    aggregates, which leaks val labels into val features when called
    on the full train. Use make_features_static + fit_fs_a + apply_fs_a
    instead with proper per-fold FS_A fitting.
    """
    df, state = make_features_static(df_in, fit=fit, state=state)
    if fit:
        if "PitNextLap" not in df.columns:
            raise ValueError("legacy make_features_A requires PitNextLap with fit=True")
        state["FS_A"] = fit_fs_a(df)
    df = apply_fs_a(df, state["FS_A"])
    return df.fillna(0), state


def cv_target_encode(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     group_cols: list[str], target: pd.Series,
                     fold_list: list, smoothing: int = 30
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold target encoding with global-mean smoothing."""
    global_mean = float(target.mean())
    n = len(train_df)
    oof_enc = np.full(n, global_mean, dtype=np.float32)

    def _key(df):
        s = df[group_cols[0]].fillna("MISSING").astype(str)
        for c in group_cols[1:]:
            s = s + "__" + df[c].fillna("MISSING").astype(str)
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
