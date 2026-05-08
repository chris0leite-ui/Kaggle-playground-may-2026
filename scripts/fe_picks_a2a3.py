"""scripts/fe_picks_a2a3.py — Tier-A2 + Tier-A3 FE pick registry.

Each pick is a class with `.fit(train_with_target)` and `.transform(df)`
methods. The CV loop in p1_single_lgbm_v3.py (with --feature-add NAME)
calls `fit` on the fold's training partition (where TARGET is present)
and `transform` on each of the train/val/test frames.

Picks marked **fold-safe by construction** use only feature columns
(esp. `PitStop`, the prior-lap pit indicator — NOT the `PitNextLap`
target). Their `.fit()` is a no-op; `.transform()` is deterministic.

Picks marked **label-derived** require per-fold refit per Rule 24.

Origin: `audit/2026-05-08-fe-research-{survey,code-grounded,extended}.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

TARGET = "PitNextLap"


# -----------------------------------------------------------------------
# Lifecycle protocol expected by p1_single_lgbm_v3.py --feature-add
# -----------------------------------------------------------------------
#   pick = PICKS[name]()
#   pick.fit(train_ti)              # train_ti has TARGET column
#   train_ti = pick.transform(train_ti)
#   train_va = pick.transform(train_va)
#   test_fold = pick.transform(test_fold)
#   pick.new_columns                # list[str] of column names produced
# -----------------------------------------------------------------------


@dataclass
class _FeBase:
    """Base class. Pick implementations override .fit and .transform."""
    new_columns: list[str] = field(default_factory=list)

    def fit(self, train_with_target: pd.DataFrame) -> "_FeBase":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# =====================================================================
# A2-2 — Mandatory 2-compound rule (svanikkolli v12 §F6) with dry-race
#        gate (TUMFTM RL-state formulation). FOLD-SAFE.
# =====================================================================
@dataclass
class A2_2_MandatoryCompoundRule(_FeBase):
    new_columns: list[str] = field(default_factory=lambda: [
        "n_compounds_used", "mandatory_pit_pending", "mandatory_urgency",
        "dry_race",
    ])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        grp = ["Driver", "Race", "Year"]
        # Dry race: no driver in this race uses INTERMEDIATE/WET.
        out["dry_race"] = (1 - out.groupby("Race")["Compound"].transform(
            lambda s: s.isin(["INTERMEDIATE", "WET"]).any()).astype(int))
        first_compound = out.groupby(grp)["Compound"].transform("first")
        comp_changed = (out["Compound"] != first_compound).astype(int)
        out["n_compounds_used"] = (out.groupby(grp).apply(
            lambda g: comp_changed.loc[g.index].cummax() + 1
        ).reset_index(level=grp, drop=True).astype(int))
        out["mandatory_pit_pending"] = (
            (out["dry_race"] == 1)
            & (out["n_compounds_used"] < 2)
            & (out["RaceProgress"] > 0.6)
        ).astype(int)
        out["mandatory_urgency"] = (
            out["mandatory_pit_pending"] * out["RaceProgress"]
        ).astype(np.float32)
        return out


# =====================================================================
# A3-2 — Per-track learned fuel coefficient (corrects A2-6's 0.035 s/lap
#        constant with mid-stint LapTime regression slope).
#        Not strictly label-derived (uses LapTime, a feature), but the
#        regression is fit on training rows only for stability.
# =====================================================================
@dataclass
class A3_2_PerTrackFuelCoef(_FeBase):
    new_columns: list[str] = field(default_factory=lambda: [
        "track_fuel_coef", "fuel_corrected_deg",
    ])
    coefs_: dict = field(default_factory=dict)

    def fit(self, train_with_target: pd.DataFrame) -> "A3_2_PerTrackFuelCoef":
        df = train_with_target
        # Mid-stint clean laps: drop first 2 + last 2 of each (Driver,
        # Race, Year, Stint) group; drop laps where PitStop=1.
        grp_stint = ["Driver", "Race", "Year", "Stint"]
        in_stint_idx = df.groupby(grp_stint).cumcount()
        stint_len = df.groupby(grp_stint)["LapNumber"].transform("count")
        mid_mask = (
            (in_stint_idx >= 2) & (in_stint_idx < stint_len - 2)
            & (df.get("PitStop", 0) == 0)
        )
        mid = df.loc[mid_mask, ["Race", "Year", "LapNumber", "LapTime (s)"]]
        coefs: dict = {}
        for (race, year), g in mid.groupby(["Race", "Year"]):
            if len(g) < 30:
                coefs[(race, year)] = 0.051  # post-2019 default
                continue
            x = g["LapNumber"].values.astype(float)
            yv = g["LapTime (s)"].values.astype(float)
            # OLS slope; clip to reasonable F1 range.
            slope = np.polyfit(x, yv, 1)[0]
            coefs[(race, year)] = float(np.clip(slope, -0.2, 0.2))
        self.coefs_ = coefs
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        default_coef = 0.051  # post-2019 published constant (in seconds/lap)
        out["track_fuel_coef"] = (
            list(zip(out["Race"].astype(str), out["Year"].astype(int)))
        )
        out["track_fuel_coef"] = out["track_fuel_coef"].map(
            lambda k: self.coefs_.get(k, default_coef)).astype(np.float32)
        grp = ["Driver", "Race", "Year"]
        first_lt = out.groupby(grp)["LapTime (s)"].transform("first")
        first_lap = out.groupby(grp)["LapNumber"].transform("first")
        out["fuel_corrected_deg"] = (
            (out["LapTime (s)"] - first_lt
             - out["track_fuel_coef"] * (out["LapNumber"] - first_lap))
            .clip(-5, 20).astype(np.float32))
        return out


# =====================================================================
# A2-7 — Field-state competitor features (svanikkolli v12 §F3).
#        FOLD-SAFE — uses PitStop (feature) not PitNextLap (target).
# =====================================================================
@dataclass
class A2_7_FieldStateF3(_FeBase):
    new_columns: list[str] = field(default_factory=lambda: [
        "avg_field_tyre_age", "max_field_tyre_age", "min_field_tyre_age",
        "field_tyre_age_pct", "is_oldest_tyre", "cars_older_tyres",
        "n_diff_compounds", "n_pitted_this_lap", "n_pitted_race_last5",
        "field_pit_rate", "tyre_age_vs_field",
    ])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        g_lap = out.groupby(["Race", "Year", "LapNumber"])
        out["avg_field_tyre_age"] = g_lap["TyreLife"].transform("mean").astype(np.float32)
        out["max_field_tyre_age"] = g_lap["TyreLife"].transform("max").astype(np.float32)
        out["min_field_tyre_age"] = g_lap["TyreLife"].transform("min").astype(np.float32)
        out["field_tyre_age_pct"] = (
            (out["TyreLife"] / (out["max_field_tyre_age"] + 1.0)).clip(0, 1)
            .astype(np.float32))
        out["is_oldest_tyre"] = (out["TyreLife"] == out["max_field_tyre_age"]).astype(np.int8)
        out["cars_older_tyres"] = g_lap["TyreLife"].transform(
            lambda x: (x > x.mean()).sum()).astype(np.int16)
        out["n_diff_compounds"] = g_lap["Compound"].transform("nunique").astype(np.int8)
        ps = out.get("PitStop", pd.Series(0, index=out.index)).fillna(0)
        out["n_pitted_this_lap"] = g_lap.transform(lambda _: ps.loc[_.index].sum()).iloc[:, 0].astype(np.int16) \
            if False else (out.assign(_ps=ps).groupby(["Race", "Year", "LapNumber"])["_ps"]
                            .transform("sum").astype(np.int16))
        g_drv = out.groupby(["Driver", "Race", "Year"])
        out["n_pitted_race_last5"] = g_drv["PitStop"].transform(
            lambda x: x.shift(1).fillna(0).rolling(5, min_periods=1).sum()
        ).fillna(0).astype(np.int16)
        out["field_pit_rate"] = (out.assign(_ps=ps)
            .groupby(["Race", "Year", "LapNumber"])["_ps"].transform("mean")
            .astype(np.float32))
        out["tyre_age_vs_field"] = ((out["TyreLife"] - out["avg_field_tyre_age"])
            .clip(-20, 20).astype(np.float32))
        # Drop helper if any leaked
        out = out.drop(columns=[c for c in ["_ps"] if c in out.columns])
        return out


# =====================================================================
# A3-1 — Rank-sorted multi-neighbour gaps + pit-pressure
#        (NFL Big Data Bowl 2020 "The Zoo" pattern; BDB 2023 pocket
#        pressure). FOLD-SAFE.
# =====================================================================
@dataclass
class A3_1_RankSortedGaps(_FeBase):
    new_columns: list[str] = field(default_factory=lambda: [
        "gap_to_car_ahead_1", "gap_to_car_ahead_2",
        "gap_to_car_behind_1", "gap_to_car_behind_2",
        "in_drs_range", "undercut_viable", "threat_from_behind",
        "gap_ahead_delta", "pit_pressure_field",
        "tirechange_pursuer",
    ])
    pit_delta_seconds: float = 22.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        # Cumulative race time per (Driver, Race, Year), pit-loss subtracted.
        ps = out.get("PitStop", pd.Series(0, index=out.index)).fillna(0)
        clean_lt = out["LapTime (s)"] - ps * self.pit_delta_seconds
        out["_cum_rt"] = (clean_lt.groupby(
            [out["Driver"], out["Race"], out["Year"]]).cumsum())
        # Position-sorted rank within (Race, Year, LapNumber) lap.
        keep = ["Race", "Year", "LapNumber", "Position", "_cum_rt"]
        gap = out[keep + ["PitStop"]].copy() if "PitStop" in out.columns \
            else out[keep].copy().assign(PitStop=ps.values)
        gap["_orig_idx"] = np.arange(len(gap))
        gap = gap.sort_values(["Race", "Year", "LapNumber", "Position"])
        glap = gap.groupby(["Race", "Year", "LapNumber"], sort=False)
        for k in (1, 2):
            gap[f"_cum_ahead_{k}"] = glap["_cum_rt"].shift(k)
            gap[f"_cum_behind_{k}"] = glap["_cum_rt"].shift(-k)
        gap["_ahead_pitted_lag1"] = glap["PitStop"].shift(1)
        gap["_behind_pitted_lag1"] = glap["PitStop"].shift(-1)
        gap = gap.sort_values("_orig_idx").reset_index(drop=True)
        for k in (1, 2):
            out[f"gap_to_car_ahead_{k}"] = (
                (out["_cum_rt"].values - gap[f"_cum_ahead_{k}"].values)
                .clip(-5, 60)).astype(np.float32)
            out[f"gap_to_car_behind_{k}"] = (
                (gap[f"_cum_behind_{k}"].values - out["_cum_rt"].values)
                .clip(-5, 60)).astype(np.float32)
        # Strategic flags.
        out["in_drs_range"] = (
            (out["gap_to_car_ahead_1"] < 1.2)
            & (out["gap_to_car_ahead_1"] > -1)
        ).astype(np.int8)
        out["undercut_viable"] = (
            (out["gap_to_car_ahead_1"] > 0.5)
            & (out["gap_to_car_ahead_1"] < 4.0)
        ).astype(np.int8)
        out["threat_from_behind"] = (
            (out["gap_to_car_behind_1"] < 2.0)
            & (out["gap_to_car_behind_1"] > -1)
        ).astype(np.int8)
        # 3-lap delta of gap_to_car_ahead_1 within (Driver, Race, Year).
        out["gap_ahead_delta"] = (out.groupby(["Driver", "Race", "Year"])[
            "gap_to_car_ahead_1"].diff(3).fillna(0).clip(-10, 10)
            .astype(np.float32))
        # BDB 2023 U Toronto pit-pressure scalar:
        # sum_over_other_cars_in_lap(exp(-gap/tau) * I(rival_TyreLife<5))
        # Approximated per (Race, Year, LapNumber): all rivals contribute.
        # Use the simpler proxy: count(rivals with fresh tyres in same lap).
        out["pit_pressure_field"] = out.groupby(
            ["Race", "Year", "LapNumber"])["TyreLife"].transform(
            lambda x: ((x < 5).sum() - (x.values < 5).astype(int))
        ).astype(np.int16)
        # tirechange_pursuer = TUMFTM RL state's defendable-undercut signal:
        # the car at Position+1 just transitioned (PitStop on prior lap).
        out["tirechange_pursuer"] = gap["_behind_pitted_lag1"].fillna(0).astype(np.int8).values
        # Cleanup
        out = out.drop(columns=["_cum_rt"])
        return out


# =====================================================================
# A2-4 — VSC vs Full-SC proxy split (svanikkolli v12 §F4). FOLD-SAFE.
# =====================================================================
@dataclass
class A2_4_VscFullScSplit(_FeBase):
    new_columns: list[str] = field(default_factory=lambda: [
        "field_median_lt", "sc_pace_ratio",
        "is_vsc_proxy", "is_full_sc_proxy",
        "laps_since_sc",
    ])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        g_lap = out.groupby(["Race", "Year", "LapNumber"])
        out["field_median_lt"] = g_lap["LapTime (s)"].transform("median").astype(np.float32)
        roll5 = out.groupby(["Driver", "Race", "Year"])["LapTime (s)"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        denom = roll5.fillna(out["LapTime (s)"]).clip(lower=60)
        out["sc_pace_ratio"] = (out["field_median_lt"] / denom).astype(np.float32)
        # Thresholds: svanikkolli used field_median_lt / roll5 > 1.08 / 1.30.
        is_sc = (out["sc_pace_ratio"] > 1.08).astype(np.int8)
        out["is_vsc_proxy"] = ((out["sc_pace_ratio"] > 1.08)
                               & (out["sc_pace_ratio"] <= 1.30)).astype(np.int8)
        out["is_full_sc_proxy"] = (out["sc_pace_ratio"] > 1.30).astype(np.int8)
        out["laps_since_sc"] = (out.assign(_sc=is_sc)
            .groupby(["Driver", "Race", "Year"])["_sc"].transform(
                lambda x: _laps_since_marker(x.values))
            .astype(np.int16))
        return out


def _laps_since_marker(arr: np.ndarray, cap: int = 30) -> list[int]:
    """For an indicator series, compute laps-since-last-1, clipped to cap."""
    out = []
    cnt = cap
    for v in arr:
        if v == 1:
            cnt = 0
        else:
            cnt = min(cnt + 1, cap)
        out.append(cnt)
    return out


# =====================================================================
# A3-3 — `tirechange_pursuer` lagged window 1-3 + DriverAheadPit lagged.
#        Stand-alone variant of A3-1's tirechange_pursuer with explicit
#        multi-lag window (Frontiers AI 2025). FOLD-SAFE.
# =====================================================================
@dataclass
class A3_3_TirechangePursuerLagged(_FeBase):
    new_columns: list[str] = field(default_factory=lambda: [
        "driver_ahead_pitted_lag1", "driver_ahead_pitted_lag2", "driver_ahead_pitted_lag3",
        "driver_behind_pitted_lag1", "driver_behind_pitted_lag2", "driver_behind_pitted_lag3",
    ])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        keep = ["Race", "Year", "LapNumber", "Position", "PitStop"]
        if "PitStop" not in out.columns:
            out["PitStop"] = 0
        gap = out[keep].copy()
        gap["_orig_idx"] = np.arange(len(gap))
        gap = gap.sort_values(["Race", "Year", "LapNumber", "Position"])
        glap = gap.groupby(["Race", "Year", "LapNumber"], sort=False)
        for k in (1, 2, 3):
            gap[f"_ahead_pit_lag{k}"] = glap["PitStop"].shift(k)
            gap[f"_behind_pit_lag{k}"] = glap["PitStop"].shift(-k)
        gap = gap.sort_values("_orig_idx").reset_index(drop=True)
        for k in (1, 2, 3):
            out[f"driver_ahead_pitted_lag{k}"] = gap[f"_ahead_pit_lag{k}"].fillna(0).astype(np.int8).values
            out[f"driver_behind_pitted_lag{k}"] = gap[f"_behind_pit_lag{k}"].fillna(0).astype(np.int8).values
        return out


# =====================================================================
# A2-6 — Original svanikkolli-style 0.035 fuel-coef (kept for ablation
#        against A3-2 per-track learned). FOLD-SAFE.
# =====================================================================
@dataclass
class A2_6_FuelCorrectionConstant(_FeBase):
    new_columns: list[str] = field(default_factory=lambda: [
        "fuel_adj_lt_const", "fuel_corrected_deg_const",
    ])
    coef: float = 0.035

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["fuel_adj_lt_const"] = (
            out["LapTime (s)"] + out["LapNumber"] * self.coef
        ).astype(np.float32)
        grp = ["Driver", "Race", "Year"]
        first_adj = out.groupby(grp)["fuel_adj_lt_const"].transform("first")
        out["fuel_corrected_deg_const"] = (
            (out["fuel_adj_lt_const"] - first_adj).clip(-5, 20).astype(np.float32))
        return out


# =====================================================================
# Registry
# =====================================================================
PICKS: dict = {
    "a2_2_mandatory_compound_rule": A2_2_MandatoryCompoundRule,
    "a2_4_vsc_fullsc_split":        A2_4_VscFullScSplit,
    "a2_6_fuel_correction_const":   A2_6_FuelCorrectionConstant,
    "a2_7_field_state_f3":          A2_7_FieldStateF3,
    "a3_1_rank_sorted_gaps":        A3_1_RankSortedGaps,
    "a3_2_per_track_fuel_coef":     A3_2_PerTrackFuelCoef,
    "a3_3_tirechange_pursuer_lagged": A3_3_TirechangePursuerLagged,
}


# Convenience for p1_single_lgbm_v3.py: callable(train, test, fold_state)
# returning (train_aug, test_aug, new_cols).
def call_pick(name: str, train_df, test_df, fold_state):
    """Compatibility shim for the LGBM v3 --feature-add integration.

    fold_state["phase"] ∈ {"sample_only", "fit_train", "transform_val",
    "transform_test"}. The shim caches a fitted instance on fold_state so
    transform-phase calls reuse the fold's trained pick.
    """
    cls = PICKS.get(name)
    if cls is None:
        raise KeyError(f"unknown pick: {name}; available: {sorted(PICKS)}")
    phase = fold_state.get("phase", "sample_only")
    if phase == "sample_only":
        # Discovery: instantiate, do not fit; return new column names.
        pick = cls()
        return train_df, test_df, list(pick.new_columns)
    cache_key = f"_pick_instance_{name}"
    if phase == "fit_train":
        pick = cls()
        if hasattr(pick, "fit"):
            pick.fit(train_df)
        out = pick.transform(train_df)
        fold_state[cache_key] = pick
        return out, None, list(pick.new_columns)
    # transform phases — reuse the instance from fit_train
    pick = fold_state.get(cache_key)
    if pick is None:
        # Fallback: refit on `fitted_on` if provided; else on train_df.
        pick = cls()
        if hasattr(pick, "fit"):
            pick.fit(fold_state.get("fitted_on", train_df))
        fold_state[cache_key] = pick
    return pick.transform(train_df), None, list(pick.new_columns)


if __name__ == "__main__":
    # Smoke test: instantiate every pick, verify new_columns are populated.
    print("Available picks:")
    for name, cls in PICKS.items():
        pick = cls()
        print(f"  {name:36s}  → {len(pick.new_columns)} cols: "
              f"{pick.new_columns[:3]}...")
