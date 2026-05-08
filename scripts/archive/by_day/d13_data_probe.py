"""Day-13 data probe — 6 questions before launching G1/G2/G3.

Q1. PitNextLap target structure: when does target=1?
Q2. Stint boundaries: does Stint change exactly when Compound changes?
Q3. Year=2023 mechanism: where do the 0.96% positives concentrate?
Q4. Driver-strategy persistence: within-driver pit-TyreLife std vs cross-driver.
Q5. Cross-driver intra-race state: signal for a γ4 feature pack?
Q6. LapTime_Delta / Cumulative_Degradation: distribution + target correlation.

Pure pandas, ~30 s wall. Writes JSON to scripts/artifacts/d13_probe.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
OUT = ART / "d13_probe.json"

print("Loading train + test...")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
print(f"  train: {len(tr):,}  test: {len(te):,}")

results: dict = {}


# Q1 — PitNextLap target structure
def q1_target_structure() -> dict:
    out: dict = {}
    out["pos_rate_overall"] = float(tr["PitNextLap"].mean())

    # positives per stint
    grp = tr.groupby(["Race", "Driver", "Year", "Stint"])
    pos_per_stint = grp["PitNextLap"].sum()
    sz_per_stint = grp.size()
    out["stint_count"] = int(len(pos_per_stint))
    out["stint_with_0_pos"] = int((pos_per_stint == 0).sum())
    out["stint_with_1_pos"] = int((pos_per_stint == 1).sum())
    out["stint_with_2plus_pos"] = int((pos_per_stint >= 2).sum())
    out["max_pos_per_stint"] = int(pos_per_stint.max())

    # is target=1 the last lap of the stint by LapNumber?
    tr_sorted = tr.sort_values(["Race", "Driver", "Year", "Stint", "LapNumber"])
    is_last = (
        tr_sorted.groupby(["Race", "Driver", "Year", "Stint"])["LapNumber"].transform("max")
        == tr_sorted["LapNumber"]
    )
    out["frac_pos_at_last_obs_lap"] = float(
        ((tr_sorted["PitNextLap"] == 1) & is_last).sum() / max(1, (tr_sorted["PitNextLap"] == 1).sum())
    )
    out["frac_last_obs_lap_pos"] = float(
        ((tr_sorted["PitNextLap"] == 1) & is_last).sum() / max(1, is_last.sum())
    )

    # PitStop=1 vs PitNextLap=1 lag-1 alignment within (Race,Driver,Year)
    g = tr_sorted.groupby(["Race", "Driver", "Year"])
    tr_sorted["PitStop_next"] = g["PitStop"].shift(-1)
    mask = tr_sorted["PitStop_next"].notna()
    overlap = (
        (tr_sorted.loc[mask, "PitNextLap"] == 1)
        & (tr_sorted.loc[mask, "PitStop_next"] == 1)
    ).sum()
    pos_with_next = (tr_sorted.loc[mask, "PitNextLap"] == 1).sum()
    out["frac_pos_aligned_with_next_pitstop"] = float(overlap / max(1, pos_with_next))
    return out


# Q2 — Stint vs Compound boundaries
def q2_stint_compound() -> dict:
    out: dict = {}
    tr_sorted = tr.sort_values(["Race", "Driver", "Year", "LapNumber"])
    g = tr_sorted.groupby(["Race", "Driver", "Year"])
    tr_sorted["Compound_next"] = g["Compound"].shift(-1)
    tr_sorted["Stint_next"] = g["Stint"].shift(-1)
    mask = tr_sorted["Stint_next"].notna()

    s_changes = tr_sorted.loc[mask, "Stint_next"] != tr_sorted.loc[mask, "Stint"]
    c_changes = tr_sorted.loc[mask, "Compound_next"] != tr_sorted.loc[mask, "Compound"]
    out["frac_stint_change_with_compound_change"] = float((s_changes & c_changes).sum() / max(1, s_changes.sum()))
    out["frac_compound_change_with_stint_change"] = float((s_changes & c_changes).sum() / max(1, c_changes.sum()))
    out["frac_stint_change_at_pit_next_lap"] = float(
        (s_changes & (tr_sorted.loc[mask, "PitNextLap"] == 1)).sum() / max(1, s_changes.sum())
    )
    out["stint_min"] = int(tr["Stint"].min())
    out["stint_max"] = int(tr["Stint"].max())
    return out


# Q3 — Year=2023 mechanism
def q3_year_2023() -> dict:
    out: dict = {}
    by_year = tr.groupby("Year").agg(
        rows=("id", "size"),
        pos_rate=("PitNextLap", "mean"),
        unique_drivers=("Driver", "nunique"),
        unique_races=("Race", "nunique"),
    )
    out["per_year"] = by_year.reset_index().to_dict(orient="records")

    # within 2023, where do positives live?
    tr23 = tr[tr["Year"] == 2023]
    out["y2023_pos_count"] = int(tr23["PitNextLap"].sum())
    by_race = tr23.groupby("Race")["PitNextLap"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    out["y2023_top5_races_by_pos_rate"] = by_race.head(5).reset_index().to_dict(orient="records")
    out["y2023_bot5_races_by_pos_rate"] = by_race.tail(5).reset_index().to_dict(orient="records")

    # mean stint length (train) per year
    g = tr.groupby(["Race", "Driver", "Year", "Stint"]).size()
    g_yr = g.groupby(level="Year").agg(["mean", "median", "max"])
    out["stint_size_per_year"] = g_yr.reset_index().to_dict(orient="records")
    return out


# Q4 — Driver-strategy persistence
def q4_driver_persistence() -> dict:
    out: dict = {}
    pit_rows = tr[tr["PitNextLap"] == 1]
    by_driver = pit_rows.groupby("Driver")["TyreLife"].agg(["count", "mean", "std"])
    by_driver = by_driver[by_driver["count"] >= 20]
    out["n_drivers_with_20plus_pits"] = int(len(by_driver))
    out["within_driver_pit_tyrelife_std_median"] = float(by_driver["std"].median())
    out["across_driver_pit_tyrelife_std"] = float(pit_rows["TyreLife"].std())

    # by (Driver, Compound)
    by_dc = pit_rows.groupby(["Driver", "Compound"])["TyreLife"].agg(["count", "mean", "std"])
    by_dc = by_dc[by_dc["count"] >= 20]
    out["n_driver_compound_with_20plus_pits"] = int(len(by_dc))
    out["within_driver_compound_pit_tyrelife_std_median"] = float(by_dc["std"].median())
    return out


# Q5 — Cross-driver intra-race state (γ4 candidate)
def q5_cross_driver() -> dict:
    out: dict = {}
    # for each (Race, Year, LapNumber) block, compute std and counts over drivers
    g = tr.groupby(["Race", "Year", "LapNumber"])
    block = g.agg(
        n_drivers=("Driver", "nunique"),
        tyrelife_std=("TyreLife", "std"),
        pos_rate=("PitNextLap", "mean"),
        soft_frac=("Compound", lambda s: float((s == "SOFT").mean())),
        hard_frac=("Compound", lambda s: float((s == "HARD").mean())),
    )
    block = block[block["n_drivers"] >= 3]
    out["n_blocks"] = int(len(block))
    out["mean_drivers_per_block"] = float(block["n_drivers"].mean())
    out["pos_rate_corr_tyrelife_std"] = float(block[["pos_rate", "tyrelife_std"]].corr().iloc[0, 1])
    out["pos_rate_corr_soft_frac"] = float(block[["pos_rate", "soft_frac"]].corr().iloc[0, 1])
    out["pos_rate_corr_hard_frac"] = float(block[["pos_rate", "hard_frac"]].corr().iloc[0, 1])

    # join block-level stats back as features and compute lift
    tr2 = tr.merge(
        block.reset_index().rename(columns={
            "tyrelife_std": "block_tyrelife_std",
            "soft_frac": "block_soft_frac",
            "hard_frac": "block_hard_frac",
        })[["Race", "Year", "LapNumber", "block_tyrelife_std", "block_soft_frac", "block_hard_frac"]],
        on=["Race", "Year", "LapNumber"], how="left",
    )
    out["row_corr_block_tyrelife_std_with_target"] = float(
        tr2[["PitNextLap", "block_tyrelife_std"]].corr().iloc[0, 1]
    )
    out["row_corr_block_soft_frac_with_target"] = float(
        tr2[["PitNextLap", "block_soft_frac"]].corr().iloc[0, 1]
    )
    out["row_corr_block_hard_frac_with_target"] = float(
        tr2[["PitNextLap", "block_hard_frac"]].corr().iloc[0, 1]
    )
    return out


# Q6 — LapTime_Delta / Cumulative_Degradation distribution + target correlation
def q6_feature_check() -> dict:
    out: dict = {}
    for col in ["LapTime_Delta", "Cumulative_Degradation", "TyreLife", "Position", "Position_Change"]:
        s = tr[col].dropna()
        out[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p05": float(s.quantile(0.05)),
            "p50": float(s.quantile(0.50)),
            "p95": float(s.quantile(0.95)),
            "corr_target": float(tr[[col, "PitNextLap"]].corr().iloc[0, 1]),
        }
    return out


for q_name, q_fn in [
    ("q1_target_structure", q1_target_structure),
    ("q2_stint_compound", q2_stint_compound),
    ("q3_year_2023", q3_year_2023),
    ("q4_driver_persistence", q4_driver_persistence),
    ("q5_cross_driver", q5_cross_driver),
    ("q6_feature_check", q6_feature_check),
]:
    print(f"running {q_name}...")
    results[q_name] = q_fn()

OUT.write_text(json.dumps(results, indent=2, default=str))
print(f"\nwrote {OUT}")
print(json.dumps(results, indent=2, default=str)[:4000])
