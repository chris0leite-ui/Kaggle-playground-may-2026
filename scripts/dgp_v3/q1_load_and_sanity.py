"""Phase A sanity probe — load synth + orig and re-validate F1, F5.

Fast probe (<2 min CPU). No new compute artefacts.

Confirms:
  - Synth shape matches eda-summary (627,305 rows)
  - F5 quantization grid (lap-counters integer; LapTime/Delta/CumDeg/RP float)
  - Stint-coherence: median std of stint_start_imputed within
    (Driver, Race, Year, Stint) groups (F1 says ≈ 2.43 laps)
  - Orig schema after dropping Normalized_TyreLife
  - Per-column overlap of values between synth and orig (literal-copy)

Output: scripts/artifacts/dgp_v3_q1_sanity.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"
ART.mkdir(exist_ok=True, parents=True)


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main() -> None:
    out: dict = {}
    ts = time.time()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    synth = pd.concat([train.drop(columns=["PitNextLap"]), test], ignore_index=True)
    t(f"loaded synth train+test: {train.shape}, {test.shape}", ts)

    out["train_shape"] = list(train.shape)
    out["test_shape"] = list(test.shape)
    out["synth_total"] = len(synth)
    out["target_prior"] = float(train["PitNextLap"].mean())
    out["columns"] = list(train.columns)

    # F5: quantization grid
    int_cols = ["LapNumber", "Stint", "TyreLife", "Position", "Year", "PitStop"]
    cont_cols = ["LapTime", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]
    grid = {}
    for c in int_cols:
        if c in synth.columns:
            v = synth[c].dropna().values
            is_int = np.all(np.isclose(v, np.round(v)))
            grid[c] = {"all_int": bool(is_int), "n_unique": int(synth[c].nunique())}
    for c in cont_cols:
        if c in synth.columns:
            v = synth[c].dropna().values
            is_int = np.all(np.isclose(v, np.round(v)))
            grid[c] = {"all_int": bool(is_int), "n_unique": int(synth[c].nunique())}
    out["F5_quantization"] = grid
    t("F5 grid done", ts)

    # F1: stint coherence
    if {"Driver", "Race", "Year", "Stint", "LapNumber", "TyreLife"}.issubset(synth.columns):
        sample = synth.sample(min(200_000, len(synth)), random_state=0).copy()
        sample["sst"] = sample["LapNumber"] - sample["TyreLife"] + 1
        gb = sample.groupby(["Driver", "Race", "Year", "Stint"])["sst"]
        sizes = gb.size()
        multi = gb.std().loc[sizes[sizes >= 2].index].dropna()
        coherent = (multi == 0).mean()
        out["F1_stint_coherence"] = {
            "n_groups_sampled_ge2": int(len(multi)),
            "median_std_within_group": float(multi.median()),
            "p90_std_within_group": float(multi.quantile(0.9)),
            "frac_coherent": float(coherent),
        }
        t("F1 stint coherence done", ts)

    # Orig load (best-effort)
    orig = None
    for fname in ["F1_pitstop_data.csv", "f1_pitstop_data.csv", "F1_strategy_dataset.csv",
                  "PitStopData.csv", "F1_pit_stops.csv"]:
        p = EXT / fname
        if p.exists():
            orig = pd.read_csv(p)
            t(f"orig loaded: {fname} {orig.shape}", ts)
            break
    if orig is None:
        # try whatever the first csv is
        csvs = list(EXT.glob("*.csv"))
        if csvs:
            orig = pd.read_csv(csvs[0])
            t(f"orig loaded (fallback): {csvs[0].name} {orig.shape}", ts)
            out["orig_filename_used"] = csvs[0].name
    if orig is not None:
        if "LapTime (s)" in orig.columns:
            orig = orig.rename(columns={"LapTime (s)": "LapTime"})
        out["orig_shape"] = list(orig.shape)
        out["orig_columns"] = list(orig.columns)
        # Per-column literal overlap
        overlap = {}
        common_cols = [c for c in cont_cols + int_cols if c in synth.columns and c in orig.columns]
        for c in common_cols:
            sset = set(synth[c].dropna().unique().tolist())
            oset = set(orig[c].dropna().unique().tolist())
            inter = sset & oset
            overlap[c] = {
                "n_synth_unique": len(sset),
                "n_orig_unique": len(oset),
                "n_common_values": len(inter),
                "frac_synth_in_orig_set": (
                    float(synth[c].isin(oset).mean()) if c in synth.columns else None
                ),
            }
        out["literal_overlap"] = overlap
        t("orig overlap done", ts)
    else:
        out["orig_status"] = "not yet downloaded; rerun once external/aadigupta is populated"

    # Save
    fp = ART / "dgp_v3_q1_sanity.json"
    fp.write_text(json.dumps(out, indent=2))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
