"""qX — per-cell first/second moments comparison synth vs orig.

For each (Y, C, PS, Race, Stint, LapNumber) cell with ≥10 rows in both,
compare:
  - mean(LapTime), std(LapTime), skew(LapTime)
  - mean(LapTime_Delta), std(LapTime_Delta)
  - mean(Cumulative_Degradation), std(Cumulative_Degradation)
  - mean(RaceProgress), std(RaceProgress)

If the host's generator outputs values with the SAME mean+std as orig's
per-cell, then the disc must be using higher moments (skewness, multi-
column joint structure) to discriminate.

If mean/std differ systematically, the host applies a per-cell affine
transformation we haven't characterised.

Output: scripts/artifacts/dgp_v3_qX_moments.json
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

FLOAT_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main() -> None:
    out: dict = {}
    ts = time.time()

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna()
    orig = orig.reset_index(drop=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
        test = test.rename(columns={"LapTime (s)": "LapTime"})
    synth = pd.concat(
        [train[[c for c in train.columns if c != "PitNextLap"]], test],
        ignore_index=True,
    )
    t(f"orig {orig.shape} synth {synth.shape}", ts)

    # Use a coarser cell key first so we have enough samples per cell
    by = ["Year", "Compound", "PitStop"]
    out["cell_key"] = by

    o_groups = orig.groupby(by)
    s_groups = synth.groupby(by)
    common_keys = set(o_groups.groups.keys()) & set(s_groups.groups.keys())

    out["per_cell_moments"] = {}
    for k in common_keys:
        og = o_groups.get_group(k)
        sg = s_groups.get_group(k)
        if len(og) < 100 or len(sg) < 100:
            continue
        moments = {}
        for c in FLOAT_COLS:
            o_v = og[c].dropna()
            s_v = sg[c].dropna()
            moments[c] = {
                "mean_o": float(o_v.mean()),
                "mean_s": float(s_v.mean()),
                "mean_diff": float(s_v.mean() - o_v.mean()),
                "std_o": float(o_v.std()),
                "std_s": float(s_v.std()),
                "std_ratio": float(s_v.std() / o_v.std()) if o_v.std() > 0 else None,
                "skew_o": float(o_v.skew()),
                "skew_s": float(s_v.skew()),
            }
        out["per_cell_moments"][str(k)] = {
            "n_o": int(len(og)),
            "n_s": int(len(sg)),
            "moments": moments,
        }

    fp = ART / "dgp_v3_qX_moments.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    # Aggregate
    print("\n=== qX per-cell first/second moments (LapTime, LapTime_Delta, CumDeg, RP) ===")
    for c in FLOAT_COLS:
        all_mean_diff = []
        all_std_ratio = []
        all_skew_diff = []
        for cell, v in out["per_cell_moments"].items():
            m = v["moments"][c]
            all_mean_diff.append(m["mean_diff"])
            if m["std_ratio"] is not None:
                all_std_ratio.append(m["std_ratio"])
            all_skew_diff.append(m["skew_s"] - m["skew_o"])
        mean_diff_arr = np.array(all_mean_diff)
        std_ratio_arr = np.array(all_std_ratio)
        skew_diff_arr = np.array(all_skew_diff)
        print(f"\n  {c}:")
        print(f"    mean diff (synth - orig): med={np.median(mean_diff_arr):.4f} p10={np.percentile(mean_diff_arr,10):.4f} p90={np.percentile(mean_diff_arr,90):.4f}")
        print(f"    std ratio (synth / orig): med={np.median(std_ratio_arr):.4f} p10={np.percentile(std_ratio_arr,10):.4f} p90={np.percentile(std_ratio_arr,90):.4f}")
        print(f"    skew diff (synth - orig): med={np.median(skew_diff_arr):.4f} p10={np.percentile(skew_diff_arr,10):.4f} p90={np.percentile(skew_diff_arr,90):.4f}")


if __name__ == "__main__":
    main()
