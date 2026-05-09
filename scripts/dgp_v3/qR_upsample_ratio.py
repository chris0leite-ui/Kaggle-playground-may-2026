"""qR — per-LapTime upsampling ratio histogram.

Hypothesis: maybe host literally upsamples orig values by ~6x. Check
the distribution of (synth.count / orig.count) per LapTime value.

If host upsamples 6x:
  - Most ratios cluster near 6
  - Few outliers

If host generates new values:
  - Many ratios are 0 (orig values absent in synth) or inf (synth values
    absent in orig)
  - Ratio distribution wide

Output: scripts/artifacts/dgp_v3_qR_upsample_ratio.json
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


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main() -> None:
    out: dict = {}
    ts = time.time()

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.dropna()
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

    expected_ratio = len(synth) / len(orig)
    out["expected_ratio"] = expected_ratio
    print(f"  expected uniform-upsample ratio: {expected_ratio:.3f}")

    # Per-LapTime ratio
    o_counts = orig["LapTime"].value_counts()
    s_counts = synth["LapTime"].value_counts()
    common = sorted(set(o_counts.index) & set(s_counts.index))
    only_orig = sorted(set(o_counts.index) - set(s_counts.index))
    only_synth = sorted(set(s_counts.index) - set(o_counts.index))
    out["unique_LapTime_in_orig"] = int(o_counts.size)
    out["unique_LapTime_in_synth"] = int(s_counts.size)
    out["common_LapTime"] = len(common)
    out["only_orig_LapTime"] = len(only_orig)
    out["only_synth_LapTime"] = len(only_synth)
    t(f"unique LapTime: orig={o_counts.size}, synth={s_counts.size}, common={len(common)}", ts)

    if common:
        ratios = np.array([s_counts[v] / o_counts[v] for v in common])
        out["common_LapTime_ratio"] = {
            "p10": float(np.percentile(ratios, 10)),
            "p25": float(np.percentile(ratios, 25)),
            "median": float(np.percentile(ratios, 50)),
            "mean": float(np.mean(ratios)),
            "p75": float(np.percentile(ratios, 75)),
            "p90": float(np.percentile(ratios, 90)),
            "frac_within_2x_of_expected": float(
                ((ratios >= expected_ratio / 2) & (ratios <= expected_ratio * 2)).mean()
            ),
            "frac_below_quarter": float((ratios < expected_ratio / 4).mean()),
            "frac_above_4x": float((ratios > expected_ratio * 4).mean()),
        }
        t(f"common LapTime ratios: median={np.median(ratios):.2f} mean={np.mean(ratios):.2f}", ts)

    # Per-(LapTime, Year, Compound) ratio — finer grain
    o_cell = orig.groupby(["Year", "Compound", "LapTime"]).size()
    s_cell = synth.groupby(["Year", "Compound", "LapTime"]).size()
    common_keys = list(set(o_cell.index) & set(s_cell.index))
    if common_keys:
        ratios2 = np.array([s_cell[k] / o_cell[k] for k in common_keys])
        out["per_yc_LapTime_ratio"] = {
            "p10": float(np.percentile(ratios2, 10)),
            "median": float(np.percentile(ratios2, 50)),
            "p90": float(np.percentile(ratios2, 90)),
            "n_common_keys": len(common_keys),
            "n_orig_only_keys": int(len(set(o_cell.index)) - len(common_keys)),
            "n_synth_only_keys": int(len(set(s_cell.index)) - len(common_keys)),
        }
        t(f"per-(Y,C,LapTime) ratio: median={np.median(ratios2):.2f}", ts)

    fp = ART / "dgp_v3_qR_upsample_ratio.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qR upsampling ratio ===")
    print(f"  Global expected: {expected_ratio:.3f}")
    print(f"  Unique LapTime: orig={out['unique_LapTime_in_orig']}, synth={out['unique_LapTime_in_synth']}")
    print(f"  Common LapTime values: {out['common_LapTime']}")
    if "common_LapTime_ratio" in out:
        r = out["common_LapTime_ratio"]
        print(f"\n  Per-LapTime ratios across common values:")
        print(f"    p10/p25/med/mean/p75/p90: {r['p10']:.2f}/{r['p25']:.2f}/{r['median']:.2f}/{r['mean']:.2f}/{r['p75']:.2f}/{r['p90']:.2f}")
        print(f"    fraction within 2x of expected ({expected_ratio:.2f}): {r['frac_within_2x_of_expected']:.3f}")
        print(f"    fraction < 0.25x of expected: {r['frac_below_quarter']:.3f}")
        print(f"    fraction > 4x of expected:    {r['frac_above_4x']:.3f}")
    if "per_yc_LapTime_ratio" in out:
        r = out["per_yc_LapTime_ratio"]
        print(f"\n  Per-(Y, C, LapTime) ratios:")
        print(f"    p10/median/p90: {r['p10']:.2f}/{r['median']:.2f}/{r['p90']:.2f}")
        print(f"    n common keys: {r['n_common_keys']}")
        print(f"    n orig-only keys: {r['n_orig_only_keys']}")
        print(f"    n synth-only keys: {r['n_synth_only_keys']}")


if __name__ == "__main__":
    main()
