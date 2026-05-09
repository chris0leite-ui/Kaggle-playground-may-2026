"""Phase A planning probe — conditional MI within (Year, Compound, PitStop) cells.

Q7 found 0% 6-tuple match but high per-column overlap, suggesting
columns are sampled near-independently within cells. Test this
directly:

  MI(col_i; col_j | cell) = E_{cell}[MI(col_i; col_j)]

Compare orig and synth. If synth's conditional MI is much lower than
orig's, the host's CTGAN is throwing away cross-column structure within
cells. That is the precise mechanism we need to model in any forward
surrogate (Phase B).

Output: scripts/artifacts/dgp_v3_q8_cond_mi.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"

PAIRS = [
    ("LapTime", "LapTime_Delta"),
    ("LapTime", "Cumulative_Degradation"),
    ("LapTime", "RaceProgress"),
    ("LapTime_Delta", "Cumulative_Degradation"),
    ("RaceProgress", "Cumulative_Degradation"),
    ("TyreLife", "Cumulative_Degradation"),
    ("TyreLife", "LapTime"),
    ("Position", "LapTime"),
    ("Position_Change", "Position"),
]


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def mi_pair(df: pd.DataFrame, c1: str, c2: str, n_max: int = 5000) -> float:
    sub = df[[c1, c2]].dropna()
    if len(sub) > n_max:
        sub = sub.sample(n_max, random_state=0)
    if len(sub) < 100:
        return float("nan")
    mi = mutual_info_regression(sub[[c1]].values, sub[c2].values, random_state=0)
    return float(mi[0])


def main() -> None:
    out: dict = {"pairs": [list(p) for p in PAIRS]}
    ts = time.time()

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"])
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
    t(f"loaded orig {orig.shape}, synth {synth.shape}", ts)

    out["marginal_mi"] = {}
    for c1, c2 in PAIRS:
        mi_o = mi_pair(orig, c1, c2)
        mi_s = mi_pair(synth, c1, c2)
        out["marginal_mi"][f"{c1}__{c2}"] = {
            "orig": mi_o,
            "synth": mi_s,
            "ratio_synth_over_orig": (mi_s / mi_o) if mi_o > 0 else None,
        }
    t("marginal MI done", ts)

    # Conditional MI: average per-cell MI, weighted by cell size
    out["conditional_mi"] = {}
    cells_o = orig.groupby(["Year", "Compound", "PitStop"])
    cells_s = synth.groupby(["Year", "Compound", "PitStop"])
    cell_keys = sorted(set(cells_o.groups.keys()) & set(cells_s.groups.keys()))
    for c1, c2 in PAIRS:
        wsum_o, wsum_s, w_o_total, w_s_total = 0.0, 0.0, 0.0, 0.0
        per_cell = []
        for k in cell_keys:
            o_cell = cells_o.get_group(k)
            s_cell = cells_s.get_group(k)
            if len(o_cell) < 200 or len(s_cell) < 200:
                continue
            mi_o = mi_pair(o_cell, c1, c2, n_max=2000)
            mi_s = mi_pair(s_cell, c1, c2, n_max=2000)
            wsum_o += mi_o * len(o_cell)
            wsum_s += mi_s * len(s_cell)
            w_o_total += len(o_cell)
            w_s_total += len(s_cell)
            per_cell.append({"cell": str(k), "n_o": int(len(o_cell)), "n_s": int(len(s_cell)),
                             "mi_o": mi_o, "mi_s": mi_s})
        if w_o_total > 0:
            out["conditional_mi"][f"{c1}__{c2}"] = {
                "orig_avg": wsum_o / w_o_total,
                "synth_avg": wsum_s / w_s_total,
                "ratio_synth_over_orig": (wsum_s / w_s_total) / (wsum_o / w_o_total) if wsum_o > 0 else None,
                "n_cells": len(per_cell),
            }
    t("conditional MI done", ts)

    fp = ART / "dgp_v3_q8_cond_mi.json"
    fp.write_text(json.dumps(out, indent=2))
    t(f"wrote {fp.name}", ts)

    print("\n=== Marginal MI (synth vs orig, ratio) ===")
    for c, v in out["marginal_mi"].items():
        ratio = v["ratio_synth_over_orig"]
        print(f"  {c:40s} orig={v['orig']:.4f} synth={v['synth']:.4f} synth/orig={ratio:.3f}" if ratio is not None else f"  {c:40s} orig={v['orig']:.4f} synth={v['synth']:.4f}")

    print("\n=== Conditional MI within (Year, Compound, PitStop) cells ===")
    for c, v in out["conditional_mi"].items():
        ratio = v["ratio_synth_over_orig"]
        print(f"  {c:40s} orig={v['orig_avg']:.4f} synth={v['synth_avg']:.4f} synth/orig={ratio:.3f}" if ratio is not None else f"  {c:40s} orig={v['orig_avg']:.4f} synth={v['synth_avg']:.4f}")


if __name__ == "__main__":
    main()
