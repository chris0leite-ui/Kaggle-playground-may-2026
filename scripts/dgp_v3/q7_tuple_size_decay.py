"""Phase A2 parallel — tuple-size sensitivity for synth→orig matching.

Q6 showed only 27/627,305 synth rows match orig on a 6-tuple, despite
per-column literal-overlap of 97-100%. Test how the match rate decays
with tuple size to characterise per-column independence in the host's
CTGAN sampler.

For tuple sizes K = 1..6 over the 6 literal-copy / preserved columns,
report:
  - synth match rate against orig keys
  - implied conditional independence factor (product of marginals vs joint)

Also report the per-column conditional-on-cond independence: for each
synth row, conditional on (Year, Compound, PitStop), how often does its
single-column value lie in orig's per-cell set?

Output: scripts/artifacts/dgp_v3_q7_tuple_decay.json
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"

FP_COLS = ["LapTime", "RaceProgress", "LapTime_Delta",
           "Position", "TyreLife", "Position_Change"]


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def keyfunc(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = []
    for c in cols:
        v = df[c]
        if v.dtype.kind == "f":
            v = v.round(6)
        parts.append(v.astype(str))
    return pd.Series(["|".join(t) for t in zip(*[p.values for p in parts])])


def main() -> None:
    out: dict = {}
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

    # Q7.1 — match rate as a function of tuple-size K
    out["match_rate_by_K"] = {}
    for K in [1, 2, 3, 4, 5, 6]:
        # Use the first K columns of FP_COLS for a deterministic comparison
        cols = FP_COLS[:K]
        sk = keyfunc(synth, cols)
        ok = set(keyfunc(orig, cols).tolist())
        rate = float(sk.isin(ok).mean())
        out["match_rate_by_K"][K] = {"cols": cols, "match_rate": rate}
        t(f"K={K} cols={cols} match_rate={rate:.4f}", ts)

    # Q7.2 — match rate for ALL combinations at K=2 (sanity)
    out["match_rate_K2_pairs"] = {}
    for cols in combinations(FP_COLS, 2):
        sk = keyfunc(synth, list(cols))
        ok = set(keyfunc(orig, list(cols)).tolist())
        rate = float(sk.isin(ok).mean())
        out["match_rate_K2_pairs"]["+".join(cols)] = rate
    t("K=2 pairs done", ts)

    # Q7.3 — within-(Year, Compound, PitStop) cell, fraction of synth values in orig per-cell set
    out["within_cell_overlap"] = {}
    for c in FP_COLS:
        results = []
        for (y, cmp_, ps), s_grp in synth.groupby(["Year", "Compound", "PitStop"]):
            o_grp = orig[(orig["Year"] == y) & (orig["Compound"] == cmp_) & (orig["PitStop"] == ps)]
            if len(o_grp) < 50 or len(s_grp) < 50:
                continue
            oset = set(o_grp[c].dropna().unique().tolist())
            if not oset:
                continue
            ov = float(s_grp[c].isin(oset).mean())
            results.append({"cell": f"{y}_{cmp_}_{ps}", "n_s": int(len(s_grp)), "overlap": ov})
        if results:
            mean_ov = float(np.mean([r["overlap"] for r in results]))
            out["within_cell_overlap"][c] = {
                "n_cells": len(results),
                "mean_within_cell_overlap": mean_ov,
                "min_overlap": float(min(r["overlap"] for r in results)),
                "max_overlap": float(max(r["overlap"] for r in results)),
            }
    t("within-cell overlap done", ts)

    # Save
    fp = ART / "dgp_v3_q7_tuple_decay.json"
    fp.write_text(json.dumps(out, indent=2))
    t(f"wrote {fp.name}", ts)

    # Print summary
    print("\n=== Q7.1 match rate by tuple size K ===")
    for K, v in out["match_rate_by_K"].items():
        print(f"  K={K} {'+'.join(v['cols']):60s} → {v['match_rate']:.4f}")

    print("\n=== Q7.3 within-cell single-column overlap (host samples within cell?) ===")
    for c, v in out["within_cell_overlap"].items():
        print(
            f"  {c:30s} mean={v['mean_within_cell_overlap']:.4f} "
            f"min={v['min_overlap']:.4f} max={v['max_overlap']:.4f} "
            f"n_cells={v['n_cells']}"
        )


if __name__ == "__main__":
    main()
