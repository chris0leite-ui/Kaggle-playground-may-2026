"""Phase A6 probe — per-cell row-count comparison (F8 direct test).

For each (Year, Compound, PitStop) cell, compute:
  p_orig(cell) = orig fraction
  p_synth(cell) = synth fraction
  ratio = p_synth / p_orig

If F8 holds, ratio should deviate strongly from 1 in cells like
(2023, _, 0) and (2023, _, 1).

Output: scripts/artifacts/dgp_v3_q10_cell_ratio.json + printed table.
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
    t(f"orig {orig.shape} synth {synth.shape}", ts)

    o_p = (orig.groupby(["Year", "Compound", "PitStop"]).size() / len(orig)).rename("p_orig")
    s_p = (synth.groupby(["Year", "Compound", "PitStop"]).size() / len(synth)).rename("p_synth")
    df = pd.concat([o_p, s_p], axis=1).fillna(0)
    df["ratio"] = df["p_synth"] / df["p_orig"].replace(0, np.nan)
    df["ratio_synth_over_orig"] = df["ratio"]
    df = df.sort_values("ratio", ascending=False)

    out: dict = {
        "n_cells": int(len(df)),
        "n_cells_orig_only": int(((df["p_orig"] > 0) & (df["p_synth"] == 0)).sum()),
        "n_cells_synth_only": int(((df["p_synth"] > 0) & (df["p_orig"] == 0)).sum()),
        "max_ratio": float(df["ratio"].dropna().max()),
        "min_ratio": float(df["ratio"].dropna().min()),
        "median_ratio": float(df["ratio"].dropna().median()),
        "table": df.reset_index().to_dict(orient="records"),
    }

    fp = ART / "dgp_v3_q10_cell_ratio.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== Top-10 most-oversampled cells (synth >> orig) ===")
    top = df.dropna(subset=["ratio"]).head(10)
    print(top.to_string())
    print("\n=== Bottom-10 most-undersampled cells (synth << orig) ===")
    bot = df.dropna(subset=["ratio"]).tail(10)
    print(bot.to_string())


if __name__ == "__main__":
    main()
