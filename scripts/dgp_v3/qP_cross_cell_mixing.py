"""qP — where do synth's continuous values come from across cells?

Q7.3 said within-cell LapTime literal overlap is 19%. The other 81%
of synth LapTime values are in orig globally but NOT in the row's
(Y, C, PS) cell. Where?

For each synth row, find nearest orig row by standardised float
distance. Track the (Y, C, PS) cell of that orig row. Compare to the
synth row's own cell.

  - If the nearest orig row is usually in the synth row's own cell
    (despite the 19% literal-overlap), the host's CTGAN samples from
    a smoothed within-cell density.
  - If the nearest orig row is from a different cell, the host mixes
    values across cells.

Output: scripts/artifacts/dgp_v3_qP_cross_cell.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"

FLOAT_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation",
              "RaceProgress"]


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

    sc = StandardScaler().fit(orig[FLOAT_COLS].values)
    Xo = sc.transform(orig[FLOAT_COLS].values)

    # Sample 30k synth rows
    s_sample = synth.sample(30_000, random_state=0).reset_index(drop=True)
    Xs = sc.transform(s_sample[FLOAT_COLS].values)

    knn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Xo)
    d, idx = knn.kneighbors(Xs)
    t(f"NN search done: median d = {np.median(d):.4f}", ts)

    # Compare cells
    nearest_cell = orig.iloc[idx.flatten()][["Year", "Compound", "PitStop"]].reset_index(drop=True)
    s_cell = s_sample[["Year", "Compound", "PitStop"]].reset_index(drop=True)

    same_year = (nearest_cell["Year"] == s_cell["Year"]).mean()
    same_compound = (nearest_cell["Compound"] == s_cell["Compound"]).mean()
    same_pitstop = (nearest_cell["PitStop"] == s_cell["PitStop"]).mean()
    same_all3 = (
        (nearest_cell["Year"] == s_cell["Year"]) &
        (nearest_cell["Compound"] == s_cell["Compound"]) &
        (nearest_cell["PitStop"] == s_cell["PitStop"])
    ).mean()
    out["NN_cell_match"] = {
        "same_year": float(same_year),
        "same_compound": float(same_compound),
        "same_pitstop": float(same_pitstop),
        "same_all3": float(same_all3),
        "median_NN_d": float(np.median(d)),
        "p10_NN_d": float(np.percentile(d, 10)),
        "p90_NN_d": float(np.percentile(d, 90)),
        "n_synth": int(len(s_sample)),
    }
    t(f"NN cell match: y={same_year:.3f} c={same_compound:.3f} ps={same_pitstop:.3f} all3={same_all3:.3f}", ts)

    # Bonus: also k=1 self-distance within orig (intra-orig NN)
    kself = NearestNeighbors(n_neighbors=2).fit(Xo)
    do, _ = kself.kneighbors(Xo)
    intra_d = do[:, 1]
    out["intra_orig_NN_d"] = {
        "median": float(np.median(intra_d)),
        "p10": float(np.percentile(intra_d, 10)),
        "p90": float(np.percentile(intra_d, 90)),
    }
    t(f"intra-orig NN d: median={out['intra_orig_NN_d']['median']:.4f}", ts)

    fp = ART / "dgp_v3_qP_cross_cell.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qP cross-cell mixing ===")
    print(f"  For each synth row, find its nearest orig row by float distance:")
    print(f"    same Year:     {out['NN_cell_match']['same_year']:.3f}")
    print(f"    same Compound: {out['NN_cell_match']['same_compound']:.3f}")
    print(f"    same PitStop:  {out['NN_cell_match']['same_pitstop']:.3f}")
    print(f"    same all 3:    {out['NN_cell_match']['same_all3']:.3f}")
    print(f"    median NN d:   {out['NN_cell_match']['median_NN_d']:.4f}")
    print(f"  vs intra-orig NN d median: {out['intra_orig_NN_d']['median']:.4f}")
    print()
    print("  Reading: if same_all3 ~ 1, host samples within cell.")
    print("           if same_all3 < 1 substantially, host mixes across cells.")


if __name__ == "__main__":
    main()
