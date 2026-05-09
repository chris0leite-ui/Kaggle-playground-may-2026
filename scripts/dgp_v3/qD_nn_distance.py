"""Phase B alternative — is synth = noisy-orig?

If the host's generator is just `orig + per-column noise`, then:
  - Each synth row's nearest neighbour in orig is at distance comparable
    to orig's intra-NN distance (within standardised feature space).
  - Each orig row appears as the NN of ~6.2 synth rows (627k/101k).

Test this by 1-NN search in standardised 7-KS-low subspace.

Output: scripts/artifacts/dgp_v3_qD_nn.json
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

FEATS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation",
         "RaceProgress", "Position", "TyreLife", "Position_Change"]


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main() -> None:
    out: dict = {}
    ts = time.time()

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
        test = test.rename(columns={"LapTime (s)": "LapTime"})
    synth = pd.concat([train[[c for c in train.columns if c != "PitNextLap"]],
                       test], ignore_index=True)
    t(f"orig {orig.shape} synth {synth.shape}", ts)

    # Per-(Year, Compound, PitStop) cell, do NN within cell.
    out["per_cell_nn"] = {}
    cells = sorted(set(orig.groupby(["Year", "Compound", "PitStop"]).groups.keys())
                   & set(synth.groupby(["Year", "Compound", "PitStop"]).groups.keys()))
    rng = np.random.default_rng(0)
    aggregate_intra_orig = []
    aggregate_synth_to_orig = []
    n_cells_done = 0
    for cell in cells:
        o_cell = orig[(orig["Year"] == cell[0]) & (orig["Compound"] == cell[1]) & (orig["PitStop"] == cell[2])]
        s_cell = synth[(synth["Year"] == cell[0]) & (synth["Compound"] == cell[1]) & (synth["PitStop"] == cell[2])]
        if len(o_cell) < 50 or len(s_cell) < 200:
            continue
        # Subsample synth for speed
        if len(s_cell) > 5000:
            s_cell = s_cell.sample(5000, random_state=0)
        if len(o_cell) > 5000:
            o_cell_sub = o_cell.sample(5000, random_state=0)
        else:
            o_cell_sub = o_cell

        sc = StandardScaler().fit(o_cell_sub[FEATS].values)
        Xo = sc.transform(o_cell_sub[FEATS].values)
        Xs = sc.transform(s_cell[FEATS].values)

        # intra-orig NN (k=2 because k=1 is itself)
        knn = NearestNeighbors(n_neighbors=2).fit(Xo)
        d_intra, _ = knn.kneighbors(Xo)
        d_intra = d_intra[:, 1]  # nearest other
        # synth-to-orig NN
        knn2 = NearestNeighbors(n_neighbors=1).fit(Xo)
        d_so, _ = knn2.kneighbors(Xs)
        d_so = d_so[:, 0]

        intra_med = float(np.median(d_intra))
        so_med = float(np.median(d_so))
        out["per_cell_nn"][f"{cell[0]}_{cell[1]}_{cell[2]}"] = {
            "n_o": int(len(o_cell_sub)),
            "n_s": int(len(s_cell)),
            "median_intra_orig_d": intra_med,
            "median_synth_to_orig_d": so_med,
            "ratio_so_over_intra": so_med / intra_med if intra_med > 0 else None,
        }
        aggregate_intra_orig.extend(d_intra.tolist())
        aggregate_synth_to_orig.extend(d_so.tolist())
        n_cells_done += 1
        if n_cells_done <= 8 or n_cells_done % 5 == 0:
            t(f"cell {cell} intra={intra_med:.3f} synth_to_orig={so_med:.3f} ratio={so_med/intra_med:.3f}", ts)
    t(f"all {n_cells_done} cells done", ts)

    out["aggregate"] = {
        "n_cells": n_cells_done,
        "median_intra_orig_d": float(np.median(aggregate_intra_orig)),
        "median_synth_to_orig_d": float(np.median(aggregate_synth_to_orig)),
        "p10_intra": float(np.percentile(aggregate_intra_orig, 10)),
        "p90_intra": float(np.percentile(aggregate_intra_orig, 90)),
        "p10_so": float(np.percentile(aggregate_synth_to_orig, 10)),
        "p90_so": float(np.percentile(aggregate_synth_to_orig, 90)),
    }

    fp = ART / "dgp_v3_qD_nn.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== Aggregate (per-cell, standardised) ===")
    a = out["aggregate"]
    print(f"  n cells used: {a['n_cells']}")
    print(f"  median intra-orig  NN distance: {a['median_intra_orig_d']:.3f} (p10 {a['p10_intra']:.3f}, p90 {a['p90_intra']:.3f})")
    print(f"  median synth->orig NN distance: {a['median_synth_to_orig_d']:.3f} (p10 {a['p10_so']:.3f}, p90 {a['p90_so']:.3f})")
    ratio = a['median_synth_to_orig_d'] / a['median_intra_orig_d']
    print(f"  ratio synth->orig / intra-orig: {ratio:.3f}")
    print(f"  IF synth = noisy orig: ratio ~ 1")
    print(f"  IF synth is independent draw: ratio > 1.5")


if __name__ == "__main__":
    main()
