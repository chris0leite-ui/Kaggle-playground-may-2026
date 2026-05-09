"""Phase 11 — per-cell calibration of K=4 PRIMARY.

K=4 PRIMARY shows mild systematic residual correlation with Stint
(rho +0.144), RaceProgress (+0.119), TyreLife (+0.091), Year (-0.062).
Per-fold per-cell calibration adjustment may recover small lift.

Mechanism: per (Stint, Year) cell on train, compute mean(y - p_primary).
On test, add cell-mean offset to p_primary.

Per-fold-safe per Rule 24: cell stats computed from train fold only.

Cells: (Stint, Year) — 8 × 4 = 32 cells.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
SEED, N_FOLDS = 42, 5


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["PitNextLap"].astype(int).values
    oof_primary = np.load(ART / "oof_K4_fwd_pathb.npy")[:, 1].astype(np.float64)
    test_primary = np.load(ART / "test_K4_fwd_pathb.npy")[:, 1].astype(np.float64)

    # Cell axis options
    train["ss"] = train["LapNumber"] - train["TyreLife"] + 1
    test["ss"] = test["LapNumber"] - test["TyreLife"] + 1
    bins = [0, 1, 5, 10, 15, 20, 25, 30, 80]
    train["ss_bin"] = pd.cut(train["ss"], bins=bins, labels=False).fillna(0).astype(int)
    test["ss_bin"] = pd.cut(test["ss"], bins=bins, labels=False).fillna(0).astype(int)

    cell_specs = [
        ("Stint_Year", lambda df: df["Stint"].astype(str) + "_" + df["Year"].astype(str)),
        ("Compound_Stint", lambda df: df["Compound"] + "_" + df["Stint"].astype(str)),
        ("Year_Compound", lambda df: df["Year"].astype(str) + "_" + df["Compound"]),
        ("Year_ss_bin", lambda df: df["Year"].astype(str) + "_" + df["ss_bin"].astype(str)),
        ("Compound_Year_ss_bin", lambda df: df["Compound"] + "_" + df["Year"].astype(str) + "_" + df["ss_bin"].astype(str)),
    ]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    base_auc = roc_auc_score(y, oof_primary)
    print(f"K=4 PRIMARY OOF AUC: {base_auc:.5f}", flush=True)

    for cell_name, cell_fn in cell_specs:
        train_cell = cell_fn(train)
        test_cell = cell_fn(test)
        # Per-fold calibration
        oof_cal = np.zeros(len(y), dtype=np.float64)
        for tr_idx, va_idx in splits:
            tr_df = train.iloc[tr_idx].copy()
            tr_df["resid"] = y[tr_idx] - oof_primary[tr_idx]
            tr_df["cell"] = train_cell.iloc[tr_idx].values
            cell_mean = tr_df.groupby("cell")["resid"].mean()
            cell_size = tr_df.groupby("cell")["resid"].size()
            # smoothing: shrink toward 0 (no adjustment)
            smooth = 200
            cell_adj = (cell_mean * cell_size) / (cell_size + smooth)
            cell_adj_d = cell_adj.to_dict()
            va_cell = train_cell.iloc[va_idx].values
            adj = np.array([cell_adj_d.get(c, 0.0) for c in va_cell])
            oof_cal[va_idx] = oof_primary[va_idx] + adj
        oof_cal = np.clip(oof_cal, 1e-9, 1 - 1e-9)
        auc = roc_auc_score(y, oof_cal)
        print(f"  cell={cell_name:30s}: OOF {auc:.5f}  Δ {(auc-base_auc)*1e4:+.2f}bp",
              flush=True)

    # Best cell: write submission
    print(f"\nTotal: {time.time()-ts:.0f}s", flush=True)


if __name__ == "__main__":
    main()
