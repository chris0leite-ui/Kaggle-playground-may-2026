"""scripts/d15d_knn_features.py — Per-segment KNN distance features.

Builds 10 unsupervised distance features per row:
  - 5 stats (mean/min/max/std/top1) of k=5 nearest neighbours within
    same Compound (5 levels)
  - 5 stats of k=5 nearest neighbours within same Driver (887 levels;
    Driver clusters with <50 rows fall back to global KNN within that row)

Distances computed on standardized numerics (~11 features). Uses
NearestNeighbors with kd_tree (low-dim, ~11 features). Per-segment
iteration:
  - Compound: 5 sub-problems ~125k rows each
  - Driver: 887 sub-problems, mostly small (median <1000 rows)

Train+test combined for unsupervised distance computation (Jahrer DAE
pattern; no target leakage).

Save:
  scripts/artifacts/d15d_knn_X_train.npy  (439140, 10)
  scripts/artifacts/d15d_knn_X_test.npy   (188165, 10)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop", "Year",
]
K = 5
MIN_CLUSTER_SIZE = 50


def _stats_from_dist(dist: np.ndarray) -> np.ndarray:
    """Given (n_rows, K) distance array (excluding self), return 5 stats."""
    # dist is sorted ascending; top1 = nearest neighbour distance
    top1 = dist[:, 0]
    mn = dist.min(axis=1)
    mx = dist.max(axis=1)
    mean = dist.mean(axis=1)
    std = dist.std(axis=1)
    return np.column_stack([mean, mn, mx, std, top1])


def _knn_within_segment(X: np.ndarray, seg_ids: np.ndarray,
                        k: int = K, min_cluster: int = MIN_CLUSTER_SIZE,
                        global_nn: NearestNeighbors | None = None,
                        label: str = "") -> np.ndarray:
    """For each row, compute k-NN distance stats within same segment.

    Rows in segments smaller than `min_cluster` fall back to global_nn
    (must be provided). Returns (n_rows, 5) feature array.
    """
    n = len(X)
    out = np.zeros((n, 5), dtype=np.float64)
    unique_segs, inv = np.unique(seg_ids, return_inverse=True)
    sanity = {}

    fallback_count = 0
    for s_idx, seg in enumerate(unique_segs):
        mask = inv == s_idx
        idx_in_seg = np.where(mask)[0]
        m = len(idx_in_seg)
        if m == 0:
            continue
        Xs = X[idx_in_seg]
        if m < min_cluster:
            # fallback to global NN
            if global_nn is not None:
                # request k+1 neighbours, slice
                d, _ = global_nn.kneighbors(Xs, n_neighbors=min(k + 1, n))
                # exclude self if global_nn was fit on same data
                # global_nn fit on full X, so first neighbour is self -> drop
                d_eff = d[:, 1:k + 1] if d.shape[1] > k else d[:, :k]
                out[idx_in_seg] = _stats_from_dist(d_eff)
                fallback_count += m
            continue
        # within-segment NN: ask for k+1 (will drop self); cap at m
        n_query = min(k + 1, m)
        nn = NearestNeighbors(n_neighbors=n_query, algorithm="kd_tree")
        nn.fit(Xs)
        d, _ = nn.kneighbors(Xs)
        # drop self (first column is dist 0). If segment is exactly k rows,
        # we asked for k+1=k+1 capped to k → no self drop possible; pad zeros.
        if d.shape[1] > k:
            d_eff = d[:, 1:k + 1]
        else:
            # m == k: only k cols including self; drop self col, pad
            d_eff = d[:, 1:]
            pad = np.zeros((m, k - d_eff.shape[1]))
            d_eff = np.hstack([d_eff, pad])
        out[idx_in_seg] = _stats_from_dist(d_eff)
        sanity[str(seg)] = float(d_eff.mean())

    print(f"  [{label}] segments={len(unique_segs)} fallback_rows={fallback_count}")
    if sanity:
        # show a few sample mean distances
        for k_, v in list(sanity.items())[:6]:
            print(f"    seg={k_!s:<35} mean_dist={v:.4f}")
    return out


def main() -> None:
    t0 = time.time()
    print("=== d15d KNN feature build ===")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    # Combined numeric matrix for unsupervised distances
    all_df = pd.concat([train, test], axis=0, ignore_index=True)
    n_train = len(train)
    n_test = len(test)
    n_all = n_train + n_test

    X_num = all_df[NUMERIC_COLS].astype(np.float64).values
    print(f"Numeric matrix: {X_num.shape}, cols: {NUMERIC_COLS}")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_num)
    print("StandardScaler fit on train+test combined.")

    # Global NN for Driver fallback (very small driver clusters)
    print(f"\nFitting global NN (n={n_all}) for Driver-fallback ...")
    t_g = time.time()
    global_nn = NearestNeighbors(n_neighbors=K + 1, algorithm="kd_tree")
    global_nn.fit(X_std)
    print(f"  global NN fit: {time.time() - t_g:.1f}s")

    # ---- Per-Compound KNN ------------------------------------------
    print("\n--- Per-Compound KNN ---")
    t_c = time.time()
    comp_seg = all_df["Compound"].astype(str).values
    feat_comp = _knn_within_segment(
        X_std, comp_seg, k=K, min_cluster=MIN_CLUSTER_SIZE,
        global_nn=global_nn, label="Compound",
    )
    print(f"  Compound KNN wall: {time.time() - t_c:.1f}s")

    # ---- Per-Driver KNN --------------------------------------------
    print("\n--- Per-Driver KNN ---")
    t_d = time.time()
    drv_seg = all_df["Driver"].astype(str).values
    feat_drv = _knn_within_segment(
        X_std, drv_seg, k=K, min_cluster=MIN_CLUSTER_SIZE,
        global_nn=global_nn, label="Driver",
    )
    print(f"  Driver KNN wall: {time.time() - t_d:.1f}s")

    # ---- Stack & split ----------------------------------------------
    feats = np.hstack([feat_comp, feat_drv])  # (n_all, 10)
    print(f"\nFinal feature matrix: {feats.shape}")
    print(f"  feature names: comp_mean,comp_min,comp_max,comp_std,comp_top1,"
          f"drv_mean,drv_min,drv_max,drv_std,drv_top1")

    X_train = feats[:n_train]
    X_test = feats[n_train:]
    assert X_train.shape == (n_train, 10), X_train.shape
    assert X_test.shape == (n_test, 10), X_test.shape

    np.save(ART / "d15d_knn_X_train.npy", X_train.astype(np.float32))
    np.save(ART / "d15d_knn_X_test.npy", X_test.astype(np.float32))
    print(f"\nSaved: d15d_knn_X_train.npy {X_train.shape}")
    print(f"       d15d_knn_X_test.npy  {X_test.shape}")
    print(f"\nTotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
