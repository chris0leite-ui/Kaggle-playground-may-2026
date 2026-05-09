"""Phase 17 — per-feature KS dissection (decoding analysis).

Computes per-column Kolmogorov-Smirnov distance between:
  - host synth (627k rows)
  - aadigupta orig (101k rows)
  - P3 recursive-on-synth replay (200k)
  - P13 v2 CTGAN-on-orig replay (200k)
  - P16 TVAE-on-orig replay (200k, when available)

The KS distance per feature reveals WHICH columns each surrogate
matches host on and which it fails on. d18 f1 only reported MEAN KS;
P17 dissects per-feature.

Hypothesis: host's signature is in mode-specific normalization of
continuous features (LapTime, LapTime_Delta). If host KS-matches on
categoricals + discretes but diverges on continuous tails, it's the
VGM mode count or normalization scheme. If host KS-matches on
continuous but diverges on categoricals, it's the conditional vector
schema.

Output: a table per feature showing KS(orig, host), KS(P3, host),
KS(P13, host), KS(P16, host) — and their relative magnitudes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"

CTGAN_FEATS = [
    "Driver", "Compound", "Race", "Year", "PitStop",
    "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
CAT_COLS = ["Driver", "Compound", "Race"]
NUM_COLS = [c for c in CTGAN_FEATS if c not in CAT_COLS]


def load_replay(path):
    if not path.exists(): return None
    return pd.read_parquet(path)


def feature_distance(host_col, other_col, is_cat):
    """KS for numeric, TV-distance for categorical."""
    h = host_col.dropna()
    o = other_col.dropna()
    if is_cat:
        h_counts = h.value_counts(normalize=True)
        o_counts = o.value_counts(normalize=True)
        all_cats = set(h_counts.index) | set(o_counts.index)
        tv = 0.5 * sum(abs(h_counts.get(c, 0) - o_counts.get(c, 0))
                       for c in all_cats)
        return float(tv)
    else:
        # KS on numeric (downsample for speed)
        n = min(50000, len(h), len(o))
        h_s = h.sample(n=n, random_state=42).to_numpy() if len(h) > n else h.to_numpy()
        o_s = o.sample(n=n, random_state=42).to_numpy() if len(o) > n else o.to_numpy()
        ks = ks_2samp(h_s, o_s)
        return float(ks.statistic)


def main():
    print("Loading host synth + orig...", flush=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    host = pd.concat([train[CTGAN_FEATS], test[CTGAN_FEATS]], ignore_index=True)
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv")
    orig = orig.dropna(subset=["Compound"])[CTGAN_FEATS]
    print(f"  host {host.shape} | orig {orig.shape}", flush=True)

    sources = {"orig": orig}
    for name, path in [
        ("P3 recursive replay", ART / "p3_ctgan_replay_disc_replay.parquet"),
        ("P13 CTGAN-on-orig", ART / "p13_orig_surrogate_v1_replay.parquet"),
        ("P16 TVAE-on-orig", ART / "p16_tvae_on_orig_replay.parquet"),
    ]:
        df = load_replay(path)
        if df is not None:
            sources[name] = df
            print(f"  {name}: {df.shape}", flush=True)
        else:
            print(f"  {name}: not available (skipping)", flush=True)

    # Compute distance per feature for each source
    rows = []
    for col in CTGAN_FEATS:
        is_cat = col in CAT_COLS
        row = {"feature": col, "is_cat": is_cat}
        for name, df in sources.items():
            d = feature_distance(host[col], df[col], is_cat)
            row[f"d_{name}"] = d
        rows.append(row)
    df_dist = pd.DataFrame(rows)
    print("\n=== Per-feature distance to host synth ===")
    print(df_dist.to_string(index=False))
    print(f"\n=== Mean distance ===")
    for col in df_dist.columns:
        if col.startswith("d_"):
            print(f"  {col[2:]:<25s}: {df_dist[col].mean():.4f}")

    summary = {"per_feature": rows,
               "mean_distance": {col[2:]: float(df_dist[col].mean())
                                 for col in df_dist.columns if col.startswith("d_")}}
    (ART / "p17_perfeature_ks.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved p17_perfeature_ks.json", flush=True)


if __name__ == "__main__":
    main()
