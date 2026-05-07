"""scripts/probe_id_order_audit.py — id / row-order artifact probe.

Synthetic-data lens: the host's generator may leave sequence
regularities (target rate drift across id, lap_number cycles, etc.).
Read-only audit; no model. ~30 sec.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    n = len(train)

    summary = {}

    # 1. Target rate by id buckets (50 equal-size)
    buckets = np.linspace(0, n, 51).astype(int)
    means = []
    for i in range(50):
        means.append(float(y[buckets[i]:buckets[i+1]].mean()))
    rates = np.array(means)
    summary["id_buckets_50"] = dict(
        min=float(rates.min()), max=float(rates.max()),
        std=float(rates.std()), span_bp=(rates.max() - rates.min()) * 1e4,
        first_5=rates[:5].tolist(), last_5=rates[-5:].tolist(),
    )

    # 2. By LapNumber (mod 10, mod 5, mod 7)
    lap = train["LapNumber"].astype(int).values
    for mod in [3, 5, 7, 10]:
        bins = lap % mod
        rates = np.array([y[bins == k].mean() for k in range(mod)])
        summary[f"LapNumber_mod_{mod}"] = dict(
            rates=rates.tolist(), span_bp=(rates.max() - rates.min()) * 1e4,
        )

    # 3. id mod K — does id encode anything?
    for mod in [2, 3, 5, 7, 11, 13, 100, 1000]:
        bins = train["id"].values % mod
        # Just check std of target rate across bins
        rates = []
        for k in range(min(mod, 20)):
            sel = bins == k
            if sel.sum() > 100:
                rates.append(float(y[sel].mean()))
        if rates:
            arr = np.array(rates)
            summary[f"id_mod_{mod}"] = dict(
                std=float(arr.std()), span_bp=(arr.max() - arr.min()) * 1e4,
                n_bins=len(rates),
            )

    # 4. Same id-mod check on test (to see if test order is similar to train)
    tid = test["id"].values
    test_id_dist = dict(
        min=int(tid.min()), max=int(tid.max()),
        train_min=int(train["id"].min()), train_max=int(train["id"].max()),
        test_in_train_range=int(((tid >= train["id"].min()) &
                                  (tid <= train["id"].max())).sum()),
    )
    summary["id_dist"] = test_id_dist

    # Decision rule: span > 50 bp on a row-order axis is a signal.
    print("=== id / order audit ===")
    print(f"id_buckets_50 span: {summary['id_buckets_50']['span_bp']:.1f} bp")
    print(f"  first 5 buckets: {summary['id_buckets_50']['first_5']}")
    print(f"  last  5 buckets: {summary['id_buckets_50']['last_5']}")
    for k in summary:
        if k.startswith("LapNumber_mod_"):
            v = summary[k]
            print(f"{k} span: {v['span_bp']:.1f} bp, rates {v['rates']}")
        elif k.startswith("id_mod_"):
            v = summary[k]
            print(f"{k} std={v['std']:.4f} span {v['span_bp']:.1f} bp "
                  f"(n_bins={v['n_bins']})")

    print(f"\nid range: train [{summary['id_dist']['train_min']}, "
          f"{summary['id_dist']['train_max']}]; test in-range "
          f"{summary['id_dist']['test_in_train_range']}/{len(test)}")

    out = ART / "probe_id_order_audit.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
