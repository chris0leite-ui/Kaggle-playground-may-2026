"""T1.3 prerequisite — Q12 group atomicity check.

The F1 domain agent flagged: Q12 mandatory-2-compound feature
depends on (Driver, Race, Year) groups being ATOMICALLY sampled
in the synthetic DGP. If the host's CTGAN-style generator
preserved groups jointly, n_distinct_compounds within a group
should peak at 2-3 (matching real F1 regulation: ≥2 distinct
dry compounds required per race).

If it peaks at 1 or has high mass at 1, groups are independently
sampled rows, the regulatory constraint is broken, and the Q12
forced-pit feature would be noisy.

Probe: only ~1 min CPU.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")


def main():
    t = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"train {train.shape}; test {test.shape}; loaded {time.time()-t:.1f}s")

    # Combine train + test for group analysis (rows are i.i.d. shuffled but
    # the (Driver, Race, Year) tuple defines the group)
    df = pd.concat([train.assign(src="train"), test.assign(src="test")],
                   ignore_index=True)
    print(f"combined {df.shape}")

    # Distinct compounds per (Driver, Race, Year) group
    g = df.groupby(["Driver", "Race", "Year"])
    n_compounds = g["Compound"].nunique()
    grp_size = g.size()
    print(f"\n=== n_distinct_compounds per (Driver, Race, Year) ===")
    print(f"  total groups: {len(n_compounds)}")
    vc = n_compounds.value_counts().sort_index()
    for k, v in vc.items():
        pct = v / len(n_compounds) * 100
        print(f"  {k} compounds: {v:>6d} groups ({pct:.1f}%)")

    # Conditional on group size: if group has >=10 laps, what's the dist?
    big = n_compounds[grp_size >= 10]
    print(f"\n=== conditional on group size >=10 (n={len(big)}) ===")
    vc = big.value_counts().sort_index()
    for k, v in vc.items():
        pct = v / len(big) * 100 if len(big) > 0 else 0
        print(f"  {k} compounds: {v:>6d} groups ({pct:.1f}%)")

    big = n_compounds[grp_size >= 30]
    print(f"\n=== conditional on group size >=30 (full-race-ish, n={len(big)}) ===")
    vc = big.value_counts().sort_index()
    for k, v in vc.items():
        pct = v / len(big) * 100 if len(big) > 0 else 0
        print(f"  {k} compounds: {v:>6d} groups ({pct:.1f}%)")

    # By Year (does 2023 break the rule? P3 says 2023 is mode-collapsed)
    print(f"\n=== by Year ===")
    for yr in sorted(df["Year"].unique()):
        g_yr = df[df["Year"] == yr].groupby(["Driver", "Race"])
        n_yr = g_yr["Compound"].nunique()
        sz_yr = g_yr.size()
        big = n_yr[sz_yr >= 10]
        if len(big) == 0:
            continue
        share_2plus = (big >= 2).mean()
        print(f"  {yr}: {len(n_yr):>5d} groups; "
              f"size>=10 {len(big):>5d}; "
              f">=2 distinct compounds {share_2plus:.1%}")

    # Critical: in real F1 dry races, n_distinct should be >=2 for ~all
    # rows in groups with size >= 10 (the race must have completed enough
    # laps for the constraint to bite). If much less, generator broke
    # groups.

    # Also: are there WET-only groups? In real F1, WET races may not need
    # the 2-compound rule. Let's check.
    df_dry = df[~df["Compound"].isin(["WET", "INTERMEDIATE"])]
    print(f"\n=== dry-only (excluding WET/INTERMEDIATE) ===")
    g_dry = df_dry.groupby(["Driver", "Race", "Year"])
    n_dry = g_dry["Compound"].nunique()
    sz_dry = g_dry.size()
    big_dry = n_dry[sz_dry >= 10]
    print(f"  groups with >=10 laps: {len(big_dry)}")
    if len(big_dry) > 0:
        share_1 = (big_dry == 1).mean()
        share_2 = (big_dry == 2).mean()
        share_3 = (big_dry >= 3).mean()
        print(f"  share with exactly 1 compound: {share_1:.1%}")
        print(f"  share with exactly 2 compounds: {share_2:.1%}")
        print(f"  share with >=3 compounds: {share_3:.1%}")

    # Verdict logic
    print(f"\n=== Verdict ===")
    # In real F1: dry races with >=10 laps should have >=2 compounds for ~99%.
    # If our synthetic data shows <80%, the regulatory rule was broken.
    if len(big_dry) > 0:
        share_2plus = (big_dry >= 2).mean()
        if share_2plus >= 0.95:
            print(f"  GROUP ATOMICITY: STRONG (>=95% multi-compound on size>=10 dry)")
            print(f"  -> Q12 forced-pit feature is well-defined; PROCEED")
        elif share_2plus >= 0.80:
            print(f"  GROUP ATOMICITY: MODERATE ({share_2plus:.1%}) — feature works"
                  f" but with noise on {1-share_2plus:.1%} of groups")
            print(f"  -> Q12 forced-pit feature OK; expect modest lift; PROCEED")
        else:
            print(f"  GROUP ATOMICITY: WEAK ({share_2plus:.1%}) — generator broke"
                  f" groups; feature will be noisy")
            print(f"  -> consider holding Q12; pivot to Q7/Q3 features")

    print(f"\nwall {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()
