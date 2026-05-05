"""Per-segment OOF AUC diagnostic on M5h (Strat anchor).

Strategy critique 2026-05-04 (audit/2026-05-04-strategy-critique.md
item "Data-understanding gaps"): we have aggregate AUC but not
per-Race / per-Stint / per-Year / per-Compound. A 38bp headroom could
be 10 races at +5bp and 16 at −20bp; lift surface lives where the
model is *bad*.

Output: console tables + audit/2026-05-04-d3-per-segment-oof.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5H_AGG_S = 0.95043


def safe_auc(y, p):
    """AUC with both classes required."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, p))


def per_group(train, oof, by, label):
    rows = []
    for g, sub in train.groupby(by, sort=True):
        idx = sub.index.values
        n = len(idx)
        pos = int(sub[TARGET].sum())
        if n < 50:  # skip vanishingly small groups
            continue
        auc = safe_auc(sub[TARGET].values.astype(int), oof[idx])
        rows.append((g, n, pos, auc))
    df = pd.DataFrame(rows, columns=[label, "n", "pos", "auc"])
    df["n"] = df["n"].astype(int)
    df["pos"] = df["pos"].astype(int)
    df["pos_rate"] = df["pos"] / df["n"]
    df["delta_bp"] = (df["auc"] - M5H_AGG_S) * 1e4
    return df.sort_values("auc")


def fmt_table(df, label, max_rows=None):
    show = df if max_rows is None else df.head(max_rows)
    body = "\n".join(
        f"| {row[label]} | {int(row['n']):>6d} | {int(row['pos']):>5d} | "
        f"{row['pos_rate']:.4f} | {row['auc']:.5f} | {row['delta_bp']:+.1f} |"
        for _, row in show.iterrows()
    )
    head = (f"| {label} | n | pos | pos_rate | OOF AUC | Δ M5h-agg (bp) |\n"
            f"|---|---:|---:|---:|---:|---:|")
    return head + "\n" + body


def main():
    train = pd.read_csv("data/train.csv")
    oof = np.load(ART / "oof_m5h_strat.npy")[:, 1].astype(np.float64)
    print(f"Aggregate Strat OOF AUC (sanity): {safe_auc(train[TARGET].astype(int).values, oof):.5f}")
    print(f"Reference M5h Strat: {M5H_AGG_S:.5f}\n")

    # Stint may bin large; LapNumber bin to deciles for readability
    train = train.copy()
    train["LapDecile"] = pd.qcut(train["LapNumber"], q=10, labels=False, duplicates="drop")
    train["TyreDecile"] = pd.qcut(train["TyreLife"], q=10, labels=False, duplicates="drop")

    segs = [
        ("Race", "Race", None),
        ("Year", "Year", None),
        ("Stint", "Stint", None),
        ("Compound", "Compound", None),
        ("LapDecile", "LapDecile", None),
        ("TyreDecile", "TyreDecile", None),
    ]

    body = ["# M5h per-segment OOF (Strat) — 2026-05-04\n",
            f"Aggregate Strat OOF: **{M5H_AGG_S:.5f}**. "
            f"Per-segment AUC with Δ vs aggregate.\n"]

    print_blocks = []
    for label, col, max_rows in segs:
        print(f"\n=== {label} ===")
        df = per_group(train, oof, col, label)
        # Show all (most segments are small enough)
        tbl = fmt_table(df, label, max_rows=None)
        print(tbl)
        body.append(f"\n## {label}\n\n{tbl}\n")

        # Summary stat: spread (worst − best), median, weighted mean
        wmean = (df["auc"] * df["n"]).sum() / df["n"].sum()
        body.append(
            f"\n- spread: {df['auc'].min():.5f} → {df['auc'].max():.5f} "
            f"({(df['auc'].max()-df['auc'].min())*1e4:+.1f}bp)  \n"
            f"- median: {df['auc'].median():.5f}  \n"
            f"- weighted-by-n mean: {wmean:.5f}\n"
        )

    out = Path("audit/2026-05-04-d3-per-segment-oof.md")
    out.write_text("\n".join(body))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
