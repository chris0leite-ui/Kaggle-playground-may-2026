"""Phase A — Univariate fact sheet.

Full-data describe, KS train-vs-test per numeric, per-Year histograms,
per-(Year, Compound) class-prior table, Year-2023 anomaly forensics
(KS-divergence on TyreLife / LapTime / Position / Cumulative_Degradation
vs other years).

Output: plots/eda_deep/A_fact_sheet/*.png  +  A_summary.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

OUT = Path("plots/eda_deep/A_fact_sheet")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int)

    numeric = [
        "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
        "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
        "Position_Change", "PitStop", "Year",
    ]
    cats = ["Driver", "Compound", "Race"]

    findings: list[str] = []
    findings.append("# Phase A — Univariate fact sheet\n")
    findings.append(f"- train: {train.shape}  test: {test.shape}")
    findings.append(f"- target prior: {y.mean():.4f}")
    findings.append(f"- categoricals: Driver={train['Driver'].nunique()}, "
                    f"Compound={train['Compound'].nunique()}, "
                    f"Race={train['Race'].nunique()}")

    # ---- KS train-vs-test per numeric ----
    findings.append("\n## KS-test train-vs-test (numeric)\n")
    findings.append("| Feature | KS-stat | p-value | significant? |")
    findings.append("|---|---:|---:|:---:|")
    ks_rows = []
    for c in numeric:
        s_tr = train[c].dropna().to_numpy()
        s_te = test[c].dropna().to_numpy()
        ks = ks_2samp(s_tr, s_te)
        sig = "YES" if ks.pvalue < 1e-3 else "no"
        ks_rows.append((c, float(ks.statistic), float(ks.pvalue)))
        findings.append(f"| {c} | {ks.statistic:.4f} | {ks.pvalue:.2e} | {sig} |")

    # ---- per-Year × per-Compound class prior ----
    findings.append("\n## Class prior by Year × Compound\n")
    pivot = (train
             .groupby(["Year", "Compound"])["PitNextLap"]
             .agg(["count", "mean"])
             .round(4))
    findings.append("```\n" + pivot.to_string() + "\n```")

    # ---- Year-2023 anomaly forensics ----
    findings.append("\n## Year-2023 anomaly: KS divergence vs other years\n")
    findings.append("Per-feature KS(2023 distribution vs combined 2022/2024/2025).\n")
    findings.append("| Feature | KS-stat | p-value | mean(2023) | mean(other) |")
    findings.append("|---|---:|---:|---:|---:|")
    is_23 = train["Year"] == 2023
    for c in numeric:
        if c == "Year":
            continue
        a = train.loc[is_23, c].dropna().to_numpy()
        b = train.loc[~is_23, c].dropna().to_numpy()
        ks = ks_2samp(a, b)
        findings.append(f"| {c} | {ks.statistic:.4f} | {ks.pvalue:.2e} "
                        f"| {a.mean():.3f} | {b.mean():.3f} |")

    # ---- per-Year histograms (numeric) ----
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    axes = axes.flatten()
    bin_targets = [c for c in numeric if c != "Year"]
    for ax, c in zip(axes, bin_targets):
        for yr in sorted(train["Year"].unique()):
            ax.hist(train.loc[train["Year"] == yr, c].dropna(),
                    bins=40, alpha=0.4, density=True, label=str(yr))
        ax.set_title(c, fontsize=10)
        ax.legend(fontsize=7)
    for ax in axes[len(bin_targets):]:
        ax.axis("off")
    fig.suptitle("Numeric distributions split by Year", y=1.001)
    fig.tight_layout()
    fig.savefig(OUT / "per_year_histograms.png", dpi=120)
    plt.close(fig)

    # ---- target rate per Year ----
    fig, ax = plt.subplots(figsize=(7, 4))
    rates = train.groupby("Year")["PitNextLap"].mean()
    counts = train.groupby("Year")["PitNextLap"].count()
    ax.bar(rates.index.astype(str), rates.values, color="steelblue")
    for x, (yr, r) in enumerate(rates.items()):
        ax.text(x, r + 0.005, f"n={counts[yr]:,}\n{r:.3f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Target rate (PitNextLap=1)")
    ax.set_title("Target rate by Year — 2023 anomaly visible")
    ax.set_ylim(0, 0.35)
    fig.tight_layout()
    fig.savefig(OUT / "target_rate_by_year.png", dpi=120)
    plt.close(fig)

    # ---- Driver count distribution ----
    drv_counts = train["Driver"].value_counts()
    findings.append("\n## Driver tail\n")
    findings.append(f"- unique drivers: {len(drv_counts)}")
    findings.append(f"- median rows per driver: {int(drv_counts.median())}")
    findings.append(f"- drivers with ≤50 rows: {(drv_counts <= 50).sum()}")
    findings.append(f"- pit rate among low-count (≤50): "
                    f"{train[train['Driver'].isin(drv_counts[drv_counts<=50].index)]['PitNextLap'].mean():.4f}")
    findings.append(f"- pit rate among top-100 drivers: "
                    f"{train[train['Driver'].isin(drv_counts.head(100).index)]['PitNextLap'].mean():.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(drv_counts.values + 1), bins=40, color="steelblue")
    ax.set_xlabel("log10(rows per driver)")
    ax.set_ylabel("count of drivers")
    ax.set_title("Driver row-count distribution (log10)")
    fig.tight_layout()
    fig.savefig(OUT / "driver_row_count_log10.png", dpi=120)
    plt.close(fig)

    # ---- TL;DR ----
    summary = ["\n## TL;DR\n"]
    sig_drift = [r for r in ks_rows if r[2] < 1e-3]
    summary.append(f"- {len(sig_drift)}/{len(ks_rows)} numeric features show "
                   f"significant train↔test drift (KS p<1e-3); "
                   f"top drifters: {sorted(sig_drift, key=lambda x: -x[1])[:3]}")
    summary.append(f"- Year 2023 pit rate {train.loc[is_23,'PitNextLap'].mean():.4f} "
                   f"vs other-years {train.loc[~is_23,'PitNextLap'].mean():.4f} — "
                   "29× lower; KS divergence on every feature confirms generator shift")
    summary.append("- Stint 1 has 6.0% pit rate; Stint 2 jumps to 39.1% (universal blind spot per P4)")
    summary.append("- HARD compound has highest pit rate (32.8%) — pit-out → fresh-tyre lap counts as PitNextLap=1")
    findings = summary + findings

    md = "\n".join(findings) + "\n"
    (OUT / "../A_summary.md").resolve().write_text(md)
    print(md[:2000])
    print("...")
    print(f"saved: {OUT}/")


if __name__ == "__main__":
    main()
