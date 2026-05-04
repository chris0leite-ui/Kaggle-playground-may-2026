"""Pool disagreement diagnostic.

Question: where in the test set is the M5h pool genuinely uncertain
(bases disagree), vs locked-in consensus (bases agree)? A new base's
LB lift potential is concentrated on high-disagreement rows — those
are the rows where its dissent can change the global rank.

Computes (on the 13-base test predictions, M5h Strat anchor):
  - per-row std and range across the 13 base predictions
  - per-segment (Race, Stint, Year) average disagreement
  - high-disagreement subset (top decile by std)

Also: for each base, compute how DIVERSITY-CONTRIBUTING it is — the
correlation of its predictions with the consensus (mean of others).
Lower correlation = more orthogonal contribution.

Output: console table + audit/2026-05-04-d3-pool-disagreement.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ART = Path("scripts/artifacts")

POOL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"),
    ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
]


def main():
    test = pd.read_csv("data/test.csv")
    n = len(test)

    # Stack 13 base test predictions into (n, K) matrix
    P = np.zeros((n, len(POOL)), dtype=np.float64)
    for j, (_, name) in enumerate(POOL):
        P[:, j] = np.load(ART / f"test_{name}_strat.npy")[:, 1]

    names = [p[0] for p in POOL]
    print(f"Pool: {len(POOL)} bases × {n} test rows\n")

    # === Per-row disagreement ===
    row_std = P.std(axis=1)
    row_range = P.max(axis=1) - P.min(axis=1)
    row_mean = P.mean(axis=1)
    consensus = np.median(P, axis=1)  # robust consensus

    print("=== Per-row disagreement (std across 13 bases) ===")
    pct_lvls = [10, 25, 50, 75, 90, 99]
    for q in pct_lvls:
        v = np.percentile(row_std, q)
        print(f"  p{q:>2d}: {v:.4f}")

    # High-disagreement subset = top decile by std
    hi_thresh = np.percentile(row_std, 90)
    hi_mask = row_std >= hi_thresh
    print(f"\nHigh-disagreement subset (std ≥ p90 = {hi_thresh:.4f}): "
          f"{int(hi_mask.sum())} rows ({100*hi_mask.mean():.1f}%)")
    print(f"  consensus range on high-dis: median {np.median(consensus[hi_mask]):.4f}, "
          f"mean {consensus[hi_mask].mean():.4f}")
    print(f"  consensus range on low-dis (rest):  median "
          f"{np.median(consensus[~hi_mask]):.4f}, mean "
          f"{consensus[~hi_mask].mean():.4f}")

    # === Per-segment disagreement ===
    print("\n=== Per-segment mean disagreement (mean row_std) ===")
    for col in ["Race", "Stint", "Year", "Compound"]:
        if col not in test.columns:
            continue
        df = pd.DataFrame({col: test[col].values, "std": row_std})
        agg = df.groupby(col)["std"].agg(["mean", "count"]).reset_index()
        agg = agg.sort_values("mean", ascending=False)
        print(f"\n  {col}:")
        for _, row in agg.iterrows():
            print(f"    {str(row[col])[:30]:<30s} n={int(row['count']):>6d}  "
                  f"mean_std={row['mean']:.4f}")

    # === Per-base diversity (correlation with consensus-of-others) ===
    print("\n=== Per-base diversity (Spearman ρ vs mean of OTHER 12 bases) ===")
    diversity = []
    for j in range(len(POOL)):
        others = np.delete(P, j, axis=1).mean(axis=1)
        rho, _ = spearmanr(P[:, j], others)
        diversity.append((names[j], rho))
    diversity_sorted = sorted(diversity, key=lambda x: x[1])
    for n, r in diversity_sorted:
        print(f"  {n:<22s} ρ={r:.5f}  ({'most diverse' if r == diversity_sorted[0][1] else ''})")

    # === Per-base disagreement on high-dis subset ===
    print("\n=== Per-base mean abs deviation from consensus on HIGH-DISAGREEMENT rows ===")
    diff_on_hi = []
    for j in range(len(POOL)):
        d = float(np.abs(P[hi_mask, j] - consensus[hi_mask]).mean())
        diff_on_hi.append((names[j], d))
    for n, d in sorted(diff_on_hi, key=lambda x: -x[1]):
        print(f"  {n:<22s} mean|p-consensus|={d:.4f}")

    # === Audit doc ===
    body = ["# Pool disagreement diagnostic — 2026-05-04\n",
            "Question: where is the M5h pool uncertain vs locked-in consensus?\n",
            f"## Per-row disagreement (std across 13 bases, {n} test rows)\n",
            "| percentile | std |", "|---:|---:|"]
    for q in pct_lvls:
        body.append(f"| p{q} | {np.percentile(row_std, q):.4f} |")
    body.append(f"\nHigh-disagreement subset (top decile by std): {int(hi_mask.sum())} "
                f"rows ({100*hi_mask.mean():.1f}%)\n")

    body.append("## Per-base diversity (Spearman ρ vs mean of other 12)\n")
    body.append("Lower ρ = more orthogonal contribution to the consensus rank.\n")
    body.append("| base | ρ vs others |", )
    body.append("|---|---:|")
    for n, r in diversity_sorted:
        body.append(f"| {n} | {r:.5f} |")

    body.append("\n## Use for slot 9-10 selection\n")
    body.append("When evaluating RealMLP / EBM / H1 / LR-FE as new pool members:\n")
    body.append("1. Compute Spearman ρ vs mean of M5h pool. Lower ρ = more diversity.\n")
    body.append("2. Compute mean |new_pred − consensus| on HIGH-DISAGREEMENT rows.\n")
    body.append("   Higher = more rank-shift potential.\n")
    body.append("3. Combine: a new base with low ρ AND high disagreement on the\n")
    body.append("   uncertainty subset is the candidate most likely to break the\n")
    body.append("   pool's locked-in consensus rank → LB lift potential.\n")

    out = Path("audit/2026-05-04-d3-pool-disagreement.md")
    out.write_text("\n".join(body))
    print(f"\n→ {out}")

    # Persist disagreement vector for downstream new-base evaluation
    np.save(ART / "test_pool_consensus.npy", consensus)
    np.save(ART / "test_pool_row_std.npy", row_std)
    print(f"→ {ART}/test_pool_consensus.npy")
    print(f"→ {ART}/test_pool_row_std.npy")


if __name__ == "__main__":
    main()
