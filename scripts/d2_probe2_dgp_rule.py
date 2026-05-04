"""Day-2 probe #2 — DGP rule probe (irrigation-water transfer).

Fit a shallow decision tree on PitNextLap; check if leaf entropy is
low for a non-trivial fraction of rows. If yes, the comp likely has
rule-structure and trees rediscover interactions natively
(irrigation: +84 bp from the closed-form rule; hand-FE on top
regressed -52 bp).

Reports per-depth (3, 5, 7):
  - tree accuracy / AUC
  - per-leaf entropy distribution
  - fraction of rows in low-entropy leaves (<0.10)
  - top splits (the candidate "rule")

Writes audit/<date>-d2-probe2-dgp-rule.md.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

train = pd.read_csv("data/train.csv")
y = train["PitNextLap"].astype(int).values
X = train.drop(columns=["id", "PitNextLap"], errors="ignore")
cat_cols = X.select_dtypes(include="object").columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category").cat.codes  # tree wants numeric


def leaf_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def probe(depth: int) -> dict:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42,
                                 min_samples_leaf=200)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, proba))

    # per-leaf statistics
    leaf_ids = clf.apply(X)
    leaf_df = pd.DataFrame({"leaf": leaf_ids, "y": y})
    g = leaf_df.groupby("leaf")["y"].agg(["count", "mean"])
    g["entropy"] = g["mean"].apply(leaf_entropy)
    g = g.sort_values("count", ascending=False)

    n_total = len(y)
    n_low_ent = int(g.loc[g.entropy < 0.10, "count"].sum())
    frac_low_ent = n_low_ent / n_total

    return dict(
        depth=depth,
        n_leaves=len(g),
        auc=auc,
        n_low_ent_rows=n_low_ent,
        frac_low_ent_rows=frac_low_ent,
        clf=clf,
        leaf_table=g,
    )


results = [probe(d) for d in (3, 5, 7)]

date = dt.date.today().isoformat()
out = Path(f"audit/{date}-d2-probe2-dgp-rule.md")
lines = [
    f"# Day-2 probe #2 — DGP rule probe ({date})",
    "",
    "Method: shallow `DecisionTreeClassifier` (max_depth=3,5,7; "
    "min_samples_leaf=200) on PitNextLap. If a meaningful fraction of "
    "rows fall into low-entropy leaves (entropy < 0.10), the data has "
    "rule-structure and trees rediscover interactions natively.",
    "",
    "Reference: irrigation-water PM-03 §1 — closed-form 6-feature rule "
    "drove +84bp; hand-FE on top of DGP regressed -52bp.",
    "",
    "## Results",
    "",
    "| depth | n_leaves | AUC | rows in low-entropy leaves | frac |",
    "|---:|---:|---:|---:|---:|",
]
for r in results:
    lines.append(
        f"| {r['depth']} | {r['n_leaves']} | {r['auc']:.5f} | "
        f"{r['n_low_ent_rows']:,} | {r['frac_low_ent_rows']:.4f} |"
    )

# Verdict
best = max(results, key=lambda r: r["frac_low_ent_rows"])
if best["frac_low_ent_rows"] >= 0.50:
    verdict = (
        f"**RULE-STRUCTURED.** At depth {best['depth']}, "
        f"{best['frac_low_ent_rows']*100:.1f}% of rows are in "
        f"low-entropy leaves. The DGP appears to have a closed-form "
        f"component. Worth probing for an integer-thresholded rule "
        f"on top features (irrigation pattern). Hand-FE on top of "
        f"the rule is likely to regress."
    )
elif best["frac_low_ent_rows"] >= 0.20:
    verdict = (
        f"**PARTIALLY RULE-STRUCTURED.** {best['frac_low_ent_rows']*100:.1f}% "
        f"of rows in low-entropy leaves at depth {best['depth']}. The DGP "
        f"has some deterministic structure but isn't a simple rule. "
        f"Trees should still capture it; hand-FE risk is moderate."
    )
else:
    verdict = (
        f"**NOT RULE-STRUCTURED.** Only {best['frac_low_ent_rows']*100:.1f}% "
        f"of rows are in low-entropy leaves. PitNextLap is a smooth "
        f"function of features, not a closed-form rule. Hand-FE / NN "
        f"approximators (RealMLP, EmbMLP) should compete with trees "
        f"— consistent with the cross-comp research finding that NN "
        f"is load-bearing on this DGP (analyticaobscura)."
    )

lines += ["", "## Verdict", "", verdict]

# Top splits at depth=3 (readable)
lines += ["", "## Tree text (depth=3)", "", "```",
          export_text(results[0]["clf"], feature_names=list(X.columns),
                      max_depth=3),
          "```", ""]

# Top 10 leaves at depth=5 (largest population)
lines += ["## Top 10 leaves at depth=5", "",
          "| leaf | count | target_rate | entropy |",
          "|---:|---:|---:|---:|"]
for leaf_id, row in results[1]["leaf_table"].head(10).iterrows():
    lines.append(
        f"| {leaf_id} | {int(row['count']):,} | {row['mean']:.4f} | "
        f"{row['entropy']:.4f} |"
    )

out.write_text("\n".join(lines) + "\n")
print("\n".join(lines[:60]))
print(f"\n→ written to {out}")
