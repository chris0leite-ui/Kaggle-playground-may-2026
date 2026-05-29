"""R22 v3 — full deduplication + diversity scan across all 23 scraped sources."""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import glob

ROOT = "/home/user/Kaggle-playground-may-2026"
PUB = f"{ROOT}/submissions/public_scrape"
PRIMARY = f"{ROOT}/submissions/submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv"

# Walk all CSVs in PUB (skip _blends and rho matrices)
candidates = []
for root, dirs, files in os.walk(PUB):
    if "_blends" in root or root == PUB:
        continue
    for fn in files:
        if fn.endswith(".csv") and "report" not in fn.lower() and "oof" not in fn.lower() and "diagnostic" not in fn.lower():
            candidates.append(os.path.join(root, fn))

# Load PRIMARY
df_pri = pd.read_csv(PRIMARY)
pri_ids = df_pri["id"].values
pri = df_pri.set_index("id")["PitNextLap"]
N = len(pri)
print(f"PRIMARY n={N}")

records = {"PRIMARY_C5": pri.values}
labels = {"PRIMARY_C5": PRIMARY}

for p in sorted(candidates):
    try:
        df = pd.read_csv(p)
        if "id" not in df.columns:
            continue
        cols = [c for c in df.columns if c.lower() != "id"]
        if len(cols) != 1 or len(df) != N:
            print(f"  SKIP {p} cols={cols} n={len(df)}")
            continue
        s = df.set_index("id")[cols[0]].reindex(pri_ids)
        if s.isna().any():
            print(f"  SKIP {p} (na after reindex)")
            continue
        # short label from path
        parent = os.path.basename(os.path.dirname(p))
        if parent == "max" or parent == "pro":
            grand = os.path.basename(os.path.dirname(os.path.dirname(p)))
            base = f"{grand[:18]}_{parent}_{os.path.splitext(os.path.basename(p))[0]}"
        else:
            short = parent.split("_")[0][:8]
            tail = os.path.splitext(os.path.basename(p))[0][:18]
            base = f"{short}__{tail}"
        if base in records:
            base = base + "_2"
        records[base] = s.values
        labels[base] = p
    except Exception as e:
        print(f"  SKIP {p}: {e}")

print(f"\nLoaded {len(records)-1} public sources + 1 PRIMARY = {len(records)} columns")
arr = np.array(list(records.values()))
names = list(records.keys())

# Rank-normalize and compute ρ matrix
ranks = np.array([rankdata(row, method="average") for row in arr])
rho = np.corrcoef(ranks)
rho_df = pd.DataFrame(rho, index=names, columns=names)

# Find duplicates (ρ >= 0.99999)
print("\n--- Identical-pair groups (ρ ≥ 0.99999) ---")
seen = set()
dups = {}
for i, n1 in enumerate(names):
    if n1 in seen:
        continue
    group = [n1]
    for j, n2 in enumerate(names):
        if i == j or n2 in seen:
            continue
        if rho[i, j] >= 0.99999:
            group.append(n2)
            seen.add(n2)
    if len(group) > 1:
        dups[n1] = group
        for g in group:
            seen.add(g)
        print(f"  {group[0]} = {group[1:]}")

# Distinct lineages: keep first member of each dup group, plus all singletons
keep = []
seen_clean = set()
for n in names:
    if n in seen_clean:
        continue
    keep.append(n)
    if n in dups:
        for g in dups[n]:
            seen_clean.add(g)
    seen_clean.add(n)
print(f"\nDistinct lineages: {len(keep)} (was {len(names)})")
for k in keep:
    print(f"  {k}")

# Compute ρ to PRIMARY for each distinct lineage
print("\n--- ρ vs PRIMARY (sorted most-diverse first) ---")
rho_pri = rho_df["PRIMARY_C5"].drop("PRIMARY_C5").loc[[k for k in keep if k != "PRIMARY_C5"]].sort_values()
for n, r in rho_pri.items():
    p = labels[n]
    sz = os.path.getsize(p)
    print(f"  {n:50s} ρ={r:.5f}  size={sz}")

# Also show ρ vs A_raunakdey for anchor-relative diversity
# need the raunakdey name in keep
raunakdey_key = None
for k in keep:
    if "raunakdey" in k.lower():
        raunakdey_key = k
        break
if raunakdey_key:
    print(f"\n--- ρ vs anchor ({raunakdey_key}) ---")
    rho_a = rho_df[raunakdey_key].drop(raunakdey_key).loc[[k for k in keep if k != raunakdey_key]].sort_values()
    for n, r in rho_a.items():
        print(f"  {n:50s} ρ_A={r:.5f}")

# Save deduplicated ρ matrix for reference
rho_df.loc[keep, keep].to_csv(f"{PUB}/_blends/rho_matrix_v3_distinct.csv")
print(f"\nSaved distinct-lineage ρ matrix to {PUB}/_blends/rho_matrix_v3_distinct.csv")
