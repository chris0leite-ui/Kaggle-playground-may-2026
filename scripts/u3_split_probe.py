"""U3 probe — is the train/test split i.i.d. row-level or
contiguous-within-(Race, Driver)?

Walk each (Race, Driver) group with rows in BOTH train AND test,
sort by LapNumber, count source alternations.
- alt_ratio ≈ 0.5 → fully interleaved (i.i.d. row split)
- alt_ratio ≈ 0    → contiguous (train-then-test)

Writes audit/<date>-u3-split-probe.md.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train["src"] = "train"
test["src"] = "test"
both = pd.concat([train, test], ignore_index=True)

rows = []
for (race, drv), g in both.groupby(["Race", "Driver"]):
    if g.src.nunique() < 2:
        continue
    g = g.sort_values("LapNumber")
    src = g.src.values
    alt = (src[1:] != src[:-1]).sum()
    rows.append({"race": race, "driver": drv, "n": len(g),
                 "train_n": (src == "train").sum(),
                 "test_n": (src == "test").sum(),
                 "alt": int(alt),
                 "ratio": float(alt / max(len(g) - 1, 1)),
                 "lap_min_train": g[g.src == "train"].LapNumber.min(),
                 "lap_max_train": g[g.src == "train"].LapNumber.max(),
                 "lap_min_test": g[g.src == "test"].LapNumber.min(),
                 "lap_max_test": g[g.src == "test"].LapNumber.max()})

df = pd.DataFrame(rows)
mean_r = df.ratio.mean()
median_r = df.ratio.median()
iid_like = (df.ratio > 0.40).sum()
contiguous = (df.ratio < 0.05).sum()
overlap = ((df.lap_min_test <= df.lap_max_train)
           & (df.lap_min_train <= df.lap_max_test)).sum()

if mean_r > 0.40:
    verdict = ("**i.i.d. row-level split** — within-group laps "
               "interleave between train and test. lead(PitStop) "
               "features are computable on test directly. "
               "GroupKFold(Race) is the right anchor.")
elif mean_r < 0.10:
    verdict = ("**contiguous within-group split** — train holds "
               "early laps, test holds late (or reverse). "
               "lead(PitStop) on test has tail-of-race gaps. "
               "Temporal CV is the honest validator.")
else:
    verdict = "**mixed** — neither pure i.i.d. nor pure contiguous."

# 3 random sample sequences (T = train row, t = test row, sorted by lap)
sample_lines = []
for _, row in df.sample(min(5, len(df)), random_state=42).iterrows():
    g = both[(both.Race == row.race)
             & (both.Driver == row.driver)].sort_values("LapNumber")
    seq = "".join("T" if s == "train" else "t" for s in g.src.values)
    sample_lines.append(
        f"- ({row.race[:24]:24s}, {row.driver:5s}): n={row.n:3d}, "
        f"alt={row.alt:3d}, ratio={row.ratio:.3f}, seq={seq[:80]}"
    )

out_lines = [
    f"# U3 probe — train/test split structure ({dt.date.today()})",
    "",
    f"Total (Race, Driver) groups present in BOTH train AND test: {len(df)}",
    "",
    "## Headline",
    f"- mean alt-ratio per group: **{mean_r:.4f}**",
    f"- median: {median_r:.4f}",
    f"- groups with alt-ratio > 0.40 (i.i.d.-like): {iid_like} / {len(df)}",
    f"- groups with alt-ratio < 0.05 (contiguous-like): {contiguous} / {len(df)}",
    f"- groups where train lap-range OVERLAPS test lap-range: {overlap} / {len(df)}",
    "",
    "## Verdict",
    verdict,
    "",
    "## Sample sequences (5 random groups, T=train row, t=test row, sorted by lap)",
    "",
    *sample_lines,
]

out = Path(f"audit/{dt.date.today().isoformat()}-u3-split-probe.md")
out.write_text("\n".join(out_lines) + "\n")
print("\n".join(out_lines))
print(f"\n→ written to {out}")
