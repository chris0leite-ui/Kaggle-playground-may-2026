"""Day-2 probe #1 — external-dataset join (S4E1 `Exited_Orig` pattern).

Original (aadigupta1601) has the same 15 columns + PitNextLap +
Normalized_TyreLife. Try to recover PitNextLap for our train and
test rows by joining on row-identifier keys.

Reports:
  - per-key match rate on train and test
  - target-value agreement on matched train rows (fraction where
    recovered PitNextLap == actual PitNextLap)
  - per-key recommendation

Writes audit/<date>-d2-probe1-external-join.md.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ext = pd.read_csv("data/external/f1_strategy_dataset_v4.csv")

print(f"train: {train.shape}  test: {test.shape}  ext: {ext.shape}")
print(f"ext drivers: {ext.Driver.nunique()};  train drivers: {train.Driver.nunique()};  overlap: {len(set(ext.Driver) & set(train.Driver))}")
print(f"ext races:   {ext.Race.nunique()};   train races:   {train.Race.nunique()};   overlap: {len(set(ext.Race) & set(train.Race))}")
print(f"ext years:   {sorted(ext.Year.unique())};  train years:   {sorted(train.Year.unique())}")

KEYS = [
    ["Driver", "Race", "Year", "LapNumber"],
    ["Driver", "Race", "Year", "LapNumber", "Compound"],
    ["Driver", "Race", "Year", "LapNumber", "Compound", "Stint", "TyreLife"],
    ["Driver", "Race", "Year", "LapNumber", "Stint", "TyreLife", "Position"],
]

results = []
for key in KEYS:
    if not all(c in ext.columns for c in key):
        continue
    # left-merge train onto ext to recover ext.PitNextLap
    j = train.merge(ext[key + ["PitNextLap"]].drop_duplicates(key),
                    on=key, how="left", suffixes=("", "_ext"))
    matched = j["PitNextLap_ext"].notna()
    train_match_rate = matched.mean()
    if matched.sum() > 0:
        target_agree = (j.loc[matched, "PitNextLap"]
                        == j.loc[matched, "PitNextLap_ext"]).mean()
    else:
        target_agree = float("nan")

    j_test = test.merge(ext[key + ["PitNextLap"]].drop_duplicates(key),
                        on=key, how="left")
    test_match_rate = j_test["PitNextLap"].notna().mean()
    results.append(dict(
        key="+".join(key),
        n_key_cols=len(key),
        train_match_rate=train_match_rate,
        train_matched_target_agree=target_agree,
        test_match_rate=test_match_rate,
    ))
    print(f"key={key}: train_match={train_match_rate:.4f}, "
          f"train_agree(if matched)={target_agree:.4f}, "
          f"test_match={test_match_rate:.4f}")

# Also: pure value-based fuzzy match — for train rows where row hash differs
# slightly, do they exist in ext after rounding floats to fewer decimals?
# (Skip for now; if exact (Driver, Race, Year, LapNumber) hits ≥10%, we have
# enough signal.)

df = pd.DataFrame(results)

# Pick best key
recoverable_test_rows = df.test_match_rate.max() if not df.empty else 0
verdict_lines = []
if recoverable_test_rows >= 0.10:
    best = df.loc[df.test_match_rate.idxmax()]
    verdict_lines.append(
        f"**JOIN HITS** at key=`{best['key']}`. "
        f"train_match={best['train_match_rate']:.4f}, "
        f"train_target_agree(if matched)={best['train_matched_target_agree']:.4f}, "
        f"test_match={best['test_match_rate']:.4f}."
    )
    if best["train_matched_target_agree"] >= 0.99:
        verdict_lines.append(
            "Target values agree 99%+ on matched train rows — the original "
            "dataset is essentially the SOURCE of our train labels. Joining "
            "on test recovers ground truth for the matched fraction. "
            "**This is structural information, not leakage from train labels.**"
        )
    elif best["train_matched_target_agree"] >= 0.80:
        verdict_lines.append(
            "Target agreement is high but not perfect on matched train rows. "
            "The host applied stochastic perturbation — the join is a strong "
            "feature but not a deterministic lookup."
        )
    else:
        verdict_lines.append(
            "Target agreement on matched train rows is weak — the host "
            "shuffled labels. Join indicator may still help via leakage of "
            "OTHER columns (e.g. Normalized_TyreLife)."
        )
else:
    verdict_lines.append(
        f"**JOIN MISSES** — best test match rate is "
        f"{recoverable_test_rows:.4f} (< 10% threshold). The host "
        f"shuffled or synthesized rows beyond the original. "
        f"Move on to probe #2 (DGP-rule probe)."
    )

# Bonus: check Normalized_TyreLife recoverability (host removed this column;
# if we can recover it via the same join, it's a known-strong leaked feature)
if "Normalized_TyreLife" in ext.columns and not df.empty:
    best_key = df.loc[df.test_match_rate.idxmax(), "key"].split("+")
    j_test = test.merge(
        ext[best_key + ["Normalized_TyreLife"]].drop_duplicates(best_key),
        on=best_key, how="left")
    ntl_recovery = j_test["Normalized_TyreLife"].notna().mean()
    verdict_lines.append(
        f"\n`Normalized_TyreLife` (host-removed) recoverable on "
        f"{ntl_recovery*100:.2f}% of test rows via the same key. "
        + ("**Host explicitly forbade reintroducing this column** "
           "(brief.md). Do NOT use it." if ntl_recovery > 0
           else "")
    )

date = dt.date.today().isoformat()
out = Path(f"audit/{date}-d2-probe1-external-join.md")
lines = [
    f"# Day-2 probe #1 — external-dataset join ({date})",
    "",
    f"Original dataset: `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`",
    f"Original shape: {ext.shape};  s6e5 train: {train.shape};  s6e5 test: {test.shape}",
    "",
    "## Match rate by key",
    "",
    "| key | n_cols | train_match | train_target_agree | test_match |",
    "|---|---:|---:|---:|---:|",
]
for r in results:
    lines.append(
        f"| `{r['key']}` | {r['n_key_cols']} | "
        f"{r['train_match_rate']:.4f} | "
        f"{r['train_matched_target_agree']:.4f} | "
        f"{r['test_match_rate']:.4f} |"
    )
lines += ["", "## Verdict", "", *verdict_lines]

out.write_text("\n".join(lines) + "\n")
print(f"\n→ written to {out}")
