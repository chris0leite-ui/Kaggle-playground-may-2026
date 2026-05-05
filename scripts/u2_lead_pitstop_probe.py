"""U2 probe — single-feature OOF AUC for lead(PitStop).

Concat train+test, sort by (Race, Driver, LapNumber), shift PitStop
by -1 within group → lead_PitStop. (Uses test's PitStop column,
which is a feature not the target — not leakage.)

Train 5-fold StratifiedKFold LGBM on lead_PitStop alone (and three
sanity-comparison single features). Report OOF AUC per feature.

Writes audit/<date>-u2-lead-pitstop-probe.md.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Tag origin and concat to compute within-(Race, Driver) lead/lag
train2 = train.assign(_orig=np.arange(len(train)), _src="train")
test2 = test.assign(_orig=np.arange(len(test)), _src="test")
both = pd.concat([train2, test2], ignore_index=True)
both = both.sort_values(["Race", "Driver", "LapNumber"])

g = both.groupby(["Race", "Driver"])
both["lead_PitStop"] = g["PitStop"].shift(-1)
both["lag_PitStop"] = g["PitStop"].shift(1)

train_aug = (both[both._src == "train"]
             .sort_values("_orig").reset_index(drop=True))
y = train_aug["PitNextLap"].astype(int).values


def cv_auc(X: np.ndarray, name: str) -> tuple[float, list[float]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    fold_scores = []
    for tr, va in skf.split(X, y):
        dtrain = lgb.Dataset(X[tr], y[tr])
        dval = lgb.Dataset(X[va], y[va])
        m = lgb.train(
            dict(objective="binary", learning_rate=0.05,
                 num_leaves=15, min_data_in_leaf=200, verbose=-1, seed=42),
            dtrain, num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )
        p = m.predict(X[va])
        oof[va] = p
        fold_scores.append(float(roc_auc_score(y[va], p)))
    return float(roc_auc_score(y, oof)), fold_scores


probes = {
    "lead_PitStop (next-lap PitStop, with -1 sentinel for last lap)": (
        train_aug[["lead_PitStop"]].fillna(-1).values),
    "PitStop (this-lap, baseline reference)": (
        train_aug[["PitStop"]].values.astype(float)),
    "lag_PitStop (prev-lap, with -1 sentinel for first lap)": (
        train_aug[["lag_PitStop"]].fillna(-1).values),
    "TyreLife (top numeric by F-stat)": (
        train_aug[["TyreLife"]].values),
    "lead_PitStop + PitStop + TyreLife (3-feature heuristic)": (
        train_aug[["lead_PitStop", "PitStop", "TyreLife"]]
        .fillna(-1).values),
}

results = {}
for name, X in probes.items():
    oof_auc, folds = cv_auc(X, name)
    results[name] = (oof_auc, folds)
    print(f"  {name}: OOF AUC = {oof_auc:.5f}  folds={[f'{f:.4f}' for f in folds]}")

# Also: how often is lead_PitStop usable (non-NaN) on test rows?
test_aug = both[both._src == "test"]
test_lead_avail = test_aug["lead_PitStop"].notna().sum() / len(test_aug)

lines = [
    f"# U2 probe — lead(PitStop) single-feature strength ({dt.date.today()})",
    "",
    f"Train rows: {len(train_aug):,}; Test rows: {len(test_aug):,}",
    f"Test rows with computable lead_PitStop: "
    f"**{test_lead_avail*100:.2f}%**",
    "",
    "## Single-feature 5-fold StratifiedKFold OOF AUC",
    "",
    "| feature(s) | OOF AUC | per-fold |",
    "|---|---:|---|",
]
for name, (oof, folds) in results.items():
    fs = ", ".join(f"{f:.4f}" for f in folds)
    lines.append(f"| {name} | **{oof:.5f}** | {fs} |")

# Verdict
lead_auc = results["lead_PitStop (next-lap PitStop, with -1 sentinel for last lap)"][0]
lines += ["", "## Verdict"]
if lead_auc >= 0.92:
    lines.append(
        f"**Dominant feature.** Single-feature OOF = {lead_auc:.5f}. "
        "Baseline submission MUST include lead_PitStop; submitting "
        "without it burns sub #1 on an obsolete model."
    )
elif lead_auc >= 0.80:
    lines.append(
        f"**Strong feature.** Single-feature OOF = {lead_auc:.5f}. "
        "Likely top-3 feature in any model; should be in baseline. "
        "Not dominant — other signal contributes substantially."
    )
else:
    lines.append(
        f"**Normal feature.** Single-feature OOF = {lead_auc:.5f}. "
        "One useful signal among many; baseline-without-it is still "
        "a meaningful calibration anchor."
    )

out = Path(f"audit/{dt.date.today().isoformat()}-u2-lead-pitstop-probe.md")
out.write_text("\n".join(lines) + "\n")
print(f"\n→ written to {out}")
