"""Phase B alt — does a DEAD-SIMPLE noisy-orig generator beat CTGAN?

Hypothesis (after Q3+Q5 falsified marginal-only and Q6+Q7 retracted
per-row literal-copy): the host's "synth" might actually be a simple
non-deep-learning pipeline:

  1. Take orig.
  2. Sample rows from orig with a custom (Year, Compound, PitStop)
     marginal (oversample PS=0).
  3. For each sampled row, perturb continuous columns by per-column
     Gaussian noise (σ small relative to column std).
  4. Re-assign Driver / Stint labels from the synth vocabulary.
  5. Drop Normalized_TyreLife.

If true, this trivial generator should beat any CTGAN family on
disc-AUC. Test by building it, sampling 20k, and computing disc-AUC.

Sweep over noise levels σ ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 1.00} ×
column std. Report which σ minimises disc-AUC.

Output: scripts/artifacts/dgp_v3_qF_noisy_orig.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"

PERTURB_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation",
                "RaceProgress"]


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def disc_auc(replay: pd.DataFrame, synth: pd.DataFrame) -> float:
    common = sorted(set(replay.columns) & set(synth.columns))
    df = pd.concat(
        [replay[common].assign(_lbl=0), synth[common].assign(_lbl=1)],
        ignore_index=True,
    )
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df[c] = pd.Categorical(df[c]).codes
    X = df.drop(columns=["_lbl"]).values
    y = df["_lbl"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, n_jobs=-1, verbosity=-1,
        )
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, oof))


def make_replay(orig: pd.DataFrame, synth_marginal: pd.DataFrame,
                synth_drivers: list, synth_stints: list, sigma: float, n: int,
                seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Step 1+2: sample orig rows with synth marginal
    cells = synth_marginal.sample(n, weights="prob", replace=True, random_state=seed)
    rows = []
    orig_by_cell = {}
    for k, g in orig.groupby(["Year", "Compound", "PitStop"]):
        orig_by_cell[k] = g
    for cell_yr, cell_cmp, cell_ps, _ in cells.itertuples(index=False):
        key = (cell_yr, cell_cmp, cell_ps)
        if key in orig_by_cell:
            sub = orig_by_cell[key]
            i = rng.integers(0, len(sub))
            rows.append(sub.iloc[i])
    df = pd.DataFrame(rows).reset_index(drop=True)

    # Step 3: per-column Gaussian noise scaled by column std
    for c in PERTURB_COLS:
        if c in df.columns:
            std = float(df[c].std())
            df[c] = df[c] + rng.normal(0, sigma * std, size=len(df))

    # Step 4: re-assign Driver and Stint from synth vocab
    df["Driver"] = rng.choice(synth_drivers, size=len(df))
    df["Stint"] = rng.choice(synth_stints, size=len(df))

    return df


def main() -> None:
    out: dict = {}
    ts = time.time()

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna()
    orig = orig.reset_index(drop=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
        test = test.rename(columns={"LapTime (s)": "LapTime"})
    synth = pd.concat(
        [train[[c for c in train.columns if c != "PitNextLap"]], test],
        ignore_index=True,
    )
    t(f"orig {orig.shape} synth {synth.shape}", ts)

    # synth marginal P(Year, Compound, PitStop)
    synth_marginal = (
        synth[["Year", "Compound", "PitStop"]]
        .value_counts(normalize=True)
        .reset_index(name="prob")
    )
    synth_drivers = synth["Driver"].unique().tolist()
    synth_stints = synth["Stint"].unique().tolist()
    t(f"synth marginal cells: {len(synth_marginal)}, drivers: {len(synth_drivers)}, "
      f"stints: {len(synth_stints)}", ts)

    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)

    out["sigmas"] = []
    for sigma in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
        replay = make_replay(orig, synth_marginal, synth_drivers, synth_stints,
                              sigma, n=20_000, seed=0)
        auc = disc_auc(replay, synth_disc)
        t(f"sigma={sigma:.3f}: disc AUC = {auc:.4f}", ts)
        out["sigmas"].append({"sigma": sigma, "disc_auc": auc, "n": int(len(replay))})

    fp = ART / "dgp_v3_qF_noisy_orig.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== Noisy-orig sweep ===")
    print(f"  reference: SDV CTGAN 10ep = 0.9993, GaussianCopula = 0.9988, TVAE = 0.9991")
    for v in out["sigmas"]:
        mark = " <- HIT" if v["disc_auc"] < 0.95 else ""
        print(f"  sigma={v['sigma']:.3f}  disc_auc={v['disc_auc']:.4f}{mark}")


if __name__ == "__main__":
    main()
