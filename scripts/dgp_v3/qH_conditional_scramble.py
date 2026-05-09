"""qH — improved noisy-orig: conditional Driver/Stint sampling matched to
synth.

qG showed Stint and Driver carry most of the remaining disc-AUC gap when
the noisy-orig replay scrambles them uniformly. Replace with conditional
sampling: for each replay row, draw Driver and Stint from synth's empirical
distribution conditional on (Year, Compound, PitStop).

Sweep two variants:
  H1: orig + synth-marginal resample + conditional Driver/Stint
  H2: H1 + small Gaussian noise on continuous columns

Output: scripts/artifacts/dgp_v3_qH_conditional.json
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


def make_replay_conditional(
    orig: pd.DataFrame,
    synth_marginal: pd.DataFrame,
    synth_cond_driver: dict,
    synth_cond_stint: dict,
    n: int,
    seed: int = 0,
    sigma: float = 0.0,
) -> pd.DataFrame:
    """Cell-conditional Driver/Stint sampling from synth empirical."""
    rng = np.random.default_rng(seed)
    cells = synth_marginal.sample(n, weights="prob", replace=True, random_state=seed)
    rows = []
    orig_by_cell = {k: g for k, g in orig.groupby(["Year", "Compound", "PitStop"])}
    for cell_yr, cell_cmp, cell_ps, _ in cells.itertuples(index=False):
        key = (cell_yr, cell_cmp, cell_ps)
        if key in orig_by_cell:
            sub = orig_by_cell[key]
            i = rng.integers(0, len(sub))
            rows.append(sub.iloc[i])
    df = pd.DataFrame(rows).reset_index(drop=True)

    # Sample Driver and Stint conditional on cell
    driver_assigned = np.empty(len(df), dtype=object)
    stint_assigned = np.empty(len(df), dtype=int)
    for k, idx in df.groupby(["Year", "Compound", "PitStop"]).groups.items():
        idx_arr = np.array(list(idx))
        # Driver
        if k in synth_cond_driver:
            opts = synth_cond_driver[k]
            choices = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
            driver_assigned[idx_arr] = choices
        # Stint
        if k in synth_cond_stint:
            opts = synth_cond_stint[k]
            choices = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
            stint_assigned[idx_arr] = choices
    df["Driver"] = driver_assigned
    df["Stint"] = stint_assigned

    if sigma > 0:
        for c in PERTURB_COLS:
            if c in df.columns:
                std = float(df[c].std())
                df[c] = df[c] + rng.normal(0, sigma * std, size=len(df))

    return df


def build_cond(synth: pd.DataFrame, col: str) -> dict:
    cond = {}
    for k, g in synth.groupby(["Year", "Compound", "PitStop"]):
        vals = g[col].value_counts(normalize=True)
        cond[k] = {"values": vals.index.tolist(), "p": vals.values.tolist()}
    return cond


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

    synth_marginal = (
        synth[["Year", "Compound", "PitStop"]]
        .value_counts(normalize=True)
        .reset_index(name="prob")
    )
    synth_cond_driver = build_cond(synth, "Driver")
    synth_cond_stint = build_cond(synth, "Stint")
    t(f"built conditional Driver/Stint distributions over "
      f"{len(synth_cond_driver)} cells", ts)

    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)

    out["sweeps"] = []
    for sigma in [0.0, 0.05, 0.10, 0.25]:
        replay = make_replay_conditional(
            orig, synth_marginal, synth_cond_driver, synth_cond_stint,
            n=20_000, seed=0, sigma=sigma,
        )
        auc = disc_auc(replay, synth_disc)
        t(f"sigma={sigma}: disc_auc = {auc:.4f}", ts)
        out["sweeps"].append({"sigma": sigma, "disc_auc": auc, "n": int(len(replay))})

    fp = ART / "dgp_v3_qH_conditional.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qH conditional Driver/Stint sampling sweep ===")
    print("  reference: qF sigma=0 uniform scramble = 0.9716")
    print("             SDV CTGAN 10ep              = 0.9993")
    for v in out["sweeps"]:
        mark = " <- HIT" if v["disc_auc"] < 0.85 else ""
        print(f"  sigma={v['sigma']:.3f}  disc_auc={v['disc_auc']:.4f}{mark}")


if __name__ == "__main__":
    main()
