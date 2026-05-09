"""qQ — sample continuous values from orig GLOBAL (no cell) + categorical
per-cell empirical.

qP found synth row's NN orig row is in the same (Y,C,PS) cell only 45% of
the time. Hypothesis: the host samples continuous values from orig's
GLOBAL empirical (no cell conditioning), then attaches categorical labels
from synth's per-cell empirical.

Test:
  1. Sample N target cells from synth marginal (Y, C, PS, Race, Stint, LapN)
  2. Sample continuous values from orig globally (no cell)
  3. Sample Driver/Position from synth empirical per cell
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

FLOAT_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation",
              "RaceProgress", "Position_Change", "TyreLife"]


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


def build_cond(synth: pd.DataFrame, col: str, by: list[str]) -> dict:
    cond = {}
    for k, g in synth.groupby(by):
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

    by = ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]
    sm = synth[by].value_counts(normalize=True).reset_index(name="prob")
    cd_driver = build_cond(synth, "Driver", by)
    cd_position = build_cond(synth, "Position", by)
    t(f"by={by}; cells = {len(sm)}", ts)

    rng = np.random.default_rng(0)
    cells = sm.sample(20_000, weights="prob", replace=True, random_state=0)
    n_replay = len(cells)

    # Sample continuous values from orig GLOBALLY (not per-cell)
    orig_idx = rng.integers(0, len(orig), size=n_replay)
    cont_vals = orig.iloc[orig_idx][FLOAT_COLS].values

    # Build replay df with cell columns + global cont values
    df_rows = []
    for i, tup in enumerate(cells.itertuples(index=False)):
        row = {c: getattr(tup, c) for c in by}
        for j, c in enumerate(FLOAT_COLS):
            row[c] = cont_vals[i][j]
        df_rows.append(row)
    df = pd.DataFrame(df_rows)
    t(f"replay df {df.shape}", ts)

    # Conditional Driver and Position
    driver_assigned = np.empty(len(df), dtype=object)
    pos_assigned = np.empty(len(df), dtype=int)
    for k, idx in df.groupby(by).groups.items():
        idx_arr = np.array(list(idx))
        if isinstance(k, tuple):
            key = k
        else:
            key = (k,)
        if key in cd_driver:
            opts = cd_driver[key]
            driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
        if key in cd_position:
            opts = cd_position[key]
            pos_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
    df["Driver"] = driver_assigned
    df["Position"] = pos_assigned

    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)
    auc = disc_auc(df, synth_disc)
    t(f"qQ global-floats + cond categorical: disc_auc = {auc:.4f}", ts)
    out["disc_auc"] = auc

    fp = ART / "dgp_v3_qQ_global_floats.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qQ summary ===")
    print(f"  qM (per-cell continuous) =  0.7160")
    print(f"  qN synth-self lower bound = 0.4944")
    print(f"  qQ (global continuous)   =  {auc:.4f}")


if __name__ == "__main__":
    main()
