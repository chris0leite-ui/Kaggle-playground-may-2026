"""qO — per-cell BGMM ONLY on true float columns; integers preserved.

qK fit BGMM on all 6 'continuous' columns and got disc=1.0 because
TyreLife/Position_Change are functionally integer and BGMM emits
floats. Retry with BGMM only on truly float columns:

  Float (use BGMM): LapTime, LapTime_Delta, Cumulative_Degradation,
                    RaceProgress
  Integer (use orig as-is): TyreLife, Position_Change, Stint, Position,
                            LapNumber, Year, PitStop

For each (Y,C,PS,R,S,LapN) cell:
  - Sample integer columns + Race + Compound from orig as-is
  - Sample float columns from per-cell BGMM
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"

FLOAT_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation",
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
    orig_keys = set(map(tuple, orig[by].values.tolist()))
    sm["in_orig"] = sm[by].apply(lambda r: tuple(r) in orig_keys, axis=1)
    sm = sm[sm["in_orig"]].drop(columns=["in_orig"])
    sm["prob"] = sm["prob"] / sm["prob"].sum()
    cd_driver = build_cond(synth, "Driver", by)
    t(f"by={by}; cells in both = {len(sm)}", ts)

    # Fit per-cell BGMM on float cols (only cells with >=4 orig rows)
    bgmm_per_cell = {}
    for k, g in orig.groupby(by):
        if len(g) < 4:
            continue
        n_components = min(3, max(1, len(g) // 5))
        try:
            bgmm = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=0,
                max_iter=100,
                n_init=1,
            )
            bgmm.fit(g[FLOAT_COLS].values)
            bgmm_per_cell[k] = bgmm
        except Exception:
            pass
    t(f"BGMM fit on {len(bgmm_per_cell)} cells", ts)

    # Sample
    rng = np.random.default_rng(0)
    cells = sm.sample(20_000, weights="prob", replace=True, random_state=0)
    rows = []
    orig_by_cell = {k: g for k, g in orig.groupby(by)}
    for tup in cells.itertuples(index=False):
        key = tuple(getattr(tup, c) for c in by)
        if key not in orig_by_cell:
            continue
        sub = orig_by_cell[key]
        i = rng.integers(0, len(sub))
        row = sub.iloc[i].copy()
        if key in bgmm_per_cell:
            sample_vals, _ = bgmm_per_cell[key].sample(1)
            for j, c in enumerate(FLOAT_COLS):
                row[c] = sample_vals[0][j]
        rows.append(row)
    df = pd.DataFrame(rows).reset_index(drop=True)
    t(f"sampled rows: {df.shape}", ts)

    # Conditional Driver
    driver_assigned = np.empty(len(df), dtype=object)
    for k, idx in df.groupby(by).groups.items():
        idx_arr = np.array(list(idx))
        if isinstance(k, tuple):
            key = k
        else:
            key = (k,)
        if key in cd_driver:
            opts = cd_driver[key]
            driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
    df["Driver"] = driver_assigned

    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)
    auc = disc_auc(df, synth_disc)
    t(f"qO BGMM-on-floats disc_auc = {auc:.4f}", ts)
    out["disc_auc"] = auc
    out["n_cells_with_bgmm"] = len(bgmm_per_cell)

    fp = ART / "dgp_v3_qO_bgmm_floats.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qO summary ===")
    print(f"  reference: qM (orig values) = 0.7160")
    print(f"             synth-self bound = 0.4944")
    print(f"  qO (BGMM on floats):           {auc:.4f}")


if __name__ == "__main__":
    main()
