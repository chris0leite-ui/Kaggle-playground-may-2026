"""qL — extend conditioning axis: condition on (Year, Compound, PitStop, Race).

qH used (Year, Compound, PitStop) cells (~33). qL refines to (Y, C, PS, Race)
cells (~520 max). Test if more granular conditioning closes disc-AUC.

  qL.1 = orig rows sampled per (Y, C, PS, Race) cell with synth marginal,
         + cond Driver/Stint per (Y, C, PS, Race)
  qL.2 = same as L1 but Driver and Stint conditioned only on (Y, C, PS),
         to isolate which axis carries the lift

Output: scripts/artifacts/dgp_v3_qL_race_cond.json
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


def make_replay(orig: pd.DataFrame, synth_marginal: pd.DataFrame,
                cond_driver: dict, cond_stint: dict, by: list[str],
                n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = synth_marginal.sample(n, weights="prob", replace=True, random_state=seed)
    rows = []
    orig_by_cell = {k: g for k, g in orig.groupby(by)}

    # cells columns are by columns + 'prob' — iterate
    by_cols = by
    for tup in cells.itertuples(index=False):
        key = tuple(getattr(tup, c) for c in by_cols)
        if key in orig_by_cell:
            sub = orig_by_cell[key]
            i = rng.integers(0, len(sub))
            rows.append(sub.iloc[i])
    df = pd.DataFrame(rows).reset_index(drop=True)

    driver_assigned = np.empty(len(df), dtype=object)
    stint_assigned = np.empty(len(df), dtype=int)
    for k, idx in df.groupby(by_cols).groups.items():
        idx_arr = np.array(list(idx))
        if isinstance(k, tuple):
            key = k
        else:
            key = (k,)
        if key in cond_driver:
            opts = cond_driver[key]
            driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
        if key in cond_stint:
            opts = cond_stint[key]
            stint_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
    df["Driver"] = driver_assigned
    df["Stint"] = stint_assigned

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

    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)
    out["variants"] = []

    # Variant L1: by = (Year, Compound, PitStop, Race)
    by = ["Year", "Compound", "PitStop", "Race"]
    sm = synth[by].value_counts(normalize=True).reset_index(name="prob")
    # Need to ensure orig has at least one row per cell we draw
    orig_keys = set(map(tuple, orig[by].values.tolist()))
    sm["in_orig"] = sm[by].apply(lambda r: tuple(r) in orig_keys, axis=1)
    sm = sm[sm["in_orig"]].drop(columns=["in_orig"])
    sm["prob"] = sm["prob"] / sm["prob"].sum()
    cd = build_cond(synth, "Driver", by)
    cs = build_cond(synth, "Stint", by)
    t(f"L1 cells in both: {len(sm)}; mean rows per cell (synth): {len(synth)/len(sm):.0f}", ts)
    replay = make_replay(orig, sm, cd, cs, by, n=20_000, seed=0)
    auc = disc_auc(replay, synth_disc)
    t(f"L1 (Y, C, PS, Race) cond: disc_auc = {auc:.4f}", ts)
    out["variants"].append({"by": by, "disc_auc": auc, "n_cells": int(len(sm))})

    # Variant L3: by = (Year, Compound, PitStop, Race, Stint)
    by = ["Year", "Compound", "PitStop", "Race", "Stint"]
    sm = synth[by].value_counts(normalize=True).reset_index(name="prob")
    orig_keys = set(map(tuple, orig[by].values.tolist()))
    sm["in_orig"] = sm[by].apply(lambda r: tuple(r) in orig_keys, axis=1)
    sm = sm[sm["in_orig"]].drop(columns=["in_orig"])
    sm["prob"] = sm["prob"] / sm["prob"].sum()
    cd = build_cond(synth, "Driver", by)
    cs = {k: {"values": [k[-1]], "p": [1.0]} for k in cd}  # Stint is in by; deterministic
    t(f"L3 cells in both: {len(sm)}", ts)
    replay = make_replay(orig, sm, cd, cs, by, n=20_000, seed=0)
    auc = disc_auc(replay, synth_disc)
    t(f"L3 (Y, C, PS, Race, Stint) cond: disc_auc = {auc:.4f}", ts)
    out["variants"].append({"by": by, "disc_auc": auc, "n_cells": int(len(sm))})

    fp = ART / "dgp_v3_qL_race_cond.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qL Race-conditioning sweep ===")
    print(f"  qH baseline (Y, C, PS) cond: 0.8323")
    for v in out["variants"]:
        mark = " <- HIT" if v["disc_auc"] < 0.85 else ""
        mark2 = " *** STRONG" if v["disc_auc"] < 0.7 else ""
        print(f"  cond on {v['by']}: disc_auc = {v['disc_auc']:.4f} ({v['n_cells']} cells){mark}{mark2}")


if __name__ == "__main__":
    main()
