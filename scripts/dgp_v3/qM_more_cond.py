"""qM — push conditioning further: add LapNumber and Position.

qL got us from 0.83 (Y,C,PS cond) to 0.72 (Y,C,PS,Race,Stint cond).
Now extend to (Y,C,PS,Race,Stint,LapNumber) and (Y,C,PS,Race,Stint,
LapNumber,Position).

Output: scripts/artifacts/dgp_v3_qM_more_cond.json
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
                cond_driver: dict, by: list[str], n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = synth_marginal.sample(n, weights="prob", replace=True, random_state=seed)
    rows = []
    orig_by_cell = {k: g for k, g in orig.groupby(by)}
    by_cols = by
    for tup in cells.itertuples(index=False):
        key = tuple(getattr(tup, c) for c in by_cols)
        if key in orig_by_cell:
            sub = orig_by_cell[key]
            i = rng.integers(0, len(sub))
            rows.append(sub.iloc[i])
    df = pd.DataFrame(rows).reset_index(drop=True)

    driver_assigned = np.empty(len(df), dtype=object)
    for k, idx in df.groupby(by_cols).groups.items():
        idx_arr = np.array(list(idx))
        if isinstance(k, tuple):
            key = k
        else:
            key = (k,)
        if key in cond_driver:
            opts = cond_driver[key]
            driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
    df["Driver"] = driver_assigned

    return df


def run_variant(orig, synth, synth_disc, by, ts):
    sm = synth[by].value_counts(normalize=True).reset_index(name="prob")
    orig_keys = set(map(tuple, orig[by].values.tolist()))
    sm["in_orig"] = sm[by].apply(lambda r: tuple(r) in orig_keys, axis=1)
    sm = sm[sm["in_orig"]].drop(columns=["in_orig"])
    sm["prob"] = sm["prob"] / sm["prob"].sum()
    cd = build_cond(synth, "Driver", by)
    t(f"by={by}: cells in both = {len(sm)}", ts)
    replay = make_replay(orig, sm, cd, by, n=20_000, seed=0)
    auc = disc_auc(replay, synth_disc)
    t(f"  disc_auc = {auc:.4f}", ts)
    return {"by": by, "disc_auc": auc, "n_cells": int(len(sm))}


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

    BY_GRID = [
        ["Year", "Compound", "PitStop", "Race", "Stint"],
        ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"],
        ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber", "Position"],
        ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber", "Position", "TyreLife"],
    ]
    for by in BY_GRID:
        out["variants"].append(run_variant(orig, synth, synth_disc, by, ts))

    fp = ART / "dgp_v3_qM_more_cond.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qM extended conditioning sweep ===")
    print(f"  reference: qH (Y,C,PS) = 0.8323; qL3 (Y,C,PS,R,S) = 0.7247")
    for v in out["variants"]:
        mark = " *** LOW" if v["disc_auc"] < 0.6 else (" <- HIT" if v["disc_auc"] < 0.7 else "")
        print(f"  cond on {v['by']}: disc_auc = {v['disc_auc']:.4f} ({v['n_cells']} cells){mark}")


if __name__ == "__main__":
    main()
