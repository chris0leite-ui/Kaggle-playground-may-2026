"""qJ — per-cell Gaussian noise + conditional Driver/Stint.

Hypothesis: the host's CTGAN (or whatever) generates LapTime, LapTime_Delta,
Cumulative_Degradation values that have orig's CELL-CONDITIONAL marginal,
but are perturbed by cell-specific Gaussian noise. Test by:
  1. Sample orig rows from per-cell pool with synth marginal.
  2. Sample Driver/Stint conditional on cell from synth empirical.
  3. Perturb continuous columns by Gaussian noise with sigma * within-cell-std.

Sweep sigma. Expect disc-AUC to bottom out at the sigma that matches host.

Output: scripts/artifacts/dgp_v3_qJ_cell_noise.json
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


def build_cond(synth: pd.DataFrame, col: str) -> dict:
    cond = {}
    for k, g in synth.groupby(["Year", "Compound", "PitStop"]):
        vals = g[col].value_counts(normalize=True)
        cond[k] = {"values": vals.index.tolist(), "p": vals.values.tolist()}
    return cond


def make_replay(orig: pd.DataFrame, synth_marginal: pd.DataFrame,
                synth_cond_driver: dict, synth_cond_stint: dict,
                cell_stds: dict, n: int, seed: int = 0,
                sigma: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = synth_marginal.sample(n, weights="prob", replace=True, random_state=seed)
    rows = []
    cell_keys = []
    orig_by_cell = {k: g for k, g in orig.groupby(["Year", "Compound", "PitStop"])}
    for cell_yr, cell_cmp, cell_ps, _ in cells.itertuples(index=False):
        key = (cell_yr, cell_cmp, cell_ps)
        if key in orig_by_cell:
            sub = orig_by_cell[key]
            i = rng.integers(0, len(sub))
            rows.append(sub.iloc[i])
            cell_keys.append(key)
    df = pd.DataFrame(rows).reset_index(drop=True)

    # Apply cell-specific noise
    if sigma > 0:
        cell_keys_arr = pd.Series(cell_keys, index=df.index)
        for c in PERTURB_COLS:
            if c not in df.columns:
                continue
            stds = np.array([cell_stds.get(k, {}).get(c, 1.0) for k in cell_keys])
            df[c] = df[c] + rng.normal(0, 1, size=len(df)) * sigma * stds

    # Conditional Driver/Stint
    driver_assigned = np.empty(len(df), dtype=object)
    stint_assigned = np.empty(len(df), dtype=int)
    for k, idx in df.groupby(["Year", "Compound", "PitStop"]).groups.items():
        idx_arr = np.array(list(idx))
        if k in synth_cond_driver:
            opts = synth_cond_driver[k]
            driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
        if k in synth_cond_stint:
            opts = synth_cond_stint[k]
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

    synth_marginal = (
        synth[["Year", "Compound", "PitStop"]]
        .value_counts(normalize=True).reset_index(name="prob")
    )
    synth_cond_driver = build_cond(synth, "Driver")
    synth_cond_stint = build_cond(synth, "Stint")

    # Compute per-cell within-cell std for each continuous column from ORIG
    cell_stds = {}
    for k, g in orig.groupby(["Year", "Compound", "PitStop"]):
        cell_stds[k] = {c: float(g[c].std()) for c in PERTURB_COLS if c in g.columns}

    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)

    out["sweeps"] = []
    for sigma in [0.00, 0.02, 0.05, 0.10, 0.20, 0.50]:
        replay = make_replay(
            orig, synth_marginal, synth_cond_driver, synth_cond_stint,
            cell_stds, n=20_000, seed=0, sigma=sigma,
        )
        auc = disc_auc(replay, synth_disc)
        t(f"sigma={sigma}: disc_auc = {auc:.4f}", ts)
        out["sweeps"].append({"sigma": sigma, "disc_auc": auc, "n": int(len(replay))})

    fp = ART / "dgp_v3_qJ_cell_noise.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qJ cell-specific noise sweep ===")
    print("  reference: qH cond Driver/Stint sigma=0 = 0.8323")
    for v in out["sweeps"]:
        mark = " <- HIT" if v["disc_auc"] < 0.85 else ""
        print(f"  sigma={v['sigma']:.3f}  disc_auc={v['disc_auc']:.4f}{mark}")


if __name__ == "__main__":
    main()
