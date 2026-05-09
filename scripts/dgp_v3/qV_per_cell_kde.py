"""qV — per-cell KDE on float columns.

qO BGMM-on-floats gave 0.8643 (worse than orig values). KDE is more
flexible: places a Gaussian kernel at every orig point with bandwidth
chosen by Scott's rule. Sweep bandwidth scaling factor.

For each cell, fit KDE on orig's 4 float columns, sample N rows,
combine with orig's integer columns + cond Driver.

Output: scripts/artifacts/dgp_v3_qV_kde.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
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

    out["sweeps"] = []
    rng = np.random.default_rng(0)
    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)

    for bw_factor in [0.0, 0.05, 0.10, 0.20, 0.50]:
        cells = sm.sample(20_000, weights="prob", replace=True, random_state=0)
        rows = []
        orig_by_cell = {k: g for k, g in orig.groupby(by)}

        # Cache KDE per cell (only when we have ≥3 rows)
        kde_cache = {}
        for k, g in orig.groupby(by):
            if len(g) >= 3 and bw_factor > 0:
                try:
                    kde = gaussian_kde(g[FLOAT_COLS].values.T,
                                       bw_method=bw_factor)
                    kde_cache[k] = kde
                except Exception:
                    pass

        for tup in cells.itertuples(index=False):
            key = tuple(getattr(tup, c) for c in by)
            if key not in orig_by_cell:
                continue
            sub = orig_by_cell[key]
            i = rng.integers(0, len(sub))
            row = sub.iloc[i].copy()
            if bw_factor > 0 and key in kde_cache:
                # Sample from KDE (1 sample, 4 floats)
                sample = kde_cache[key].resample(size=1).flatten()
                for j, c in enumerate(FLOAT_COLS):
                    row[c] = sample[j]
            rows.append(row)
        df = pd.DataFrame(rows).reset_index(drop=True)

        # Cond Driver
        driver_assigned = np.empty(len(df), dtype=object)
        for k, idx in df.groupby(by).groups.items():
            idx_arr = np.array(list(idx))
            if isinstance(k, tuple):
                ck = k
            else:
                ck = (k,)
            if ck in cd_driver:
                opts = cd_driver[ck]
                driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
        df["Driver"] = driver_assigned

        auc = disc_auc(df, synth_disc)
        t(f"bw_factor={bw_factor}: replay {df.shape}, disc_auc = {auc:.4f}", ts)
        out["sweeps"].append({"bw_factor": bw_factor, "disc_auc": auc, "n": int(len(df))})

    fp = ART / "dgp_v3_qV_kde.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qV per-cell KDE bandwidth sweep ===")
    print("  reference: qM (orig values, no KDE) = 0.7160")
    for v in out["sweeps"]:
        mark = " <- HIT" if v["disc_auc"] < 0.71 else ""
        print(f"  bw={v['bw_factor']:.3f}: disc_auc={v['disc_auc']:.4f}{mark}")


if __name__ == "__main__":
    main()
