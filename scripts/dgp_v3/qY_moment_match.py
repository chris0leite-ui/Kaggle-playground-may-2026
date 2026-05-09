"""qY — per-cell moment matching: rescale orig to synth's per-cell mean/std.

qX revealed synth's per-cell mean/std DIFFER from orig's per-cell mean/std
in non-trivial ways (e.g. LapTime mean shift -2.8, std ratio 0.87 per
(Y, C, PS) cell).

For each replay row drawn from orig per (Y, C, PS, R, S, LapN):
  - Compute target per-(Y, C, PS) mean and std from synth.
  - Compute source per-(Y, C, PS) mean and std from orig.
  - Standardise orig values: z = (x - mu_o) / std_o
  - Rescale to synth: x' = z * std_s + mu_s

This DOES NOT use synth's individual values — only its per-cell
mean/std. It tests if synth's per-cell distribution can be modelled
as orig's distribution shifted/scaled.
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

FLOAT_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]


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
    t(f"by={by}; cells = {len(sm)}", ts)

    # Test multiple moment-matching scope levels
    out["sweeps"] = []
    rng = np.random.default_rng(0)
    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)

    for moment_by in [["Year", "Compound", "PitStop"],
                       ["Year", "Compound"],
                       ["Compound"]]:
        # Build per-cell mean/std for orig and synth at moment_by level
        orig_cell_stats = orig.groupby(moment_by)[FLOAT_COLS].agg(["mean", "std"])
        synth_cell_stats = synth.groupby(moment_by)[FLOAT_COLS].agg(["mean", "std"])

        # Build the replay
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

            # Apply moment-matching for FLOAT_COLS
            mb_key = tuple(getattr(tup, c) for c in moment_by)
            if len(moment_by) == 1:
                mb_key = mb_key[0]
            try:
                for c in FLOAT_COLS:
                    mu_o = orig_cell_stats.loc[mb_key, (c, "mean")]
                    sd_o = orig_cell_stats.loc[mb_key, (c, "std")]
                    mu_s = synth_cell_stats.loc[mb_key, (c, "mean")]
                    sd_s = synth_cell_stats.loc[mb_key, (c, "std")]
                    if sd_o > 0 and not np.isnan(sd_s) and not np.isnan(mu_s):
                        z = (row[c] - mu_o) / sd_o
                        row[c] = z * sd_s + mu_s
            except (KeyError, TypeError):
                pass
            rows.append(row)
        df = pd.DataFrame(rows).reset_index(drop=True)

        # Cond Driver
        driver_assigned = np.empty(len(df), dtype=object)
        for k, idx in df.groupby(by).groups.items():
            idx_arr = np.array(list(idx))
            ck = k if isinstance(k, tuple) else (k,)
            if ck in cd_driver:
                opts = cd_driver[ck]
                driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
        df["Driver"] = driver_assigned

        auc = disc_auc(df, synth_disc)
        t(f"moment_by={moment_by}: replay {df.shape}, disc_auc = {auc:.4f}", ts)
        out["sweeps"].append({"moment_by": moment_by, "disc_auc": auc, "n": int(len(df))})

    fp = ART / "dgp_v3_qY_moment_match.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qY moment-matching sweep ===")
    print("  reference: qM (no moment match) = 0.7160")
    print("             synth-self bound      = 0.4944")
    for v in out["sweeps"]:
        mark = " <- HIT" if v["disc_auc"] < 0.65 else (" <- BETTER" if v["disc_auc"] < 0.71 else "")
        print(f"  moment_by={v['moment_by']}: disc_auc={v['disc_auc']:.4f}{mark}")


if __name__ == "__main__":
    main()
