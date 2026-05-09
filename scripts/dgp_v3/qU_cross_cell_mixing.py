"""qU — cross-cell value mixing in the analytic resampler.

qP found 45% of synth's NN orig rows share all 3 of (Year, Compound,
PitStop). 55% are from OTHER cells. qR found 73% of synth's
(Y, C, LapTime) keys are NOT in orig. Both findings point to the same
mechanism: the host's continuous-value generator mixes values across
cells.

Build a sampler that:
  - For each target cell c (Y,C,PS,Race,Stint,LapNumber), sample row
    from BOTH orig in cell c AND orig in NEIGHBOURING cells (matching
    on Compound + Year only, or Compound only).
  - Weight by closeness in cell hierarchy.

Sweep mixing fraction:
  - 100% in-cell  (= qM, 0.7160)
  - 50/50 in-cell + same-Y/C
  - 100% same-Y/C (Y, C anywhere)
  - 100% global  (= qQ, 0.9907)

If 50/50 beats both, the host's mixing fraction is partial.
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


def make_replay_with_mixing(orig: pd.DataFrame, synth_marginal: pd.DataFrame,
                             cd_driver: dict, by: list[str], n: int,
                             mix_p_outcell: float, seed: int = 0) -> pd.DataFrame:
    """For each sampled cell, with prob mix_p_outcell sample from same
    (Year, Compound) ignoring (PS, Race, Stint, LapNumber); else from cell."""
    rng = np.random.default_rng(seed)
    cells = synth_marginal.sample(n, weights="prob", replace=True, random_state=seed)
    by_cols = by
    orig_by_cell = {k: g for k, g in orig.groupby(by_cols)}
    orig_by_yc = {k: g for k, g in orig.groupby(["Year", "Compound"])}

    rows = []
    for tup in cells.itertuples(index=False):
        key = tuple(getattr(tup, c) for c in by_cols)
        target_cell_meta = {c: getattr(tup, c) for c in by_cols}
        if rng.random() < mix_p_outcell:
            # Sample from same (Year, Compound), keep target cell meta
            yc_key = (target_cell_meta["Year"], target_cell_meta["Compound"])
            if yc_key in orig_by_yc:
                pool = orig_by_yc[yc_key]
                i = rng.integers(0, len(pool))
                row = pool.iloc[i].copy()
                # Override the cell meta back to target
                for c in by_cols:
                    row[c] = target_cell_meta[c]
                rows.append(row)
        else:
            # Sample from target cell exactly
            if key in orig_by_cell:
                pool = orig_by_cell[key]
                i = rng.integers(0, len(pool))
                rows.append(pool.iloc[i])

    df = pd.DataFrame(rows).reset_index(drop=True)

    # Conditional Driver
    driver_assigned = np.empty(len(df), dtype=object)
    for k, idx in df.groupby(by_cols).groups.items():
        idx_arr = np.array(list(idx))
        if isinstance(k, tuple):
            ck = k
        else:
            ck = (k,)
        if ck in cd_driver:
            opts = cd_driver[ck]
            driver_assigned[idx_arr] = rng.choice(opts["values"], size=len(idx_arr), p=opts["p"])
    df["Driver"] = driver_assigned

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

    by = ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]
    sm = synth[by].value_counts(normalize=True).reset_index(name="prob")
    orig_keys = set(map(tuple, orig[by].values.tolist()))
    sm["in_orig"] = sm[by].apply(lambda r: tuple(r) in orig_keys, axis=1)
    sm = sm[sm["in_orig"]].drop(columns=["in_orig"])
    sm["prob"] = sm["prob"] / sm["prob"].sum()
    cd_driver = build_cond(synth, "Driver", by)
    t(f"by={by}; cells = {len(sm)}", ts)

    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)
    out["sweeps"] = []
    for mix_p in [0.0, 0.25, 0.50, 0.75, 1.0]:
        replay = make_replay_with_mixing(
            orig, sm, cd_driver, by, n=20_000,
            mix_p_outcell=mix_p, seed=0,
        )
        if len(replay) < 1000:
            t(f"mix={mix_p}: replay too small ({len(replay)}), skip", ts)
            continue
        auc = disc_auc(replay, synth_disc)
        t(f"mix={mix_p}: replay {replay.shape}, disc_auc = {auc:.4f}", ts)
        out["sweeps"].append({"mix_p_outcell": mix_p, "disc_auc": auc, "n": int(len(replay))})

    fp = ART / "dgp_v3_qU_cross_cell.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qU cross-cell mixing sweep ===")
    print("  reference: qM in-cell only (mix=0): 0.7160")
    print("             qQ all out-of-cell (~global): 0.9907")
    for v in out["sweeps"]:
        mark = " <- BETTER" if v["disc_auc"] < 0.71 else ""
        print(f"  mix_p_outcell={v['mix_p_outcell']:.2f}: disc_auc={v['disc_auc']:.4f}{mark}")


if __name__ == "__main__":
    main()
