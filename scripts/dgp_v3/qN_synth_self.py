"""qN — synth-self-bootstrap upper bound.

Sample continuous + integer columns from synth (not orig), conditional on
(Y, C, PS, Race, Stint, LapNumber). This is the upper bound on what
analytic conditional sampling can achieve. disc-AUC should be ~0.5.

If it's, then the gap from qM (0.7160) to qN (~0.5) is exactly the
continuous-density gap we need a generator for in Phase C.
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


def main() -> None:
    out: dict = {}
    ts = time.time()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
        test = test.rename(columns={"LapTime (s)": "LapTime"})
    synth = pd.concat(
        [train[[c for c in train.columns if c != "PitNextLap"]], test],
        ignore_index=True,
    ).reset_index(drop=True)
    t(f"synth {synth.shape}", ts)

    # synth-self-bootstrap: sample one half from a uniform random sample,
    # the other from a different random sample. Both from synth.
    rng = np.random.default_rng(0)
    idx_a = rng.choice(len(synth), size=20_000, replace=False)
    idx_b = rng.choice(len(synth), size=20_000, replace=False)
    a = synth.iloc[idx_a].reset_index(drop=True)
    b = synth.iloc[idx_b].reset_index(drop=True)

    auc = disc_auc(a, b)
    t(f"synth random vs synth random: disc_auc = {auc:.4f}", ts)
    out["synth_random_vs_synth_random"] = auc

    # Now: synth bootstrap sampling using qM's marginal but from synth itself
    by = ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]
    sm = synth[by].value_counts(normalize=True).reset_index(name="prob")
    cells = sm.sample(20_000, weights="prob", replace=True, random_state=0)
    rows = []
    synth_by_cell = {k: g for k, g in synth.groupby(by)}
    for tup in cells.itertuples(index=False):
        key = tuple(getattr(tup, c) for c in by)
        if key in synth_by_cell:
            sub = synth_by_cell[key]
            i = rng.integers(0, len(sub))
            rows.append(sub.iloc[i])
    df = pd.DataFrame(rows).reset_index(drop=True)
    t(f"synth-self qM-style: {df.shape}", ts)
    synth_disc = synth.sample(20_000, random_state=0).reset_index(drop=True)
    auc2 = disc_auc(df, synth_disc)
    t(f"synth-self bootstrap (qM-style): disc_auc = {auc2:.4f}", ts)
    out["synth_self_qM_style"] = auc2

    fp = ART / "dgp_v3_qN_synth_self.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qN synth-self upper bound ===")
    print(f"  synth random vs synth random:                   {auc:.4f} (chance)")
    print(f"  synth bootstrap (qM cells, synth values):       {auc2:.4f}")
    print(f"  qM (orig values, qM cells):                     0.7160")
    print(f"\n  Gap closable by per-cell density estimation: {0.7160 - auc2:.4f}")


if __name__ == "__main__":
    main()
