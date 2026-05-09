"""Phase B baseline — disc-AUC for raw orig vs synth and orig-vs-orig.

Two reference points without any surrogate:

  1. orig vs synth: how different is host synth from orig itself?
     If high (e.g. > 0.9), the host's generator produces a clearly
     different distribution than orig (regardless of which generator
     it is).
     If low (e.g. < 0.7), the host's synth is essentially orig
     re-weighted — there's almost nothing for our generator to fit.

  2. orig sample-1 vs orig sample-2: chance baseline. Should be ~0.5
     (same distribution sampled twice).

Output: scripts/artifacts/dgp_v3_qE_baseline_disc.json
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


def disc_auc(a: pd.DataFrame, b: pd.DataFrame) -> float:
    common = sorted(set(a.columns) & set(b.columns))
    df = pd.concat(
        [a[common].assign(_lbl=0), b[common].assign(_lbl=1)],
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

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna()
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

    # Take 20k from each
    rng = np.random.default_rng(0)
    o1 = orig.sample(20_000, random_state=0)
    o2 = orig.sample(20_000, random_state=1)
    s1 = synth.sample(20_000, random_state=0)

    # Match columns
    common = sorted(set(orig.columns) & set(synth.columns))
    o1 = o1[common]
    o2 = o2[common]
    s1 = s1[common]

    out["orig_vs_synth"] = disc_auc(o1, s1)
    t(f"orig_vs_synth disc-AUC = {out['orig_vs_synth']:.4f}", ts)

    out["orig_vs_orig"] = disc_auc(o1, o2)
    t(f"orig_vs_orig disc-AUC = {out['orig_vs_orig']:.4f} (chance baseline)", ts)

    fp = ART / "dgp_v3_qE_baseline_disc.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== Baseline disc-AUC ===")
    print(f"  orig vs synth (raw): {out['orig_vs_synth']:.4f}")
    print(f"  orig vs orig (chance): {out['orig_vs_orig']:.4f}")
    print(f"  Q3 SDV CTGAN(orig) vs synth: 0.9993")
    print()
    print("Reading: orig vs synth at ~1.0 means host's generator produces a")
    print("structurally different distribution from orig. orig vs synth ~0.5")
    print("would mean synth = orig re-weighted (no real generator).")


if __name__ == "__main__":
    main()
