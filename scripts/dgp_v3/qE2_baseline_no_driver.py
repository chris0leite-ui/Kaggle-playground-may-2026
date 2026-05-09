"""qE2 — baseline disc-AUC excluding Driver and high-cardinality categoricals.

Driver alone is a perfect discriminator (orig has 31 codes; synth has 887).
Strip Driver to see if orig vs synth is physics-different too, or only
label-different.

Output: scripts/artifacts/dgp_v3_qE2_no_driver.json
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


def disc_auc(a: pd.DataFrame, b: pd.DataFrame, drop: list[str] = ()) -> float:
    common = sorted(set(a.columns) & set(b.columns))
    common = [c for c in common if c not in drop]
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
    common = sorted(set(orig.columns) & set(synth.columns))
    o1 = orig[common].sample(20_000, random_state=0)
    s1 = synth[common].sample(20_000, random_state=0)

    out["all_features"] = disc_auc(o1, s1)
    t(f"all features: {out['all_features']:.4f}", ts)

    out["drop_Driver"] = disc_auc(o1, s1, drop=["Driver"])
    t(f"drop Driver: {out['drop_Driver']:.4f}", ts)

    out["drop_Driver_Race"] = disc_auc(o1, s1, drop=["Driver", "Race"])
    t(f"drop Driver, Race: {out['drop_Driver_Race']:.4f}", ts)

    out["drop_Driver_Race_Stint"] = disc_auc(o1, s1, drop=["Driver", "Race", "Stint"])
    t(f"drop Driver, Race, Stint: {out['drop_Driver_Race_Stint']:.4f}", ts)

    out["only_continuous"] = disc_auc(
        o1, s1, drop=["Driver", "Race", "Stint", "Compound", "Year",
                       "PitStop", "Position", "LapNumber"],
    )
    t(f"only continuous (LT, LTD, CD, RP, PC, TL): {out['only_continuous']:.4f}", ts)

    fp = ART / "dgp_v3_qE2_no_driver.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== orig vs synth disc-AUC by feature subset ===")
    for k, v in out.items():
        print(f"  {k:30s} {v:.4f}")


if __name__ == "__main__":
    main()
