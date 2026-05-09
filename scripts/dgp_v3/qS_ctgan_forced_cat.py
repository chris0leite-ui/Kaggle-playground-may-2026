"""qS — CTGAN with explicit categorical metadata for cell axes.

SDV CTGAN auto-detects discrete vs continuous. By default LapNumber,
Position, TyreLife are auto-numerical but they're integer-categorical.
Explicitly mark Year, Compound, PitStop, Stint, Position, LapNumber,
TyreLife as categorical, and train for 30 epochs.

If this drops disc-AUC below 0.99, the cond-vector schema axis matters
more than we thought.
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
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

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


def main() -> None:
    out: dict = {}
    ts = time.time()

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna()
    train = pd.read_csv(DATA / "train.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
    synth_disc = train.drop(columns=["PitNextLap"]).sample(20_000, random_state=0).reset_index(drop=True)
    t(f"orig {orig.shape} synth_disc {synth_disc.shape}", ts)

    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer

    metadata = Metadata.detect_from_dataframe(data=orig)
    # Force as categorical — only low-cardinality cols where auto-detect may
    # treat them as numerical.
    forced_cat = ["Year", "Compound", "PitStop", "Stint", "Position"]
    for col in forced_cat:
        if col in orig.columns:
            try:
                metadata.update_column(column_name=col, sdtype="categorical")
            except Exception as e:
                print(f"   could not force {col}: {e}")
    t("metadata forced categorical for 5 cols", ts)

    model = CTGANSynthesizer(metadata, epochs=30, cuda=False, verbose=False)
    model.fit(orig)
    t("CTGAN fit done (30 ep, forced cat)", ts)

    s = model.sample(20_000)
    t(f"sample {s.shape}", ts)
    auc = disc_auc(s, synth_disc)
    t(f"qS CTGAN forced-cat 30ep: disc_auc = {auc:.4f}", ts)
    out["disc_auc"] = auc

    fp = ART / "dgp_v3_qS_ctgan_forced_cat.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    print("\n=== qS summary ===")
    print(f"  reference: SDV CTGAN 10ep default: 0.9993")
    print(f"  qS CTGAN forced 9-cat 30ep:        {auc:.4f}")
    if auc < 0.95:
        print("  *** HIT - explicit cat metadata closes the gap")


if __name__ == "__main__":
    main()
