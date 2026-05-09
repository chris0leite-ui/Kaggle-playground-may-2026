"""Phase A2 smoke probe — minimal SDV CTGAN train on a 20k orig sample.

Purpose: validate the pipeline (5 epochs, ~3 min CPU) before committing
to a full 20-epoch run on full orig (~25 min). Outputs the same artefacts
the production probe will, just at a smaller scale, so the production
probe can re-use the disc-AUC harness.

Output:
  scripts/artifacts/dgp_v3_q3_smoke_disc_auc.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main() -> None:
    ts = time.time()

    # Load orig, drop Normalized_TyreLife per host rule.
    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"])
    orig = orig.dropna()  # 66 NaNs in Compound
    orig_samp = orig.sample(20_000, random_state=0)
    t(f"loaded orig {orig.shape}, sampled {orig_samp.shape}", ts)

    # Load synth (only what the disc needs)
    train = pd.read_csv(DATA / "train.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
    synth = train.drop(columns=["PitNextLap"]).sample(20_000, random_state=0)
    t(f"loaded synth subsample {synth.shape}", ts)

    # SDV CTGAN — newer SDV API uses synthesizers.CTGANSynthesizer
    try:
        from sdv.metadata import Metadata
        from sdv.single_table import CTGANSynthesizer

        metadata = Metadata.detect_from_dataframe(data=orig_samp)
        t("SDV metadata detected", ts)

        synth_model = CTGANSynthesizer(
            metadata,
            epochs=5,
            verbose=False,
            cuda=False,
        )
        synth_model.fit(orig_samp)
        t("CTGAN fit done", ts)

        replay = synth_model.sample(20_000)
        t(f"replay sampled {replay.shape}", ts)
    except Exception as e:
        print(f"SDV CTGAN failed: {type(e).__name__}: {e}")
        sys.exit(1)

    # 2-class disc: replay (label=0) vs synth (label=1)
    common_cols = sorted(set(replay.columns) & set(synth.columns))
    print(f"  common cols ({len(common_cols)}): {common_cols}")

    df_disc = pd.concat(
        [
            replay[common_cols].assign(_lbl=0),
            synth[common_cols].assign(_lbl=1),
        ],
        ignore_index=True,
    )

    # Encode categoricals as ints (LightGBM-friendly)
    cat_cols = df_disc.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df_disc[c] = pd.Categorical(df_disc[c]).codes
    X = df_disc.drop(columns=["_lbl"]).values
    y = df_disc["_lbl"].values
    t(f"prepared disc matrix X={X.shape}", ts)

    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            n_jobs=-1,
            verbosity=-1,
        )
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    auc = float(roc_auc_score(y, oof))
    t(f"5-fold disc AUC = {auc:.4f}", ts)

    out = {
        "smoke": True,
        "epochs": 5,
        "n_orig_train": int(len(orig_samp)),
        "n_replay": int(len(replay)),
        "n_synth_disc": int(len(synth)),
        "common_cols": common_cols,
        "disc_auc": auc,
        "wall_seconds": time.time() - ts,
    }
    fp = ART / "dgp_v3_q3_smoke_disc_auc.json"
    fp.write_text(json.dumps(out, indent=2))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
