"""Phase 3 — recursive CTGAN replay-discriminator (synth-only).

This is the E3 probe that was deferred in 2026-05-07 because the
sandbox lacked torch+sdv. Both are now installed.

DGP-finding hypothesis: the host's synthesizer is CTGAN-class with
custom preprocessing/wrapper. Off-the-shelf CTGAN trained on synth
itself will produce "replay" rows that match the host's CTGAN
distribution where they overlap, and miss the host-specific signature
where they differ. A 2-class discriminator {synth, replay} captures
this signature; the discriminator's output for synth rows is a feature
quantifying host-specific bias.

Steps:

  1. Subsample 100k synth-train rows for CTGAN training (CPU budget).
  2. Train SDV CTGANSynthesizer with epochs=30, batch=500.
  3. Sample 200k replay rows.
  4. Train LightGBM 5-fold discriminator on
     {synth (label=1), replay (label=0)}.
  5. Predict on full synth-train + synth-test.
  6. Save OOF + test predictions in (n, 2) format.

Output: oof_p3_ctgan_replay_disc_strat.npy / test_*.npy.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
SEED = 42

NAME = "p3_ctgan_replay_disc"

CTGAN_FEATS = [
    "Driver", "Compound", "Race", "Year", "PitStop",
    "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
CAT_COLS = ["Driver", "Compound", "Race"]
NUM_COLS = [c for c in CTGAN_FEATS if c not in CAT_COLS]


def main():
    ts = time.time()
    print("Loading data...", flush=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"  train {train.shape} | test {test.shape} [{time.time()-ts:.1f}s]",
          flush=True)

    # Subsample for CTGAN training
    rng = np.random.default_rng(SEED)
    n_ctgan = 80_000
    train_idx = rng.choice(len(train), n_ctgan, replace=False)
    ctgan_train = train.iloc[train_idx][CTGAN_FEATS].copy()

    # Convert dtypes for SDV
    for c in CAT_COLS:
        ctgan_train[c] = ctgan_train[c].astype("string")
    print(f"  CTGAN training subset: {ctgan_train.shape} "
          f"[{time.time()-ts:.1f}s]", flush=True)

    # SDV metadata + synthesizer
    print("Building SDV metadata + CTGANSynthesizer...", flush=True)
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer
    md = SingleTableMetadata()
    md.detect_from_dataframe(ctgan_train)
    # Force sdtypes
    for c in CAT_COLS:
        md.update_column(c, sdtype="categorical")
    md.update_column("Year", sdtype="categorical")
    md.update_column("PitStop", sdtype="categorical")
    md.update_column("Stint", sdtype="categorical")
    md.update_column("LapNumber", sdtype="numerical")
    md.update_column("TyreLife", sdtype="numerical")
    md.update_column("Position", sdtype="numerical")
    md.update_column("Position_Change", sdtype="numerical")

    syn = CTGANSynthesizer(
        md,
        epochs=20,
        batch_size=500,
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        embedding_dim=64,
        verbose=True,
    )
    print(f"  fitting CTGAN... [{time.time()-ts:.1f}s]", flush=True)
    syn.fit(ctgan_train)
    print(f"  done fitting [{time.time()-ts:.1f}s]", flush=True)

    # Sample replay
    print("Sampling replay rows...", flush=True)
    n_replay = 200_000
    replay = syn.sample(num_rows=n_replay)
    print(f"  replay shape: {replay.shape} [{time.time()-ts:.1f}s]",
          flush=True)

    # Build discriminator dataset
    full_synth = pd.concat([train[CTGAN_FEATS], test[CTGAN_FEATS]],
                           ignore_index=True)
    print(f"  full_synth: {full_synth.shape}", flush=True)

    # All synth (train+test) labeled 1; replay labeled 0
    disc_synth = full_synth.copy()
    disc_synth["__src"] = 1
    disc_replay = replay[CTGAN_FEATS].copy()
    disc_replay["__src"] = 0
    disc_df = pd.concat([disc_synth, disc_replay], ignore_index=True)
    print(f"  disc_df: {disc_df.shape}", flush=True)

    # Encode categoricals
    for c in CAT_COLS:
        cats = pd.Categorical(disc_df[c]).categories
        disc_df[c] = pd.Categorical(disc_df[c], categories=cats).codes.astype("int32")

    # 5-fold disc within disc_df
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    src = disc_df["__src"].to_numpy()
    X = disc_df[CTGAN_FEATS]

    PARAMS = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, n_estimators=1000,
        num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=4, seed=SEED, verbose=-1,
    )

    # Predict P(synth=1) for synth_train + synth_test rows
    # Strategy: train on (replay + synth-NOT-this-fold), predict on synth-this-fold
    # We do regular 5-fold on the disc_df, then aggregate predictions for synth rows
    # (synth rows are at idx 0..n_synth-1; replay at n_synth..)

    n_synth = len(full_synth)
    n_train = len(train)

    # Per-fold disc training, predicting synth rows that are in the val fold
    # Synth rows are in the disc_df at idx [0, n_synth); replay at [n_synth, end)
    # We need OOF (per-row predictions where each row is a hold-out)
    # Use 5-fold on the disc rows; predictions on val are OOF
    disc_oof = np.zeros(len(disc_df), dtype=np.float32)
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(disc_df)), src)):
        fts = time.time()
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(
            X.iloc[tr], src[tr],
            eval_set=[(X.iloc[va], src[va])],
            categorical_feature=CAT_COLS + ["Year", "PitStop", "Stint"],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)],
        )
        disc_oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(src[va], disc_oof[va])
        print(f"  fold {fold} disc AUC {auc:.4f}  "
              f"[{time.time()-fts:.0f}s, {time.time()-ts:.0f}s]", flush=True)

    overall_auc = roc_auc_score(src, disc_oof)
    print(f"\n  Overall disc AUC: {overall_auc:.5f}", flush=True)

    # Extract predictions for synth_train + synth_test
    synth_disc_pred = disc_oof[:n_synth]
    train_disc_pred = synth_disc_pred[:n_train]
    test_disc_pred = synth_disc_pred[n_train:]

    # Save in (n, 2) format
    train_2d = np.column_stack([1 - train_disc_pred, train_disc_pred])
    test_2d = np.column_stack([1 - test_disc_pred, test_disc_pred])
    np.save(ART / f"oof_{NAME}_strat.npy", train_2d)
    np.save(ART / f"test_{NAME}_strat.npy", test_2d)

    # Compute the discriminator-as-feature stats
    summary = {
        "name": NAME,
        "overall_disc_auc": float(overall_auc),
        "synth_disc_pred_mean": float(synth_disc_pred.mean()),
        "synth_disc_pred_std": float(synth_disc_pred.std()),
        "n_ctgan_train": int(n_ctgan),
        "n_replay": int(n_replay),
        "ctgan_epochs": 20,
        "interpretation": (
            "disc AUC > 0.7 means CTGAN replay distinguishable from host "
            "synth — i.e. our CTGAN replicates poorly. High disc-pred "
            "for a synth row means it's host-specific (off-the-shelf CTGAN "
            "wouldn't have generated it that way)."
        ),
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {NAME}_results.json", flush=True)


if __name__ == "__main__":
    main()
