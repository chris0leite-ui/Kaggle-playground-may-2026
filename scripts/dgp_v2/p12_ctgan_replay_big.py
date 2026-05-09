"""Phase 12 — Larger CTGAN surrogate (Phase 2-mini of decode-NN plan).

P3 used SDV defaults: epochs=20, batch=500, generator/discriminator=(128,128),
embedding=64. Disc AUC vs host = 0.9993 (near-perfect distinguishability).

P12 trains a bigger surrogate on the same 80k synth-train subsample to
test whether the disc AUC gap is from training budget or genuine
architectural mismatch:
  - epochs: 40 (2x P3)
  - batch: 1000 (2x P3)
  - generator/discriminator: (256, 256) (2x P3)
  - embedding: 128 (2x P3)
  - pac: 10 (default)

If disc AUC drops materially (< 0.95), the host signature is largely a
training-budget artifact and bigger surrogates close the gap.
If disc AUC stays > 0.99, the host has structural differences (custom
preprocessing, custom conditioning vector, custom mode counts) that
hyperparameter scaling alone won't capture.

Outputs same artifact format as P3 for direct comparison.
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

NAME = "p12_ctgan_replay_disc_big"

CTGAN_FEATS = [
    "Driver", "Compound", "Race", "Year", "PitStop",
    "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
CAT_COLS = ["Driver", "Compound", "Race"]


def main():
    ts = time.time()
    print("Loading data...", flush=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    rng = np.random.default_rng(SEED)
    n_ctgan = 80_000
    train_idx = rng.choice(len(train), n_ctgan, replace=False)
    ctgan_train = train.iloc[train_idx][CTGAN_FEATS].copy()
    for c in CAT_COLS:
        ctgan_train[c] = ctgan_train[c].astype("string")
    print(f"  CTGAN training subset: {ctgan_train.shape} [{time.time()-ts:.0f}s]",
          flush=True)

    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer
    md = SingleTableMetadata()
    md.detect_from_dataframe(ctgan_train)
    for c in CAT_COLS:
        md.update_column(c, sdtype="categorical")
    for c in ["Year", "PitStop", "Stint"]:
        md.update_column(c, sdtype="categorical")
    for c in ["LapNumber", "TyreLife", "Position", "Position_Change"]:
        md.update_column(c, sdtype="numerical")

    syn = CTGANSynthesizer(
        md,
        epochs=40,
        batch_size=1000,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        embedding_dim=128,
        pac=10,
        verbose=True,
    )
    print(f"  fitting (bigger) CTGAN... [{time.time()-ts:.0f}s]", flush=True)
    syn.fit(ctgan_train)
    print(f"  done fitting [{time.time()-ts:.0f}s]", flush=True)

    n_replay = 200_000
    replay = syn.sample(num_rows=n_replay)
    print(f"  replay shape: {replay.shape} [{time.time()-ts:.0f}s]", flush=True)

    full_synth = pd.concat([train[CTGAN_FEATS], test[CTGAN_FEATS]],
                           ignore_index=True)
    disc_synth = full_synth.copy(); disc_synth["__src"] = 1
    disc_replay = replay[CTGAN_FEATS].copy(); disc_replay["__src"] = 0
    disc_df = pd.concat([disc_synth, disc_replay], ignore_index=True)

    for c in CAT_COLS:
        cats = pd.Categorical(disc_df[c]).categories
        disc_df[c] = pd.Categorical(disc_df[c], categories=cats).codes.astype("int32")

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
    n_synth = len(full_synth); n_train = len(train)
    disc_oof = np.zeros(len(disc_df), dtype=np.float32)
    fold_aucs = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(disc_df)), src)):
        fts = time.time()
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X.iloc[tr], src[tr], eval_set=[(X.iloc[va], src[va])],
              categorical_feature=CAT_COLS + ["Year", "PitStop", "Stint"],
              callbacks=[lgb.early_stopping(80, verbose=False),
                         lgb.log_evaluation(0)])
        disc_oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(src[va], disc_oof[va])
        fold_aucs.append(float(auc))
        print(f"  fold {fold} disc AUC {auc:.4f} [{time.time()-fts:.0f}s]",
              flush=True)
    overall_auc = roc_auc_score(src, disc_oof)
    print(f"\n  Overall disc AUC: {overall_auc:.5f}", flush=True)

    synth_disc_pred = disc_oof[:n_synth]
    train_disc_pred = synth_disc_pred[:n_train]
    test_disc_pred = synth_disc_pred[n_train:]
    np.save(ART / f"oof_{NAME}_strat.npy",
            np.column_stack([1 - train_disc_pred, train_disc_pred]))
    np.save(ART / f"test_{NAME}_strat.npy",
            np.column_stack([1 - test_disc_pred, test_disc_pred]))
    summary = {
        "name": NAME,
        "overall_disc_auc": float(overall_auc),
        "p3_overall_disc_auc": 0.99926,
        "delta_vs_p3_bp": float((overall_auc - 0.99926) * 1e4),
        "synth_disc_pred_mean": float(synth_disc_pred.mean()),
        "synth_disc_pred_std": float(synth_disc_pred.std()),
        "n_ctgan_train": int(n_ctgan),
        "n_replay": int(n_replay),
        "ctgan_epochs": 40,
        "ctgan_batch_size": 1000,
        "ctgan_generator_dim": [256, 256],
        "ctgan_discriminator_dim": [256, 256],
        "ctgan_embedding_dim": 128,
        "ctgan_pac": 10,
        "fold_aucs": fold_aucs,
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {NAME}_results.json", flush=True)


if __name__ == "__main__":
    main()
