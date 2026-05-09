"""Phase 16 — TVAE-on-orig surrogate (decoding data point #2).

P13 v2 trains CTGAN on orig (data point #1). P16 trains TVAE on the
same orig data with comparable budget. Compare disc AUCs vs host
synth: if TVAE disc AUC < CTGAN disc AUC, host might be VAE-class
not GAN-class; if equal, both architecture classes fail equally
(host's signature is in custom preprocessing/cond, not arch).

Per d18 f1: at SDV defaults, TVAE disc AUC was 0.9894, CTGAN 0.9884
(TVAE slightly less close). With longer training + same orig, P16
re-tests this with comparable budget to P13 v2.
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

NAME = "p16_tvae_on_orig"

CTGAN_FEATS = [
    "Driver", "Compound", "Race", "Year", "PitStop",
    "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
CAT_COLS = ["Driver", "Compound", "Race"]
DISCRETE_AS_CAT = ["Year", "PitStop", "Stint"]


def main():
    ts = time.time()
    print("Loading orig + host synth...", flush=True)
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    orig = orig.dropna(subset=["Compound"])
    orig_train = orig[CTGAN_FEATS].copy()
    for c in CAT_COLS:
        orig_train[c] = orig_train[c].astype("string")
    print(f"  orig {orig_train.shape}", flush=True)

    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer
    md = SingleTableMetadata()
    md.detect_from_dataframe(orig_train)
    for c in CAT_COLS:
        md.update_column(c, sdtype="categorical")
    for c in DISCRETE_AS_CAT:
        md.update_column(c, sdtype="categorical")
    for c in ["LapNumber", "TyreLife", "Position", "Position_Change"]:
        md.update_column(c, sdtype="numerical")

    syn = TVAESynthesizer(
        md,
        epochs=80,
        batch_size=500,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        embedding_dim=128,
        verbose=True,
    )
    print(f"  fitting TVAE on orig... [{time.time()-ts:.0f}s]", flush=True)
    syn.fit(orig_train)
    print(f"  done fitting [{time.time()-ts:.0f}s]", flush=True)

    n_replay = 200_000
    print(f"  sampling {n_replay} replay rows...", flush=True)
    replay = syn.sample(num_rows=n_replay)
    print(f"  replay shape: {replay.shape} [{time.time()-ts:.0f}s]", flush=True)
    replay.to_parquet(ART / f"{NAME}_replay.parquet", index=False)

    full_host = pd.concat([train[CTGAN_FEATS], test[CTGAN_FEATS]],
                          ignore_index=True)
    disc_host = full_host.copy(); disc_host["__src"] = 1
    disc_replay = replay[CTGAN_FEATS].copy(); disc_replay["__src"] = 0
    disc_df = pd.concat([disc_host, disc_replay], ignore_index=True)

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
        bagging_freq=4, seed=SEED, verbose=-1, n_jobs=4,
    )
    disc_oof = np.zeros(len(disc_df), dtype=np.float32)
    fold_aucs = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(disc_df)), src)):
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X.iloc[tr], src[tr], eval_set=[(X.iloc[va], src[va])],
              categorical_feature=CAT_COLS + DISCRETE_AS_CAT,
              callbacks=[lgb.early_stopping(80, verbose=False),
                         lgb.log_evaluation(0)])
        disc_oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        fold_aucs.append(float(roc_auc_score(src[va], disc_oof[va])))
        print(f"  fold {fold} disc AUC {fold_aucs[-1]:.4f}", flush=True)
    overall_auc = roc_auc_score(src, disc_oof)
    print(f"\n  Overall disc AUC (host vs P16-TVAE): {overall_auc:.5f}",
          flush=True)

    summary = {
        "name": NAME,
        "trained_on": "aadigupta1601 orig (101k rows, 14 cols)",
        "n_orig_train": int(len(orig_train)),
        "n_replay": int(n_replay),
        "tvae_epochs": 80,
        "tvae_batch_size": 500,
        "tvae_compress_dims": [128, 128],
        "tvae_decompress_dims": [128, 128],
        "tvae_embedding_dim": 128,
        "overall_disc_auc_vs_host": float(overall_auc),
        "fold_aucs": fold_aucs,
        "p3_recursive_disc_auc": 0.99926,
        "d18_f1_tvae_default_disc_auc": 0.9894,
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {NAME}_results.json", flush=True)


if __name__ == "__main__":
    main()
