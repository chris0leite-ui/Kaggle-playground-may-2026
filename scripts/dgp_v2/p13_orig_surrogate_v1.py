"""Phase 2 v1 — proper orig→synth surrogate (P13).

Trains CTGAN on aadigupta1601 ORIG (101k rows, 14 cols matching host's
output after Normalized_TyreLife drop). Samples 200k. Measures
discriminator AUC vs HOST's full 627k synth.

Goal: minimize disc AUC. The host's pipeline produced disc AUC 0.9993
vs my P3's recursive-on-synth surrogate. P13 is the FIRST surrogate
trained on the actual orig (the host's likely training input), so this
is the proper Phase-2 setup.

Critic notes embedded:
  - Q: drop NTL from training? A: yes — host says NTL "makes prediction
    trivial", so likely was dropped pre-training. Will test both
    variants if first run is high.
  - Q: what config? A: SDV CTGAN defaults (epochs=80 vs default 300 to
    save compute, batch=500, dim=(128,128), embed=64, pac=10). Iterate
    if disc AUC > 0.95.
  - Q: do we condition on PitStop explicitly? A: SDV CTGAN's default
    cond samples a discrete column per row uniformly. PitStop being in
    the mix is automatic. We won't override unless P13 is poor.
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

NAME = "p13_orig_surrogate_v1"

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
    print(f"  orig {orig.shape} | host train {train.shape} | "
          f"host test {test.shape}", flush=True)

    # Filter orig to match host's Compound vocab and drop NaN
    orig = orig.dropna(subset=["Compound"])
    orig_train = orig[CTGAN_FEATS].copy()
    for c in CAT_COLS:
        orig_train[c] = orig_train[c].astype("string")
    print(f"  orig after dropna: {orig_train.shape}", flush=True)

    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer
    md = SingleTableMetadata()
    md.detect_from_dataframe(orig_train)
    for c in CAT_COLS:
        md.update_column(c, sdtype="categorical")
    for c in DISCRETE_AS_CAT:
        md.update_column(c, sdtype="categorical")
    for c in ["LapNumber", "TyreLife", "Position", "Position_Change"]:
        md.update_column(c, sdtype="numerical")

    syn = CTGANSynthesizer(
        md,
        epochs=80,
        batch_size=500,
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        embedding_dim=64,
        pac=10,
        verbose=True,
    )
    print(f"  fitting CTGAN on orig... [{time.time()-ts:.0f}s]", flush=True)
    syn.fit(orig_train)
    print(f"  done fitting [{time.time()-ts:.0f}s]", flush=True)

    n_replay = 200_000
    print(f"  sampling {n_replay} replay rows...", flush=True)
    replay = syn.sample(num_rows=n_replay)
    print(f"  replay shape: {replay.shape} [{time.time()-ts:.0f}s]", flush=True)

    # Save replay for later inversion encoder training
    replay.to_parquet(ART / f"{NAME}_replay.parquet", index=False)
    print(f"  saved replay parquet", flush=True)

    # Disc vs host synth
    full_host_synth = pd.concat([train[CTGAN_FEATS], test[CTGAN_FEATS]],
                                ignore_index=True)
    print(f"  building disc dataset: host {len(full_host_synth)} vs "
          f"replay {len(replay)}", flush=True)
    disc_host = full_host_synth.copy(); disc_host["__src"] = 1
    disc_replay = replay[CTGAN_FEATS].copy(); disc_replay["__src"] = 0
    disc_df = pd.concat([disc_host, disc_replay], ignore_index=True)

    # Encode categoricals (union vocab)
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
        fts = time.time()
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X.iloc[tr], src[tr], eval_set=[(X.iloc[va], src[va])],
              categorical_feature=CAT_COLS + DISCRETE_AS_CAT,
              callbacks=[lgb.early_stopping(80, verbose=False),
                         lgb.log_evaluation(0)])
        disc_oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(src[va], disc_oof[va])
        fold_aucs.append(float(auc))
        print(f"  fold {fold} disc AUC {auc:.4f} [{time.time()-fts:.0f}s]",
              flush=True)
    overall_auc = roc_auc_score(src, disc_oof)
    print(f"\n  Overall disc AUC (host vs P13_replay): {overall_auc:.5f}",
          flush=True)
    print(f"  vs P3 recursive-on-synth: 0.99926", flush=True)
    print(f"  vs off-the-shelf default (d18 f1): 0.9884",
          flush=True)

    summary = {
        "name": NAME,
        "trained_on": "aadigupta1601 (orig, 14 cols, NaN-Compound dropped)",
        "n_orig_train": int(len(orig_train)),
        "n_replay": int(n_replay),
        "ctgan_epochs": 80,
        "ctgan_batch_size": 500,
        "ctgan_generator_dim": [128, 128],
        "ctgan_discriminator_dim": [128, 128],
        "ctgan_embedding_dim": 64,
        "ctgan_pac": 10,
        "overall_disc_auc_vs_host": float(overall_auc),
        "fold_aucs": fold_aucs,
        "p3_baseline_disc_auc": 0.99926,
        "d18_f1_off_shelf_disc_auc": 0.9884,
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {NAME}_results.json", flush=True)


if __name__ == "__main__":
    main()
