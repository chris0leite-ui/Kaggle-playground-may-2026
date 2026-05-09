"""Phase 2.5 — LGBM trained on the surrogate's REPLAY (P13b).

After P13 produces a CTGAN trained on orig and 200k replay samples,
P13b trains LGBM on the replay (which now has CTGAN-mediated
PitNextLap labels). Then applies to host's synth.

The hypothesis: if our P13 surrogate is close to host's pipeline, an
LGBM trained on our replay produces predictions close to what host
would have produced on the same input distribution. Use as a new base
candidate at K=4+1.

Compare to d15_orig_transfer (LGBM trained directly on orig, +0.778 bp
at K=2 min-meta) — P13b is the surrogate-mediated variant.

Critic notes:
  - The replay's PitNextLap is sampled from CTGAN's conditional
    distribution, not from a deterministic orig source. So the LGBM
    learns the synth-marginal P(y|x). This may match host synth's
    P(y|x) if surrogate matches host.
  - The disc AUC from P13 will tell us how well surrogate matches.
    If disc AUC > 0.99, P13b is unlikely to add lift (we replicate
    host's marginals poorly).
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
SEED = 42

NAME = "p13b_lgbm_on_replay"
REPLAY_PATH = ART / "p13_orig_surrogate_v1_replay.parquet"

CTGAN_FEATS = [
    "Driver", "Compound", "Race", "Year", "PitStop",
    "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
CAT_COLS = ["Driver", "Compound", "Race"]


def main():
    ts = time.time()
    print("Loading replay + host data...", flush=True)
    replay = pd.read_parquet(REPLAY_PATH)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"  replay {replay.shape} | train {train.shape} | test {test.shape}",
          flush=True)
    print(f"  replay PitNextLap rate: {replay['PitNextLap'].astype(int).mean():.4f}",
          flush=True)
    print(f"  host train PitNextLap rate: {train['PitNextLap'].mean():.4f}",
          flush=True)

    # Build joint train: train on REPLAY only (the host's "marginal" surrogate)
    # Apply to host train (for OOF) + test
    feat_cols = CTGAN_FEATS

    # Encode categoricals on union of replay + host
    full = pd.concat([replay[CTGAN_FEATS], train[CTGAN_FEATS], test[CTGAN_FEATS]],
                     ignore_index=True)
    for c in CAT_COLS:
        cats = pd.Categorical(full[c]).categories
        full[c] = pd.Categorical(full[c], categories=cats).codes.astype("int32")
    n_replay = len(replay); n_train = len(train); n_test = len(test)
    Xreplay = full.iloc[:n_replay]
    Xtrain = full.iloc[n_replay:n_replay+n_train]
    Xtest = full.iloc[n_replay+n_train:]
    yreplay = replay["PitNextLap"].astype(int).to_numpy()

    # Train ONE LGBM on full replay (no CV — replay is the "lookup table",
    # train is the holdout we're predicting; OOF on host train is a single
    # forward pass)
    PARAMS = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, n_estimators=2000,
        num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=4, seed=SEED, verbose=-1, n_jobs=4,
    )
    print(f"\nTraining LGBM on replay (n={n_replay})... [{time.time()-ts:.0f}s]",
          flush=True)
    # 80/20 internal val for early stopping
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_replay)
    val_idx = perm[:int(n_replay*0.2)]
    tr_idx = perm[int(n_replay*0.2):]
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(
        Xreplay.iloc[tr_idx], yreplay[tr_idx],
        eval_set=[(Xreplay.iloc[val_idx], yreplay[val_idx])],
        categorical_feature=CAT_COLS + ["Year", "PitStop", "Stint"],
        callbacks=[lgb.early_stopping(80, verbose=False),
                   lgb.log_evaluation(0)],
    )
    val_auc = roc_auc_score(yreplay[val_idx], m.predict_proba(Xreplay.iloc[val_idx])[:, 1])
    print(f"  internal val AUC (replay): {val_auc:.5f}", flush=True)

    # Predict on host train + test
    print(f"\nPredicting on host... [{time.time()-ts:.0f}s]", flush=True)
    host_train_pred = m.predict_proba(Xtrain)[:, 1]
    host_test_pred = m.predict_proba(Xtest)[:, 1]

    # OOF-equivalent: on host train, evaluate AUC vs host PitNextLap
    y_train = train["PitNextLap"].astype(int).to_numpy()
    host_train_auc = roc_auc_score(y_train, host_train_pred)
    print(f"  AUC on host train: {host_train_auc:.5f}", flush=True)
    print(f"  AUC for d15_orig_transfer (lgbm on orig): 0.85138", flush=True)

    # Save in (n, 2) format
    np.save(ART / f"oof_{NAME}_strat.npy",
            np.column_stack([1 - host_train_pred, host_train_pred]))
    np.save(ART / f"test_{NAME}_strat.npy",
            np.column_stack([1 - host_test_pred, host_test_pred]))
    summary = {
        "name": NAME,
        "trained_on": "P13 replay (200k rows)",
        "internal_val_auc_on_replay": float(val_auc),
        "auc_on_host_train": float(host_train_auc),
        "d15_orig_transfer_baseline": 0.85138,
        "n_replay": int(n_replay),
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {NAME}_results.json [{time.time()-ts:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
