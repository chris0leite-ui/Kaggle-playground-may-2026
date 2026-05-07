"""Day-16 H9 — Full-test transductive pseudo-label.

ζ6 axis. Different from d5 confidence-extreme (top/bot 5%, NULL).
Here we use PRIMARY's predicted probabilities on ALL test rows as
SOFT pseudo-labels (not hard 0/1; the predicted P), and retrain
ONE LGBM on (train + test_pseudo) with half-weight on test pseudo.

Mechanism: standard transductive learning. PRIMARY captures the
signal that's IN the K=22 hier-meta; an LGBM trained on
synth_train + test_pseudo extends the GBDT's training distribution
to the full data manifold (covering test-row areas that train doesn't
densely cover). Different from d5 confidence-extreme which used hard
labels and only 10% of test.

Output:
  oof_d16_h9_transductive_pseudo_strat.npy   (n_train, 2)
  test_d16_h9_transductive_pseudo_strat.npy  (n_test, 2)
  d16_h9_transductive_pseudo_results.json
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

ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5

PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
CATS = ["Driver", "Compound", "Race", "Year"]

LGBM_PARAMS = dict(objective="binary", metric="auc",
                   num_leaves=31, learning_rate=0.05,
                   min_child_samples=200, feature_fraction=0.85,
                   bagging_fraction=0.85, bagging_freq=1,
                   verbose=-1, seed=SEED)
LGBM_BOOST = 500


def main():
    t0 = time.time()
    print("[h9] loading data + PRIMARY test pred ...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).values
    n_train, n_test = len(train), len(test)
    primary_test = np.load(PRIMARY_TEST)[:, 1].astype(np.float32)
    print(f"[h9] train {n_train}  test {n_test}  primary-test mean={primary_test.mean():.4f}",
          flush=True)

    # Encode cats.
    encoders = {}
    full = pd.concat([train[CATS], test[CATS]], axis=0, ignore_index=True)
    for c in CATS:
        vals = full[c].astype(str).unique().tolist()
        encoders[c] = {v: i for i, v in enumerate(vals)}
    for df in (train, test):
        for c in CATS:
            df[c + "_idx"] = df[c].astype(str).map(encoders[c]).astype(np.int32)

    feat_cols = NUMERICS + [c + "_idx" for c in CATS]
    cat_idx_cols = [c + "_idx" for c in CATS]

    # Build extended (train + test_pseudo) feature matrix once
    # Sample weight: train rows weight=1.0, test pseudo rows weight=0.5
    Xtr_synth = train[feat_cols].copy()
    Xte_synth = test[feat_cols].copy()
    print(f"[h9] extended dataset: train+pseudo = {n_train + n_test} rows",
          flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n_train), y))

    oof = np.zeros(n_train, dtype=np.float32)
    test_pred = np.zeros(n_test, dtype=np.float32)

    pseudo_w = 0.5
    for fold, (tr, va) in enumerate(splits):
        # Train portion = synth_train_fold + ALL test pseudo
        Xtr_fold = Xtr_synth.iloc[tr]
        ytr_fold = y[tr]
        wtr_fold = np.ones(len(tr), dtype=np.float32)
        # Combine with test pseudo
        X_combined = pd.concat([Xtr_fold, Xte_synth], axis=0, ignore_index=True)
        y_combined = np.concatenate([ytr_fold, primary_test]).astype(np.float32)
        w_combined = np.concatenate([wtr_fold,
                                     np.full(n_test, pseudo_w, dtype=np.float32)])

        Xva_fold = Xtr_synth.iloc[va]
        yva_fold = y[va]

        dtr = lgb.Dataset(X_combined, y_combined, weight=w_combined,
                          categorical_feature=cat_idx_cols)
        dva = lgb.Dataset(Xva_fold, yva_fold, categorical_feature=cat_idx_cols,
                          reference=dtr)
        model = lgb.train(LGBM_PARAMS, dtr, num_boost_round=LGBM_BOOST,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80, verbose=False)])
        n_iter = model.best_iteration or LGBM_BOOST
        oof[va] = model.predict(Xva_fold, num_iteration=n_iter)
        test_pred += model.predict(Xte_synth, num_iteration=n_iter) / N_FOLDS
        print(f"  fold {fold}: best_iter={n_iter}  fold_auc={roc_auc_score(yva_fold, oof[va]):.5f}",
              flush=True)

    full_auc = float(roc_auc_score(y, oof))
    print(f"\n[h9] standalone OOF AUC = {full_auc:.6f}", flush=True)

    np.save(ART / "oof_d16_h9_transductive_pseudo_strat.npy",
            np.column_stack([1.0 - oof, oof]))
    np.save(ART / "test_d16_h9_transductive_pseudo_strat.npy",
            np.column_stack([1.0 - test_pred, test_pred]))
    res = dict(pseudo_weight=pseudo_w,
               primary_test_mean=float(primary_test.mean()),
               primary_test_std=float(primary_test.std()),
               standalone_oof_auc=full_auc,
               n_train=n_train, n_test=n_test,
               wall_s=time.time() - t0)
    (ART / "d16_h9_transductive_pseudo_results.json").write_text(json.dumps(res, indent=2))
    print(f"[h9] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
