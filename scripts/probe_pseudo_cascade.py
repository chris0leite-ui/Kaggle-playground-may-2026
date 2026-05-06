"""scripts/probe_pseudo_cascade.py — confidence-filtered pseudo-label cascade.

Synthetic-data lens: synth test rows often have unusually decisive
PRIMARY meta confidence; partial pseudo-labelling at confidence
extremes is cheap and may transfer (vs full d5 cascade which over-amplified).

Procedure (one round):
  1. Use d12_lr_meta (= K=21 LR-meta-OOF) as PRIMARY confidence proxy.
  2. Top 5% test by confidence → soft-pseudo-label = 1
     Bot 5% test by confidence → soft-pseudo-label = 0
     Middle 90% → not used.
  3. Retrain a single LightGBM base on (train ∪ pseudo-train).
  4. Save OOF + test, gate via probe_min_meta.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))

    # Confidence-extreme pseudo
    p_test = primary_test
    hi_thr = float(np.quantile(p_test, 0.95))
    lo_thr = float(np.quantile(p_test, 0.05))
    pseudo_pos = p_test >= hi_thr
    pseudo_neg = p_test <= lo_thr
    print(f"pseudo: hi_thr={hi_thr:.4f} ({pseudo_pos.sum()} test rows → 1), "
          f"lo_thr={lo_thr:.4f} ({pseudo_neg.sum()} test rows → 0); "
          f"middle {(~pseudo_pos & ~pseudo_neg).sum()} skipped")

    feat_cols = ["TyreLife", "RaceProgress", "LapTime_Delta",
                 "Cumulative_Degradation", "Position", "LapTime (s)",
                 "Stint", "Year", "Position_Change", "LapNumber",
                 "Driver", "Compound", "Race"]
    cat_cols = ["Driver", "Compound", "Race"]

    X = train[feat_cols].copy()
    X_test = test[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    pseudo_idx = np.where(pseudo_pos | pseudo_neg)[0]
    pseudo_y = np.where(pseudo_pos[pseudo_idx], 1, 0).astype(np.int8)
    X_pseudo = X_test.iloc[pseudo_idx].copy()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  min_data_in_leaf=200, verbose=-1, seed=SEED)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        # Combine outer-train rows + pseudo (pseudo half-weighted)
        X_combined = pd.concat([X.iloc[tr], X_pseudo], axis=0, ignore_index=True)
        # Re-cast categoricals after concat (concat reverts to object dtype)
        for c in cat_cols:
            X_combined[c] = X_combined[c].astype("category")
        y_combined = np.concatenate([y[tr], pseudo_y])
        w_combined = np.concatenate([np.ones(len(tr), dtype=np.float32),
                                      0.5 * np.ones(len(pseudo_y), dtype=np.float32)])
        dtr = lgb.Dataset(X_combined, y_combined, weight=w_combined,
                          categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        test_pred += m.predict(X_test) / N_FOLDS
        s = float(roc_auc_score(y[va], oof[va]))
        print(f"  fold {k}: AUC {s:.5f} best_iter {m.best_iteration} "
              f"wall {time.time()-t:.1f}s")

    auc = float(roc_auc_score(y, oof))
    rho, _ = spearmanr(test_pred, primary_test)
    print(f"\n=== pseudo_cascade base ===")
    print(f"  std OOF: {auc:.5f}  Δ vs PRIMARY {(auc-auc_primary)*1e4:+.2f} bp")
    print(f"  ρ vs PRIMARY: {rho:.6f}")

    np.save(ART / "oof_pseudo_cascade_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_pseudo_cascade_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))
    summary = dict(std_oof=auc, delta_vs_primary_bp=(auc - auc_primary)*1e4,
                   rho_vs_primary=float(rho),
                   pseudo_pos_count=int(pseudo_pos.sum()),
                   pseudo_neg_count=int(pseudo_neg.sum()),
                   wall_s=time.time() - t0)
    (ART / "probe_pseudo_cascade.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/probe_pseudo_cascade.json (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
