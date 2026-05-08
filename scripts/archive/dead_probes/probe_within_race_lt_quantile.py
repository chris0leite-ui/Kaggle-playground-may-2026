"""scripts/probe_within_race_lt_quantile.py — within-Race quantile of LapTime_Delta.

EDA Phase F finding: LapTime_Delta single-feature LR has +922 bp
Strat→GroupKF gap. The raw form leaks race-specific scale. This probe
tests whether replacing/normalizing it via within-Race-Year quantile
rank (5 buckets) creates a leak-robust single feature whose signal
adds to K=21.

Build: a minimal LightGBM base trained on (existing features +
LapTime_Delta_q5_per_race_year) → 5-fold StratKF OOF + test averaged.
Save artifacts; min-meta gate via probe_min_meta.py.

If signal lifts at the meta gate, this is a HEDGE-track candidate
(leak-robust signal addition).
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


def within_race_year_quantile(df, col="LapTime_Delta", n_q=5):
    """Per-(Race, Year) quantile rank of `col`, returned as int 0..n_q-1.
    Uses df-internal ranks → deterministic, no fold leakage."""
    out = np.zeros(len(df), dtype=np.int8)
    g = df.groupby(["Race", "Year"], observed=True)
    for (race, year), idx in g.groups.items():
        idx_arr = np.asarray(idx)
        vals = df.loc[idx_arr, col].values
        if len(vals) < n_q:
            out[idx_arr] = 0
            continue
        # rank to [0,1) then bucket
        ranks = vals.argsort().argsort().astype(np.float64) / max(len(vals) - 1, 1)
        out[idx_arr] = np.clip(np.floor(ranks * n_q), 0, n_q - 1).astype(np.int8)
    return out


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)

    # New feature: within-Race-Year quantile of LapTime_Delta
    train["LT_q5_race_year"] = within_race_year_quantile(train, "LapTime_Delta", 5)
    test["LT_q5_race_year"] = within_race_year_quantile(test, "LapTime_Delta", 5)
    print(f"new feature LT_q5_race_year: train dist {np.bincount(train['LT_q5_race_year']+0)}, "
          f"test dist {np.bincount(test['LT_q5_race_year']+0)}")

    # Build LGBM base: existing numerics + categoricals + new feature
    feat_cols = ["Driver", "Compound", "Race", "Year", "Stint", "TyreLife",
                 "Position", "LapTime_Delta", "Cumulative_Degradation",
                 "RaceProgress", "Position_Change", "LT_q5_race_year"]
    cat_cols = ["Driver", "Compound", "Race"]

    X = train[feat_cols].copy()
    X_test = test[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  min_data_in_leaf=200, verbose=-1, seed=SEED)

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        dtrain = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dval = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
        model = lgb.train(params, dtrain, num_boost_round=1500,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(80),
                                     lgb.log_evaluation(0)])
        oof[va] = model.predict(X.iloc[va])
        test_pred += model.predict(X_test) / N_FOLDS
        s = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(s)
        print(f"  fold {k}: AUC {s:.5f}  best_iter {model.best_iteration}  "
              f"wall {time.time()-t_fold:.1f}s")

    auc = float(roc_auc_score(y, oof))
    rho, _ = spearmanr(test_pred, primary_test)
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    new_pos = test_pred >= rare_thr
    flips_neg = int(np.sum(primary_pos & ~new_pos))
    flips_pos = int(np.sum(~primary_pos & new_pos))

    print(f"\n=== within_race_lt_q5 base ===")
    print(f"  std OOF: {auc:.5f} (PRIMARY {roc_auc_score(y, primary_oof):.5f})")
    print(f"  ρ vs PRIMARY: {rho:.6f}")
    print(f"  flips +→− {flips_neg}, −→+ {flips_pos}")
    print(f"  fold std: {np.std(fold_aucs):.5f}")

    np.save(ART / "oof_within_race_lt_q5_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_within_race_lt_q5_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))
    sub = sample_sub.copy(); sub[TARGET] = test_pred
    sub.to_csv("submissions/submission_within_race_lt_q5.csv", index=False)
    summary = dict(std_oof=auc, rho_vs_primary=float(rho),
                   flips_to_neg=flips_neg, flips_to_pos=flips_pos,
                   fold_aucs=fold_aucs, fold_std=float(np.std(fold_aucs)),
                   wall_s=time.time() - t0)
    (ART / "probe_within_race_lt_q5.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ {ART / 'probe_within_race_lt_q5.json'} (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
