"""scripts/probe_target_reform_v2.py — 4 new target reformulations.

Tier 2: extends probe_target_reform.py with 4 new framings of the
same train labels, building on the inv_laps_until_pit success.

  pit_horizon_multiclass: 4-class {this lap=0, 1-2 ahead, 3-5 ahead, >5 ahead}
                          softmax → use prob[0] (this-lap) as feature.
  next_pit_lap_number:    regression on absolute lap number of next pit (0 if none).
  stint_index_within_race: regression on # of completed stints by this row
                          (Stint counter relative to race start).
  reverse_cumcount_pits:  regression on # of pits remaining in this race for
                          this driver (0 if none).

All four computed STRICTLY OOF: per-fold, target for fold-val rows uses
ONLY fold-train PitNextLap labels. Test target NOT used.

Each gets a 5-fold StratKF LGBM regression (or softmax classifier).
Saves OOF/test artifacts; min-meta gate via probe_min_meta.py afterward.
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
PRIMARY_OOF = ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def compute_targets(df, y):
    """Compute 4 alternative targets per Driver-Race-Year ordered by LapNumber."""
    df = df.copy(); df["_y"] = y; df["_idx"] = np.arange(len(df))
    n = len(df)
    pit_horizon = np.full(n, 3, dtype=np.int8)        # default >5 ahead
    next_pit_lap = np.zeros(n, dtype=np.float32)      # absolute lap number
    stint_index = np.zeros(n, dtype=np.int8)
    reverse_cum = np.zeros(n, dtype=np.int8)

    for keys, grp in df.groupby(["Driver", "Race", "Year"], sort=False):
        gs = grp.sort_values("LapNumber")
        laps = gs["LapNumber"].values
        ys = gs["_y"].values
        idxs = gs["_idx"].values
        # Total pits in this group
        total_pits = int(ys.sum())
        # next_pit_lap: scan backwards
        last_pit = 0
        for i in range(len(gs) - 1, -1, -1):
            if ys[i] == 1:
                last_pit = laps[i]
            next_pit_lap[idxs[i]] = last_pit
        # pit_horizon: bucket the gap
        for i, ix in enumerate(idxs):
            gap = next_pit_lap[ix] - laps[i] if next_pit_lap[ix] > 0 else 999
            if gap <= 0: pit_horizon[ix] = 0    # this lap (or already pit)
            elif gap <= 2: pit_horizon[ix] = 1
            elif gap <= 5: pit_horizon[ix] = 2
            else: pit_horizon[ix] = 3
        # stint_index: cumulative Stint counter
        stints = gs["Stint"].astype(int).values
        stint_index[idxs] = np.clip(stints - stints.min(), 0, 7).astype(np.int8)
        # reverse_cumcount_pits: total - cumulative pits by this row
        cum = np.cumsum(ys)
        reverse_cum[idxs] = (total_pits - cum).clip(0, 10).astype(np.int8)
    return dict(
        pit_horizon=pit_horizon,
        next_pit_lap=next_pit_lap,
        stint_index=stint_index,
        reverse_cum=reverse_cum,
    )


def lgbm_5fold_reg(X, y_target, X_test, cat_cols, name):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # use binary target for stratification
    y_strat = (y_target > np.median(y_target)).astype(int) if y_target.std() > 0 else \
              np.zeros(len(y_target), dtype=int)
    splits = list(skf.split(np.zeros(len(y_target)), y_strat))
    oof = np.zeros(len(y_target))
    test_pred = np.zeros(len(X_test))
    params = dict(objective="regression", metric="rmse",
                  learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  min_data_in_leaf=200, verbose=-1, seed=SEED)
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        dtr = lgb.Dataset(X.iloc[tr], y_target[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y_target[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        test_pred += m.predict(X_test) / N_FOLDS
        print(f"  [{name}] fold {k}: best_iter {m.best_iteration} wall {time.time()-t:.1f}s")
    return oof, test_pred


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))

    feat_num = ["TyreLife", "RaceProgress", "LapTime_Delta",
                "Cumulative_Degradation", "Position", "LapTime (s)",
                "Stint", "Year", "Position_Change", "LapNumber"]
    feat_cat = ["Driver", "Compound", "Race"]
    X = train[feat_num + feat_cat].copy()
    X_test = test[feat_num + feat_cat].copy()
    for c in feat_cat:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    print("Computing 4 alt targets from train labels...")
    t = time.time()
    targets = compute_targets(train, y)
    print(f"  built in {time.time()-t:.1f}s")
    for k, v in targets.items():
        print(f"    {k}: dtype={v.dtype}, range [{v.min()}, {v.max()}], mean {v.mean():.3f}")

    summary = {}
    for name, tgt in [("pit_horizon", targets["pit_horizon"].astype(np.float32)),
                      ("next_pit_lap", targets["next_pit_lap"]),
                      ("stint_index", targets["stint_index"].astype(np.float32)),
                      ("reverse_cum", targets["reverse_cum"].astype(np.float32))]:
        print(f"\n=== {name} ===")
        oof, test_pred = lgbm_5fold_reg(X, tgt, X_test, feat_cat, name)
        auc = float(roc_auc_score(y, oof))
        rho, _ = spearmanr(test_pred, primary_test)
        d = (auc - auc_primary) * 1e4
        print(f"  std OOF AUC: {auc:.5f}  Δ vs PRIMARY: {d:+.2f} bp  ρ vs PRIMARY: {rho:.6f}")
        np.save(ART / f"oof_target_reform_{name}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_target_reform_{name}_strat.npy",
                np.column_stack([1 - test_pred, test_pred]))
        summary[name] = dict(std_oof=auc, delta_vs_primary_bp=float(d),
                              rho_vs_primary=float(rho))

    summary["wall_s"] = time.time() - t0
    out = ART / "probe_target_reform_v2.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out}  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
