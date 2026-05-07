"""d18 E2 — per-row preimage join via kNN in orig.

For each synth row, find K nearest orig rows (per-Compound partition) on the
7 marginal-aligned features (KS-low per Phase-1 audit: TyreLife KS=0.017,
Position KS=0.019, LapTime KS=0.056, CumDeg KS=0.071, RaceProgress KS=0.186,
LapTime_Delta KS=0.179, LapNumber KS=0.188).

Aggregate top-K orig neighbours into per-row features:
  preimage_y_mean    : mean(PitNextLap) over neighbours = soft pseudo-label
  preimage_y_std     : std(PitNextLap) = neighbourhood disagreement
  preimage_dist_mean : mean L2 distance to top-K neighbours (match quality)
  preimage_dist_min  : min L2 distance (best single match)
  preimage_ntl_mean  : mean Normalized_TyreLife of neighbours (recovered DGP)
  preimage_year_match: fraction of top-K with matching Year
  preimage_race_match: fraction of top-K with matching Race

Then 5-fold LGBM on (raw 14 features + 7 preimage features).

Leakage-clean: kNN matches are queried in orig by row features only;
PitNextLap of orig is used only as a feature aggregation, not as the
synth's training label. Different from leak-lookup (d15) which used
EB-smoothed P(y|feature_bin) — this is row-level kNN.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"
LAPTIME = "LapTime (s)"
K_NN = 10  # number of neighbours

# Marginal-aligned features per Phase-1 KS (low orig↔synth divergence).
# Same axis the d16 win exploited.
KNN_FEATS = ["TyreLife", "Position", LAPTIME, "Cumulative_Degradation",
             "RaceProgress", "LapTime_Delta", "LapNumber"]
CAT_OK = ["Compound", "Race"]
NUM_FEATS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             LAPTIME, "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change"]


def _scale(orig_X, others):
    """Per-feature standardize using orig stats."""
    mu = orig_X.mean(axis=0); sd = orig_X.std(axis=0) + 1e-8
    return [(orig_X - mu) / sd] + [(o - mu) / sd for o in others]


def main():
    t0 = time.time()
    print("[E2 per-row preimage]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    print(f"  train {tr.shape}  test {te.shape}  orig {orig.shape}")

    # Compound mapping (consistent across orig + synth)
    cmps = sorted(set(orig["Compound"].astype(str).unique()) |
                  set(tr["Compound"].astype(str).unique()) |
                  set(te["Compound"].astype(str).unique()))
    cm = {c: i for i, c in enumerate(cmps)}
    orig_cmp = orig["Compound"].astype(str).map(cm).astype(int).values
    tr_cmp = tr["Compound"].astype(str).map(cm).astype(int).values
    te_cmp = te["Compound"].astype(str).map(cm).astype(int).values

    # Per-Compound kNN with sklearn.NearestNeighbors (kd_tree).
    feats_arr_orig = orig[KNN_FEATS].astype(float).values
    feats_arr_tr = tr[KNN_FEATS].astype(float).values
    feats_arr_te = te[KNN_FEATS].astype(float).values
    # Replace any NaN (Position_Change has some early-lap NaN in orig)
    for arr in (feats_arr_orig, feats_arr_tr, feats_arr_te):
        np.nan_to_num(arr, copy=False, nan=0.0)

    # Standardize per Compound (per-Compound feature means differ).
    # Build per-cmp scalers using orig only.
    out_tr_feats = np.zeros((len(tr), 7), dtype=np.float32)
    out_te_feats = np.zeros((len(te), 7), dtype=np.float32)
    out_cols = ["preimage_y_mean", "preimage_y_std", "preimage_dist_mean",
                "preimage_dist_min", "preimage_ntl_mean",
                "preimage_year_match", "preimage_race_match"]

    orig_y = np.asarray(orig[TARGET].astype(int).values, dtype=np.int64)
    if "Normalized_TyreLife" in orig.columns:
        orig_ntl = np.asarray(orig["Normalized_TyreLife"].astype(float)
                              .fillna(0.0).values, dtype=np.float64)
    else:
        orig_ntl = np.zeros(len(orig))
    orig_year = np.asarray(orig["Year"].astype(int).values, dtype=np.int64)
    orig_race = np.asarray(orig["Race"].astype(str).values, dtype=object)

    tr_year = np.asarray(tr["Year"].astype(int).values, dtype=np.int64)
    tr_race = np.asarray(tr["Race"].astype(str).values, dtype=object)
    te_year = np.asarray(te["Year"].astype(int).values, dtype=np.int64)
    te_race = np.asarray(te["Race"].astype(str).values, dtype=object)

    for c, cidx in cm.items():
        mask_o = orig_cmp == cidx
        mask_tr = tr_cmp == cidx
        mask_te = te_cmp == cidx
        n_o, n_tr, n_te = int(mask_o.sum()), int(mask_tr.sum()), int(mask_te.sum())
        print(f"  Compound={c} (idx={cidx})  orig={n_o}  tr={n_tr}  te={n_te}")
        if n_o == 0 or n_tr == 0 + n_te == 0:
            continue
        Xo = feats_arr_orig[mask_o]
        Xtr = feats_arr_tr[mask_tr]
        Xte = feats_arr_te[mask_te]
        Xo_s, Xtr_s, Xte_s = _scale(Xo, [Xtr, Xte])
        k_use = min(K_NN, n_o)
        nn = NearestNeighbors(n_neighbors=k_use, algorithm="kd_tree", n_jobs=-1)
        nn.fit(Xo_s)

        for src_X, src_mask, out, src_year, src_race, label in [
            (Xtr_s, mask_tr, out_tr_feats, tr_year, tr_race, "tr"),
            (Xte_s, mask_te, out_te_feats, te_year, te_race, "te"),
        ]:
            if len(src_X) == 0:
                continue
            t1 = time.time()
            dist, ind = nn.kneighbors(src_X, return_distance=True)
            print(f"    {label} kNN {len(src_X)}×{k_use}  {time.time()-t1:.1f}s")
            ny = orig_y[mask_o][ind]                  # (rows, k)
            nntl = orig_ntl[mask_o][ind]
            nyear = orig_year[mask_o][ind]
            nrace = orig_race[mask_o][ind]
            row_ix = np.where(src_mask)[0]
            out[row_ix, 0] = ny.mean(axis=1)
            out[row_ix, 1] = ny.std(axis=1)
            out[row_ix, 2] = dist.mean(axis=1)
            out[row_ix, 3] = dist.min(axis=1)
            out[row_ix, 4] = nntl.mean(axis=1)
            out[row_ix, 5] = (nyear == src_year[src_mask][:, None]).mean(axis=1)
            out[row_ix, 6] = (nrace == src_race[src_mask][:, None]).mean(axis=1)

    tr_X_pre = pd.DataFrame(out_tr_feats, columns=out_cols)
    te_X_pre = pd.DataFrame(out_te_feats, columns=out_cols)
    print(f"\n  preimage features built  wall {time.time()-t0:.0f}s")
    print("  per-feature stats (train):")
    print(tr_X_pre.describe().to_string())

    # Downstream LGBM (raw + preimage)
    cmps_l = list(cm.keys())
    cmpmap = {c: i for i, c in enumerate(cmps_l)}
    raw_cols = ["Compound", "Race"] + NUM_FEATS
    tr_raw = tr[raw_cols].copy()
    te_raw = te[raw_cols].copy()
    tr_raw["Compound"] = tr_cmp
    te_raw["Compound"] = te_cmp
    races_l = sorted(set(tr["Race"].astype(str).unique()) |
                     set(te["Race"].astype(str).unique()))
    rmap = {r: i for i, r in enumerate(races_l)}
    tr_raw["Race"] = tr["Race"].astype(str).map(rmap).astype(int).values
    te_raw["Race"] = te["Race"].astype(str).map(rmap).astype(int).values

    tr_X = pd.concat([tr_raw.reset_index(drop=True),
                      tr_X_pre.reset_index(drop=True)], axis=1)
    te_X = pd.concat([te_raw.reset_index(drop=True),
                      te_X_pre.reset_index(drop=True)], axis=1)
    y = tr[TARGET].astype(int).values

    print("\n[downstream LGBM raw + preimage]")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test_avg = np.zeros(len(te_X), dtype=np.float64)
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=127, min_data_in_leaf=80, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbosity=-1, seed=SEED)
    cat_idx = [tr_X.columns.get_loc(c) for c in ["Compound", "Race"]]
    for fi, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        ds_tr = lgb.Dataset(tr_X.iloc[tr_i], label=y[tr_i],
                            categorical_feature=cat_idx, free_raw_data=False)
        ds_va = lgb.Dataset(tr_X.iloc[va_i], label=y[va_i],
                            categorical_feature=cat_idx, reference=ds_tr,
                            free_raw_data=False)
        m = lgb.train(p, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_i] = m.predict(tr_X.iloc[va_i], num_iteration=m.best_iteration)
        test_avg += m.predict(te_X, num_iteration=m.best_iteration) / N_FOLDS
        print(f"    fold {fi}: AUC={roc_auc_score(y[va_i], oof[va_i]):.5f}")

    auc_oof = float(roc_auc_score(y, oof))
    print(f"  OOF AUC = {auc_oof:.5f}")

    # Save
    np.save(ART / "oof_d18_e2_preimage_knn_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d18_e2_preimage_knn_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))
    summary = dict(oof_auc=auc_oof, k=K_NN, knn_feats=KNN_FEATS,
                   wall_s=time.time() - t0)
    (ART / "d18_e2_preimage_knn_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done E2]  wall {time.time()-t0:.0f}s  OOF {auc_oof:.5f}")


if __name__ == "__main__":
    main()
