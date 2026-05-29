"""scripts/d21_membership_inference.py — Membership inference / row-copy base.

ISSUES leaf 7j. Synthetic-only-plan pivot after M1+M2 NULL.

Hypothesis: the synthesizer near-copied some orig rows. For each synth
row, the nearest-orig-row's PitNextLap should leak orig's actual label
when distance is small. This is the only mechanism class under the
synthetic frame that bypasses the "conditionally near-independent"
DGP signature (the DGP residual diagnostic showed within-row features
are absorbed; this exploits cross-row proximity instead).

Mechanism:
  - Standardize the 11 shared numeric features on orig.
  - For each synth row, find k=5 nearest orig neighbors WITHIN the same
    (Compound, Race) cell. Cell-conditioning enforces structural
    plausibility (orig and synth share the same race/compound universe
    per d18 chain decomp; Driver excluded — synth has 856 ghost codes).
  - Fallback to global NN for synth rows in cells with no orig coverage.
  - 3 features per synth row: mean_y_top5 / min_dist / mean_dist_top5.
  - Downstream LGBM 5-fold StratifiedKFold → OOF + test predictions.

Outputs:
  scripts/artifacts/oof_d21_membership_inference_strat.npy  (n_train, 2)
  scripts/artifacts/test_d21_membership_inference_strat.npy (n_test, 2)
  scripts/artifacts/d21_membership_inference_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import BallTree

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
LAPTIME = "LapTime (s)"

NUM_FEATS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             LAPTIME, "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change"]
CELL_KEYS = ["Compound", "Race"]
K = 5
K18_OOF = ART / "oof_K18_pathb_driverclass_stint_tau100000.npy"
K18_TEST = ART / "test_K18_pathb_driverclass_stint_tau100000.npy"


def _pos(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, 1].astype(np.float64)
    return arr.astype(np.float64).ravel()


def _standardize_fit(orig_X: np.ndarray):
    mu = orig_X.mean(axis=0)
    sd = orig_X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def _standardize_apply(X: np.ndarray, mu, sd) -> np.ndarray:
    return (X - mu) / sd


def _nn_features_for_split(orig_X, orig_y, query_X, orig_cell, query_cell,
                            k=K, global_tree=None, global_y=None,
                            log=print):
    """Per-row top-k NN features in (Compound, Race) cells.

    Falls back to global NN for query rows whose cell has < k orig neighbors.
    Returns (mean_y_topk, min_dist, mean_dist_topk) as columns (n_query, 3).
    """
    n_q = len(query_X)
    mean_y = np.full(n_q, np.nan, dtype=np.float32)
    min_d = np.full(n_q, np.nan, dtype=np.float32)
    mean_d = np.full(n_q, np.nan, dtype=np.float32)

    # Group orig and query by cell
    orig_groups = pd.DataFrame({"cell": orig_cell, "idx": np.arange(len(orig_cell))})\
        .groupby("cell")["idx"].apply(np.asarray).to_dict()
    query_groups = pd.DataFrame({"cell": query_cell, "idx": np.arange(len(query_cell))})\
        .groupby("cell")["idx"].apply(np.asarray).to_dict()

    n_cells = len(query_groups)
    n_fallback = 0
    t_start = time.time()
    for ci, (cell, q_idx) in enumerate(query_groups.items(), 1):
        o_idx = orig_groups.get(cell, np.array([], dtype=int))
        if len(o_idx) >= k:
            tree = BallTree(orig_X[o_idx], leaf_size=40)
            dists, nbrs = tree.query(query_X[q_idx], k=k)
            mean_y[q_idx] = orig_y[o_idx[nbrs]].mean(axis=1).astype(np.float32)
            min_d[q_idx] = dists[:, 0].astype(np.float32)
            mean_d[q_idx] = dists.mean(axis=1).astype(np.float32)
        else:
            n_fallback += len(q_idx)
            dists, nbrs = global_tree.query(query_X[q_idx], k=k)
            mean_y[q_idx] = global_y[nbrs].mean(axis=1).astype(np.float32)
            min_d[q_idx] = dists[:, 0].astype(np.float32)
            mean_d[q_idx] = dists.mean(axis=1).astype(np.float32)
        if ci % 50 == 0 or ci == n_cells:
            log(f"    cells {ci}/{n_cells}  fallback rows so far: {n_fallback}  "
                f"({time.time()-t_start:.1f}s)")

    return np.column_stack([mean_y, min_d, mean_d]), n_fallback


def downstream_lgbm(tr_X, tr_y, te_X, smoke=False, log=print):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr_y), dtype=np.float64)
    test_avg = np.zeros(len(te_X), dtype=np.float64)
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=200, feature_fraction=1.0,
             bagging_fraction=0.85, bagging_freq=5, verbosity=-1, seed=SEED)
    n_round = 400 if not smoke else 60
    for fi, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(tr_y)), tr_y), 1):
        ds_tr = lgb.Dataset(tr_X[tr_i], label=tr_y[tr_i], free_raw_data=False)
        ds_va = lgb.Dataset(tr_X[va_i], label=tr_y[va_i], reference=ds_tr,
                            free_raw_data=False)
        m = lgb.train(p, ds_tr, num_boost_round=n_round, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        oof[va_i] = m.predict(tr_X[va_i], num_iteration=m.best_iteration)
        test_avg += m.predict(te_X, num_iteration=m.best_iteration) / N_FOLDS
        auc = roc_auc_score(tr_y[va_i], oof[va_i])
        log(f"    fold {fi}: AUC={auc:.5f}  best_iter={m.best_iteration}")
    log(f"    OOF AUC = {roc_auc_score(tr_y, oof):.5f}")
    return oof, test_avg


def _save_oof_test(name, oof_pos, test_pos):
    oof = np.column_stack([1.0 - oof_pos, oof_pos]).astype(np.float64)
    test = np.column_stack([1.0 - test_pos, test_pos]).astype(np.float64)
    np.save(ART / f"oof_{name}_strat.npy", oof)
    np.save(ART / f"test_{name}_strat.npy", test)
    print(f"  saved oof_{name}_strat.npy {oof.shape}  test_{name}_strat.npy {test.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-synth-sample", type=int, default=0)
    ap.add_argument("--out-prefix", default="d21_membership_inference")
    args = ap.parse_args()

    t0 = time.time()
    smoke = args.smoke
    print(f"[d21 membership-inference{' SMOKE' if smoke else ''}]  loading data")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test",
                                     "Pre-Season Track Session"])].copy()
    print(f"  train {tr.shape}  test {te.shape}  orig {orig.shape}")

    if args.n_synth_sample > 0:
        tr = tr.sample(n=args.n_synth_sample,
                       random_state=SEED).reset_index(drop=True)
        te = te.sample(n=min(args.n_synth_sample // 2, len(te)),
                       random_state=SEED).reset_index(drop=True)
        print(f"  smoke-sampled: train {tr.shape}  test {te.shape}")

    # Numerics — drop any rows with NaN in the search-space cols (orig may
    # have nulls; coerce to float, then drop NaN rows from orig).
    for df in [orig, tr, te]:
        for c in NUM_FEATS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    pre_n = len(orig)
    orig = orig.dropna(subset=NUM_FEATS).reset_index(drop=True)
    print(f"  orig after NaN drop: {len(orig)} (lost {pre_n - len(orig)})")

    orig_X = orig[NUM_FEATS].values.astype(np.float64)
    tr_X = tr[NUM_FEATS].fillna(0).values.astype(np.float64)
    te_X = te[NUM_FEATS].fillna(0).values.astype(np.float64)
    orig_y = orig[TARGET].astype(int).values

    mu, sd = _standardize_fit(orig_X)
    orig_X_z = _standardize_apply(orig_X, mu, sd)
    tr_X_z = _standardize_apply(tr_X, mu, sd)
    te_X_z = _standardize_apply(te_X, mu, sd)

    orig_cell = (orig["Compound"].astype(str) + "|" +
                 orig["Race"].astype(str)).values
    tr_cell = (tr["Compound"].astype(str) + "|" +
               tr["Race"].astype(str)).values
    te_cell = (te["Compound"].astype(str) + "|" +
               te["Race"].astype(str)).values

    print(f"\n[build global BallTree on orig (fallback)]")
    t_g = time.time()
    global_tree = BallTree(orig_X_z, leaf_size=40)
    print(f"  built ({time.time()-t_g:.1f}s)")

    print(f"\n[per-cell NN on synth train]")
    tr_nn, tr_fb = _nn_features_for_split(
        orig_X_z, orig_y, tr_X_z, orig_cell, tr_cell, k=K,
        global_tree=global_tree, global_y=orig_y)
    print(f"  train fallback rows: {tr_fb} / {len(tr)} ({tr_fb/len(tr):.1%})")

    print(f"\n[per-cell NN on synth test]")
    te_nn, te_fb = _nn_features_for_split(
        orig_X_z, orig_y, te_X_z, orig_cell, te_cell, k=K,
        global_tree=global_tree, global_y=orig_y)
    print(f"  test fallback rows: {te_fb} / {len(te)} ({te_fb/len(te):.1%})")

    # Sanity: feature stats
    feat_names = ["nn_mean_y_top5", "nn_min_dist", "nn_mean_dist_top5"]
    print(f"\n[feature stats]")
    for i, name in enumerate(feat_names):
        print(f"  {name:<22s}  train mean={tr_nn[:,i].mean():+.4f} "
              f"std={tr_nn[:,i].std():.4f}  test mean={te_nn[:,i].mean():+.4f} "
              f"std={te_nn[:,i].std():.4f}")

    # Class-conditional KS on nn_mean_y_top5 (the load-bearing feature)
    from scipy.stats import ks_2samp
    y_tr = tr[TARGET].astype(int).values
    pos = y_tr == 1
    ks_stat = ks_2samp(tr_nn[pos, 0], tr_nn[~pos, 0])
    print(f"  KS nn_mean_y_top5  y=1 vs y=0  stat={ks_stat.statistic:.4f}")

    print(f"\n[downstream LGBM on 3-feat NN block]")
    oof, test = downstream_lgbm(tr_nn, y_tr, te_nn, smoke=smoke)
    _save_oof_test(args.out_prefix, oof, test)

    standalone_auc = float(roc_auc_score(y_tr, oof))

    summary = dict(
        n_train=int(len(tr)), n_test=int(len(te)), n_orig=int(len(orig)),
        k=K, train_fallback_rows=int(tr_fb), test_fallback_rows=int(te_fb),
        feature_names=feat_names,
        ks_nn_mean_y_top5_y1_vs_y0=float(ks_stat.statistic),
        standalone_oof_auc=standalone_auc,
    )

    if K18_OOF.exists() and K18_TEST.exists() and args.n_synth_sample == 0:
        k18_oof = _pos(np.load(K18_OOF))
        k18_te = _pos(np.load(K18_TEST))
        if len(k18_oof) == len(oof):
            k18_auc = float(roc_auc_score(y_tr, k18_oof))
            rho_oof = float(spearmanr(oof, k18_oof)[0])
            rho_test = float(spearmanr(test, k18_te)[0])
            gap_bp = (standalone_auc - k18_auc) * 1e4
            summary.update(
                k18_oof_auc=k18_auc, gap_vs_k18_bp=gap_bp,
                rho_oof_vs_k18=rho_oof, rho_test_vs_k18=rho_test)
            print(f"\n  K=18 PRIMARY OOF AUC : {k18_auc:.5f}")
            print(f"  d21 standalone OOF   : {standalone_auc:.5f}")
            print(f"  gap                  : {gap_bp:+.2f} bp")
            print(f"  ρ_OOF  d21↔K18       : {rho_oof:.5f}")
            print(f"  ρ_test d21↔K18       : {rho_test:.5f}")

    (ART / f"{args.out_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n[done]  total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
