"""d4_cb_yetirank — CatBoost YetiRank pairwise ranking base (NH14).

All current CB bases (year-cat, lossguide, slow-wide-bag) use Logloss.
YetiRank is a pairwise ranking loss → different loss surface, structurally
aligned with AUC. Likely orthogonal predictions to the GBDT-Logloss pool.

Design choices:
  group_id = (Year, Race, Driver) — natural F1 stint unit; 40,869 groups,
             median 11 rows, ~1 positive/group (5% have 0 positives, fine
             for YetiRank).
  cat_cols = Driver, Race, Compound, Year — Year promoted (mech #1 win).
  hyperparams = slow+wide (lr=0.03, iter=4000, l2=8, depth=6, od_wait=100),
                echoing cb_slow-wide-bag which is BEST Strat CB.

Output handling:
  CatBoostRanker.predict() returns raw ranking scores (unbounded). For
  stack compatibility, we rank-normalize each fold's val/test predictions
  to (0,1) before saving — preserves AUC, keeps logit-features in the LR
  meta from blowing up.

Modes (Strat-only per Rule R1):
  smoke   1 fold, 50k stratified rows  → ~30s API sanity
  probe   1 fold, full data            → 1-fold time gate (Rule 2)
  anchor  5-fold Strat                 → final OOF/test artifact

Usage:
  python scripts/d4_cb_yetirank.py --mode smoke
  python scripts/d4_cb_yetirank.py --mode probe
  python scripts/d4_cb_yetirank.py --mode anchor
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
CAT_COLS = ["Driver", "Race", "Compound", "Year"]
GROUP_COLS = ["Year", "Race", "Driver"]
BASE_S = 0.94075
M5H_S = 0.95043

P_YETIRANK = dict(
    loss_function="YetiRank",
    iterations=4000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=8.0,
    random_seed=SEED,
    eval_metric="AUC",
    od_type="Iter",
    od_wait=100,
    verbose=0,
    thread_count=-1,
    allow_writing_files=False,
)


def load_data(smoke: bool = False):
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    if smoke:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(train), 50000, replace=False)
        train = train.iloc[idx].reset_index(drop=True)
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    g_train_str = (X["Year"] + "_" + X["Race"] + "_" + X["Driver"]).astype(str).values
    g_test_str = (X_test["Year"] + "_" + X_test["Race"] + "_" + X_test["Driver"]).astype(str).values
    g_train = pd.Categorical(g_train_str).codes.astype(np.int64)
    g_test = pd.Categorical(g_test_str).codes.astype(np.int64)
    return train, X, y, X_test, g_train, g_test


def fit_one_fold(X, y, X_test, g_train, tr, va, fit_test=True):
    """YetiRank requires Pool data sorted contiguously by group_id."""
    Xtr = X.iloc[tr].reset_index(drop=True)
    Xva = X.iloc[va].reset_index(drop=True)
    ytr, yva = y[tr], y[va]
    gtr, gva = g_train[tr], g_train[va]
    o_tr = np.argsort(gtr, kind="stable")
    o_va = np.argsort(gva, kind="stable")
    Xtr_s = Xtr.iloc[o_tr].reset_index(drop=True)
    Xva_s = Xva.iloc[o_va].reset_index(drop=True)
    ytr_s, yva_s = ytr[o_tr], yva[o_va]
    gtr_s, gva_s = gtr[o_tr], gva[o_va]

    ptr = Pool(Xtr_s, ytr_s, group_id=gtr_s, cat_features=CAT_COLS)
    pva = Pool(Xva_s, yva_s, group_id=gva_s, cat_features=CAT_COLS)
    m = CatBoostRanker(**P_YETIRANK)
    m.fit(ptr, eval_set=pva)

    p_va_s = m.predict(Xva_s)
    inv_va = np.empty_like(o_va)
    inv_va[o_va] = np.arange(len(o_va))
    p_va_raw = p_va_s[inv_va]

    p_test_raw = None
    if fit_test:
        p_test_raw = m.predict(X_test)

    return p_va_raw, p_test_raw, int(m.get_best_iteration())


def smoke_mode():
    _, X, y, X_test, g_train, _ = load_data(smoke=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[0]
    t0 = time.time()
    p_va, _, bi = fit_one_fold(X, y, X_test, g_train, tr, va, fit_test=False)
    auc = roc_auc_score(y[va], p_va)
    wall = time.time() - t0
    print(f"[yetirank] SMOKE: AUC={auc:.5f} best_iter={bi} wall={wall:.1f}s")
    return dict(mode="smoke", auc=auc, best_iter=bi, wall=wall)


def probe_mode():
    _, X, y, X_test, g_train, _ = load_data(smoke=False)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[0]
    t0 = time.time()
    p_va, _, bi = fit_one_fold(X, y, X_test, g_train, tr, va, fit_test=False)
    auc = roc_auc_score(y[va], p_va)
    wall = time.time() - t0
    proj = wall * N_FOLDS
    print(f"[yetirank] PROBE f0: AUC={auc:.5f} best_iter={bi} wall={wall:.1f}s "
          f"5-fold-Strat proj={proj/60:.1f}min")
    if proj > 3600:
        print(f"  WARNING: projection > 1h — Rule 2 says shrink")
    return dict(mode="probe", auc_fold0=auc, best_iter_fold0=bi,
                wall_fold0=wall, strat_5fold_proj_s=proj)


def anchor_mode():
    _, X, y, X_test, g_train, g_test = load_data(smoke=False)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof_rank = np.zeros(len(y), dtype=np.float32)
    test_rank_sum = np.zeros(len(X_test), dtype=np.float64)
    scores, biters, walls = [], [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        p_va, p_test, bi = fit_one_fold(X, y, X_test, g_train, tr, va, fit_test=True)
        oof_rank[va] = (rankdata(p_va) / len(p_va)).astype(np.float32)
        test_rank_sum += rankdata(p_test) / len(p_test)
        s = float(roc_auc_score(y[va], p_va))
        scores.append(s); biters.append(bi); walls.append(time.time() - t0)
        print(f"  [yetirank/strat] f{k}: AUC={s:.5f} bi={bi} wall={walls[-1]:.1f}s")

    test_rank = (test_rank_sum / N_FOLDS).astype(np.float32)
    auc = float(roc_auc_score(y, oof_rank))
    delta = (auc - BASE_S) * 1e4
    delta_m5h = (auc - M5H_S) * 1e4
    print(f"[yetirank/strat] OOF={auc:.5f}  Δbase={delta:+.1f}bp  "
          f"Δm5h={delta_m5h:+.1f}bp  sd={np.std(scores):.5f}")

    save_oof("d4_cb_yetirank_strat",
             np.column_stack([1 - oof_rank, oof_rank]),
             np.column_stack([1 - test_rank, test_rank]),
             dict(oof_score=auc, fold_std=float(np.std(scores)),
                  fold_scores=scores, cv="strat", metric="roc_auc",
                  delta_vs_baseline_bp=delta, delta_vs_m5h_bp=delta_m5h,
                  params_used=P_YETIRANK, cat_cols=CAT_COLS,
                  group_cols=GROUP_COLS, best_iters=biters, fold_walls=walls,
                  output="rank-normalized within fold/test"))
    return dict(mode="anchor", strat_oof=auc, delta_bp=delta,
                delta_m5h_bp=delta_m5h, std=float(np.std(scores)),
                best_iters=biters, walls=walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "probe", "anchor"], default="smoke")
    args = ap.parse_args()
    print(f"=== d4_cb_yetirank ({args.mode}) ===")
    print(f"  cat_cols={CAT_COLS}  group_cols={GROUP_COLS}")
    print(f"  params={ {k: v for k, v in P_YETIRANK.items() if k != 'verbose'} }")

    if args.mode == "smoke":
        result = smoke_mode()
    elif args.mode == "probe":
        result = probe_mode()
    else:
        result = anchor_mode()

    out_path = Path(f"scripts/artifacts/d4_cb_yetirank_{args.mode}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
