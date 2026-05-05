"""CatBoost variants — Stage-B0/B/C exploration script.

Variants (per audit/2026-05-04-catboost-research.md):
  m3-baseline   reference to confirm we match M3 (depth=6, default CTR)
  year-cat      #1: move Year (4 vals, top-importance) into CAT_COLS
  onehot        #2: one_hot_max_size=10 (Compound + Year auto one-hot)
  ctr-complex   #3: max_ctr_complexity=6 + explicit combinations_ctr
  counter-only  #4: simple/combinations_ctr=Counter only (no target leak)
  slow-wide     #5: lr=0.03, iter=2500, l2=8, od_wait=100
  mvs           #6: bootstrap_type=MVS, subsample=0.7, mvs_reg=0.1
  lossguide     #7: grow_policy=Lossguide, num_leaves=64, max_depth=8
  ordered       #8: boosting_type=Ordered (forced)

Modes:
  smoke   1 fold, 50k stratified rows  → ~5-15s, sanity check
  probe   1 fold, full data            → ~70-300s, 5-fold time estimate
  anchor  5-fold both anchors          → ~5-25min, gated artifact

Usage:
  python scripts/cb_variants_two_anchor.py --variant year-cat --mode smoke
  python scripts/cb_variants_two_anchor.py --variant year-cat --mode probe
  python scripts/cb_variants_two_anchor.py --variant year-cat --mode anchor
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
CAT_BASE = ["Driver", "Race", "Compound"]
CAT_PLUS_YEAR = ["Driver", "Race", "Compound", "Year"]
BASE_S, BASE_G = 0.94075, 0.92059
M3_S, M3_G = 0.94612, 0.91645

P_BASE = dict(iterations=800, learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
              random_seed=42, eval_metric="AUC", od_type="Iter", od_wait=50,
              verbose=0, thread_count=-1, allow_writing_files=False)


def variant_config(name: str):
    """Return (cat_cols, params) for the named variant."""
    if name == "m3-baseline":
        return CAT_BASE, dict(P_BASE)
    if name == "year-cat":
        return CAT_PLUS_YEAR, dict(P_BASE)
    if name == "onehot":
        p = dict(P_BASE); p["one_hot_max_size"] = 10
        return CAT_PLUS_YEAR, p
    if name == "ctr-complex":
        p = dict(P_BASE)
        p["max_ctr_complexity"] = 6
        p["simple_ctr"] = ["Borders:CtrBorderCount=15:Prior=0/1"]
        p["combinations_ctr"] = ["Borders:CtrBorderCount=15:Prior=0/1"]
        return CAT_PLUS_YEAR, p
    if name == "counter-only":
        p = dict(P_BASE)
        p["simple_ctr"] = ["Counter:CtrBorderCount=15:Prior=0/1"]
        p["combinations_ctr"] = ["Counter:CtrBorderCount=15:Prior=0/1"]
        return CAT_PLUS_YEAR, p
    if name == "slow-wide":
        p = dict(P_BASE)
        p["learning_rate"] = 0.03
        p["iterations"] = 2500
        p["l2_leaf_reg"] = 8.0
        p["od_wait"] = 100
        return CAT_PLUS_YEAR, p
    if name == "mvs":
        p = dict(P_BASE)
        p["bootstrap_type"] = "MVS"
        p["subsample"] = 0.7
        p["mvs_reg"] = 0.1
        return CAT_PLUS_YEAR, p
    if name == "lossguide":
        p = dict(P_BASE)
        p["grow_policy"] = "Lossguide"
        p["num_leaves"] = 64
        p["max_depth"] = 8
        p.pop("depth", None)
        return CAT_PLUS_YEAR, p
    if name == "ordered":
        p = dict(P_BASE)
        p["boosting_type"] = "Ordered"
        p["iterations"] = 500
        return CAT_PLUS_YEAR, p
    raise ValueError(f"unknown variant: {name}")


def load_data(cat_cols, smoke=False):
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    if smoke:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(train), 50000, replace=False)
        train = train.iloc[idx].reset_index(drop=True)
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    return train, X, y, X_test


def fit_one_fold(X, y, X_test, tr, va, cat_cols, params, fit_test=True):
    ptr = Pool(X.iloc[tr], y[tr], cat_features=cat_cols)
    pva = Pool(X.iloc[va], y[va], cat_features=cat_cols)
    m = CatBoostClassifier(**params)
    m.fit(ptr, eval_set=pva)
    p_va = m.predict_proba(pva)[:, 1]
    p_test = None
    if fit_test:
        ptest = Pool(X_test, cat_features=cat_cols)
        p_test = m.predict_proba(ptest)[:, 1]
    return p_va, p_test, int(m.get_best_iteration())


def smoke_mode(name, cat_cols, params):
    train, X, y, X_test = load_data(cat_cols, smoke=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[0]
    t0 = time.time()
    p_va, _, bi = fit_one_fold(X, y, X_test, tr, va, cat_cols, params,
                                fit_test=False)
    auc = roc_auc_score(y[va], p_va)
    wall = time.time() - t0
    print(f"[{name}] SMOKE: AUC={auc:.5f}  best_iter={bi}  wall={wall:.1f}s")
    return dict(mode="smoke", variant=name, auc=auc, best_iter=bi, wall=wall)


def probe_mode(name, cat_cols, params):
    train, X, y, X_test = load_data(cat_cols, smoke=False)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[0]
    t0 = time.time()
    p_va, _, bi = fit_one_fold(X, y, X_test, tr, va, cat_cols, params,
                                fit_test=False)
    auc = roc_auc_score(y[va], p_va)
    wall = time.time() - t0
    proj = wall * N_FOLDS * 2
    print(f"[{name}] PROBE: f0 AUC={auc:.5f}  best_iter={bi}  wall={wall:.1f}s  "
          f"5-fold-2anchor proj={proj/60:.1f}min")
    return dict(mode="probe", variant=name, auc_fold0=auc, best_iter_fold0=bi,
                wall_fold0=wall, two_anchor_proj_s=proj)


def anchor_mode(name, cat_cols, params):
    train, X, y, X_test = load_data(cat_cols, smoke=False)
    out = {}
    for anchor, splits in [
        ("strat", list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                        random_state=SEED).split(np.zeros(len(y)), y))),
        ("groupkf", list(GroupKFold(n_splits=N_FOLDS).split(
            np.zeros(len(y)), y, train["Race"].values))),
    ]:
        oof = np.zeros(len(y), dtype=np.float32)
        test_p = np.zeros(len(X_test), dtype=np.float32)
        scores, biters, walls = [], [], []
        for k, (tr, va) in enumerate(splits):
            t0 = time.time()
            p_va, p_test, bi = fit_one_fold(X, y, X_test, tr, va, cat_cols, params,
                                             fit_test=True)
            oof[va] = p_va
            test_p += p_test / N_FOLDS
            s = float(roc_auc_score(y[va], p_va))
            scores.append(s); biters.append(bi); walls.append(time.time() - t0)
            print(f"  [{name}/{anchor}] f{k}: AUC={s:.5f} bi={bi} "
                  f"wall={walls[-1]:.1f}s")
        auc = float(roc_auc_score(y, oof))
        base = BASE_S if anchor == "strat" else BASE_G
        delta = (auc - base) * 1e4
        print(f"[{name}/{anchor}] OOF={auc:.5f}  Δbase={delta:+.1f}bp  "
              f"sd={np.std(scores):.5f}")
        save_oof(f"cb_{name}_{anchor}",
                 np.column_stack([1 - oof, oof]),
                 np.column_stack([1 - test_p, test_p]),
                 dict(oof_score=auc, fold_std=float(np.std(scores)),
                      fold_scores=scores, cv=anchor, metric="roc_auc",
                      delta_vs_baseline_bp=delta, params_used=params,
                      cat_cols=cat_cols, best_iters=biters, fold_walls=walls))
        out[anchor] = dict(auc=auc, delta_bp=delta, std=float(np.std(scores)),
                           best_iters=biters, walls=walls)
    return dict(mode="anchor", variant=name, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--mode", choices=["smoke", "probe", "anchor"], default="smoke")
    args = ap.parse_args()

    cat_cols, params = variant_config(args.variant)
    print(f"=== {args.variant} ({args.mode}) ===")
    print(f"  cat_cols={cat_cols}")
    print(f"  params={ {k: v for k, v in params.items() if k != 'verbose'} }")

    if args.mode == "smoke":
        result = smoke_mode(args.variant, cat_cols, params)
    elif args.mode == "probe":
        result = probe_mode(args.variant, cat_cols, params)
    else:
        result = anchor_mode(args.variant, cat_cols, params)

    out_path = Path(f"scripts/artifacts/cb_{args.variant}_{args.mode}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
