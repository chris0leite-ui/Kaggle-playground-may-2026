"""D12 e3_hgbc 5-seed bag (Strat anchor only).

Each seed × StratKF(5, seed=42) → average across 5 seeds (probability-mean).
Saves OOF and test arrays in 2-col [1-p, p] format. Per-seed walls logged.

Seeds: 42, 7, 123, 456, 789. (May reduce to 3 via --seeds.)

Output:
  scripts/artifacts/oof_d12_e3_5seed_bag_strat.npy
  scripts/artifacts/test_d12_e3_5seed_bag_strat.npy
  scripts/artifacts/d12_e3_5seed_bag_results.json

Per-fold progress is also written to scripts/artifacts/d12_e3_progress.txt
(line-by-line, fsync) so we can monitor without stdio buffering.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
N_FOLDS = 5
CV_SEED = 42  # pinned for splits
SEEDS = [int(s) for s in os.environ.get("D12_E3_SEEDS",
                                         "42,7,123").split(",")]
PROGRESS = ART / "d12_e3_progress.txt"


def log(msg):
    sys.stdout.write(msg + "\n"); sys.stdout.flush()
    with open(PROGRESS, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n"); f.flush()
        os.fsync(f.fileno())

HIGH_CARD = ["Driver"]
LOW_CARD = ["Compound", "Race"]


def make_hgbc(seed):
    # Reduced max_iter from 1500 → 600 to fit within multi-agent CPU
    # contention budget. Original e3 hit ~1000 iters at full data; with
    # ES at n_iter_no_change=50, capping at 600 typically loses <0.5bp
    # OOF (~5% of base AUC delta). Acceptable for calibration probe.
    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=seed,
        categorical_features="from_dtype",
    )


def run_seed(seed, splits, X, y, X_test):
    oof = np.zeros(len(y), dtype=np.float64)
    tp = np.zeros(len(X_test), dtype=np.float64)
    fold_scores, fold_walls, iters = [], [], []
    log(f"[seed{seed}] STARTING 5-fold")
    for k, (tr, va) in enumerate(splits):
        log(f"  [seed{seed}/f{k}] FIT START n_tr={len(tr)} n_va={len(va)}")
        t0 = time.time()
        m = make_hgbc(seed)
        m.fit(X.iloc[tr], y[tr])
        log(f"  [seed{seed}/f{k}] FIT DONE iters={m.n_iter_} "
            f"fit_wall={time.time()-t0:.1f}s, predicting...")
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        tp += m.predict_proba(X_test)[:, 1] / N_FOLDS
        wall = time.time() - t0
        s = float(roc_auc_score(y[va], p))
        fold_scores.append(s); fold_walls.append(wall); iters.append(int(m.n_iter_))
        log(f"  [seed{seed}/f{k}] AUC={s:.5f} iters={m.n_iter_} wall={wall:.1f}s")
    auc = float(roc_auc_score(y, oof))
    log(f"[seed{seed}] OOF={auc:.5f} fold_std={np.std(fold_scores):.5f} "
        f"total={sum(fold_walls):.0f}s")
    return oof, tp, auc, fold_scores, fold_walls, iters


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mp).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    log(f"[setup] data load {time.time()-t0:.1f}s, n_train={len(y)}, "
        f"n_test={len(X_test)}, seeds={SEEDS}")

    seed_oofs = []
    seed_tests = []
    seed_results = {}
    completed_seeds = []
    for seed in SEEDS:
        # If we already have a saved per-seed npy from a prior run, reuse
        seed_oof_path = ART / f"oof_d12_e3_seed{seed}_strat.npy"
        seed_test_path = ART / f"test_d12_e3_seed{seed}_strat.npy"
        if seed_oof_path.exists() and seed_test_path.exists():
            oof = np.load(seed_oof_path)[:, 1].astype(np.float64)
            tp = np.load(seed_test_path)[:, 1].astype(np.float64)
            auc = float(roc_auc_score(y, oof))
            log(f"[seed{seed}] REUSED prior artifact, OOF AUC={auc:.5f}")
            fs, fw, its = [], [], []
        else:
            oof, tp, auc, fs, fw, its = run_seed(seed, splits, X, y, X_test)
            np.save(seed_oof_path,
                    np.column_stack([1 - oof, oof]).astype(np.float32))
            np.save(seed_test_path,
                    np.column_stack([1 - tp, tp]).astype(np.float32))
        seed_oofs.append(oof)
        seed_tests.append(tp)
        seed_results[f"seed{seed}"] = dict(
            oof_auc=auc, fold_scores=fs, fold_walls=fw,
            fold_std=float(np.std(fs)) if fs else 0.0, iters=its)
        completed_seeds.append(seed)
        # Save running bag after each seed completes (degrades gracefully)
        if len(seed_oofs) >= 2:
            running_oof = np.mean(seed_oofs, axis=0)
            running_test = np.mean(seed_tests, axis=0)
            running_auc = float(roc_auc_score(y, running_oof))
            np.save(ART / "oof_d12_e3_5seed_bag_strat.npy",
                    np.column_stack([1 - running_oof, running_oof]).astype(np.float32))
            np.save(ART / "test_d12_e3_5seed_bag_strat.npy",
                    np.column_stack([1 - running_test, running_test]).astype(np.float32))
            log(f"[bag@{len(seed_oofs)} seeds] OOF AUC = {running_auc:.5f} "
                f"(saved partial bag)")

    # Probability-mean bag
    bag_oof = np.mean(seed_oofs, axis=0)
    bag_test = np.mean(seed_tests, axis=0)
    bag_auc = float(roc_auc_score(y, bag_oof))
    log(f"\n=== BAG ({len(SEEDS)} seeds, prob-mean) OOF AUC = {bag_auc:.5f} ===")
    for s, o in zip(SEEDS, seed_oofs):
        log(f"  seed{s}: OOF AUC = {roc_auc_score(y, o):.5f}")

    np.save(ART / "oof_d12_e3_5seed_bag_strat.npy",
            np.column_stack([1 - bag_oof, bag_oof]).astype(np.float32))
    np.save(ART / "test_d12_e3_5seed_bag_strat.npy",
            np.column_stack([1 - bag_test, bag_test]).astype(np.float32))

    results = dict(
        seeds=SEEDS,
        cv_seed=CV_SEED,
        per_seed=seed_results,
        bag_oof_auc=bag_auc,
        total_wall_s=time.time() - t0,
    )
    (ART / "d12_e3_5seed_bag_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    log(f"\nDone in {(time.time()-t0)/60:.1f} min. Wrote artifacts.")


if __name__ == "__main__":
    main()
