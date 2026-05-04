"""E3 — HistGradientBoostingClassifier two-anchor 5-fold.

Adds a 4th GBDT family (sklearn HGBC) to the base pool. Same data,
same anchors, native categorical via `categorical_features="from_dtype"`.
Mirrors structure of `scripts/baseline_two_anchor.py`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
BASE_S, BASE_G = 0.94075, 0.92059


def make_hgbc():
    return HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
        categorical_features="from_dtype",
    )


def run_anchor(name, splits, X, y, X_test):
    oof = np.zeros(len(y), dtype=np.float32)
    tp = np.zeros(len(X_test), dtype=np.float32)
    fs, walls = [], []
    m0 = None
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        m = make_hgbc()
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        tp += m.predict_proba(X_test)[:, 1] / N_FOLDS
        wall = time.time() - t0
        s = float(roc_auc_score(y[va], p))
        fs.append(s); walls.append(wall)
        if k == 0:
            m0 = m
        print(f"  [{name}] f{k}: AUC={s:.5f} iters={m.n_iter_} wall={wall:.1f}s")
    return oof, tp, float(roc_auc_score(y, oof)), fs, float(np.std(fs)), walls, m0


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    # HGBC caps native categorical cardinality at 255. Driver=874 -> label-encode to int
    # numeric column. Compound (5) and Race (26) stay as pandas category for native cat-handling.
    HIGH_CARD = ["Driver"]
    LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True).astype(str).unique()
            mapping = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mapping).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mapping).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    print("=== Anchor A: StratifiedKFold(5, seed=42) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y)), y))
    oof_a, test_a, auc_a, fs_a, sd_a, w_a, _ = run_anchor("STRAT", splits_a, X, y, X_test)

    print("=== Anchor B: GroupKFold(5) on Race ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, train["Race"].values))
    oof_b, test_b, auc_b, fs_b, sd_b, w_b, _ = run_anchor("GROUP", splits_b, X, y, X_test)

    da = (auc_a - BASE_S) * 1e4
    db = (auc_b - BASE_G) * 1e4
    total = time.time() - t0
    print(f"\nStrat: {auc_a:.5f}  Δ={da:+.1f}bp  std={sd_a:.5f}")
    print(f"GroupKF: {auc_b:.5f}  Δ={db:+.1f}bp  std={sd_b:.5f}")
    print(f"total wall: {total:.0f}s")

    save_oof("e3_hgbc_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=sd_a, fold_scores=fs_a,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=da, fold_walls_s=w_a))
    save_oof("e3_hgbc_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=sd_b, fold_scores=fs_b,
                  cv="GroupKFold(Race)", metric="roc_auc",
                  delta_vs_baseline_bp=db, fold_walls_s=w_b))

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_e3_hgbc.csv", index=False)

    g1s = "PASS" if auc_a >= BASE_S - 5e-4 else ("SOFT" if auc_a >= BASE_S - 1e-3 else "FAIL")
    g1g = "PASS" if auc_b >= BASE_G - 5e-4 else ("SOFT" if auc_b >= BASE_G - 1e-3 else "FAIL")

    body = (
        f"# E3 — HistGradientBoostingClassifier two-anchor (2026-05-04)\n\n"
        f"4th GBDT family (sklearn HGBC) for stack diversity.\n\n"
        f"## Two-anchor results\n\n"
        f"| anchor | OOF AUC | fold_std | per-fold | Δ baseline (bp) | G1 |\n"
        f"|---|---:|---:|---|---:|---|\n"
        f"| Strat | **{auc_a:.5f}** | {sd_a:.5f} | "
        f"{[f'{x:.4f}' for x in fs_a]} | {da:+.1f} | {g1s} |\n"
        f"| GroupKF | **{auc_b:.5f}** | {sd_b:.5f} | "
        f"{[f'{x:.4f}' for x in fs_b]} | {db:+.1f} | {g1g} |\n\n"
        f"## Wall times\n\n"
        f"- Strat 5-fold: {sum(w_a):.0f}s ({np.mean(w_a):.1f}s/fold)\n"
        f"- GroupKF 5-fold: {sum(w_b):.0f}s ({np.mean(w_b):.1f}s/fold)\n"
        f"- Total: {total:.0f}s\n"
    )
    Path("audit/2026-05-04-e3-hgbc.md").write_text(body)
    print(f"audit written")


if __name__ == "__main__":
    main()
