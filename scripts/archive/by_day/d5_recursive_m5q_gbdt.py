"""D5 Path C — Recursive GBDT with M5q_oof_proba as a feature (HANDOVER NH11).

Hypothesis: M5q's LR meta over 14 bases blends rank-correlated columns
linearly. A new GBDT base that sees M5q_oof_proba alongside the raw
features can split on the consensus prediction itself and learn
row-level corrections (e.g., "when M5q sits in [0.3, 0.7] AND
Compound=SOFT, push up"). Cross-row interactions through the rank
structure of M5q_oof_proba become visible to a tree-based meta in
ways no LR meta can express.

Leakage: M5q OOF was generated under
StratifiedKFold(shuffle=True, random_state=42, n_splits=5). We pin
the SAME split here so per-row m5q_oof_proba was produced by an
M5q model that never trained on that row.

Reports
  (a) recursive standalone OOF (alone_score)
  (b) LR-meta of [M5q, recursive] OOF (stack_score)
  (c) Δ vs M5q anchor (0.95057)
  (d) test-corr ρ of recursive vs M5q (diversity check)

Outputs
  scripts/artifacts/oof_d5_recursive_m5q_strat.npy
  scripts/artifacts/test_d5_recursive_m5q_strat.npy
  scripts/artifacts/d5_recursive_m5q_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import ART, N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
M5Q_S = 0.95057  # M5q Strat OOF anchor (CLAUDE.md ladder)
BASE_S = 0.94075


def make_hgbc():
    return HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
        categorical_features="from_dtype",
    )


def expand(P):
    """expand(P) -> [raw, rank/n, logit]; pattern from m5qrs."""
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    eps = 1e-9
    Pc = np.clip(P, eps, 1 - eps)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rank, logit])


def lr_meta_oof(F_oof, y, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    return meta_oof


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")

    # Same prep as e3_hgbc_two_anchor (HGBC-compatible categoricals)
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

    # Load M5q OOF/test (column 1 = positive class)
    m5q_oof = np.load(ART / "oof_m5q_strat.npy")[:, 1].astype(np.float64)
    m5q_test = np.load(ART / "test_m5q_strat.npy")[:, 1].astype(np.float64)
    print(f"M5q OOF AUC sanity: {roc_auc_score(y, m5q_oof):.5f} "
          f"(should be ~{M5Q_S:.5f})")

    # Inject M5q proba as a feature. HGBC handles float well; trees can split
    # on raw proba directly. We do NOT add logit/rank views — single column,
    # cleanest first probe.
    X = X.copy()
    X_test = X_test.copy()
    X["m5q_proba"] = m5q_oof.astype(np.float32)
    X_test["m5q_proba"] = m5q_test.astype(np.float32)

    # Pin SAME split as M5q (StratifiedKFold(shuffle=True, random_state=42))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float32)
    tp = np.zeros(len(test), dtype=np.float32)
    fold_aucs, walls = [], []
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        m = make_hgbc()
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        tp += m.predict_proba(X_test)[:, 1] / N_FOLDS
        wall = time.time() - t_fold
        s = float(roc_auc_score(y[va], p))
        fold_aucs.append(s); walls.append(wall)
        print(f"  fold {k}: AUC={s:.5f}  iters={m.n_iter_}  wall={wall:.1f}s")

    alone_score = float(roc_auc_score(y, oof))
    delta_baseline_bp = (alone_score - BASE_S) * 1e4
    print(f"\nRecursive standalone OOF: {alone_score:.5f}  "
          f"Δ baseline={delta_baseline_bp:+.1f}bp  "
          f"std={np.std(fold_aucs):.5f}")

    # Diversity: test corr (Spearman == Pearson on rank/n)
    rho_test = float(np.corrcoef(
        rankdata(tp.astype(np.float64)),
        rankdata(m5q_test),
    )[0, 1])
    print(f"ρ(recursive_test, m5q_test) = {rho_test:.5f}")

    # 2-base LR stack: [M5q, recursive]
    P_oof = np.column_stack([m5q_oof, oof.astype(np.float64)])
    F_oof = expand(P_oof)
    meta_oof = lr_meta_oof(F_oof, y)
    stack_score = float(roc_auc_score(y, meta_oof))
    delta_m5q_bp = (stack_score - M5Q_S) * 1e4
    print(f"\n[M5q, recursive] LR stack OOF: {stack_score:.5f}  "
          f"Δ M5q anchor={delta_m5q_bp:+.1f}bp")

    # Save artifacts
    save_oof(
        "d5_recursive_m5q_strat",
        np.column_stack([1 - oof, oof]),
        np.column_stack([1 - tp, tp]),
        dict(
            alone_oof_score=alone_score,
            stack_oof_score=stack_score,
            delta_vs_baseline_bp=delta_baseline_bp,
            delta_vs_m5q_bp=delta_m5q_bp,
            rho_vs_m5q_test=rho_test,
            fold_aucs=fold_aucs,
            fold_walls_s=walls,
            cv="StratifiedKFold(5, shuffle=True, random_state=42)",
            metric="roc_auc",
            n_folds=N_FOLDS, seed=SEED,
            features_added=["m5q_proba"],
            notes=("Recursive base: HGBC + raw features + M5q_oof_proba. "
                   "Same fold split as M5q for leakage-clean OOF. "
                   "Stack = LR-meta on expand([M5q, recursive])."),
            total_wall_s=time.time() - t0,
        ),
    )
    print(f"\nartifacts saved → {ART}/oof_d5_recursive_m5q_strat.npy")
    print(f"total wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
