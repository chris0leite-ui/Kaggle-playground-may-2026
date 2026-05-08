"""SVM specialists — fit one SVM per segment level, pool decision scores.

Five cheap variants on the same 45-feature vanilla-LR recipe used by
`svm_kernel_probe.py`:

  1. linear_global       LinearSVC (no kernel) on all rows of fold-train
  2. linear_per_year     LinearSVC per Year (~5-6 levels)
  3. linear_per_compound LinearSVC per Compound (5 levels)
  4. linear_per_stint    LinearSVC per clipped Stint (1..5+)
  5. rbf_per_year        Nyström-RBF (gamma=0.02, n_components=600)
                         + LinearSVC, one per Year

For specialists, predictions for a row come ONLY from the model fit on
that row's segment level (training rows of the same level, same fold).
Decision scores → sigmoid → [0,1] for OOF/test convention.

Output (per variant):
  scripts/artifacts/oof_<NAME>_strat.npy   (n_train, 2)
  scripts/artifacts/test_<NAME>_strat.npy  (n_test, 2)
  scripts/artifacts/<NAME>_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import N_FOLDS, SEED, folds, save_oof
from svm_kernel_probe import build_features

TARGET = "PitNextLap"


def _fit_linear(X_tr, y_tr, X_va, X_te, *, seed: int):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    X_va_s = sc.transform(X_va).astype(np.float32) if len(X_va) else X_va
    X_te_s = sc.transform(X_te).astype(np.float32)
    clf = LinearSVC(
        C=1.0, loss="squared_hinge", penalty="l2", dual=False,
        class_weight="balanced", max_iter=2000, tol=1e-4,
        random_state=seed,
    )
    clf.fit(X_tr_s, y_tr)
    p_va = expit(clf.decision_function(X_va_s)) if len(X_va_s) else np.zeros(0)
    p_te = expit(clf.decision_function(X_te_s))
    return p_va, p_te


def _fit_rbf(X_tr, y_tr, X_va, X_te, *, gamma: float, seed: int):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    X_va_s = sc.transform(X_va).astype(np.float32) if len(X_va) else X_va
    X_te_s = sc.transform(X_te).astype(np.float32)
    n_comp = min(600, max(50, len(X_tr_s) // 4))
    ns = Nystroem(kernel="rbf", gamma=gamma, n_components=n_comp,
                  random_state=seed, n_jobs=1)
    Z_tr = ns.fit_transform(X_tr_s).astype(np.float32)
    Z_va = ns.transform(X_va_s).astype(np.float32) if len(X_va_s) else X_va_s
    Z_te = ns.transform(X_te_s).astype(np.float32)
    clf = LinearSVC(
        C=1.0, loss="squared_hinge", penalty="l2", dual=False,
        class_weight="balanced", max_iter=2000, tol=1e-4,
        random_state=seed,
    )
    clf.fit(Z_tr, y_tr)
    p_va = expit(clf.decision_function(Z_va)) if len(Z_va) else np.zeros(0)
    p_te = expit(clf.decision_function(Z_te))
    return p_va, p_te


def _segment_codes(train: pd.DataFrame, test: pd.DataFrame, col: str
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Encode segments. Stint clipped to 1..5; otherwise sorted-unique."""
    if col == "Stint":
        s_tr = np.clip(train["Stint"].astype(int).values, 1, 5)
        s_te = np.clip(test["Stint"].astype(int).values, 1, 5)
        return s_tr, s_te
    levels = sorted(set(train[col].astype(str).unique()) |
                    set(test[col].astype(str).unique()))
    m = {v: i for i, v in enumerate(levels)}
    return (train[col].astype(str).map(m).astype(int).values,
            test[col].astype(str).map(m).astype(int).values)


def run_global(variant: str, X_train: np.ndarray, X_test: np.ndarray,
               y: np.ndarray, *, gamma: float):
    """One model on all rows of fold-train; standard 5-fold."""
    name = f"svm_{variant}_strat"
    print(f"\n=== {variant} (global) ===")
    n_train, n_test = len(X_train), len(X_test)
    oof = np.zeros(n_train, dtype=np.float32)
    test_proba = np.zeros(n_test, dtype=np.float32)
    fold_aucs, fold_secs = [], []

    for k, tr, va in folds(y, task="classification"):
        t0 = time.time()
        if variant == "linear_global":
            p_va, p_te = _fit_linear(
                X_train[tr], y[tr], X_train[va], X_test, seed=SEED + k,
            )
        else:
            raise ValueError(f"unknown global variant {variant}")
        secs = time.time() - t0
        fold_secs.append(secs)
        oof[va] = p_va
        test_proba += p_te.astype(np.float32) / N_FOLDS
        auc = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(auc)
        print(f"   fold {k}: AUC={auc:.5f}  ({secs:.1f}s)")

    oof_full = float(roc_auc_score(y, oof))
    print(f"   full OOF: {oof_full:.5f}  total {sum(fold_secs):.0f}s")
    save_oof(name,
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(variant=variant, oof_score=oof_full, fold_aucs=fold_aucs,
                  fold_secs=fold_secs))
    return name, oof_full, fold_secs


def run_specialist(variant: str, segment_col: str, kernel: str,
                   X_train: np.ndarray, X_test: np.ndarray,
                   y: np.ndarray, seg_tr: np.ndarray, seg_te: np.ndarray,
                   *, gamma: float):
    """One model per segment level. For each fold:
       - For each segment level s:
           fit on training rows where seg_tr==s in this fold
           predict validation rows where seg_tr==s  (this fold)
           predict test rows where seg_te==s        (averaged over folds)
    """
    name = f"svm_{variant}_strat"
    levels = sorted(set(seg_tr.tolist()) | set(seg_te.tolist()))
    print(f"\n=== {variant} (per-{segment_col}, kernel={kernel}, "
          f"{len(levels)} levels) ===")
    n_train, n_test = len(X_train), len(X_test)
    oof = np.zeros(n_train, dtype=np.float32)
    test_proba = np.zeros(n_test, dtype=np.float32)
    fold_aucs, fold_secs = [], []

    for k, tr, va in folds(y, task="classification"):
        t0 = time.time()
        for s in levels:
            tr_mask = seg_tr[tr] == s
            va_mask = seg_tr[va] == s
            te_mask = seg_te == s
            tr_idx = tr[tr_mask]
            va_idx = va[va_mask]
            te_idx = np.flatnonzero(te_mask)

            if len(tr_idx) < 200 or y[tr_idx].sum() < 10 \
                    or y[tr_idx].sum() == len(tr_idx):
                # Degenerate — fall back to global mean for this level/fold
                fallback = float(y[tr].mean())
                if len(va_idx):
                    oof[va_idx] = fallback
                if len(te_idx):
                    test_proba[te_idx] += fallback / N_FOLDS
                continue

            if kernel == "linear":
                p_va, p_te = _fit_linear(
                    X_train[tr_idx], y[tr_idx],
                    X_train[va_idx] if len(va_idx) else X_train[:0],
                    X_test[te_idx] if len(te_idx) else X_test[:0],
                    seed=SEED + k,
                )
            elif kernel == "rbf":
                p_va, p_te = _fit_rbf(
                    X_train[tr_idx], y[tr_idx],
                    X_train[va_idx] if len(va_idx) else X_train[:0],
                    X_test[te_idx] if len(te_idx) else X_test[:0],
                    gamma=gamma, seed=SEED + k,
                )
            else:
                raise ValueError(f"unknown kernel {kernel}")

            if len(va_idx):
                oof[va_idx] = p_va
            if len(te_idx):
                test_proba[te_idx] += p_te.astype(np.float32) / N_FOLDS

        secs = time.time() - t0
        fold_secs.append(secs)
        auc = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(auc)
        print(f"   fold {k}: AUC={auc:.5f}  ({secs:.1f}s)")

    oof_full = float(roc_auc_score(y, oof))
    print(f"   full OOF: {oof_full:.5f}  total {sum(fold_secs):.0f}s")
    save_oof(name,
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(variant=variant, segment_col=segment_col, kernel=kernel,
                  n_levels=len(levels),
                  oof_score=oof_full, fold_aucs=fold_aucs,
                  fold_secs=fold_secs))
    return name, oof_full, fold_secs


VARIANTS = {
    "linear_global":       ("global", None, "linear"),
    "linear_per_year":     ("specialist", "Year", "linear"),
    "linear_per_compound": ("specialist", "Compound", "linear"),
    "linear_per_stint":    ("specialist", "Stint", "linear"),
    "rbf_per_year":        ("specialist", "Year", "rbf"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="all")
    ap.add_argument("--gamma", type=float, default=0.02,
                    help="RBF gamma (specialists only)")
    args = ap.parse_args()

    print(f"loading data/train.csv + data/test.csv ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X_train, X_test, names = build_features(train, test)
    y = train[TARGET].astype(int).values
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}  "
          f"feats: {len(names)}")

    # Pre-compute segment codes for the segments we'll use.
    seg_year = _segment_codes(train, test, "Year")
    seg_compound = _segment_codes(train, test, "Compound")
    seg_stint = _segment_codes(train, test, "Stint")

    targets = list(VARIANTS) if args.variant == "all" else [args.variant]
    summary = {}
    for v in targets:
        kind, seg, kernel = VARIANTS[v]
        if kind == "global":
            name, oof_full, fs = run_global(v, X_train, X_test, y,
                                            gamma=args.gamma)
        else:
            seg_tr, seg_te = (seg_year if seg == "Year" else
                              seg_compound if seg == "Compound" else
                              seg_stint)
            name, oof_full, fs = run_specialist(
                v, seg, kernel, X_train, X_test, y, seg_tr, seg_te,
                gamma=args.gamma,
            )
        summary[v] = (name, oof_full, sum(fs))

    print(f"\n=== summary ===")
    for v, (name, oof, total_s) in summary.items():
        print(f"  {v:24s}  OOF={oof:.5f}   {total_s:.0f}s")


if __name__ == "__main__":
    main()
