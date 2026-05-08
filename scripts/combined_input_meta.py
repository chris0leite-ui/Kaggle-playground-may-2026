"""Combined-input meta-stacker — K=4 base predictions + top-5 raw numerics.

Friction from the SVM/kNN/NCA arc:
`non-parametric-meta-on-K=4-cant-beat-LR-meta-without-new-input`. The
K=4 ensemble's logit effective rank is ~3, so any router over the K=4
predictions alone can at best tie the LR-meta. The hypothesis this
script tests: feed the meta-stacker a *combined* input — K=4 base
predictions PLUS the top-5 raw numeric features — and see whether the
extra raw-feature signal lets the meta route by tire-life / lap-number
/ stint regions where the K=4 bases disagree.

Two meta variants on the same combined input:
  v1  Plain LR meta on combined features.
  v2  Kernel-SVM meta (Nyström-RBF γ=0.02 + LinearSVC, the best
      configuration found on K=4-only input).

Reference baselines (from prior runs on this branch, identical folds):
  K=4 plain LR-meta:                 0.95399
  K=4 + Path-B C×S τ=100k (PRIMARY): 0.95403
  K=4 kernel-SVM-meta linsvc γ=0.02: 0.95403

If the combined meta beats 0.95399, that's a new candidate base in
itself: standalone-AUC > all K=4-only routers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import N_FOLDS, SEED, folds, save_oof

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

K4_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]
TOP5_NUMERIC = ["TyreLife", "LapNumber", "Stint", "RaceProgress",
                "Cumulative_Degradation"]


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def build_combined_features(train: pd.DataFrame, test: pd.DataFrame
                            ) -> tuple[np.ndarray, np.ndarray]:
    """K=4 base preds expanded raw+rank+logit (12 cols) + top-5
    standardised numerics (5 cols) = 17 features."""
    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4_FWD]
    base_tests = [_pos(ART / f"test_{n}_strat.npy") for n in K4_FWD]
    P_oof = np.column_stack(base_oofs)
    P_test = np.column_stack(base_tests)
    F_pred_oof = expand(P_oof)
    F_pred_test = expand(P_test)

    # Raw numerics: standardize on train-only stats
    X_num_train = train[TOP5_NUMERIC].values.astype(np.float64)
    X_num_test = test[TOP5_NUMERIC].values.astype(np.float64)
    sc = StandardScaler()
    X_num_train_s = sc.fit_transform(X_num_train)
    X_num_test_s = sc.transform(X_num_test)

    F_oof = np.hstack([F_pred_oof, X_num_train_s])
    F_test = np.hstack([F_pred_test, X_num_test_s])
    return F_oof, F_test


def fit_lr_meta(F_oof, F_test, y, splits):
    oof = np.zeros(len(y))
    test = np.zeros(F_test.shape[0])
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        oof[va] = lr.predict_proba(F_oof[va])[:, 1]
        test += lr.predict_proba(F_test)[:, 1] / len(splits)
    return oof, test


def fit_kernel_meta(F_oof, F_test, y, splits, *, gamma=0.02,
                    n_components=600):
    oof = np.zeros(len(y))
    test = np.zeros(F_test.shape[0])
    for k, (tr, va) in enumerate(splits):
        sc = StandardScaler()
        F_tr = sc.fit_transform(F_oof[tr]).astype(np.float32)
        F_va = sc.transform(F_oof[va]).astype(np.float32)
        F_te = sc.transform(F_test).astype(np.float32)
        ns = Nystroem(kernel="rbf", gamma=gamma, n_components=n_components,
                      random_state=SEED + k, n_jobs=1)
        Z_tr = ns.fit_transform(F_tr).astype(np.float32)
        Z_va = ns.transform(F_va).astype(np.float32)
        Z_te = ns.transform(F_te).astype(np.float32)
        clf = LinearSVC(
            C=1.0, loss="squared_hinge", penalty="l2", dual=False,
            class_weight="balanced", max_iter=2000, tol=1e-4,
            random_state=SEED + k,
        )
        clf.fit(Z_tr, y[tr])
        oof[va] = expit(clf.decision_function(Z_va))
        test += expit(clf.decision_function(Z_te)) / len(splits)
    return oof, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="lr,kernel")
    ap.add_argument("--gamma", type=float, default=0.02)
    args = ap.parse_args()

    print("Loading data + K=4 base artifacts ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    F_oof, F_test = build_combined_features(train, test)
    print(f"  combined features: F_oof {F_oof.shape}  F_test {F_test.shape}")
    print(f"    (12 expanded K=4 preds + 5 standardised top-5 numerics)")

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Reference: K=4 LR-meta (load cached OOF for ρ comparison)
    oof_k4_lr = _pos(ART / "oof_svm_kmeta_lr_meta_k4.npy")
    test_k4_lr = _pos(ART / "test_svm_kmeta_lr_meta_k4.npy")
    oof_k4_pb = _pos(ART / "oof_K4_fwd_pathb.npy")
    test_k4_pb = _pos(ART / "test_K4_fwd_pathb.npy")
    auc_k4_lr = float(roc_auc_score(y, oof_k4_lr))
    auc_k4_pb = float(roc_auc_score(y, oof_k4_pb))
    print(f"\n  reference K=4 LR-meta OOF: {auc_k4_lr:.5f}")
    print(f"  reference K=4 Path-B PRIMARY OOF: {auc_k4_pb:.5f}")

    results = {}
    variants = args.variants.split(",")
    if "lr" in variants:
        print("\n[1] LR-meta on combined input ...")
        t0 = time.time()
        oof, test_p = fit_lr_meta(F_oof, F_test, y, splits)
        auc = float(roc_auc_score(y, oof))
        rho_pb = float(spearmanr(test_p, test_k4_pb)[0])
        print(f"  OOF: {auc:.5f}  vs K=4-LR Δ {(auc-auc_k4_lr)*1e4:+.2f}bp"
              f"  vs K=4-PathB Δ {(auc-auc_k4_pb)*1e4:+.2f}bp"
              f"  ρ_test {rho_pb:.4f}  ({time.time()-t0:.1f}s)")
        save_oof("combined_lr_meta_K4_top5",
                 np.column_stack([1-oof, oof]),
                 np.column_stack([1-test_p, test_p]),
                 dict(variant="combined_lr_meta_K4_top5", oof_score=auc,
                      bases=K4_FWD, raw_features=TOP5_NUMERIC))
        results["combined_lr"] = (auc, oof, test_p)

    if "kernel" in variants:
        print(f"\n[2] Kernel-SVM-meta (γ={args.gamma}) on combined input ...")
        t0 = time.time()
        oof, test_p = fit_kernel_meta(F_oof, F_test, y, splits,
                                      gamma=args.gamma)
        auc = float(roc_auc_score(y, oof))
        rho_pb = float(spearmanr(test_p, test_k4_pb)[0])
        print(f"  OOF: {auc:.5f}  vs K=4-LR Δ {(auc-auc_k4_lr)*1e4:+.2f}bp"
              f"  vs K=4-PathB Δ {(auc-auc_k4_pb)*1e4:+.2f}bp"
              f"  ρ_test {rho_pb:.4f}  ({time.time()-t0:.1f}s)")
        save_oof(f"combined_kernel_meta_K4_top5_g{args.gamma:g}",
                 np.column_stack([1-oof, oof]),
                 np.column_stack([1-test_p, test_p]),
                 dict(variant=f"combined_kernel_meta_g{args.gamma:g}",
                      oof_score=auc, bases=K4_FWD,
                      raw_features=TOP5_NUMERIC, gamma=args.gamma))
        results["combined_kernel"] = (auc, oof, test_p)


if __name__ == "__main__":
    main()
