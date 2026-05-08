"""SVM kernel-meta probe — kernel SVM as the meta-stacker on K=4.

The new PRIMARY (set 2026-05-08 PM on
`origin/claude/review-ml-handover-IMUNP`) is a K=4 forward-greedy
sparse pool combined with a Path-B Compound × Stint stacker
(τ=100k), LB 0.95351. Logit effective rank of the K=27 pool was
3.23 (17 bases dead weight). Path-B amp was only +0.04 bp at K=27.
The K=4 sparse pool is the cleaner reference for any new direction.

This probe replaces the linear meta-stacker with a kernel SVM:
inputs are the 4 base OOF/test predictions expanded to raw + rank
+ logit (12 columns). Question: does kernel-class meta-routing find
non-linear structure that the LR-meta and Path-B stacker miss?

Variants:
  - lr_meta_k4              Reference: plain LR meta on K=4 (12 cols).
  - pathb_k4_tau100000      Reference: Path-B C×S τ=100k on K=4 (the
                             new PRIMARY artifact, rebuilt locally).
  - svm_kmeta_linsvc        Nyström-RBF + LinearSVC on the K=4 input.
  - svm_kmeta_klogreg       Nyström-RBF + LogReg on the K=4 input
                             (calibrated kernel-logistic).
  - svm_kmeta_blend         γ-weighted blend of svm_kmeta_linsvc with
                             the Path-B PRIMARY (R5 hedge structure).

Cost: ~20 min CPU total at 350k × 12 features.
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
MIN_ROWS = 1000

# K=4 forward-greedy pool from the IMUNP branch's new PRIMARY:
# d17_h1d_yekenot_full + p1_single_cb_v4_gpu + f1_hgbc_deep
# + d16_orig_continuous_only.
K4_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P: np.ndarray) -> np.ndarray:
    """Stack raw + rank + logit. Matches probe_minimal_pool_sweep convention."""
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


# --- meta-stackers ------------------------------------------------------------
def fit_lr_meta(F_oof, F_test, y, splits):
    oof = np.zeros(len(y))
    test = np.zeros(F_test.shape[0])
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        oof[va] = lr.predict_proba(F_oof[va])[:, 1]
        test += lr.predict_proba(F_test)[:, 1] / len(splits)
    return oof, test


def fit_lr_aug(F, y):
    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_lr_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def fit_path_b(F_oof, F_test, y, splits, seg_train, seg_test, n_seg, tau):
    """Compound × Stint partial-pooling LR stacker. Mirrors
    `probe_minimal_pool_sweep.fit_path_b` for the OOF leg, plus an
    explicit full-train fit for the test leg."""
    oof = np.zeros(len(y))
    for tr_idx, va_idx in splits:
        w_global = fit_lr_aug(F_oof[tr_idx], y[tr_idx])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr_idx] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr_idx][idx])) < 2:
                continue
            W_local[s] = fit_lr_aug(F_oof[tr_idx][idx], y[tr_idx][idx])
            mask[s] = True
        n_local = counts.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local
                    + (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg_train[va_idx]):
            idx = np.where(seg_train[va_idx] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            oof[va_idx[idx]] = predict_lr_aug(F_oof[va_idx][idx], w)

    # Test leg: full-train Path-B refit
    w_global = fit_lr_aug(F_oof, y)
    W_local = np.zeros((n_seg, len(w_global)))
    counts = np.zeros(n_seg, dtype=np.int64)
    mask = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        counts[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_local[s] = fit_lr_aug(F_oof[idx], y[idx])
        mask[s] = True
    n_local = counts.astype(np.float64)
    alpha = n_local / (n_local + tau)
    W_shrunk = (alpha[:, None] * W_local
                + (1 - alpha[:, None]) * w_global[None, :])
    test = np.zeros(F_test.shape[0])
    for s in np.unique(seg_test):
        idx = np.where(seg_test == s)[0]
        w = W_shrunk[s] if mask[s] else w_global
        test[idx] = predict_lr_aug(F_test[idx], w)
    return oof, test


def fit_kernel_svm(F_oof, F_test, y, splits, *, gamma, n_components,
                   classifier="linsvc", seed=SEED):
    """Nyström-RBF + linear / logistic classifier as meta-stacker."""
    oof = np.zeros(len(y))
    test = np.zeros(F_test.shape[0])
    for k, (tr, va) in enumerate(splits):
        sc = StandardScaler()
        F_tr = sc.fit_transform(F_oof[tr]).astype(np.float32)
        F_va = sc.transform(F_oof[va]).astype(np.float32)
        F_te = sc.transform(F_test).astype(np.float32)

        ns = Nystroem(kernel="rbf", gamma=gamma, n_components=n_components,
                      random_state=seed + k, n_jobs=1)
        Z_tr = ns.fit_transform(F_tr).astype(np.float32)
        Z_va = ns.transform(F_va).astype(np.float32)
        Z_te = ns.transform(F_te).astype(np.float32)

        if classifier == "linsvc":
            clf = LinearSVC(
                C=1.0, loss="squared_hinge", penalty="l2", dual=False,
                class_weight="balanced", max_iter=2000, tol=1e-4,
                random_state=seed + k,
            )
            clf.fit(Z_tr, y[tr])
            oof[va] = expit(clf.decision_function(Z_va))
            test += expit(clf.decision_function(Z_te)) / len(splits)
        elif classifier == "klogreg":
            clf = LogisticRegression(
                C=1.0, solver="lbfgs", class_weight="balanced", max_iter=500,
            )
            clf.fit(Z_tr, y[tr])
            oof[va] = clf.predict_proba(Z_va)[:, 1]
            test += clf.predict_proba(Z_te)[:, 1] / len(splits)
        else:
            raise ValueError(f"unknown classifier {classifier}")
    return oof, test


# --- driver -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-sweep", action="store_true",
                    help="Run a γ-sweep at smoke (1-fold @ 50k) before full.")
    ap.add_argument("--gammas", type=str, default="0.02,0.05,0.1",
                    help="Comma-separated γ values for the kernel-SVM-meta.")
    ap.add_argument("--n-components", type=int, default=600,
                    help="Nyström landmark count. Must keep peak RAM "
                         "under 15 GB on full 350k-row training fold "
                         "(LinearSVC promotes to float64 internally).")
    ap.add_argument("--classifiers", type=str, default="linsvc,klogreg",
                    help="Comma-separated classifier kinds.")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="Skip rebuilding LR-meta and Path-B (use existing).")
    args = ap.parse_args()
    gammas = [float(g) for g in args.gammas.split(",")]
    classifiers = args.classifiers.split(",")

    print("Loading data + K=4 base artifacts ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4_FWD]
    base_tests = [_pos(ART / f"test_{n}_strat.npy") for n in K4_FWD]
    P_oof = np.column_stack(base_oofs)
    P_test = np.column_stack(base_tests)
    print(f"  K=4 bases: {K4_FWD}")
    print(f"  P_oof {P_oof.shape}  P_test {P_test.shape}")

    F_oof = _expand(P_oof)
    F_test = _expand(P_test)
    print(f"  Feature matrix (raw+rank+logit): F_oof {F_oof.shape}")

    # Compound × Stint segments for Path-B
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp_map = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp_map).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp_map).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_train = c_tr * 6 + s_tr
    seg_test = c_te * 6 + s_te

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    results = {}

    if args.skip_baselines and (ART / "oof_svm_kmeta_lr_meta_k4.npy").exists() \
            and (ART / "oof_K4_fwd_pathb.npy").exists():
        print("\n[1+2] Reloading cached LR-meta and K=4 Path-B baselines ...")
        oof_lr = _pos(ART / "oof_svm_kmeta_lr_meta_k4.npy")
        test_lr = _pos(ART / "test_svm_kmeta_lr_meta_k4.npy")
        oof_pb = _pos(ART / "oof_K4_fwd_pathb.npy")
        test_pb = _pos(ART / "test_K4_fwd_pathb.npy")
        auc_lr = float(roc_auc_score(y, oof_lr))
        auc_pb = float(roc_auc_score(y, oof_pb))
        print(f"   LR-meta OOF: {auc_lr:.5f}   Path-B PRIMARY OOF: {auc_pb:.5f}")
    else:
        # 1. Plain LR-meta on K=4
        print("\n[1] Plain LR-meta on K=4 ...")
        t0 = time.time()
        oof_lr, test_lr = fit_lr_meta(F_oof, F_test, y, splits)
        auc_lr = float(roc_auc_score(y, oof_lr))
        print(f"   OOF AUC: {auc_lr:.5f}  ({time.time()-t0:.1f}s)")
        save_oof("svm_kmeta_lr_meta_k4",
                 np.column_stack([1 - oof_lr, oof_lr]),
                 np.column_stack([1 - test_lr, test_lr]),
                 dict(variant="lr_meta_k4", oof_score=auc_lr))

        # 2. Path-B Compound × Stint τ=100k on K=4 (the K=4 PRIMARY artefact)
        print("\n[2] Path-B C×S τ=100k on K=4 (rebuilds K=4 PRIMARY) ...")
        t0 = time.time()
        oof_pb, test_pb = fit_path_b(F_oof, F_test, y, splits,
                                     seg_train, seg_test, n_seg, 100000.0)
        auc_pb = float(roc_auc_score(y, oof_pb))
        print(f"   OOF AUC: {auc_pb:.5f}  ({time.time()-t0:.1f}s)")
        save_oof("K4_fwd_pathb",
                 np.column_stack([1 - oof_pb, oof_pb]),
                 np.column_stack([1 - test_pb, test_pb]),
                 dict(variant="K4_fwd_pathb_tau100k", oof_score=auc_pb))
    results["lr_meta_k4"] = (auc_lr, oof_lr, test_lr)
    results["pathb_k4"] = (auc_pb, oof_pb, test_pb)
    primary_oof, primary_test = oof_pb, test_pb

    # 3. γ-sweep at smoke (optional)
    if args.gamma_sweep:
        print("\n[3] γ-sweep at smoke (1-fold @ 50k) ...")
        rng = np.random.default_rng(SEED)
        sub_idx = rng.choice(splits[0][0], size=50_000, replace=False)
        va_idx = splits[0][1]
        for g in gammas:
            t0 = time.time()
            sc = StandardScaler()
            Ftr_s = sc.fit_transform(F_oof[sub_idx]).astype(np.float32)
            Fva_s = sc.transform(F_oof[va_idx]).astype(np.float32)
            ns = Nystroem(kernel="rbf", gamma=g, n_components=args.n_components,
                          random_state=SEED, n_jobs=1)
            Z_tr = ns.fit_transform(Ftr_s).astype(np.float32)
            Z_va = ns.transform(Fva_s).astype(np.float32)
            clf = LinearSVC(
                C=1.0, loss="squared_hinge", dual=False, class_weight="balanced",
                max_iter=2000, tol=1e-4, random_state=SEED,
            )
            clf.fit(Z_tr, y[sub_idx])
            p_va = expit(clf.decision_function(Z_va))
            auc = float(roc_auc_score(y[va_idx], p_va))
            print(f"   γ={g:>6g}: AUC={auc:.5f}  ({time.time()-t0:.1f}s)")

    # 4. Kernel-SVM-meta variants — full 5-fold at user-supplied γs
    for g in gammas:
        for kind in classifiers:
            print(f"\n[4] Kernel-SVM-meta ({kind}) γ={g} ...")
            t0 = time.time()
            oof_k, test_k = fit_kernel_svm(
                F_oof, F_test, y, splits,
                gamma=g, n_components=args.n_components, classifier=kind,
            )
            auc_k = float(roc_auc_score(y, oof_k))
            secs = time.time() - t0
            name = f"svm_kmeta_{kind}_g{g:g}"
            print(f"   OOF AUC: {auc_k:.5f}  vs LR {auc_lr:.5f} "
                  f"Δ {(auc_k - auc_lr) * 1e4:+.2f}bp"
                  f"  vs PathB-PRIMARY Δ {(auc_k - auc_pb) * 1e4:+.2f}bp"
                  f"  ({secs:.1f}s)")
            save_oof(name,
                     np.column_stack([1 - oof_k, oof_k]),
                     np.column_stack([1 - test_k, test_k]),
                     dict(variant=name, gamma=g, n_components=args.n_components,
                          oof_score=auc_k, secs=secs))
            results[name] = (auc_k, oof_k, test_k)

            # ρ_test vs PRIMARY + flip diagnostic
            rho, _ = spearmanr(test_k, primary_test)
            rare_thr = float(np.quantile(primary_test, 0.99))
            n_to_neg = int(((primary_test >= rare_thr) & (test_k < rare_thr)).sum())
            n_to_pos = int(((primary_test < rare_thr) & (test_k >= rare_thr)).sum())
            print(f"   ρ_test vs PRIMARY: {rho:.4f}  "
                  f"flips +→− {n_to_neg}, −→+ {n_to_pos}")

    # 5. Summary
    print("\n=== Summary (OOF AUCs vs new PRIMARY 0.95351) ===")
    for name, (auc, _, _) in sorted(results.items(),
                                    key=lambda kv: -kv[1][0]):
        delta = (auc - auc_pb) * 1e4
        print(f"  {name:<32s} {auc:.5f}  Δ vs PathB-K4 {delta:+.2f}bp")


if __name__ == "__main__":
    main()
