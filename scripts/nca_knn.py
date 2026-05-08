"""kNN with a learned manifold distance — Neighbourhood Components Analysis.

NCA learns a linear projection W : R^d -> R^k that maximises leave-one-
out kNN classification accuracy on the training set. After projection,
plain Euclidean distance in R^k IS the learned manifold metric.

Variants (PI prompt: "use our K=4, K=10 or K=27 ensemble for creating
the manifold"):

  K4   NCA on the K=4 forward-greedy ensemble's base predictions
       (4 cols → raw + rank + logit = 12 features → 3 NCA dims → kNN K=50).
  K10  NCA on the K=10 forward-greedy ensemble (10 cols → 30 features →
       3 NCA dims → kNN K=50).
  A    Raw-features apples-to-apples vs plain kNN s2 (top-5 numeric).
  B    Full 14-column raw-feature pool with learned 3-D metric.

Why ensemble-input is the better baseline: the base predictions are
already a "compressed view" of the data (4-10 numbers per row instead
of 14), so the input dimensionality is small AND the inputs are
already calibrated probabilities. NCA's pairwise-distance loss matrix
is O(n²) regardless of input dim, so we still subsample to ≤20k for
NCA fit; then apply the learned projection to all 350k rows for kNN.

Cost per fold:
  - NCA fit on 20k subsample: ~1-3 min on K=4, ~2-5 min on K=10.
  - NCA transform full data: ~1 sec.
  - kNN classify on 88k val + 188k test in 3-D: ~30 s.
  Total: ~2-6 min/fold → ~10-30 min for full 5-fold per variant.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler

from common import N_FOLDS, SEED, folds, save_oof
from knn_feature_subsets import build_feature_pool

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

# Raw-feature subsets (original A/B variants).
NCA_A_COLS = ["TyreLife", "LapNumber", "Stint", "RaceProgress",
              "Cumulative_Degradation"]
NCA_B_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
    "Compound_LE", "Race_freq", "Driver_freq",
]

# Ensemble base-prediction pools (forward-greedy pick orders from the
# IMUNP branch's `probe_minimal_pool_sweep.py`).
K4_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]
K10_FWD = K4_FWD + [
    "b_lapsuntilpit",
    "baseline_two_anchor",
    "d9_R6_next_compound",
    "cb_year-cat",
    "e5_optuna_lgbm",
    "d9f_FM_A",
]


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P: np.ndarray) -> np.ndarray:
    """Stack raw + rank + logit for the input bases. Matches the
    LR-meta convention used in `scripts/probe_min_meta.py` etc."""
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def load_ensemble_features(bases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load OOF + test predictions for a list of base names, expand
    to raw + rank + logit. Returns F_train (n_train, 3*K), F_test."""
    base_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in bases]
    base_tests = [_pos(ART / f"test_{b}_strat.npy") for b in bases]
    P_oof = np.column_stack(base_oofs)
    P_test = np.column_stack(base_tests)
    return _expand(P_oof), _expand(P_test)


def run_nca_variant(name: str, cols: list[str], pool: pd.DataFrame, n_tr: int,
                    y: np.ndarray, *,
                    n_components: int = 3,
                    nca_subsample: int = 50_000,
                    knn_k: int = 50,
                    max_iter: int = 50):
    """Per fold: subsample for NCA fit, apply learned projection to full
    data, fit kNN in projected space, predict val + test."""
    print(f"\n=== {name} ({len(cols)} input → {n_components} NCA dims) ===")
    print(f"  cols: {cols}")
    X_full = pool[cols].values.astype(np.float64)
    X_train_raw = X_full[:n_tr]
    X_test_raw = X_full[n_tr:]

    oof = np.zeros(n_tr, dtype=np.float32)
    test_proba = np.zeros(len(X_test_raw), dtype=np.float32)
    fold_aucs, fold_secs = [], []

    rng = np.random.default_rng(SEED)
    for kf, tr, va in folds(y, task="classification"):
        t0 = time.time()
        # 1. Standardise (fit on this fold's training rows)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train_raw[tr])
        X_va_s = sc.transform(X_train_raw[va])
        X_te_s = sc.transform(X_test_raw)

        # 2. Stratified subsample for NCA fit
        if len(tr) > nca_subsample:
            pos_idx = np.where(y[tr] == 1)[0]
            neg_idx = np.where(y[tr] == 0)[0]
            n_pos = int(nca_subsample * (len(pos_idx) / len(tr)))
            n_neg = nca_subsample - n_pos
            sub_pos = rng.choice(pos_idx, size=n_pos, replace=False)
            sub_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            sub = np.concatenate([sub_pos, sub_neg])
            rng.shuffle(sub)
        else:
            sub = np.arange(len(tr))

        # 3. Fit NCA
        nca = NeighborhoodComponentsAnalysis(
            n_components=n_components,
            init="pca", max_iter=max_iter, tol=1e-5,
            random_state=SEED + kf, verbose=0,
        )
        t_nca = time.time()
        nca.fit(X_tr_s[sub].astype(np.float64), y[tr][sub])
        nca_secs = time.time() - t_nca

        # 4. Apply projection to full train / val / test
        Z_tr = nca.transform(X_tr_s).astype(np.float32)
        Z_va = nca.transform(X_va_s).astype(np.float32)
        Z_te = nca.transform(X_te_s).astype(np.float32)

        # 5. kNN in learned space
        clf = KNeighborsClassifier(
            n_neighbors=knn_k, weights="distance", algorithm="auto",
            leaf_size=40, n_jobs=-1,
        )
        clf.fit(Z_tr, y[tr])
        p_va = clf.predict_proba(Z_va)[:, 1]
        p_te = clf.predict_proba(Z_te)[:, 1]

        oof[va] = p_va
        test_proba += p_te.astype(np.float32) / N_FOLDS

        secs = time.time() - t0
        auc = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(auc)
        fold_secs.append(secs)
        print(f"  fold {kf}: AUC={auc:.5f}  ({secs:.1f}s, NCA fit "
              f"{nca_secs:.1f}s on {len(sub):,} rows)")

    oof_full = float(roc_auc_score(y, oof))
    print(f"  full OOF: {oof_full:.5f}  total {sum(fold_secs):.0f}s")
    save_oof(name,
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(variant=name, cols=cols, n_components=n_components,
                  nca_subsample=nca_subsample, knn_k=knn_k,
                  max_iter=max_iter, oof_score=oof_full,
                  fold_aucs=fold_aucs, fold_secs=fold_secs))
    return oof, test_proba, oof_full


def run_nca_on_ensemble(name: str, bases: list[str], y: np.ndarray, *,
                        n_components: int = 3,
                        nca_subsample: int = 20_000,
                        knn_k: int = 50,
                        max_iter: int = 50):
    """NCA-kNN with the ensemble's base predictions as input features."""
    print(f"\n=== {name} ({len(bases)} bases → {3*len(bases)} feats "
          f"→ {n_components} NCA dims → kNN K={knn_k}) ===")
    F_full_train, F_test = load_ensemble_features(bases)
    n_tr = len(F_full_train)
    print(f"  F_train: {F_full_train.shape}  F_test: {F_test.shape}")

    oof = np.zeros(n_tr, dtype=np.float32)
    test_proba = np.zeros(len(F_test), dtype=np.float32)
    fold_aucs, fold_secs = [], []

    rng = np.random.default_rng(SEED)
    for kf, tr, va in folds(y, task="classification"):
        t0 = time.time()
        sc = StandardScaler()
        F_tr = sc.fit_transform(F_full_train[tr])
        F_va = sc.transform(F_full_train[va])
        F_te = sc.transform(F_test)

        # Stratified subsample for NCA fit (loss matrix is O(n²))
        if len(tr) > nca_subsample:
            pos_idx = np.where(y[tr] == 1)[0]
            neg_idx = np.where(y[tr] == 0)[0]
            n_pos = int(nca_subsample * (len(pos_idx) / len(tr)))
            n_neg = nca_subsample - n_pos
            sub_pos = rng.choice(pos_idx, size=n_pos, replace=False)
            sub_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            sub = np.concatenate([sub_pos, sub_neg])
            rng.shuffle(sub)
        else:
            sub = np.arange(len(tr))

        nca = NeighborhoodComponentsAnalysis(
            n_components=n_components, init="pca",
            max_iter=max_iter, tol=1e-5,
            random_state=SEED + kf, verbose=0,
        )
        t_nca = time.time()
        nca.fit(F_tr[sub].astype(np.float64), y[tr][sub])
        nca_secs = time.time() - t_nca

        Z_tr = nca.transform(F_tr).astype(np.float32)
        Z_va = nca.transform(F_va).astype(np.float32)
        Z_te = nca.transform(F_te).astype(np.float32)

        clf = KNeighborsClassifier(
            n_neighbors=knn_k, weights="distance", algorithm="auto",
            leaf_size=40, n_jobs=-1,
        )
        clf.fit(Z_tr, y[tr])
        p_va = clf.predict_proba(Z_va)[:, 1]
        p_te = clf.predict_proba(Z_te)[:, 1]

        oof[va] = p_va
        test_proba += p_te.astype(np.float32) / N_FOLDS

        secs = time.time() - t0
        auc = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(auc)
        fold_secs.append(secs)
        print(f"  fold {kf}: AUC={auc:.5f}  ({secs:.1f}s, NCA fit "
              f"{nca_secs:.1f}s on {len(sub):,} rows)")

    oof_full = float(roc_auc_score(y, oof))
    print(f"  full OOF: {oof_full:.5f}  total {sum(fold_secs):.0f}s")
    save_oof(name,
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(variant=name, bases=bases, n_components=n_components,
                  nca_subsample=nca_subsample, knn_k=knn_k,
                  max_iter=max_iter, oof_score=oof_full,
                  fold_aucs=fold_aucs, fold_secs=fold_secs))
    return oof, test_proba, oof_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="K4,K10",
                    help="Comma-separated subset of {A,B,K4,K10}.")
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument("--nca-subsample", type=int, default=20_000,
                    help="NCA fit subsample. Pairwise loss matrix is O(n²) "
                         "→ 20k uses ~3.2 GB; 30k ~7.2 GB.")
    ap.add_argument("--knn-k", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=50)
    args = ap.parse_args()

    print("Loading data ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    variants = args.variants.split(",")
    summary = {}
    t0 = time.time()

    # Lazy: only build the raw-feature pool if we need an A/B variant.
    raw_pool = None
    n_tr = None

    for v in variants:
        if v in ("A", "B"):
            if raw_pool is None:
                raw_pool, n_tr = build_feature_pool(train, test)
                print(f"  raw-feature pool cols: {raw_pool.shape[1]}")
            cols = NCA_A_COLS if v == "A" else NCA_B_COLS
            name = ("knn_nca_A_top5_numeric" if v == "A"
                    else "knn_nca_B_pool14")
            oof, test_p, score = run_nca_variant(
                name, cols, raw_pool, n_tr, y,
                n_components=args.n_components,
                nca_subsample=args.nca_subsample,
                knn_k=args.knn_k, max_iter=args.max_iter,
            )
        elif v == "K4":
            oof, test_p, score = run_nca_on_ensemble(
                "knn_nca_K4_ensemble", K4_FWD, y,
                n_components=args.n_components,
                nca_subsample=args.nca_subsample,
                knn_k=args.knn_k, max_iter=args.max_iter,
            )
            name = "knn_nca_K4_ensemble"
        elif v == "K10":
            oof, test_p, score = run_nca_on_ensemble(
                "knn_nca_K10_ensemble", K10_FWD, y,
                n_components=args.n_components,
                nca_subsample=args.nca_subsample,
                knn_k=args.knn_k, max_iter=args.max_iter,
            )
            name = "knn_nca_K10_ensemble"
        else:
            raise SystemExit(f"unknown variant {v}")
        summary[name] = score

    print(f"\n=== summary ===")
    print(f"  references:")
    print(f"    plain kNN s2 (top-5 numeric):       0.89426")
    print(f"    K=4 plain LR-meta:                  0.95399")
    print(f"    K=4 + Path-B C×S τ=100k (PRIMARY):  0.95403")
    print(f"    K=4 kernel-SVM-meta linsvc γ=0.02:  0.95403")
    print(f"  this run:")
    for nm, s in summary.items():
        print(f"    {nm:<32s} {s:.5f}")
    print(f"\ntotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
