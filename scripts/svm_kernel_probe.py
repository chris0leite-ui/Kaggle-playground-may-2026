"""SVM kernel probe — first SVM-family entry on this comp (ISSUES.md 1e).

Three variants on the same Yekenot-LR feature recipe:

  1. rff_sgd_hinge    : RBFSampler(gamma_sweep, n_components=2000)
                        + SGDClassifier(loss='hinge', class_weight='balanced')
                        Streaming-friendly on full 439k rows.
                        Cheap falsification probe — does kernel-class beat
                        the LR-bank ceiling 0.928?

  2. nystroem_linsvc  : Nystroem(rbf, n_components=1500)
                        + LinearSVC(loss='squared_hinge', dual=False)
                        The canonical scalable kernel-SVM recipe.

  3. nystroem_klogreg : Nystroem(rbf, n_components=1500)
                        + LogisticRegression(C=1.0, lbfgs)
                        Kernel-logistic — calibrated ranker check
                        (per Rule 26(i) Q6 alignment cross-check).

Feature recipe (matches scripts/lr_diag_a2_bagged_lr.py vanilla):
  - 11 numeric (StandardScaler, fit on train fold)
  - Compound one-hot (drop_first)
  - Race one-hot (drop_first)
  - Driver frequency-encoded (count from full df, standardised) — leak-safe
  - 4 cheap interactions used elsewhere in the LR-bank

Output (per variant `<NAME>`):
  - oof_<NAME>_strat.npy   shape (N_TRAIN, 2) sums to 1
  - test_<NAME>_strat.npy  shape (N_TEST, 2) sums to 1
  - <NAME>_results.json    fold AUCs, OOF AUC, gamma, timing

CV: pinned StratifiedKFold(SEED=42, N_FOLDS=5) per common.py.

Usage:
  python scripts/svm_kernel_probe.py --variant rff_sgd_hinge --gamma 0.1 --smoke
  python scripts/svm_kernel_probe.py --variant nystroem_linsvc --gamma 0.1
  python scripts/svm_kernel_probe.py --variant all --gamma 0.1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import N_FOLDS, SEED, folds, save_oof

TARGET = "PitNextLap"
ID_COL = "id"


# ---------------------------------------------------------------------------
# Feature builder (Yekenot-LR recipe; matches lr_diag_a2_bagged_lr vanilla)
# ---------------------------------------------------------------------------
def build_features(train: pd.DataFrame, test: pd.DataFrame
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_train, X_test, feature_names). No fold-conditional ops here;
    StandardScaler is fit per-fold inside the CV loop."""
    num_cols = [c for c in train.columns
                if c not in ["Driver", "Compound", "Race", TARGET, ID_COL]
                and pd.api.types.is_numeric_dtype(train[c])]

    df = pd.concat([train.assign(__split="tr"),
                    test.assign(__split="te")], ignore_index=True)

    X_num = df[num_cols].values.astype(np.float64)

    comp_cols = pd.get_dummies(df["Compound"], prefix="Cmp",
                               drop_first=True, dtype=np.float64)
    race_cols = pd.get_dummies(df["Race"], prefix="Race",
                               drop_first=True, dtype=np.float64)

    # Driver-frequency from train rows only (leak-safe, AV-AUC=0.502 anyway)
    drv_counts = train["Driver"].value_counts()
    drv_freq = df["Driver"].map(drv_counts).fillna(0).values.astype(np.float64)

    # Cheap pair interactions also used by the LR-bank vanilla recipe.
    nm_idx = {c: i for i, c in enumerate(num_cols)}
    pair_specs = [
        ("TyreLife", "Stint"),
        ("LapNumber", "Position"),
        ("Cumulative_Degradation", "TyreLife"),
        ("Position", "LapNumber"),
    ]
    pairs = np.column_stack([
        X_num[:, nm_idx[a]] * X_num[:, nm_idx[b]] for a, b in pair_specs
    ])

    X = np.hstack([X_num, comp_cols.values, race_cols.values,
                   drv_freq.reshape(-1, 1), pairs])
    names = (list(num_cols) + list(comp_cols.columns) + list(race_cols.columns)
             + ["Driver_freq"]
             + [f"{a}*{b}" for a, b in pair_specs])

    n_tr = len(train)
    return X[:n_tr], X[n_tr:], names


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def fit_predict_rff(X_tr, y_tr, X_va, X_te, *, gamma: float, seed: int):
    """RFF approximation of RBF + SGD-hinge."""
    rff = RBFSampler(gamma=gamma, n_components=2000, random_state=seed)
    Z_tr = rff.fit_transform(X_tr).astype(np.float32)
    Z_va = rff.transform(X_va).astype(np.float32)
    Z_te = rff.transform(X_te).astype(np.float32)

    clf = SGDClassifier(
        loss="hinge", penalty="l2", alpha=1e-5,
        class_weight="balanced", max_iter=20, tol=1e-4,
        random_state=seed, n_jobs=-1,
    )
    clf.fit(Z_tr, y_tr)
    # decision_function → sigmoid → pseudo-proba in (0,1) for AUC ranking
    p_va = expit(clf.decision_function(Z_va))
    p_te = expit(clf.decision_function(Z_te))
    return p_va, p_te


def fit_predict_nystroem_linsvc(X_tr, y_tr, X_va, X_te, *,
                                gamma: float, seed: int):
    """Nyström RBF + LinearSVC (squared-hinge).

    n_components reduced to 800 + n_jobs=1 in Nystroem to keep peak RAM
    under 15 GB on the full 350k-row training fold. The default n_jobs=-1
    spawned worker processes that duplicated the dense kernel-feature
    matrix and OOM-killed the run.
    """
    ns = Nystroem(kernel="rbf", gamma=gamma, n_components=800,
                  random_state=seed, n_jobs=1)
    Z_tr = ns.fit_transform(X_tr).astype(np.float32)
    Z_va = ns.transform(X_va).astype(np.float32)
    Z_te = ns.transform(X_te).astype(np.float32)

    clf = LinearSVC(
        C=1.0, loss="squared_hinge", penalty="l2", dual=False,
        class_weight="balanced", max_iter=2000, tol=1e-4,
        random_state=seed,
    )
    clf.fit(Z_tr, y_tr)
    p_va = expit(clf.decision_function(Z_va))
    p_te = expit(clf.decision_function(Z_te))
    return p_va, p_te


def fit_predict_nystroem_klogreg(X_tr, y_tr, X_va, X_te, *,
                                 gamma: float, seed: int):
    """Nyström RBF + Logistic Regression (kernel-logistic)."""
    ns = Nystroem(kernel="rbf", gamma=gamma, n_components=800,
                  random_state=seed, n_jobs=1)
    Z_tr = ns.fit_transform(X_tr).astype(np.float32)
    Z_va = ns.transform(X_va).astype(np.float32)
    Z_te = ns.transform(X_te).astype(np.float32)

    clf = LogisticRegression(
        C=1.0, solver="lbfgs",
        class_weight="balanced", max_iter=500,
    )
    clf.fit(Z_tr, y_tr)
    p_va = clf.predict_proba(Z_va)[:, 1]
    p_te = clf.predict_proba(Z_te)[:, 1]
    return p_va, p_te


VARIANTS = {
    "rff_sgd_hinge": fit_predict_rff,
    "nystroem_linsvc": fit_predict_nystroem_linsvc,
    "nystroem_klogreg": fit_predict_nystroem_klogreg,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_variant(variant: str, X_train_full: np.ndarray, X_test: np.ndarray,
                y: np.ndarray, *, gamma: float, smoke: bool):
    """5-fold CV (or 1-fold smoke) for a single variant. Saves artifacts."""
    fit_fn = VARIANTS[variant]
    name = f"svm_{variant}_g{gamma:g}{'_smoke' if smoke else ''}_strat"
    print(f"\n=== variant: {variant}  gamma={gamma}  smoke={smoke} ===")
    print(f"     name: {name}")

    n_train, n_feat = X_train_full.shape
    n_test = X_test.shape[0]
    print(f"     X_train: {X_train_full.shape}  X_test: {X_test.shape}")

    oof = np.zeros(n_train, dtype=np.float32)
    test_proba = np.zeros(n_test, dtype=np.float32)
    fold_scores = []
    fold_secs = []

    for k, tr, va in folds(y, task="classification"):
        if smoke:
            # Subsample 50k from train of fold-0 only; predict on that fold's val
            if k != 0:
                continue
            rng = np.random.default_rng(SEED + k)
            sub_idx = rng.choice(tr, size=50_000, replace=False)
            tr = sub_idx

        t0 = time.time()
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train_full[tr]).astype(np.float32)
        X_va_s = sc.transform(X_train_full[va]).astype(np.float32)
        X_te_s = sc.transform(X_test).astype(np.float32)

        p_va, p_te = fit_fn(
            X_tr_s, y[tr], X_va_s, X_te_s, gamma=gamma, seed=SEED + k,
        )
        secs = time.time() - t0
        fold_secs.append(secs)

        oof[va] = p_va
        test_proba += p_te.astype(np.float32) / N_FOLDS

        auc = float(roc_auc_score(y[va], p_va))
        fold_scores.append(auc)
        print(f"     fold {k}: AUC={auc:.5f}  ({secs:.1f}s, "
              f"n_tr={len(tr)}, n_va={len(va)})")
        if smoke:
            break

    if smoke:
        oof_score = fold_scores[0]
        proj_full_5fold_min = (sum(fold_secs) / max(len(fold_secs), 1)
                               * (n_train / 50_000) * N_FOLDS / 60)
        print(f"     SMOKE OOF (fold 0 only): {oof_score:.5f}")
        print(f"     projected full 5-fold cost: {proj_full_5fold_min:.1f} min")
    else:
        oof_score = float(roc_auc_score(y, oof))
        print(f"     full OOF AUC: {oof_score:.5f}  fold std: "
              f"{np.std(fold_scores):.5f}  total {sum(fold_secs):.1f}s")

    # Save in [n, 2] convention (1 - p, p) for stacker compatibility
    oof_2col = np.column_stack([1 - oof, oof])
    test_2col = np.column_stack([1 - test_proba, test_proba])

    save_oof(name, oof_2col, test_2col, dict(
        variant=variant, gamma=gamma, smoke=smoke,
        oof_score=oof_score, fold_scores=fold_scores,
        fold_secs=fold_secs, n_components=2000 if "rff" in variant else 1500,
        n_features=int(n_feat), n_train=int(n_train), n_test=int(n_test),
    ))
    print(f"     → scripts/artifacts/oof_{name}.npy")
    return name, oof_score, fold_scores, fold_secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=list(VARIANTS) + ["all"],
                    default="rff_sgd_hinge")
    ap.add_argument("--gamma", type=float, default=0.1,
                    help="RBF gamma (1/(n_features * scale))")
    ap.add_argument("--smoke", action="store_true",
                    help="1-fold @ 50k rows for time projection")
    args = ap.parse_args()

    print(f"loading data/train.csv + data/test.csv ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"train: {train.shape}  test: {test.shape}")

    X_train, X_test, names = build_features(train, test)
    y = train[TARGET].astype(int).values
    print(f"feature recipe: {len(names)} cols  "
          f"(num+Cmp+Race+drv_freq+pairs)")

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    results = {}
    for v in variants:
        name, oof_score, fold_scores, fold_secs = run_variant(
            v, X_train, X_test, y, gamma=args.gamma, smoke=args.smoke,
        )
        results[v] = dict(name=name, oof_score=oof_score,
                          fold_scores=fold_scores, fold_secs=fold_secs)

    print(f"\n=== summary ===")
    for v, r in results.items():
        print(f"  {v:24s} OOF={r['oof_score']:.5f}  total "
              f"{sum(r['fold_secs']):.0f}s")


if __name__ == "__main__":
    main()
