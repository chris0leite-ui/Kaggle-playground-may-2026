"""Phase 14 — Polynomial-2 LR-meta on K=4 [P, rank, logit].

Critic-driven pivot from base-side decode: rank-lock at logit-direction
level is empirically real (5 variants P2/P3/P5/P7/P8 all NULL); the
escape vector is META architecture, not base substrate.

A2-8 tested LightGBM tree-meta on 43 features (6 pairwise products +
6 abs-diffs + 6 logit-diffs + side-info) — lost -1.30 bp vs LR-meta.
Friction: `tree-stack-meta-overfits-small-K-pool`.

P14 is the COMPLEMENTARY test: keep LR (which Path-B partial-pooling
loves) but expand the input to FULL polynomial-2 of K=4 [P, rank,
logit] = 12 features. Pairwise products: 12 choose 2 = 66 + 12 squared
self-terms = 78 features total.

If linearly-separable in the original 12-feat space (rank-lock at
logit-direction), Poly2 expansion is a no-op. If interactions matter
but tree class can't fit them without overfitting, LR + L2 on Poly2
might extract them cleanly.

Cost: 5-fold LR on 350k rows x 78 feat ~ 5 min CPU.

Critic embedded: this is the structurally-clearest meta-arch test
I have time to run while P13 surrogate finishes; addresses the actual
binding constraint per critic's PIVOT recommendation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
SEED, N_FOLDS = 42, 5

K4 = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]


def load_p(path):
    a = np.load(path).astype(np.float64)
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 1]
    return a.ravel()


def expand_base(P: np.ndarray) -> np.ndarray:
    """[P, rank, logit] expansion — same as probe.py _expand."""
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    P_clip = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(P_clip / (1 - P_clip))
    return np.hstack([P, rk, logit])


def expand_poly2(F: np.ndarray) -> np.ndarray:
    """Add all pairwise products + self-squared terms.

    F has 12 cols (4 P + 4 rank + 4 logit). Output: 12 + 12*13/2 = 90 cols
    (12 raw + 78 poly2).
    """
    n, d = F.shape
    feats = [F]
    for i in range(d):
        for j in range(i, d):
            feats.append((F[:, i] * F[:, j]).reshape(-1, 1))
    return np.hstack(feats)


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train["PitNextLap"].astype(int).values

    # Load K=4 base preds
    base_oofs, base_tests = [], []
    for fname in K4:
        oo = load_p(ART / f"oof_{fname}_strat.npy")
        te = load_p(ART / f"test_{fname}_strat.npy")
        base_oofs.append(oo); base_tests.append(te)
    P_oof = np.column_stack(base_oofs)
    P_test = np.column_stack(base_tests)

    # 12-feat [P, rank, logit] expansion
    F_oof_lr = expand_base(P_oof)
    F_test_lr = expand_base(P_test)

    # Poly2 expansion
    F_oof_p2 = expand_poly2(F_oof_lr)
    F_test_p2 = expand_poly2(F_test_lr)
    print(f"Feature shapes: LR={F_oof_lr.shape}, Poly2={F_oof_p2.shape}",
          flush=True)

    # Standardize for LR (Poly2 has different scales)
    sc_lr = StandardScaler().fit(F_oof_lr)
    sc_p2 = StandardScaler().fit(F_oof_p2)
    F_oof_lr_s = sc_lr.transform(F_oof_lr)
    F_test_lr_s = sc_lr.transform(F_test_lr)
    F_oof_p2_s = sc_p2.transform(F_oof_p2)
    F_test_p2_s = sc_p2.transform(F_test_p2)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # ---- Fit standard LR-meta (12 feat) ----
    print("\n=== Standard LR-meta (12 feat) ===", flush=True)
    oof_lr = np.zeros(len(y))
    test_lr = np.zeros(len(F_test_lr_s))
    for tr, va in splits:
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        m.fit(F_oof_lr_s[tr], y[tr])
        oof_lr[va] = m.predict_proba(F_oof_lr_s[va])[:, 1]
        test_lr += m.predict_proba(F_test_lr_s)[:, 1] / N_FOLDS
    auc_lr = roc_auc_score(y, oof_lr)
    print(f"  LR-meta OOF: {auc_lr:.5f}", flush=True)

    # ---- Fit Poly2 LR-meta (90 feat) ----
    # Sweep C in {0.01, 0.1, 1.0, 10.0} to find the right regularization
    print(f"\n=== Poly2 LR-meta (90 feat) — C sweep ===", flush=True)
    results = {}
    for C in [0.01, 0.1, 1.0, 10.0]:
        oof = np.zeros(len(y))
        test = np.zeros(len(F_test_p2_s))
        fts = time.time()
        for tr, va in splits:
            m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            m.fit(F_oof_p2_s[tr], y[tr])
            oof[va] = m.predict_proba(F_oof_p2_s[va])[:, 1]
            test += m.predict_proba(F_test_p2_s)[:, 1] / N_FOLDS
        auc = roc_auc_score(y, oof)
        delta = (auc - auc_lr) * 1e4
        print(f"  C={C:>6.2f}: OOF {auc:.5f} (Δ vs LR-meta {delta:+.2f}bp) "
              f"[{time.time()-fts:.0f}s]", flush=True)
        results[f"C={C}"] = {"oof_auc": float(auc), "delta_bp": float(delta)}
        # save best
        if C == 1.0:
            np.save(ART / f"oof_p14_poly2_lr_C{C}_strat.npy",
                    np.column_stack([1 - oof, oof]))
            np.save(ART / f"test_p14_poly2_lr_C{C}_strat.npy",
                    np.column_stack([1 - test, test]))

    # Also test L1 regularization (sparse selection of interactions)
    print(f"\n=== Poly2 LR-meta L1 — C sweep ===", flush=True)
    for C in [0.01, 0.1, 1.0]:
        oof = np.zeros(len(y))
        fts = time.time()
        for tr, va in splits:
            m = LogisticRegression(C=C, max_iter=2000, solver="saga",
                                    penalty="l1")
            m.fit(F_oof_p2_s[tr], y[tr])
            oof[va] = m.predict_proba(F_oof_p2_s[va])[:, 1]
        auc = roc_auc_score(y, oof)
        delta = (auc - auc_lr) * 1e4
        print(f"  L1 C={C:>5.2f}: OOF {auc:.5f} (Δ {delta:+.2f}bp) "
              f"[{time.time()-fts:.0f}s]", flush=True)
        results[f"L1_C={C}"] = {"oof_auc": float(auc), "delta_bp": float(delta)}

    summary = {
        "name": "p14_poly2_lr_meta",
        "lr_meta_baseline_oof": float(auc_lr),
        "poly2_results": results,
        "n_features_lr": 12,
        "n_features_poly2": int(F_oof_p2.shape[1]),
    }
    (ART / "p14_poly2_lr_meta_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved p14_poly2_lr_meta_results.json [{time.time()-ts:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
