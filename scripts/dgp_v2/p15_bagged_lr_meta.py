"""Phase 15 — Bagged LR-meta on K=4 (variance reduction).

After P14 confirmed the K=4 meta-arch family is closed across LR /
Poly2 LR / tree / kernel / NCA, the only structurally-untested LR
variant is BAGGED LR-meta: average predictions from N LR-metas, each
fit on a bootstrap resample of train.

Mechanism: variance reduction at meta level. If each LR fit varies
slightly across resamples (due to fold-induced noise + sample
randomness), averaging may smooth and lift OOF.

Sweep N in {30, 100}. Compare to plain LR-meta and Path-B PRIMARY.

Predicted: NULL (LR is low-variance estimator on 350k rows; bagging
won't help much). But CHEAP and falsifies cleanly.
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
    if a.ndim == 2 and a.shape[1] == 2: return a[:, 1]
    return a.ravel()


def expand_base(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    P_clip = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(P_clip / (1 - P_clip))
    return np.hstack([P, rk, logit])


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train["PitNextLap"].astype(int).values

    base_oofs = [load_p(ART / f"oof_{f}_strat.npy") for f in K4]
    base_tests = [load_p(ART / f"test_{f}_strat.npy") for f in K4]
    F_oof = expand_base(np.column_stack(base_oofs))
    F_test = expand_base(np.column_stack(base_tests))
    print(f"K=4 expanded: F_oof {F_oof.shape}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Plain LR-meta baseline
    print("\n=== Plain LR-meta (baseline) ===", flush=True)
    oof_plain = np.zeros(len(y))
    for tr, va in splits:
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        m.fit(F_oof[tr], y[tr])
        oof_plain[va] = m.predict_proba(F_oof[va])[:, 1]
    auc_plain = roc_auc_score(y, oof_plain)
    print(f"  Plain LR-meta OOF: {auc_plain:.5f}", flush=True)

    # Bagged LR-meta
    for n_bag in [30, 100]:
        print(f"\n=== Bagged LR-meta (N={n_bag}) ===", flush=True)
        oof_bag = np.zeros(len(y))
        rng = np.random.default_rng(SEED)
        for fold, (tr, va) in enumerate(splits):
            fts = time.time()
            preds_va = np.zeros(len(va))
            for b in range(n_bag):
                # Bootstrap resample of tr (80% sample without replacement)
                n_tr = len(tr)
                bag_idx = rng.choice(tr, size=int(n_tr * 0.8), replace=False)
                m = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
                m.fit(F_oof[bag_idx], y[bag_idx])
                preds_va += m.predict_proba(F_oof[va])[:, 1] / n_bag
            oof_bag[va] = preds_va
            print(f"  fold {fold} bag-AUC {roc_auc_score(y[va], oof_bag[va]):.5f} "
                  f"[{time.time()-fts:.0f}s]", flush=True)
        auc_bag = roc_auc_score(y, oof_bag)
        delta = (auc_bag - auc_plain) * 1e4
        print(f"  Overall bagged-{n_bag} OOF: {auc_bag:.5f} "
              f"(Δ vs plain {delta:+.2f} bp)", flush=True)

        # Save (overwrites between sweeps; save bagged-100 as canonical)
        np.save(ART / f"oof_p15_bagged_lr_N{n_bag}_strat.npy",
                np.column_stack([1 - oof_bag, oof_bag]))

    print(f"\nTotal: {time.time()-ts:.0f}s", flush=True)


if __name__ == "__main__":
    main()
