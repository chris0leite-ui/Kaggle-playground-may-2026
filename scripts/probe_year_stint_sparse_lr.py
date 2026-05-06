"""scripts/probe_year_stint_sparse_lr.py — Year×Stint sparse-LR feature probe.

Round-2 critic finding: EDA Phase E showed Year×Stint field-pair magnitude
0.386 — the strongest pair in the FM. d14 Path B Year×Stint COHORT axis
already failed (cohort axis dead because Year=2023 is flat-rate generator).
But Year×Stint as a sparse-LR feature interaction is a different axis:
the LR exploits the (Year, Stint) pair signal at the row level, not as
a meta-cohort routing key.

Build: a sparse-LR base on
  - Year (one-hot)
  - Stint (one-hot)
  - Compound (one-hot)
  - Driver (hashing trick, 2^16 buckets)
  - Year × Stint (one-hot, 4×6 = 24 dims)
  - Year × Stint × Compound (3-way, larger)

5-fold StratKF, save OOF + test, gate via probe_min_meta.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _onehot(values, n_levels):
    """Build CSR one-hot matrix for integer codes 0..n_levels-1."""
    n = len(values)
    return csr_matrix((np.ones(n), (np.arange(n), values)),
                      shape=(n, n_levels))


def _hash_onehot(strings, n_buckets=2 ** 14):
    """Hashing trick one-hot for high-cardinality strings."""
    n = len(strings)
    cols = np.fromiter(((hash(s) & (n_buckets - 1)) for s in strings),
                       dtype=np.int64, count=n)
    return csr_matrix((np.ones(n), (np.arange(n), cols)),
                      shape=(n, n_buckets))


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))

    # Code Year, Stint, Compound to small ints
    years = sorted(set(train["Year"].astype(int).unique()) |
                   set(test["Year"].astype(int).unique()))
    yr_map = {y_: i for i, y_ in enumerate(years)}
    cmps = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp_map = {c: i for i, c in enumerate(cmps)}

    def transform(df):
        yr = df["Year"].astype(int).map(yr_map).astype(int).values
        st = np.clip(df["Stint"].astype(int).values, 0, 5)
        cm = df["Compound"].astype(str).map(cmp_map).astype(int).values
        # Single one-hots
        X_yr = _onehot(yr, len(years))
        X_st = _onehot(st, 6)
        X_cm = _onehot(cm, len(cmps))
        # Driver hashed
        X_dr = _hash_onehot(df["Driver"].astype(str).values, n_buckets=2 ** 14)
        # Year × Stint  (4 × 6 = 24)
        X_yr_st = _onehot(yr * 6 + st, len(years) * 6)
        # Year × Stint × Compound  (4×6×5 = 120)
        X_yr_st_cm = _onehot(yr * 30 + st * 5 + cm, len(years) * 30)
        return hstack([X_yr, X_st, X_cm, X_dr, X_yr_st, X_yr_st_cm]).tocsr()

    X_train = transform(train)
    X_test = transform(test)
    print(f"sparse-LR feature matrix: train {X_train.shape}, test {X_test.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        lr = LogisticRegression(C=1.0, max_iter=200,
                                solver="liblinear", penalty="l2")
        lr.fit(X_train[tr], y[tr])
        oof[va] = lr.predict_proba(X_train[va])[:, 1]
        test_pred += lr.predict_proba(X_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(s)
        print(f"  fold {k}: AUC {s:.5f}  wall {time.time()-t_fold:.1f}s")

    auc = float(roc_auc_score(y, oof))
    rho, _ = spearmanr(test_pred, primary_test)
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    new_pos = test_pred >= rare_thr
    flips_neg = int(np.sum(primary_pos & ~new_pos))
    flips_pos = int(np.sum(~primary_pos & new_pos))

    print(f"\n=== year_stint_sparse_lr base ===")
    print(f"  std OOF: {auc:.5f} (PRIMARY {auc_primary:.5f}, "
          f"Δ {(auc-auc_primary)*1e4:+.2f} bp)")
    print(f"  ρ vs PRIMARY: {rho:.6f}")
    print(f"  flips +→− {flips_neg}, −→+ {flips_pos}")

    np.save(ART / "oof_year_stint_sparse_lr_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_year_stint_sparse_lr_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))
    sub = sample_sub.copy(); sub[TARGET] = test_pred
    sub.to_csv("submissions/submission_year_stint_sparse_lr.csv", index=False)
    summary = dict(std_oof=auc, delta_vs_primary_bp=(auc - auc_primary) * 1e4,
                   rho_vs_primary=float(rho),
                   flips_to_neg=flips_neg, flips_to_pos=flips_pos,
                   fold_aucs=fold_aucs,
                   wall_s=time.time() - t0)
    (ART / "probe_year_stint_sparse_lr.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n→ {ART / 'probe_year_stint_sparse_lr.json'} "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
