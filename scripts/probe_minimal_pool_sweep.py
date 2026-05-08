"""scripts/probe_minimal_pool_sweep.py — EXP-5: minimal-pool sweep.

User-driven question: can we reduce the pool below K=10 without
losing meaningful LB? K=10 forward-greedy landed at LB 0.95356 vs
PRIMARY 0.95368 (Δ −1.2 bp). Test the prefixes K=2,3,4,5,6,7,8,9 of
the same forward-greedy pick order, plain LR-meta and Path-B C×S
τ=100k. Identify the smallest pool within 2 bp OOF of K=10. Stage
that as a submission candidate.

Cost: ~6-10 min CPU.
Outputs scripts/artifacts/probe_minimal_pool_sweep.json + sparse
forward-greedy submission CSVs.
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

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS, MAX_ITER = 42, 5, 500
MIN_ROWS = 1000

# E9 forward-greedy pick order (per t2_k10_primary.py / Day-18 PM)
K10_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
    "b_lapsuntilpit",
    "baseline_two_anchor",
    "d9_R6_next_compound",
    "cb_year-cat",
    "e5_optuna_lgbm",
    "d9f_FM_A",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def fit_lr_aug(F, y):
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def fit_plain(F, y, splits):
    oof = np.zeros(len(y))
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof


def fit_path_b(F, y, splits, seg, n_seg, tau):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in splits:
        w_global = fit_lr_aug(F[tr_idx], y[tr_idx])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg[tr_idx] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr_idx][idx])) < 2:
                continue
            W_local[s] = fit_lr_aug(F[tr_idx][idx], y[tr_idx][idx])
            mask[s] = True
        n_local = counts.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local +
                    (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg[va_idx]):
            idx = np.where(seg[va_idx] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            F_aug = np.column_stack([np.ones(len(idx)), F[va_idx][idx]])
            oof[va_idx[idx]] = 1.0 / (1.0 + np.exp(
                -np.clip(F_aug @ w, -30, 30)))
    return oof


def main():
    t0 = time.time()
    print("Loading K=10 forward-greedy bases ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    base_oofs, base_tests = [], []
    for n in K10_FWD:
        base_oofs.append(_pos(ART / f"oof_{n}_strat.npy"))
        base_tests.append(_pos(ART / f"test_{n}_strat.npy"))
    print(f"  K=10 bases loaded; OOF rows {len(y):,}, test rows {len(test):,}")

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

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Reference: existing K=10 result (rerun for consistency)
    print("\nReference K=10 plain LR-meta + Path-B C×S τ=100k ...")
    F_K10 = expand(np.column_stack(base_oofs))
    oof_K10_plain = fit_plain(F_K10, y, splits)
    auc_K10_plain = float(roc_auc_score(y, oof_K10_plain))
    oof_K10_pb = fit_path_b(F_K10, y, splits, seg_train, n_seg, 100000.0)
    auc_K10_pb = float(roc_auc_score(y, oof_K10_pb))
    print(f"  K=10 plain: {auc_K10_plain:.5f}  Path-B: {auc_K10_pb:.5f}")

    # Sweep K = 2..10 prefixes
    print("\nSweeping K=2..10 forward-greedy prefixes ...")
    results = {}
    for k in range(2, 11):
        Psub = np.column_stack(base_oofs[:k])
        F = expand(Psub)
        oof_p = fit_plain(F, y, splits)
        auc_p = float(roc_auc_score(y, oof_p))
        oof_pb = fit_path_b(F, y, splits, seg_train, n_seg, 100000.0)
        auc_pb = float(roc_auc_score(y, oof_pb))
        delta_pb_vs_K10 = (auc_pb - auc_K10_pb) * 1e4
        delta_p_vs_K10 = (auc_p - auc_K10_plain) * 1e4
        print(f"  k={k:>2d}: plain {auc_p:.5f} (Δ{delta_p_vs_K10:+6.2f})  "
              f"PathB {auc_pb:.5f} (Δ{delta_pb_vs_K10:+6.2f})  "
              f"bases={K10_FWD[:k]}")
        results[k] = {
            "k": k, "names": K10_FWD[:k],
            "plain_oof": auc_p, "path_b_oof": auc_pb,
            "delta_plain_vs_K10_bp": float(delta_p_vs_K10),
            "delta_path_b_vs_K10_bp": float(delta_pb_vs_K10),
        }
        # Save sparse Path-B OOF + test for any k that's within 5 bp of K=10
        if delta_pb_vs_K10 >= -5.0:
            np.save(ART / f"oof_K{k}_fwd_pathb_strat.npy",
                    np.column_stack([1 - oof_pb, oof_pb]))
            # Build test predictions (full-train Path-B)
            F_test_full = expand(np.column_stack(base_tests[:k]))
            F_train_full = expand(np.column_stack(base_oofs[:k]))
            w_global_full = fit_lr_aug(F_train_full, y)
            W_local_full = np.zeros((n_seg, len(w_global_full)))
            counts_full = np.zeros(n_seg, dtype=np.int64)
            mask_full = np.zeros(n_seg, dtype=bool)
            for s in range(n_seg):
                idx = np.where(seg_train == s)[0]
                counts_full[s] = len(idx)
                if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
                    continue
                W_local_full[s] = fit_lr_aug(F_train_full[idx], y[idx])
                mask_full[s] = True
            n_local = counts_full.astype(np.float64)
            alpha = n_local / (n_local + 100000.0)
            W_shrunk = (alpha[:, None] * W_local_full +
                        (1 - alpha[:, None]) * w_global_full[None, :])
            tp = np.zeros(len(test))
            for s in np.unique(seg_test):
                idx = np.where(seg_test == s)[0]
                w = W_shrunk[s] if mask_full[s] else w_global_full
                F_aug = np.column_stack([np.ones(len(idx)),
                                         F_test_full[idx]])
                tp[idx] = 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))
            np.save(ART / f"test_K{k}_fwd_pathb_strat.npy",
                    np.column_stack([1 - tp, tp]))
            sub = sample_sub.copy()
            sub[TARGET] = tp
            Path("submissions").mkdir(exist_ok=True)
            sub.to_csv(f"submissions/submission_K{k}_fwd_pathb.csv",
                       index=False)

    # Recommendation
    print("\n=== Smallest-pool recommendation ===")
    print("Smallest k with Path-B OOF within 2 bp of K=10:")
    chosen = None
    for k in sorted(results.keys()):
        if results[k]["delta_path_b_vs_K10_bp"] >= -2.0:
            chosen = k
            print(f"  k={k}: PathB Δ vs K=10 = "
                  f"{results[k]['delta_path_b_vs_K10_bp']:+.2f} bp  "
                  f"-- RECOMMENDED for LB calibration submit")
            break
    if chosen is None:
        chosen = 10
        print(f"  No k<10 was within 2 bp of K=10 PathB. Stay at K=10.")

    out = {
        "K10_plain_oof": auc_K10_plain,
        "K10_path_b_oof": auc_K10_pb,
        "sweep": results,
        "smallest_within_2bp_of_K10": chosen,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_minimal_pool_sweep.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nWrote scripts/artifacts/probe_minimal_pool_sweep.json. "
          f"Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
