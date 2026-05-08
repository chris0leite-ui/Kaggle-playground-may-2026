"""scripts/probe_a2_2_pathb_K4.py — Path-B amp test on K=4 + A2-2.

Phase 4 plain LR-meta gate yielded only +0.302 bp for A2-2; the actual
PRIMARY meta is K=4 + Path-B Compound×Stint τ=100k. This probe checks
whether Path-B amplifies the +1.4 bp single-LGBM lift that the plain
LR-meta absorbed.

Reference: K=4 + Path-B C×S τ=100k OOF = 0.95403 (current PRIMARY).

Reuses the Path-B fitter from `probe_minimal_pool_sweep.py`.

Cost ~3-5 min CPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS, MAX_ITER = 42, 5, 500
MIN_ROWS = 1000

K4 = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]
CANDIDATE = "p1_lgbm_v3_with_a2_2"


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
        W_shrunk = (alpha[:, None] * W_local
                    + (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg[va_idx]):
            idx = np.where(seg[va_idx] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            F_aug = np.column_stack([np.ones(len(idx)), F[va_idx][idx]])
            oof[va_idx[idx]] = 1.0 / (1.0 + np.exp(
                -np.clip(F_aug @ w, -30, 30)))
    return oof


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4]
    cand_oof = _pos(ART / f"oof_{CANDIDATE}_strat.npy")
    print(f"  K=4 bases: {K4}")
    print(f"  candidate: {CANDIDATE}")
    print(f"  rows: {len(y):,}")

    # Compound × Stint segmentation (same as PRIMARY).
    cats = sorted(train["Compound"].astype(str).unique())
    cmp_map = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp_map).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_train = c_tr * 6 + s_tr
    print(f"  segments: {n_seg} (compound × stint, capped at stint=5)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Reference: K=4 baseline, both plain and Path-B
    print("\nK=4 baseline ...")
    P_K4 = np.column_stack(base_oofs)
    F_K4 = expand(P_K4)
    t1 = time.time()
    oof_K4_plain = fit_plain(F_K4, y, splits)
    auc_K4_plain = float(roc_auc_score(y, oof_K4_plain))
    oof_K4_pb = fit_path_b(F_K4, y, splits, seg_train, n_seg, 100000.0)
    auc_K4_pb = float(roc_auc_score(y, oof_K4_pb))
    print(f"  K=4 plain:  {auc_K4_plain:.5f}")
    print(f"  K=4 Path-B: {auc_K4_pb:.5f}  (PRIMARY ref 0.95403)")
    print(f"  Path-B amp: {(auc_K4_pb - auc_K4_plain)*1e4:+.2f} bp  ({time.time()-t1:.1f}s)")

    # Candidate: K=4 + a2_2
    print("\nK=4 + a2_2 ...")
    P_K5 = np.column_stack(base_oofs + [cand_oof])
    F_K5 = expand(P_K5)
    t1 = time.time()
    oof_K5_plain = fit_plain(F_K5, y, splits)
    auc_K5_plain = float(roc_auc_score(y, oof_K5_plain))
    oof_K5_pb = fit_path_b(F_K5, y, splits, seg_train, n_seg, 100000.0)
    auc_K5_pb = float(roc_auc_score(y, oof_K5_pb))
    print(f"  K=5 plain:  {auc_K5_plain:.5f}  Δ {(auc_K5_plain-auc_K4_plain)*1e4:+.2f} bp vs K=4 plain")
    print(f"  K=5 Path-B: {auc_K5_pb:.5f}  Δ {(auc_K5_pb-auc_K4_pb)*1e4:+.2f} bp vs K=4 Path-B")
    print(f"  Path-B amp: {(auc_K5_pb - auc_K5_plain)*1e4:+.2f} bp  ({time.time()-t1:.1f}s)")

    # Verdict
    delta_pb = (auc_K5_pb - auc_K4_pb) * 1e4
    delta_plain = (auc_K5_plain - auc_K4_plain) * 1e4
    if delta_pb >= 0.5:
        verdict = "PASS"
    elif delta_pb >= 0.1:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    print(f"\n  K=4+1 Path-B Δ: {delta_pb:+.3f} bp  →  verdict: {verdict}")

    rho_with_pathb_primary = float(spearmanr(oof_K5_pb, oof_K4_pb).statistic)
    print(f"  ρ(K=4+1 Path-B, K=4 Path-B): {rho_with_pathb_primary:.6f}")

    summary = dict(
        candidate=CANDIDATE,
        K4_bases=K4,
        K4_plain_oof=auc_K4_plain,
        K4_pathb_oof=auc_K4_pb,
        K5_plain_oof=auc_K5_plain,
        K5_pathb_oof=auc_K5_pb,
        delta_plain_bp=float(delta_plain),
        delta_pathb_bp=float(delta_pb),
        pathb_amp_K4_bp=float((auc_K4_pb - auc_K4_plain)*1e4),
        pathb_amp_K5_bp=float((auc_K5_pb - auc_K5_plain)*1e4),
        rho_K5_vs_K4_pathb=rho_with_pathb_primary,
        verdict=verdict,
        wall_s=time.time() - t0,
    )
    out = ART / "probe_a2_2_pathb_K4.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  → {out}")
    print(f"  total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
