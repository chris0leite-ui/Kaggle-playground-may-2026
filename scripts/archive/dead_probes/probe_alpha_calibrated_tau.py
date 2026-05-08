"""scripts/probe_alpha_calibrated_tau.py — α-calibrated τ-resweep.

Tests the hypothesis from `audit/2026-05-06-alpha-asymmetry-verification.md`:
the τ that maximises Strat OOF in d13e is BIASED because OOF uses
fold-train counts in α=n/(n+τ) while test uses full-train counts.

Fix at OOF time only: use full-train counts for α (matching test-time)
while still fitting per-segment LR on fold-train rows. This produces an
OOF estimate calibrated to the test-time shrinkage. Re-sweep τ on this
calibrated OOF — the optimum may sit at a smaller τ.

Output: same family as d13e but with calibrated α at OOF.
Sweep τ ∈ {1k, 2k, 5k, 10k, 20k, 50k, 100k}.
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
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"
MIN_ROWS = 1000
MAX_ITER = 500

K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
]
TAUS = [1000, 2000, 5000, 10000, 20000, 50000, 100000]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _fit_lr_aug(F, y, max_iter=MAX_ITER):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def _predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_test = _pos(PRIMARY_TEST)
    primary_oof = _pos(PRIMARY_OOF)
    auc_primary = float(roc_auc_score(y, primary_oof))

    base_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]
    base_tests = [_pos(ART / f"test_{b}_strat.npy") for b in K21_BASES]
    F_oof = _expand(np.column_stack(base_oofs))
    F_test = _expand(np.column_stack(base_tests))
    print(f"K=21 pool; F shape {F_oof.shape}")

    # Compound × Stint segmentation
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    seg_tr = c_tr * 6 + s_tr
    seg_te = c_te * 6 + s_te
    n_seg = len(cats) * 6

    # KEY: pre-compute FULL-TRAIN counts (used for α at BOTH OOF and test).
    counts_full = np.bincount(seg_tr, minlength=n_seg).astype(np.float64)
    print(f"Segments: {n_seg}; full-train counts min/med/max: "
          f"{int(counts_full[counts_full>0].min())}/"
          f"{int(np.median(counts_full[counts_full>0]))}/"
          f"{int(counts_full.max())}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oofs = {tau: np.zeros(len(y)) for tau in TAUS}
    print(f"\n--- α-CALIBRATED Compound×Stint hier-meta sweep on K=21 ---")
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = _fit_lr_aug(F_oof[tr], y[tr])
        W_local = np.zeros((n_seg, len(w_global)))
        mask = np.zeros(n_seg, dtype=bool)
        # Per-segment LR fits on fold-train rows (matches d13e)
        for s in range(n_seg):
            idx = np.where(seg_tr[tr] == s)[0]
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_local[s] = _fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            mask[s] = True
        # KEY: use FULL-train counts in α, NOT fold-train counts
        for tau in TAUS:
            alpha = counts_full / (counts_full + tau)
            W_shrunk = (alpha[:, None] * W_local +
                        (1 - alpha[:, None]) * w_global[None, :])
            for s in np.unique(seg_tr[va]):
                idx = np.where(seg_tr[va] == s)[0]
                w = W_shrunk[s] if mask[s] else w_global
                oofs[tau][va[idx]] = _predict_aug(F_oof[va[idx]], w)
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s "
              f"({int(mask.sum())}/{n_seg} segments fit)")

    # Full-train fit for test predictions (same as d13e, using full counts)
    t_full = time.time()
    w_global_full = _fit_lr_aug(F_oof, y)
    W_local_full = np.zeros((n_seg, len(w_global_full)))
    mask_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_tr == s)[0]
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_local_full[s] = _fit_lr_aug(F_oof[idx], y[idx])
        mask_full[s] = True
    test_preds = {}
    for tau in TAUS:
        alpha = counts_full / (counts_full + tau)
        W_shrunk = (alpha[:, None] * W_local_full +
                    (1 - alpha[:, None]) * w_global_full[None, :])
        tp = np.zeros(len(test))
        for s in np.unique(seg_te):
            idx = np.where(seg_te == s)[0]
            w = W_shrunk[s] if mask_full[s] else w_global_full
            tp[idx] = _predict_aug(F_test[idx], w)
        test_preds[tau] = tp
    print(f"  full-train wall: {time.time()-t_full:.1f}s")

    print(f"\n=== α-calibrated τ sweep ===")
    print(f"  PRIMARY (d13e Compound × Stint τ=20k): OOF {auc_primary:.5f}")
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    summary = dict(auc_primary=auc_primary, taus={})
    for tau in TAUS:
        oof = oofs[tau]
        tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho, _ = spearmanr(tp, primary_test)
        new_pos = tp >= rare_thr
        flips_neg = int(np.sum(primary_pos & ~new_pos))
        flips_pos = int(np.sum(~primary_pos & new_pos))
        ratio = (min(flips_pos, flips_neg) / max(flips_pos, flips_neg)
                 if max(flips_pos, flips_neg) > 0 else 1.0)
        d = (auc - auc_primary) * 1e4
        print(f"  τ={tau:>6}: OOF {auc:.5f}  Δ vs PRIMARY {d:+.2f} bp  "
              f"ρ {rho:.6f}  flips {flips_neg}/{flips_pos} ratio {ratio:.3f}")
        np.save(ART / f"oof_alpha_calib_compound_stint_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_alpha_calib_compound_stint_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        summary["taus"][str(tau)] = dict(
            oof=auc, delta_vs_primary_bp=float(d),
            rho_vs_primary=float(rho),
            flips_to_neg=flips_neg, flips_to_pos=flips_pos,
            flip_ratio=float(ratio),
        )

    summary["wall_s"] = time.time() - t0
    out = ART / "probe_alpha_calibrated_tau.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out}  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
