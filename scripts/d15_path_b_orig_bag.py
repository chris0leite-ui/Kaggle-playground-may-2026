"""scripts/d15_path_b_orig_bag.py — hier-meta with multi-arch orig bag.

Probe K=23 and K=24 hier-meta(Compound×Stint, τ=20k) configurations:
  K=22 baseline:     K=21 PRIMARY pool + d15_orig_transfer
                     (already at OOF 0.95094 from earlier d15 probe)
  K=23 (this probe): K=22 + d15_orig_cb (drop redundant lgbm_t per ρ=0.988)
  K=24 (this probe): K=23 + d15_orig_xgb

Inter-arch ρ matrix (synth test):
  transfer(LGBM) vs cb=0.948, vs xgb=0.948, vs lgbm_t=0.988 (REDUNDANT)
  cb vs xgb=0.941    cb vs lgbm_t=0.950   xgb vs lgbm_t=0.949
ρ vs PRIMARY single-row:
  transfer 0.565 | cb 0.587 | xgb 0.639 | lgbm_t 0.568

Pick K=24 candidates: transfer + cb + xgb (most-diverse intra-orig
trio) + the 21 PRIMARY pool bases.

Saves new oof/test artifacts and submission CSVs for both K=23 and K=24
configurations. Reports OOF lift vs hier-meta(K=21) baseline (0.95083)
and against the K=22 baseline (0.95094).
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
MIN_ROWS = 1000
MAX_ITER = 500
TAU = 20000

POOL_KEEP = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race",
    "d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound",
    "d9f_FM_A", "d9f_FM_B",
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def fit_lr_aug(F, y, max_iter=MAX_ITER):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def hier_meta(F_oof, F_test, y, seg_train, seg_test, n_seg, splits):
    oof_meta = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = fit_lr_aug(F_oof[tr], y[tr])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_local[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            mask[s] = True
        n_local = counts.astype(np.float64)
        alpha = n_local / (n_local + TAU)
        W_shrunk = (alpha[:, None] * W_local +
                    (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg_train[va]):
            idx = np.where(seg_train[va] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            oof_meta[va[idx]] = predict_aug(F_oof[va[idx]], w)
        print(f"    fold {fold}: {time.time()-t_fold:.1f}s")

    # Full train
    w_global_full = fit_lr_aug(F_oof, y)
    W_local_full = np.zeros((n_seg, len(w_global_full)))
    counts_full = np.zeros(n_seg, dtype=np.int64)
    mask_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        counts_full[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_local_full[s] = fit_lr_aug(F_oof[idx], y[idx])
        mask_full[s] = True
    n_local = counts_full.astype(np.float64)
    alpha = n_local / (n_local + TAU)
    W_shrunk = (alpha[:, None] * W_local_full +
                (1 - alpha[:, None]) * w_global_full[None, :])
    test_pred = np.zeros(F_test.shape[0])
    for s in np.unique(seg_test):
        idx = np.where(seg_test == s)[0]
        w = W_shrunk[s] if mask_full[s] else w_global_full
        test_pred[idx] = predict_aug(F_test[idx], w)
    return oof_meta, test_pred


def load_pool(names):
    o, t = [], []
    for n in names:
        a = np.load(ART / f"oof_{n}_strat.npy")
        b = np.load(ART / f"test_{n}_strat.npy")
        o.append(a[:, 1] if a.ndim == 2 else a)
        t.append(b[:, 1] if b.ndim == 2 else b)
    return np.column_stack(o), np.column_stack(t)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_test = np.load(
        ART / "test_d13e_compound_stint_tau20000_strat.npy"
    )[:, 1].astype(np.float64)

    # Compound × Stint segmentation (same as d13e)
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    seg_train = c_tr * 6 + s_tr; seg_test = c_te * 6 + s_te
    n_seg = len(cats) * 6

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    configs = {
        "K23_orig_transfer_cb":
            POOL_KEEP + ["d15_orig_transfer", "d15_orig_cb"],
        "K24_orig_transfer_cb_xgb":
            POOL_KEEP + ["d15_orig_transfer", "d15_orig_cb", "d15_orig_xgb"],
    }

    results = {}
    for label, names in configs.items():
        print(f"\n=== {label} (K={len(names)}) ===")
        P_oof, P_test = load_pool(names)
        F_oof = expand(P_oof); F_test = expand(P_test)
        print(f"  pool shape: {P_oof.shape}, expanded F: {F_oof.shape}")
        oof_meta, test_pred = hier_meta(
            F_oof, F_test, y, seg_train, seg_test, n_seg, splits)

        auc = float(roc_auc_score(y, oof_meta))
        rho, _ = spearmanr(test_pred, primary_test)
        rare_thr = float(np.quantile(primary_test, 0.99))
        primary_pos = primary_test >= rare_thr
        new_pos = test_pred >= rare_thr
        flips_to_neg = int(np.sum(primary_pos & ~new_pos))
        flips_to_pos = int(np.sum(~primary_pos & new_pos))
        ratio = (min(flips_to_neg, flips_to_pos) /
                 max(flips_to_neg, flips_to_pos)
                 if max(flips_to_neg, flips_to_pos) > 0 else 1.0)

        d_K21 = (auc - 0.95083) * 1e4
        d_K22 = (auc - 0.95094) * 1e4
        print(f"  OOF: {auc:.5f}  Δ vs K=21 hier-meta: {d_K21:+.3f}bp  "
              f"Δ vs K=22 (orig_transfer): {d_K22:+.3f}bp")
        print(f"  ρ vs PRIMARY: {rho:.6f}")
        print(f"  flips top-1%: +→− {flips_to_neg}, −→+ {flips_to_pos}, "
              f"ratio {ratio:.3f} (R7 cap 200, total {flips_to_neg+flips_to_pos})")

        np.save(ART / f"oof_d15_path_b_{label}_strat.npy",
                np.column_stack([1 - oof_meta, oof_meta]))
        np.save(ART / f"test_d15_path_b_{label}_strat.npy",
                np.column_stack([1 - test_pred, test_pred]))
        sub = sample_sub.copy(); sub[TARGET] = test_pred
        sub_path = f"submissions/submission_d15_path_b_{label}.csv"
        sub.to_csv(sub_path, index=False)

        results[label] = dict(
            K=len(names), oof=auc,
            delta_K21_bp=float(d_K21), delta_K22_bp=float(d_K22),
            rho_vs_primary=float(rho),
            flips_to_neg=flips_to_neg, flips_to_pos=flips_to_pos,
            flip_ratio=float(ratio), submission=sub_path,
        )

    # Summary
    print("\n=== Summary ===")
    print(f"  Reference: K=21 hier-meta = 0.95083 | K=22 +orig_transfer = 0.95094 (LB tie)")
    for label, r in results.items():
        print(f"  {label:<28s}  OOF {r['oof']:.5f}  Δ_K22 {r['delta_K22_bp']:+.2f}bp  "
              f"ρ_PRIMARY {r['rho_vs_primary']:.5f}  flips {r['flips_to_neg']+r['flips_to_pos']}")

    results["wall_s"] = time.time() - t0
    (ART / "d15_path_b_orig_bag_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n→ wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
