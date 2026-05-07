"""d17 Path-B C×S on K=25 = K=21 + p1_single_cb_v4_gpu.

Mirrors d13e exactly (Compound × Stint segmentation, τ ∈ {5k, 20k,
100k, 500k}, MIN_ROWS=1000), with the K=21 PRIMARY pool extended by
the day-17 single-CB v3 GPU base.

Background: probe_min_meta__p1_single_cb_v4_gpu reports K=21+1 LR-meta
lift +12.06 bp over K=21 baseline (~15× any prior base-add). Path-B
hier-meta with Compound×Stint segmentation has historically amplified
FM-class lifts 6-12× from OOF to LB (d9c+3bp, d13e+8bp, d15b 1.4×).
This script tests whether p1_single_cb fires Path-B amp similarly.

Outputs (all under scripts/artifacts/):
  oof_d17_path_b_K25_full_merge_tau{tau}_strat.npy + test_..._strat.npy
  d17_path_b_K25_full_merge_results.json
  submissions/submission_d17_path_b_K25_full_merge_tau{tau}.csv
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
PRIMARY_LB_OOF = 0.95121  # d16 cont_only K=25 Path-B (current PRIMARY)
MIN_ROWS = 1000
MAX_ITER = 500

# K=21 PRIMARY pool (matches d13e exactly)
POOL_KEEP = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"), ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"), ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"), ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"), ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    ("realmlp", "realmlp"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]
TOP_3_D9 = [
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
]
FM_PAIR = [
    ("FM_A", "d9f_FM_A"), ("FM_B", "d9f_FM_B"),
]
# d17 22nd base
D17_NEW = [
    ("d16_cont_only",   "d16_orig_continuous_only"),
    ("d18_chain_decomp", "d18_chain_decomp"),
    ("p1_single_cb_v4",  "p1_single_cb_v4_gpu"),
    ("h1d_yekenot",      "d17_h1d_yekenot_full"),
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def fit_lr_aug(F, y, max_iter=MAX_ITER):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    # Load 22 base OOF/test pairs
    base_oofs, base_tests, names = [], [], []
    for label, fname in POOL_KEEP + TOP_3_D9 + FM_PAIR + D17_NEW:
        oo = np.load(ART / f"oof_{fname}_strat.npy")
        te = np.load(ART / f"test_{fname}_strat.npy")
        oo = oo[:, 1] if oo.ndim == 2 else oo
        te = te[:, 1] if te.ndim == 2 else te
        base_oofs.append(oo.astype(np.float64))
        base_tests.append(te.astype(np.float64))
        names.append(label)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"K={len(base_oofs)} bases ({names[-1]} = the d17 add)")
    print(f"F_oof shape {F_oof.shape}  F_test shape {F_test.shape}")

    # Compound × Stint segmentation (matches d13e)
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    n_cats = len(cats)
    seg_train = c_tr * 6 + s_tr
    seg_test = c_te * 6 + s_te
    n_seg = n_cats * 6
    sizes = np.bincount(seg_train, minlength=n_seg)
    populated = int(np.sum(sizes >= MIN_ROWS))
    print(f"Compound×Stint: n_seg={n_seg}, ≥{MIN_ROWS} rows: {populated}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Global LR baseline
    print("\n--- Global K=25 LR baseline ---")
    meta_global = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global[va] = lr.predict_proba(F_oof[va])[:, 1]
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s")
    auc_global = float(roc_auc_score(y, meta_global))
    print(f"  Global K=25 LR meta OOF: {auc_global:.5f}")

    # Hier-meta sweep
    taus = [5000, 20000, 100000, 500000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    print("\n--- Compound×Stint hier-meta on K=25 ---")
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = fit_lr_aug(F_oof[tr], y[tr])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        skipped = 0
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                skipped += 1
                continue
            W_local[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            mask[s] = True
        for tau in taus:
            n_local = counts.astype(np.float64)
            alpha = n_local / (n_local + tau)
            W_shrunk = (alpha[:, None] * W_local +
                        (1 - alpha[:, None]) * w_global[None, :])
            for s in np.unique(seg_train[va]):
                idx = np.where(seg_train[va] == s)[0]
                w = W_shrunk[s] if mask[s] else w_global
                oofs[tau][va[idx]] = predict_aug(F_oof[va[idx]], w)
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s "
              f"({int(mask.sum())}/{n_seg} segments fit; {skipped} skipped)")

    # Full-train fit for test predictions
    print("\n--- Full-train test predictions ---")
    t_full = time.time()
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
    test_preds = {}
    for tau in taus:
        n_local = counts_full.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local_full +
                    (1 - alpha[:, None]) * w_global_full[None, :])
        tp = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx = np.where(seg_test == s)[0]
            w = W_shrunk[s] if mask_full[s] else w_global_full
            tp[idx] = predict_aug(F_test[idx], w)
        test_preds[tau] = tp
    print(f"  full-train wall: {time.time()-t_full:.1f}s")

    # Evaluate vs current PRIMARY (d16 cont_only K=25 Path-B)
    primary_test = np.load(ART / "test_d16_path_b_K22_continuous_only_tau20000_strat.npy"
                          )[:, 1].astype(np.float64)
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr

    print("\n=== d17 Path-B K=25 sweep (vs PRIMARY d16 cont_only K=25 OOF 0.95121) ===")
    final = dict(global_oof=auc_global, primary_lb_oof=PRIMARY_LB_OOF,
                 names=names, taus={})
    for tau in taus:
        oof = oofs[tau]
        tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho_primary, _ = spearmanr(tp, primary_test)
        new_pos = tp >= rare_thr
        flips_to_neg = int(np.sum(primary_pos & ~new_pos))
        flips_to_pos = int(np.sum(~primary_pos & new_pos))
        ratio = (min(flips_to_neg, flips_to_pos) /
                 max(flips_to_neg, flips_to_pos)
                 if max(flips_to_neg, flips_to_pos) > 0 else 1.0)
        d_oof_primary = (auc - PRIMARY_LB_OOF) * 1e4
        d_oof_global = (auc - auc_global) * 1e4
        print(f"\n  τ={tau}:")
        print(f"    Strat OOF: {auc:.5f}  Δ vs PRIMARY OOF: {d_oof_primary:+.2f}bp  "
              f"Δ vs K=25 global: {d_oof_global:+.2f}bp")
        print(f"    ρ vs PRIMARY: {rho_primary:.6f}")
        print(f"    flips: +→− {flips_to_neg}, −→+ {flips_to_pos}, "
              f"ratio {ratio:.3f}")
        np.save(ART / f"oof_d17_path_b_K25_full_merge_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d17_path_b_K25_full_merge_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        Path("submissions").mkdir(exist_ok=True)
        sub.to_csv(f"submissions/submission_d17_path_b_K25_full_merge_tau{tau}.csv",
                   index=False)
        final["taus"][str(tau)] = dict(
            oof=auc, delta_oof_primary_bp=float(d_oof_primary),
            delta_oof_global_bp=float(d_oof_global),
            rho_vs_primary=float(rho_primary),
            flips_to_neg=flips_to_neg, flips_to_pos=flips_to_pos,
            flip_ratio=float(ratio),
        )

    final["wall_s"] = time.time() - t0
    (ART / "d17_path_b_K25_full_merge_results.json").write_text(
        json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d17_path_b_K25_full_merge_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
