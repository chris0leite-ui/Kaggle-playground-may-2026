"""Day-15 Branch B — Path B Compound x Stint hier-meta on K=22 (K=21 + d15b_dae_only).

Forked from scripts/d13e_path_b_compound_stint.py. The only change vs PRIMARY:
add d16_orig_continuous_only as the 22nd base in the pool. d16_orig_continuous_only
came in at min-meta +0.793bp OOF lift over K=21 baseline at rho 0.99547,
matching the d13 Stint hier-meta amp band (+0.86bp -> +7bp LB at 11.6x).

Sweep tau in {5000, 20000, 100000} (subset of d13e's 4-tau sweep, focused on
the band that produced d13e's PRIMARY at tau=20000).

Outputs:
  oof_d16_path_b_K22_continuous_only_tau{tau}_strat.npy
  test_d16_path_b_K22_continuous_only_tau{tau}_strat.npy
  d16_path_b_K22_continuous_only_results.json
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
PRIMARY_S = 0.95073   # d9f K=21 swap = global LR meta
MIN_ROWS = 1000
MAX_ITER = 500

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


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
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
    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy"
                            )[:, 1].astype(np.float64)
    primary_lb_winner_test = np.load(
        ART / "test_d13_path_b_stint_tau100000_strat.npy"
    )[:, 1].astype(np.float64)

    base_oofs, base_tests = [], []
    for label, fname in POOL_KEEP + TOP_3_D9:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        base_oofs.append(oo); base_tests.append(te)
    for src_oof, src_test in [
        ("oof_d9f_FM_A_strat.npy", "test_d9f_FM_A_strat.npy"),
        ("oof_d9f_FM_B_strat.npy", "test_d9f_FM_B_strat.npy"),
        # K=22 addition: Day-15 Branch B-GPU DAE-on-LGBM (latent-only variant).
        # Standalone OOF 0.94007, rho vs PRIMARY 0.9477 (most-diverse since
        # FM_A_53 d13a). K=22 min-meta +0.793bp at rho 0.99547 -- d13 Stint
        # hier-meta amp band (+0.86bp -> +7bp LB at 11.6x).
        ("oof_d16_orig_continuous_only_strat_2d.npy", "test_d16_orig_continuous_only_strat_2d.npy"),
    ]:
        oo = np.load(ART / src_oof)[:, 1].astype(np.float64)
        te = np.load(ART / src_test)[:, 1].astype(np.float64)
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"K={len(base_oofs)} bases; F shape {F_oof.shape}")

    # Compound × Stint segmentation
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
    populated_min1k = int(np.sum(sizes >= MIN_ROWS))
    print(f"Compound×Stint: n_seg={n_seg}, ≥{MIN_ROWS} rows in "
          f"{populated_min1k} segments")
    print(f"  segment sizes: min/med/max = "
          f"{sizes[sizes>0].min()}/{int(np.median(sizes[sizes>0]))}/"
          f"{sizes.max()}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Global LR baseline (matches PRIMARY)
    print("\n--- Global LR baseline ---")
    meta_global = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global[va] = lr.predict_proba(F_oof[va])[:, 1]
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s")
    auc_global = float(roc_auc_score(y, meta_global))
    print(f"  Global meta OOF: {auc_global:.5f}")

    # Hier-meta sweep (focused on d13e PRIMARY tau=20000 band)
    taus = [5000, 20000, 100000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    print("\n--- Compound×Stint hier-meta ---")
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
              f"({int(mask.sum())}/{n_seg} segments fit; "
              f"{skipped} skipped < {MIN_ROWS} rows)")

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

    # Evaluate each tau
    print("\n=== Compound × Stint sweep ===")
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    winner_pos = primary_lb_winner_test >= rare_thr
    final = dict(global_oof=auc_global, taus={})
    for tau in taus:
        oof = oofs[tau]
        tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho_d9f, _ = spearmanr(tp, primary_test)
        rho_winner, _ = spearmanr(tp, primary_lb_winner_test)
        new_pos = tp >= rare_thr
        flips_d9f_to_neg = int(np.sum(primary_pos & ~new_pos))
        flips_d9f_to_pos = int(np.sum(~primary_pos & new_pos))
        flips_w_to_neg = int(np.sum(winner_pos & ~new_pos))
        flips_w_to_pos = int(np.sum(~winner_pos & new_pos))
        ratio_d9f = (min(flips_d9f_to_neg, flips_d9f_to_pos) /
                      max(flips_d9f_to_neg, flips_d9f_to_pos)
                      if max(flips_d9f_to_neg, flips_d9f_to_pos) > 0 else 1.0)
        d_oof_d9f = (auc - PRIMARY_S) * 1e4
        d_oof_global = (auc - auc_global) * 1e4
        print(f"\n  τ={tau}:")
        print(f"    Strat OOF: {auc:.5f}  Δ vs d9f: {d_oof_d9f:+.2f}bp  "
              f"Δ vs global: {d_oof_global:+.2f}bp")
        print(f"    ρ vs d9f K=21:           {rho_d9f:.6f}")
        print(f"    ρ vs d13 Stint τ=100k:   {rho_winner:.6f}")
        print(f"    flips vs d9f:    +→− {flips_d9f_to_neg}, "
              f"−→+ {flips_d9f_to_pos}, ratio {ratio_d9f:.3f}")
        print(f"    flips vs Stint:  +→− {flips_w_to_neg}, "
              f"−→+ {flips_w_to_pos}")
        # Save (d15b namespace; do NOT overwrite d13e PRIMARY artifacts)
        np.save(ART / f"oof_d16_path_b_K22_continuous_only_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d16_path_b_K22_continuous_only_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        final["taus"][str(tau)] = dict(
            oof=auc, delta_oof_d9f_bp=float(d_oof_d9f),
            delta_oof_global_bp=float(d_oof_global),
            rho_vs_d9f=float(rho_d9f),
            rho_vs_stint_winner=float(rho_winner),
            flips_d9f_to_neg=flips_d9f_to_neg,
            flips_d9f_to_pos=flips_d9f_to_pos,
            flip_ratio_d9f=float(ratio_d9f),
            flips_winner_to_neg=flips_w_to_neg,
            flips_winner_to_pos=flips_w_to_pos,
        )

    final["wall_s"] = time.time() - t0
    (ART / "d16_path_b_K22_continuous_only_results.json").write_text(
        json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d16_path_b_K22_continuous_only_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
