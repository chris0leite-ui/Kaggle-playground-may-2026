"""Day-18 — Path-B hier-meta with Compound × Year segmentation on K=24.

Adapted from d13e_path_b_compound_stint.py. Two changes:
  1. Segmentation cross: (Compound, Stint) → (Compound, Year)
  2. Pool: K=21 → K=24 (add d16_orig_continuous_only +
     p1_single_cb_v3_gpu + d17_h1d_yekenot_full)

Motivation: LR-leverage Probe 5 found per-(Compound × Year) mega LR
gives +60.8 bp standalone over global mega (0.92776 → 0.93385). The
2023-cohort cells dominate: MEDIUM_2023 +1081 bp, SOFT_2023 +865 bp,
HARD_2023 +725 bp. Pooled-coef LR cannot represent cohort-conditional
DGP shifts; Path-B hier-meta is the principled, leak-free way to
capture the same signal at meta-level on K=24.

Per friction `path-b-amp-only-fires-on-meta-arch-not-base-add`,
segmentation cross changes ARE meta-arch redesigns (amp-eligible).
d14 sweep tested Year on K=21 (NULL); this is Year on K=24 with
v4-class bases that have year-conditional structure d14 lacked.

τ ∈ {5000, 20000, 100000, 500000}. ~15 min wall.
Spec: audit/2026-05-07-pathb-compound-year-probe-plan.md.
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

K24_BASES = [
    # K=21 PRIMARY pool
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
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
    ("FM_A", "d9f_FM_A"), ("FM_B", "d9f_FM_B"),
    # +3 Day-17 PM bases
    ("d16_cont_only", "d16_orig_continuous_only"),
    ("p1_cb_v3", "p1_single_cb_v3_gpu"),
    ("h1d_yekenot", "d17_h1d_yekenot_full"),
]


def _pos_load(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel().astype(np.float64)


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


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    # Reference for ρ + flip diagnostics: current PRIMARY (d17 K=24+h1d)
    prim_oof = _pos_load(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    prim_test = _pos_load(ART / "test_d17_K24_d18pool_h1d_strat.npy")
    # Day-17 PM PRIMARY (LB 0.95354) — d17 K=23 v4+h1d Path-B C×S τ=100k
    winner_path = ART / "test_d17_path_b_K23_v4_h1d_tau100000_strat.npy"
    primary_lb_winner_test = (_pos_load(winner_path)
                              if winner_path.exists() else prim_test)

    base_oofs, base_tests = [], []
    for label, fname in K24_BASES:
        oo = _pos_load(ART / f"oof_{fname}_strat.npy")
        te = _pos_load(ART / f"test_{fname}_strat.npy")
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"K={len(base_oofs)} bases; F shape {F_oof.shape}")

    # Compound × Year segmentation (replacing Compound × Stint)
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    years = sorted(set(train["Year"].astype(int).unique()) |
                   set(test["Year"].astype(int).unique()))
    yr_map = {yr: i for i, yr in enumerate(years)}
    y_tr_int = train["Year"].astype(int).map(yr_map).astype(int).values
    y_te_int = test["Year"].astype(int).map(yr_map).astype(int).values
    n_cats = len(cats)
    n_years = len(years)
    seg_train = c_tr * n_years + y_tr_int
    seg_test = c_te * n_years + y_te_int
    n_seg = n_cats * n_years
    sizes = np.bincount(seg_train, minlength=n_seg)
    populated_min1k = int(np.sum(sizes >= MIN_ROWS))
    print(f"Compound×Year: n_seg={n_seg} ({n_cats} compounds × "
          f"{n_years} years); ≥{MIN_ROWS} rows in {populated_min1k} segs")
    print(f"  segment sizes: min/med/max = "
          f"{sizes[sizes>0].min()}/{int(np.median(sizes[sizes>0]))}/"
          f"{sizes.max()}")
    print(f"  Compounds: {cats}")
    print(f"  Years: {years}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Global LR baseline
    print("\n--- Global LR baseline (K=24 LR-meta) ---")
    meta_global = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global[va] = lr.predict_proba(F_oof[va])[:, 1]
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s")
    auc_global = float(roc_auc_score(y, meta_global))
    print(f"  Global meta OOF: {auc_global:.5f}")

    # Hier-meta sweep
    taus = [5000, 20000, 100000, 500000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    print("\n--- Compound×Year hier-meta sweep ---")
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
              f"({int(mask.sum())}/{n_seg} segs fit; "
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
    print("\n=== Compound × Year sweep ===")
    rare_thr = float(np.quantile(prim_test, 0.99))
    prim_pos = prim_test >= rare_thr
    winner_pos = primary_lb_winner_test >= rare_thr
    auc_prim_oof = float(roc_auc_score(y, prim_oof))
    print(f"  Reference: PRIMARY OOF (d17_K24_d18pool_h1d) = {auc_prim_oof:.5f}")
    final = dict(global_oof=auc_global,
                 primary_oof_ref=auc_prim_oof,
                 taus={})
    for tau in taus:
        oof = oofs[tau]
        tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho_prim, _ = spearmanr(tp, prim_test)
        rho_winner, _ = spearmanr(tp, primary_lb_winner_test)
        new_pos = tp >= rare_thr
        f_prim_neg = int(np.sum(prim_pos & ~new_pos))
        f_prim_pos = int(np.sum(~prim_pos & new_pos))
        f_w_neg = int(np.sum(winner_pos & ~new_pos))
        f_w_pos = int(np.sum(~winner_pos & new_pos))
        ratio_prim = (min(f_prim_neg, f_prim_pos) /
                       max(f_prim_neg, f_prim_pos)
                       if max(f_prim_neg, f_prim_pos) > 0 else 1.0)
        d_oof_prim = (auc - auc_prim_oof) * 1e4
        d_oof_global = (auc - auc_global) * 1e4
        print(f"\n  τ={tau}:")
        print(f"    Strat OOF: {auc:.5f}  Δ vs PRIMARY OOF: {d_oof_prim:+.2f}bp  "
              f"Δ vs global K24-meta: {d_oof_global:+.2f}bp")
        print(f"    ρ vs PRIMARY:                  {rho_prim:.6f}")
        print(f"    ρ vs Day-17 PM Path-B winner:  {rho_winner:.6f}")
        print(f"    flips vs PRIMARY:    +→− {f_prim_neg}, "
              f"−→+ {f_prim_pos}, ratio {ratio_prim:.3f}")
        print(f"    flips vs Day-17 winner:  +→− {f_w_neg}, "
              f"−→+ {f_w_pos}")
        # Save artifacts
        np.save(ART / f"oof_d18_path_b_compound_year_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d18_path_b_compound_year_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub_path = Path("submissions")
        sub_path.mkdir(exist_ok=True)
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(sub_path / f"submission_d18_path_b_compound_year_tau{tau}.csv",
                   index=False)
        final["taus"][str(tau)] = dict(
            oof=auc, delta_oof_primary_bp=float(d_oof_prim),
            delta_oof_global_bp=float(d_oof_global),
            rho_vs_primary=float(rho_prim),
            rho_vs_winner=float(rho_winner),
            flips_prim_to_neg=f_prim_neg,
            flips_prim_to_pos=f_prim_pos,
            flip_ratio_prim=float(ratio_prim),
            flips_winner_to_neg=f_w_neg,
            flips_winner_to_pos=f_w_pos,
        )

    final["wall_s"] = time.time() - t0
    (ART / "d18_path_b_compound_year_results.json").write_text(
        json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d18_path_b_compound_year_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
