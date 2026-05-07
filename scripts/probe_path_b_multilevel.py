"""scripts/probe_path_b_multilevel.py — T4a multi-level hierarchical shrinkage.

Per friction tag `path-b-amp-only-fires-on-meta-arch-not-base-add`, base-adds
get only ~1.4× LB amp. Meta-arch redesign is the only path to Path-B's
6-11.6× amp regime.

Mechanism:
  Standard Path B: per-segment LR shrunk to global LR by α = n/(n+τ).
  Multi-level: per-row weight is a 4-tier blend of LR coefficients:
    Level 0 (deepest): per-(Compound × Stint × Year)  — n ~ 5k-30k
    Level 1:           per-(Compound × Stint)         — n ~ 14k-150k (current PRIMARY level)
    Level 2:           per-Compound                   — n ~ 80k-200k
    Level 3 (global):  global                         — n = 439k

  Each level's coefficient gets weight α_k = n_k / (n_k + τ_k) where
  τ_k can be level-specific. Final blend:
    w_row = Σ_k (α_k / Σ_j α_j) * w_k

  This generalises Path B's 2-tier shrinkage to a true multi-level Bayesian
  hierarchy. Different segments at different levels get different effective
  weights based on local data volume.

Sweep: τ_0 (deepest) ∈ {5k, 20k, 100k} at fixed τ_1=20k τ_2=50k τ_3=∞.
Sweep: τ_1 ∈ {5k, 20k, 100k} at fixed τ_0=20k τ_2=50k τ_3=∞.

Best variant compared to current PRIMARY (d15b K=22 DAE Path B Compound×Stint
τ=20k LB 0.95059).
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
PRIMARY_OOF = ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"
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
EXTRAS = ["d15b_lgbm_dae_only"]   # K=22 with DAE (matches current PRIMARY pool)
SWEEPS = [
    # (tau_0, tau_1, tau_2)  — global is always full-shrinkage (effectively τ_3=∞)
    (5000, 20000, 50000),
    (20000, 20000, 50000),
    (100000, 20000, 50000),
    (20000, 5000, 50000),
    (20000, 100000, 50000),
]


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


def _fit_segment_lrs(F_tr, y_tr, seg_tr, n_segments, min_rows=MIN_ROWS):
    n_feat = F_tr.shape[1]
    W = np.zeros((n_segments, 1 + n_feat))
    counts = np.zeros(n_segments, dtype=np.int64)
    mask = np.zeros(n_segments, dtype=bool)
    for s in range(n_segments):
        idx = np.where(seg_tr == s)[0]
        counts[s] = len(idx)
        if len(idx) < min_rows or len(np.unique(y_tr[idx])) < 2:
            continue
        W[s] = _fit_lr_aug(F_tr[idx], y_tr[idx])
        mask[s] = True
    return W, counts, mask


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))
    print(f"PRIMARY (DAE-K22 Path B τ=20k): OOF {auc_primary:.5f}")

    base_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]
    base_tests = [_pos(ART / f"test_{b}_strat.npy") for b in K21_BASES]
    for ex in EXTRAS:
        base_oofs.append(_pos(ART / f"oof_{ex}_strat.npy"))
        base_tests.append(_pos(ART / f"test_{ex}_strat.npy"))
    F_oof = _expand(np.column_stack(base_oofs))
    F_test = _expand(np.column_stack(base_tests))
    K = len(base_oofs)
    print(f"Pool: K={K}; F shape {F_oof.shape}")

    # Build 3 segmentation levels
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    years = sorted(set(train["Year"].astype(int).unique()) |
                   set(test["Year"].astype(int).unique()))
    ymp = {y_: i for i, y_ in enumerate(years)}
    y_tr = train["Year"].astype(int).map(ymp).astype(int).values
    y_te = test["Year"].astype(int).map(ymp).astype(int).values

    # Level 0: Compound × Stint × Year   (5 × 6 × 4 = 120)
    seg0_tr = c_tr * 24 + s_tr * 4 + y_tr
    seg0_te = c_te * 24 + s_te * 4 + y_te
    n_seg0 = len(cats) * 6 * len(years)
    # Level 1: Compound × Stint   (5 × 6 = 30)  — the PRIMARY level
    seg1_tr = c_tr * 6 + s_tr
    seg1_te = c_te * 6 + s_te
    n_seg1 = len(cats) * 6
    # Level 2: Compound  (5)
    seg2_tr = c_tr
    seg2_te = c_te
    n_seg2 = len(cats)

    # Counts for each level (full train, used for α at OOF and test)
    cnt0 = np.bincount(seg0_tr, minlength=n_seg0).astype(np.float64)
    cnt1 = np.bincount(seg1_tr, minlength=n_seg1).astype(np.float64)
    cnt2 = np.bincount(seg2_tr, minlength=n_seg2).astype(np.float64)
    print(f"Level sizes: L0 {n_seg0} (pop≥1k {(cnt0>=1000).sum()}), "
          f"L1 {n_seg1} (pop≥1k {(cnt1>=1000).sum()}), L2 {n_seg2}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Per-fold: fit global, level-0/1/2 LRs; per-row blend
    print(f"\n--- Multi-level Path B sweep over {len(SWEEPS)} configs ---")
    oofs = {tuple(s): np.zeros(len(y)) for s in SWEEPS}
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = _fit_lr_aug(F_oof[tr], y[tr])
        W0, _, m0 = _fit_segment_lrs(F_oof[tr], y[tr], seg0_tr[tr], n_seg0,
                                      min_rows=500)   # L0 has smaller cells
        W1, _, m1 = _fit_segment_lrs(F_oof[tr], y[tr], seg1_tr[tr], n_seg1,
                                      min_rows=1000)
        W2, _, m2 = _fit_segment_lrs(F_oof[tr], y[tr], seg2_tr[tr], n_seg2,
                                      min_rows=1000)
        # For each tau-config, build per-row blend on val rows
        for (tau0, tau1, tau2) in SWEEPS:
            # alpha levels (use FULL counts; calibrated at test-time match)
            a0 = cnt0 / (cnt0 + tau0)
            a1 = cnt1 / (cnt1 + tau1)
            a2 = cnt2 / (cnt2 + tau2)
            for j_va in range(len(va)):
                i = va[j_va]
                s0 = seg0_tr[i]; s1 = seg1_tr[i]; s2 = seg2_tr[i]
                # Stack 4 levels; missing-mask sets weight to 0
                ws = []; alphas = []
                if m0[s0]: ws.append(W0[s0]); alphas.append(a0[s0])
                if m1[s1]: ws.append(W1[s1]); alphas.append(a1[s1])
                if m2[s2]: ws.append(W2[s2]); alphas.append(a2[s2])
                ws.append(w_global); alphas.append(1.0 - sum(alphas))
                if alphas[-1] < 0:
                    # normalize if local α already > 1 cumulatively
                    total = sum(alphas[:-1])
                    alphas = [a/total*0.95 for a in alphas[:-1]] + [0.05]
                w_blend = sum(a * w for a, w in zip(alphas, ws))
                F_aug = np.concatenate([[1.0], F_oof[i]])
                oofs[(tau0, tau1, tau2)][i] = 1.0 / (1.0 + np.exp(
                    -np.clip(F_aug @ w_blend, -30, 30)))
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s "
              f"(L0 fit {int(m0.sum())}/{n_seg0}, L1 {int(m1.sum())}/{n_seg1}, "
              f"L2 {int(m2.sum())}/{n_seg2})")

    # Full-train fit for test
    print("\n--- Full-train fit for test predictions ---")
    t_full = time.time()
    w_global_full = _fit_lr_aug(F_oof, y)
    W0_full, _, m0_full = _fit_segment_lrs(F_oof, y, seg0_tr, n_seg0, min_rows=500)
    W1_full, _, m1_full = _fit_segment_lrs(F_oof, y, seg1_tr, n_seg1, min_rows=1000)
    W2_full, _, m2_full = _fit_segment_lrs(F_oof, y, seg2_tr, n_seg2, min_rows=1000)
    test_preds = {tuple(s): np.zeros(len(test)) for s in SWEEPS}
    for (tau0, tau1, tau2) in SWEEPS:
        a0 = cnt0 / (cnt0 + tau0)
        a1 = cnt1 / (cnt1 + tau1)
        a2 = cnt2 / (cnt2 + tau2)
        for i in range(len(test)):
            s0 = seg0_te[i]; s1 = seg1_te[i]; s2 = seg2_te[i]
            ws = []; alphas = []
            if m0_full[s0]: ws.append(W0_full[s0]); alphas.append(a0[s0])
            if m1_full[s1]: ws.append(W1_full[s1]); alphas.append(a1[s1])
            if m2_full[s2]: ws.append(W2_full[s2]); alphas.append(a2[s2])
            ws.append(w_global_full); alphas.append(1.0 - sum(alphas))
            if alphas[-1] < 0:
                total = sum(alphas[:-1])
                alphas = [a/total*0.95 for a in alphas[:-1]] + [0.05]
            w_blend = sum(a * w for a, w in zip(alphas, ws))
            F_aug = np.concatenate([[1.0], F_test[i]])
            test_preds[(tau0, tau1, tau2)][i] = 1.0 / (1.0 + np.exp(
                -np.clip(F_aug @ w_blend, -30, 30)))
    print(f"  full-train wall: {time.time()-t_full:.1f}s")

    print(f"\n=== Multi-level Path B sweep results ===")
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    summary = {}
    for cfg in SWEEPS:
        oof = oofs[cfg]
        tp = test_preds[cfg]
        auc = float(roc_auc_score(y, oof))
        rho, _ = spearmanr(tp, primary_test)
        new_pos = tp >= rare_thr
        flips_neg = int(np.sum(primary_pos & ~new_pos))
        flips_pos = int(np.sum(~primary_pos & new_pos))
        d = (auc - auc_primary) * 1e4
        cfg_str = f"τ0={cfg[0]}_τ1={cfg[1]}_τ2={cfg[2]}"
        print(f"  {cfg_str}: OOF {auc:.5f}  Δ {d:+.2f}bp  ρ {rho:.6f}  flips {flips_neg}/{flips_pos}")
        np.save(ART / f"oof_path_b_multilevel_{cfg_str}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_path_b_multilevel_{cfg_str}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_path_b_multilevel_{cfg_str}.csv",
                   index=False)
        summary[cfg_str] = dict(oof=auc, delta_vs_primary_bp=float(d),
                                 rho_vs_primary=float(rho),
                                 flips_to_neg=flips_neg, flips_to_pos=flips_pos)
    summary["wall_s"] = time.time() - t0
    out = ART / "probe_path_b_multilevel.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out}  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
