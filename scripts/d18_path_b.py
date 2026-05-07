"""d18 Path-B Compound × Stint hier-meta on K=22 / K=23 with d18_chain_decomp.

A0 — K=22 = K=21 + d18_chain_decomp
A1 — K=23 = K=21 + d16_orig_continuous_only + d18_chain_decomp

Forked from `scripts/d16_path_b_K22_continuous_only.py`. τ-sweep {5k, 20k,
100k}; same Compound × Stint segmentation. Reports OOF, ρ vs d9f-K21-swap
PRIMARY, flips top-1%.
"""
from __future__ import annotations

import argparse
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
PRIMARY_S = 0.95073
MIN_ROWS, MAX_ITER = 1000, 500

POOL_KEEP = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
]
TOP_3_D9 = ["d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound"]
FM_AB = ["d9f_FM_A", "d9f_FM_B"]


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


def load_pos(name):
    def _pos(p):
        a = np.load(p)
        return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)
    return (_pos(ART / f"oof_{name}_strat.npy"),
            _pos(ART / f"test_{name}_strat.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["k22_d18", "k23_d16_d18"],
                    required=True)
    args = ap.parse_args()
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    pool_names = POOL_KEEP + TOP_3_D9 + FM_AB
    if args.variant == "k22_d18":
        extras = ["d18_chain_decomp"]
        outname = "d18_path_b_K22_d18"
    elif args.variant == "k23_d16_d18":
        extras = ["d16_orig_continuous_only", "d18_chain_decomp"]
        outname = "d18_path_b_K23_d16_d18"
    pool_names = pool_names + extras
    print(f"variant={args.variant}  K={len(pool_names)} bases  "
          f"extras={extras}")

    base_oofs, base_tests = [], []
    for name in pool_names:
        oo, te = load_pos(name)
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"F shape oof={F_oof.shape} test={F_test.shape}")

    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy")[:, 1].astype(np.float64)

    # Compound × Stint segmentation
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_train = c_tr * 6 + s_tr
    seg_test = c_te * 6 + s_te

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

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

    taus = [5000, 20000, 100000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    print("\n--- Compound × Stint hier-meta ---")
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
              f"({int(mask.sum())}/{n_seg} fit)")

    print("\n--- Full-train test predictions ---")
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

    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    print("\n=== Compound × Stint sweep ===")
    final = dict(variant=args.variant, k=len(pool_names),
                 global_oof=auc_global, taus={})
    for tau in taus:
        oof = oofs[tau]; tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho, _ = spearmanr(tp, primary_test)
        new_pos = tp >= rare_thr
        f_neg = int(np.sum(primary_pos & ~new_pos))
        f_pos = int(np.sum(~primary_pos & new_pos))
        d_oof = (auc - PRIMARY_S) * 1e4
        d_oof_global = (auc - auc_global) * 1e4
        print(f"\n  τ={tau}:")
        print(f"    Strat OOF: {auc:.5f}  Δ vs PRIMARY_S 0.95073: {d_oof:+.2f} bp  "
              f"Δ vs global LR: {d_oof_global:+.2f} bp")
        print(f"    ρ vs d9f K=21 swap: {rho:.6f}")
        print(f"    flips top-1%: +→− {f_neg}, −→+ {f_pos}")
        np.save(ART / f"oof_{outname}_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_{outname}_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        final["taus"][str(tau)] = dict(
            oof=auc, delta_oof_PRIMARY_S_bp=float(d_oof),
            delta_oof_global_bp=float(d_oof_global),
            rho_vs_d9f=float(rho), flips_top1_to_neg=f_neg,
            flips_top1_to_pos=f_pos)
    final["wall_s"] = time.time() - t0
    (ART / f"{outname}_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/{outname}_results.json  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
