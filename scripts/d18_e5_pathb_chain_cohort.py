"""d18 E5 — Path-B hier-meta with chain_total_ll as a NEW cohort axis.

Per friction `path-b-amp-only-fires-on-meta-arch-not-base-add`, base-adds
realise ~1.0× amp; meta-arch redesigns realise 6-11.6×. d18 chain features
give us a NEW axis: per-row orig-DGP-likelihood quintile. Combined with
the existing Compound × Stint segmentation gives 5 × 6 × 5 = 150 cells —
or simpler 2D cohorts: Compound × chain_LL_q5 (25 cells) and
Stint × chain_LL_q5 (30 cells).

This is the cleaner version of Phase-5 r̂_q5 cohort that was NULL on
K=14 sub-pool (friction
`path-b-on-pool-subset-conflates-cohort-axis-with-pool-size`).

Cohort axes tested (in order of expected EV):
  C1  Compound × chain_LL_q5     (25 cells, ~17k rows each)
  C2  Compound × Stint × chain_LL_q3   (45 cells, ~10k rows each — fewer-bin
       chain_LL because triple-cohort empties cells fast)
  C3  Stint × chain_LL_q5        (30 cells, ~14k rows each)

Pool: K=22 = K=21 + d18_chain_decomp (the strongest base-add).
τ-sweep {5k, 20k, 100k}.
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
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


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
    ap.add_argument("--cohort", choices=["c1", "c2", "c3"], required=True,
                    help="C1=Compound×llq5, C2=Compound×Stint×llq3, C3=Stint×llq5")
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    diag_train = pd.read_parquet("data/chain_decomp_features_train.parquet")
    diag_test = pd.read_parquet("data/chain_decomp_features_test.parquet")
    y = train[TARGET].astype(int).values

    pool_names = POOL_KEEP + TOP_3_D9 + FM_AB + ["d18_chain_decomp"]
    base_oofs, base_tests = [], []
    for name in pool_names:
        oo, te = load_pos(name)
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"K={len(pool_names)} (incl d18)  F shape oof={F_oof.shape}")

    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy")[:, 1].astype(np.float64)

    # Build cohort axis from chain_total_ll.
    ll_tr = diag_train["chain_total_ll"].values
    ll_te = diag_test["chain_total_ll"].values
    cmps_l = sorted(set(train["Compound"].astype(str).unique()) |
                    set(test["Compound"].astype(str).unique()))
    cm = {c: i for i, c in enumerate(cmps_l)}
    c_tr = train["Compound"].astype(str).map(cm).astype(int).values
    c_te = test["Compound"].astype(str).map(cm).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)

    if args.cohort == "c1":
        # Compound × chain_LL_q5
        edges = np.quantile(ll_tr, np.linspace(0, 1, 6))
        ll_q_tr = np.clip(np.searchsorted(edges, ll_tr, side="right") - 1, 0, 4)
        ll_q_te = np.clip(np.searchsorted(edges, ll_te, side="right") - 1, 0, 4)
        n_seg = len(cmps_l) * 5
        seg_train = c_tr * 5 + ll_q_tr
        seg_test = c_te * 5 + ll_q_te
        outname = "d18_e5_pathb_C1_cmp_llq5"
    elif args.cohort == "c2":
        edges = np.quantile(ll_tr, np.linspace(0, 1, 4))  # q3
        ll_q_tr = np.clip(np.searchsorted(edges, ll_tr, side="right") - 1, 0, 2)
        ll_q_te = np.clip(np.searchsorted(edges, ll_te, side="right") - 1, 0, 2)
        n_seg = len(cmps_l) * 6 * 3
        seg_train = c_tr * 18 + s_tr * 3 + ll_q_tr
        seg_test = c_te * 18 + s_te * 3 + ll_q_te
        outname = "d18_e5_pathb_C2_cmp_stint_llq3"
    else:  # c3
        edges = np.quantile(ll_tr, np.linspace(0, 1, 6))
        ll_q_tr = np.clip(np.searchsorted(edges, ll_tr, side="right") - 1, 0, 4)
        ll_q_te = np.clip(np.searchsorted(edges, ll_te, side="right") - 1, 0, 4)
        n_seg = 6 * 5
        seg_train = s_tr * 5 + ll_q_tr
        seg_test = s_te * 5 + ll_q_te
        outname = "d18_e5_pathb_C3_stint_llq5"

    sizes = np.bincount(seg_train, minlength=n_seg)
    print(f"  cohort={args.cohort} n_seg={n_seg}  ≥{MIN_ROWS} rows in "
          f"{int((sizes >= MIN_ROWS).sum())} cells "
          f"(min/med/max: {sizes[sizes>0].min()}/"
          f"{int(np.median(sizes[sizes>0]))}/{sizes.max()})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    print("\n--- Global LR baseline ---")
    meta_global = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc_global = float(roc_auc_score(y, meta_global))
    print(f"  Global meta OOF: {auc_global:.5f}")

    taus = [5000, 20000, 100000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}
    print(f"\n--- Path-B {args.cohort} hier-meta ---")
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

    print("\n=== Path-B sweep ===")
    final = dict(cohort=args.cohort, n_seg=n_seg, global_oof=auc_global, taus={})
    for tau in taus:
        oof = oofs[tau]; tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho, _ = spearmanr(tp, primary_test)
        d_oof = (auc - PRIMARY_S) * 1e4
        d_oof_global = (auc - auc_global) * 1e4
        print(f"  τ={tau}: OOF {auc:.5f}  Δ vs PRIMARY_S {d_oof:+.2f}bp  "
              f"Δ vs global LR {d_oof_global:+.2f}bp  ρ={rho:.5f}")
        np.save(ART / f"oof_{outname}_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_{outname}_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        final["taus"][str(tau)] = dict(oof=auc,
                                       delta_oof_PRIMARY_S_bp=float(d_oof),
                                       delta_oof_global_bp=float(d_oof_global),
                                       rho_vs_d9f=float(rho))
    final["wall_s"] = time.time() - t0
    (ART / f"{outname}_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/{outname}_results.json  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
