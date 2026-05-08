"""scripts/probe_path_b_alt_segs.py — alternative Path-B segmentations on K=27.

Forks d18_path_b.py at variant=k27_v4h1d_d16_d18_e2_f2 (the current
PRIMARY pool, OOF 0.95432). Tries two segmentation crosses NOT yet
tested at this pool:

  seg=cs_y     Compound × Stint × Year     (3-way; ~100 cells)
  seg=c_rp     Compound × RaceProgress-bin (5 × 5 = 25 cells)

The canonical PRIMARY uses Compound × Stint (5 × 6 = 30 cells, τ=100k).
Reference: state/calibration-ladder.md row "27-base v4+h1d+DGP-class".

τ-sweep {5000, 20000, 100000}. Q6 metric-aligned (LR-meta on logits is
BCE, row-AUC matches BCE ranking). Cost: ~5-7 min per segmentation.

Outputs scripts/artifacts/probe_path_b_alt_segs_<seg>_results.json +
oof_/test_ npy files for any τ that beats PRIMARY OOF.
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
MIN_ROWS, MAX_ITER = 1000, 500

POOL_KEEP = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
]
TOP_3_D9 = ["d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound"]
FM_AB = ["d9f_FM_A", "d9f_FM_B"]
EXTRAS_K27 = [
    "p1_single_cb_v4_gpu", "d17_h1d_yekenot_full",
    "d16_orig_continuous_only", "d18_chain_decomp",
    "d18_e2_preimage_knn", "d18_f2_constraint",
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


def load_pos(name):
    def _pos(p):
        a = np.load(p)
        return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)
    return (_pos(ART / f"oof_{name}_strat.npy"),
            _pos(ART / f"test_{name}_strat.npy"))


def make_segments(seg: str, train: pd.DataFrame, test: pd.DataFrame
                  ) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Return (seg_train, seg_test, n_seg, description)."""
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp).astype(int).values
    n_cats = len(cats)

    if seg == "cs_y":
        s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
        s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
        years = sorted(set(train["Year"].astype(int).unique()) |
                       set(test["Year"].astype(int).unique()))
        yr_map = {yr: i for i, yr in enumerate(years)}
        y_tr = train["Year"].astype(int).map(yr_map).astype(int).values
        y_te = test["Year"].astype(int).map(yr_map).astype(int).values
        n_years = len(years)
        seg_tr = (c_tr * 6 + s_tr) * n_years + y_tr
        seg_te = (c_te * 6 + s_te) * n_years + y_te
        n_seg = n_cats * 6 * n_years
        desc = f"Compound × Stint × Year ({n_cats} × 6 × {n_years} = {n_seg} cells)"
    elif seg == "c_rp":
        # 5-bin RaceProgress: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        rp_tr = np.clip(train["RaceProgress"].astype(float).values, 0, 0.999)
        rp_te = np.clip(test["RaceProgress"].astype(float).values, 0, 0.999)
        rb_tr = (rp_tr * 5).astype(int)
        rb_te = (rp_te * 5).astype(int)
        seg_tr = c_tr * 5 + rb_tr
        seg_te = c_te * 5 + rb_te
        n_seg = n_cats * 5
        desc = f"Compound × RaceProgress-bin ({n_cats} × 5 = {n_seg} cells)"
    else:
        raise ValueError(f"unknown seg {seg}")
    return seg_tr, seg_te, n_seg, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", choices=["cs_y", "c_rp"], required=True)
    args = ap.parse_args()
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    pool_names = POOL_KEEP + TOP_3_D9 + FM_AB + EXTRAS_K27
    print(f"K={len(pool_names)} bases  pool=K27_v4h1d_d16_d18_e2_f2")
    base_oofs, base_tests = [], []
    for name in pool_names:
        oo, te = load_pos(name)
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"F shape oof={F_oof.shape} test={F_test.shape}")

    # Reference: current PRIMARY (K=27, Compound × Stint, τ=100k)
    prim_oof = np.load(ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")[:, 1]
    prim_test = np.load(ART / "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")[:, 1]
    auc_primary = float(roc_auc_score(y, prim_oof))
    print(f"PRIMARY OOF (K=27 Compound × Stint τ=100k) = {auc_primary:.5f}")

    seg_train, seg_test, n_seg, desc = make_segments(args.seg, train, test)
    sizes = np.bincount(seg_train, minlength=n_seg)
    populated = int(np.sum(sizes >= MIN_ROWS))
    print(f"\nSegmentation: {desc}")
    print(f"  ≥{MIN_ROWS} rows in {populated}/{n_seg} segs; "
          f"min/med/max non-empty: {sizes[sizes>0].min()}/"
          f"{int(np.median(sizes[sizes>0]))}/{sizes.max()}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    print("\n--- Global LR baseline (K=27 LR-meta) ---")
    meta_global = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global[va] = lr.predict_proba(F_oof[va])[:, 1]
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s")
    auc_global = float(roc_auc_score(y, meta_global))
    print(f"  Global LR-meta OOF: {auc_global:.5f}")

    taus = [5000, 20000, 100000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    print(f"\n--- Path-B sweep on {desc} ---")
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
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s ({int(mask.sum())}/{n_seg} fit; {skipped} skipped)")

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

    print(f"\n=== {desc} sweep ===")
    final = dict(seg=args.seg, k=len(pool_names),
                 segmentation=desc, n_seg=n_seg, populated=populated,
                 global_oof=auc_global,
                 primary_oof_ref=auc_primary,
                 taus={})
    for tau in taus:
        oof = oofs[tau]; tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho_prim, _ = spearmanr(tp, prim_test)
        d_prim = (auc - auc_primary) * 1e4
        d_global = (auc - auc_global) * 1e4
        print(f"\n  τ={tau}:  OOF {auc:.5f}   "
              f"Δ vs PRIMARY (Compound × Stint, τ=100k): {d_prim:+.3f} bp   "
              f"Δ vs global K=27 LR-meta: {d_global:+.3f} bp   "
              f"ρ vs PRIMARY: {rho_prim:.6f}")
        # Save artifacts only if it lifts vs PRIMARY
        if d_prim >= 0.0:
            np.save(ART / f"oof_path_b_{args.seg}_tau{tau}_strat.npy",
                    np.column_stack([1 - oof, oof]))
            np.save(ART / f"test_path_b_{args.seg}_tau{tau}_strat.npy",
                    np.column_stack([1 - tp, tp]))
            sub = sample_sub.copy(); sub[TARGET] = tp
            Path("submissions").mkdir(exist_ok=True)
            sub.to_csv(f"submissions/submission_path_b_{args.seg}_tau{tau}.csv",
                       index=False)
            print(f"    -> saved (held; not submitted)")
        final["taus"][str(tau)] = dict(
            oof=auc, delta_oof_primary_bp=float(d_prim),
            delta_oof_global_bp=float(d_global),
            rho_vs_primary=float(rho_prim),
        )

    final["wall_s"] = time.time() - t0
    out = ART / f"probe_path_b_alt_segs_{args.seg}_results.json"
    out.write_text(json.dumps(final, indent=2))
    print(f"\n→ {out}  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
