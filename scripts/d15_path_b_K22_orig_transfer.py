"""scripts/d15_path_b_K22_orig_transfer.py — hier-meta on K=22 with orig_transfer.

Today's K=22 LR-meta + d15_orig_transfer landed LB 0.95039 (-10bp regress
vs PRIMARY). Diagnosis: the regression is meta-arch confound (LR vs
hier-meta = ~14bp on this comp), not the base-add axis.

This probe isolates the base-add axis: same hier-meta architecture as
PRIMARY (Compound × Stint, τ=20k) but applied to a K=22 pool =
K=21 + d15_orig_transfer.

If OOF beats PRIMARY's 0.95083 → submit (slot decision: PI).
If OOF ≤ 0.95083 → orig_transfer fully falsified for this comp.

~10 min CPU.
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
TAU = 20000     # PRIMARY's tau

# Same K=21 pool as d13e (the PRIMARY)
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
    ("FM_A", "d9f_FM_A"),
    ("FM_B", "d9f_FM_B"),
]
NEW_22ND = [
    ("orig_transfer", "d15_orig_transfer"),
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    P_clip = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(P_clip / (1 - P_clip))
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

    # PRIMARY (d13e) test predictions for ρ
    primary_test = np.load(
        ART / "test_d13e_compound_stint_tau20000_strat.npy"
    )[:, 1].astype(np.float64)

    # Build pools: K=21 (PRIMARY pool) and K=22 (+ orig_transfer)
    bases_K21 = POOL_KEEP + TOP_3_D9 + FM_PAIR
    bases_K22 = bases_K21 + NEW_22ND

    def load_pool(spec):
        oofs, tests = [], []
        for label, fname in spec:
            oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
            te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
            oofs.append(oo); tests.append(te)
        return np.column_stack(oofs), np.column_stack(tests)

    P21_oof, P21_test = load_pool(bases_K21)
    P22_oof, P22_test = load_pool(bases_K22)
    F21_oof = expand(P21_oof); F21_test = expand(P21_test)
    F22_oof = expand(P22_oof); F22_test = expand(P22_test)
    print(f"K=21 pool: {P21_oof.shape}  K=22 pool: {P22_oof.shape}")

    # Compound × Stint segmentation (same as d13e)
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
    n_pop = int(np.sum(sizes >= MIN_ROWS))
    print(f"Segments ≥{MIN_ROWS}: {n_pop}/{n_seg}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    results = {}
    for label, F_oof, F_test, K in [
        ("K21_d13e_repro", F21_oof, F21_test, 21),
        ("K22_orig_transfer", F22_oof, F22_test, 22),
    ]:
        print(f"\n=== {label} (K={K}) hier-meta τ={TAU} ===")
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
            print(f"  fold {fold}: {time.time()-t_fold:.1f}s "
                  f"({int(mask.sum())} segments fit)")

        # Full-train fit for test predictions
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
        test_pred = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx = np.where(seg_test == s)[0]
            w = W_shrunk[s] if mask_full[s] else w_global_full
            test_pred[idx] = predict_aug(F_test[idx], w)

        auc = float(roc_auc_score(y, oof_meta))
        rho, _ = spearmanr(test_pred, primary_test)
        # Flip stats
        rare_thr = float(np.quantile(primary_test, 0.99))
        primary_pos = primary_test >= rare_thr
        new_pos = test_pred >= rare_thr
        flips_to_neg = int(np.sum(primary_pos & ~new_pos))
        flips_to_pos = int(np.sum(~primary_pos & new_pos))
        ratio = (min(flips_to_neg, flips_to_pos) /
                 max(flips_to_neg, flips_to_pos)
                 if max(flips_to_neg, flips_to_pos) > 0 else 1.0)

        print(f"\n  OOF AUC:           {auc:.5f}")
        print(f"  ρ vs PRIMARY:      {rho:.6f}")
        print(f"  flip ratio (top1%): {ratio:.3f} (+→− {flips_to_neg}, −→+ {flips_to_pos})")

        # Save artifacts
        np.save(ART / f"oof_d15_path_b_{label}_strat.npy",
                np.column_stack([1 - oof_meta, oof_meta]))
        np.save(ART / f"test_d15_path_b_{label}_strat.npy",
                np.column_stack([1 - test_pred, test_pred]))
        sub = sample_sub.copy(); sub[TARGET] = test_pred
        sub_path = f"submissions/submission_d15_path_b_{label}.csv"
        sub.to_csv(sub_path, index=False)
        results[label] = dict(
            K=K, oof=auc, rho_vs_primary=float(rho),
            flips_to_neg=flips_to_neg, flips_to_pos=flips_to_pos,
            flip_ratio=float(ratio), submission=sub_path,
        )

    # Compare
    print("\n=== Summary (Compound×Stint hier-meta τ=20k) ===")
    print(f"  K=21 (d13e repro):    OOF {results['K21_d13e_repro']['oof']:.5f}  "
          f"(PRIMARY benchmark = 0.95083)")
    print(f"  K=22 (+orig_transfer): OOF {results['K22_orig_transfer']['oof']:.5f}  "
          f"ρ vs PRIMARY = {results['K22_orig_transfer']['rho_vs_primary']:.5f}")
    delta_bp = (results['K22_orig_transfer']['oof']
                - results['K21_d13e_repro']['oof']) * 1e4
    print(f"  Δ K22-vs-K21:         {delta_bp:+.3f} bp")

    results["delta_bp_vs_K21"] = float(delta_bp)
    results["wall_s"] = time.time() - t0
    (ART / "d15_path_b_K22_orig_transfer_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n→ wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
