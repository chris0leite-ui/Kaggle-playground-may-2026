"""Day-13 Path B finalizer — Stint-only hierarchical meta.

The full d13 sweep showed Stint segmentation gives the strongest
OOF lift (+0.88bp at τ=20000, ρ=0.9961; +0.86bp at τ=100000,
ρ=0.9984). Compound×Stint never finished (killed at fold 2).

This script reruns just Stint segmentation to:
  1. Save OOF and test predictions for τ=20000 + τ=100000
  2. Compute G3 rare-class flip ratio (d10d failed this hard)
  3. Decide submittability: ρ + flip ratio + pred-LB
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


def fit_lr_aug(F, y):
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
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

    base_oofs, base_tests = [], []
    for label, fname in POOL_KEEP + TOP_3_D9:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        base_oofs.append(oo); base_tests.append(te)
    for src_oof, src_test in [
        ("oof_d9f_FM_A_strat.npy", "test_d9f_FM_A_strat.npy"),
        ("oof_d9f_FM_B_strat.npy", "test_d9f_FM_B_strat.npy"),
    ]:
        oo = np.load(ART / src_oof)[:, 1].astype(np.float64)
        te = np.load(ART / src_test)[:, 1].astype(np.float64)
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    K = len(base_oofs)
    print(f"K={K} bases loaded; F shape {F_oof.shape}")

    # Stint segmentation (5 populated of 6)
    seg_train = np.clip(train["Stint"].astype(int).values, 0, 5)
    seg_test = np.clip(test["Stint"].astype(int).values, 0, 5)
    n_seg = 6

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    taus = [20000, 100000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = fit_lr_aug(F_oof[tr], y[tr])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            counts[s] = len(idx)
            if len(idx) < 200 or len(np.unique(y[tr][idx])) < 2:
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
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s")

    # Full-train fit for test predictions.
    print("  fitting full-train + test predictions…")
    w_global_full = fit_lr_aug(F_oof, y)
    W_local_full = np.zeros((n_seg, len(w_global_full)))
    counts_full = np.zeros(n_seg, dtype=np.int64)
    mask_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        counts_full[s] = len(idx)
        if len(idx) < 200 or len(np.unique(y[idx])) < 2:
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

    # Compute global LR meta for reference
    meta_global_oof = np.zeros(len(y))
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc_global = float(roc_auc_score(y, meta_global_oof))
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    test_global = lr_full.predict_proba(F_test)[:, 1]

    # Evaluate
    print(f"\n=== Stint hier-meta results ===")
    print(f"  Global meta OOF: {auc_global:.5f} (PRIMARY-equiv)")
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr

    final = dict(global_oof=auc_global, taus={})
    for tau in taus:
        oof = oofs[tau]
        tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho_test, _ = spearmanr(tp, primary_test)
        rho_oof, _ = spearmanr(oof, meta_global_oof)
        new_pos = tp >= rare_thr
        flips_to_pos = int(np.sum(~primary_pos & new_pos))
        flips_to_neg = int(np.sum(primary_pos & ~new_pos))
        ratio = (min(flips_to_pos, flips_to_neg) /
                 max(flips_to_pos, flips_to_neg)) \
                if max(flips_to_pos, flips_to_neg) > 0 else 1.0
        d_oof = (auc - auc_global) * 1e4
        print(f"\n  τ={tau}:")
        print(f"    OOF: {auc:.5f}  Δ global {d_oof:+.2f}bp")
        print(f"    ρ vs PRIMARY test: {rho_test:.6f}")
        print(f"    ρ vs PRIMARY OOF:  {rho_oof:.6f}")
        print(f"    rare-class flips: + → −  {flips_to_neg}, "
              f"− → +  {flips_to_pos}, ratio {ratio:.3f}")
        # Save
        np.save(ART / f"oof_d13_path_b_stint_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d13_path_b_stint_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_d13_path_b_stint_tau{tau}.csv",
                   index=False)
        # G3 verdict
        g3_pass = ratio >= 0.5
        rho_pass = rho_test >= 0.999
        oof_pass = d_oof >= 0.5
        verdict = "SLOT-WORTHY" if (g3_pass and oof_pass) else \
                  "MARGINAL" if oof_pass else "BELOW SLOT THRESHOLD"
        print(f"    G3 flip ratio ≥ 0.5: {'PASS' if g3_pass else 'FAIL'} "
              f"({ratio:.3f})")
        print(f"    OOF Δ ≥ +0.5bp: {'PASS' if oof_pass else 'FAIL'} "
              f"({d_oof:+.2f}bp)")
        print(f"    ρ ≥ 0.999 (TIE-band): {'PASS' if rho_pass else 'FAIL'} "
              f"({rho_test:.4f})")
        print(f"    Verdict: {verdict}")
        final["taus"][str(tau)] = dict(
            oof=auc, delta_oof_bp=float(d_oof),
            rho_vs_primary_test=float(rho_test),
            rho_vs_primary_oof=float(rho_oof),
            flips_to_neg=flips_to_neg, flips_to_pos=flips_to_pos,
            flip_ratio=float(ratio),
            g3_pass=bool(g3_pass), oof_pass=bool(oof_pass),
            rho_pass=bool(rho_pass), verdict=verdict,
        )

    final["wall_s"] = time.time() - t0
    (ART / "d13b_path_b_stint_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13b_path_b_stint_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
