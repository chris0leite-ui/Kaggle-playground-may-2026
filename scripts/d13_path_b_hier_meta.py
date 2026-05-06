"""Day-13 Path B — empirical-Bayes hierarchical LR meta.

Synthesis of d10b/c/d:
  d10b/c found that under leak-blocking GKF, FM_B's L1 weight rises
  from 0.138 to 6.96 (mid-pack to dominant). d10d showed a fully
  GKF-fit meta over-corrects (rare-class flip ratio 0.001) — the
  GKF blinds GBDT bases to row-specific extremes that *are* genuine
  on the iid test set. The right answer is **per-segment partial
  pooling**: for common segments (where leakage piggybacking is
  large), keep GBDT weights high; for rare/edge segments (where
  GBDT Strat OOF is mostly leakage), shift weight to FM.

Architecture (Yao 2021 style, empirical Bayes):
  For segment s ∈ S, fit LR on rows in s → w_s_local
  Fit LR on all rows → w_global
  Shrink:  w_s = (n_s · w_s_local + τ · w_global) / (n_s + τ)
  Apply:   row i in segment s gets ŷ = sigmoid(w_s · x_i)

τ controls the shrinkage strength. τ → 0 recovers per-segment LR
(unstable on small segments). τ → ∞ recovers global LR (= PRIMARY).
Sweet spot: τ such that segment with n_s ≈ 5,000 rows gets ~50/50
local-vs-global blend.

Segments swept: (Compound), (Stint_clipped), (Compound × Stint_clipped),
(Year × Compound). All 5-fold Strat-CV — Strat is the LB proxy per
U3 i.i.d. test; we are not adding leakage protection at the meta
level (that's d10d, which gate-failed). We are adding **per-segment
weighting**, which respects the existing Strat partition.

Pool: K=21 = PRIMARY (d9f K=21 swap) — POOL_KEEP + TOP_3_D9 +
FM_A + FM_B.

EV: +1-3bp per HANDOVER Path B. ~5-15 min wall.
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
PRIMARY_LB = 0.95031   # d9f K=21 swap
RHO_TIE = 0.999

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


def fit_global_lr(F_tr, y_tr):
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr.fit(F_tr, y_tr)
    # Concatenate intercept and coef so shrinkage applies to bias too.
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def fit_segment_lrs(F_tr, y_tr, seg_tr, n_segments, min_rows=200):
    """Fit per-segment LR on rows where seg_tr == s.

    Returns:
      W_local[s] = (1 + n_features,) weight vector (intercept first)
      counts[s]  = number of train rows in segment s
      mask[s]    = True if segment had >= min_rows train rows; else
                   we fall back to global only.
    """
    n_feat = F_tr.shape[1]
    W_local = np.zeros((n_segments, 1 + n_feat), dtype=np.float64)
    counts = np.zeros(n_segments, dtype=np.int64)
    mask = np.zeros(n_segments, dtype=bool)
    for s in range(n_segments):
        idx = np.where(seg_tr == s)[0]
        counts[s] = len(idx)
        if len(idx) < min_rows or len(np.unique(y_tr[idx])) < 2:
            continue
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_tr[idx], y_tr[idx])
        W_local[s] = np.concatenate([lr.intercept_, lr.coef_.ravel()])
        mask[s] = True
    return W_local, counts, mask


def predict_hier(F_va, seg_va, W_shrunk, mask, w_global):
    """For each val row, use its segment's shrunk weights if mask[s]
    else fall back to global."""
    n_va = F_va.shape[0]
    logits = np.zeros(n_va, dtype=np.float64)
    F_aug = np.column_stack([np.ones(n_va), F_va])
    for s in np.unique(seg_va):
        idx = np.where(seg_va == s)[0]
        w = W_shrunk[s] if mask[s] else w_global
        logits[idx] = F_aug[idx] @ w
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def predict_global(F_va, w_global):
    F_aug = np.column_stack([np.ones(len(F_va)), F_va])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w_global, -30, 30)))


def hier_oof(F_oof, F_test, y, segments_train, segments_test,
              n_segments, taus, splits):
    """Full hierarchical EB stacking with per-segment LR + shrinkage.

    Returns oof[tau] and test[tau] for each tau in taus (one fit
    of per-segment LRs reused across taus — only shrinkage differs).
    """
    n_train = len(y)
    oofs = {tau: np.zeros(n_train, dtype=np.float64) for tau in taus}
    test_avgs = {tau: np.zeros(len(F_test), dtype=np.float64)
                 for tau in taus}

    # Per-fold global LR for OOF prediction.
    for fold, (tr, va) in enumerate(splits):
        t0 = time.time()
        w_global = fit_global_lr(F_oof[tr], y[tr])
        W_local, counts, mask = fit_segment_lrs(
            F_oof[tr], y[tr], segments_train[tr], n_segments)
        for tau in taus:
            n_local = counts.astype(np.float64)
            alpha = n_local / (n_local + tau)
            W_shrunk = (alpha[:, None] * W_local +
                        (1 - alpha[:, None]) * w_global[None, :])
            oofs[tau][va] = predict_hier(F_oof[va], segments_train[va],
                                          W_shrunk, mask, w_global)
        print(f"  fold {fold}: {time.time()-t0:.1f}s "
              f"(global+{int(mask.sum())}/{n_segments} segments)")

    # Full-train fit for test predictions.
    w_global_full = fit_global_lr(F_oof, y)
    W_local_full, counts_full, mask_full = fit_segment_lrs(
        F_oof, y, segments_train, n_segments)
    for tau in taus:
        n_local = counts_full.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local_full +
                    (1 - alpha[:, None]) * w_global_full[None, :])
        test_avgs[tau] = predict_hier(F_test, segments_test,
                                       W_shrunk, mask_full, w_global_full)
    return oofs, test_avgs


def make_segments(train, test, kind):
    """Return (seg_train, seg_test, n_segments, name)."""
    def stint_clip(s):
        return np.clip(s.astype(int).values, 0, 5)
    if kind == "compound":
        cats = sorted(set(train["Compound"].astype(str).unique()) |
                      set(test["Compound"].astype(str).unique()))
        mp = {c: i for i, c in enumerate(cats)}
        st = train["Compound"].astype(str).map(mp).astype(int).values
        ste = test["Compound"].astype(str).map(mp).astype(int).values
        return st, ste, len(cats), "Compound"
    elif kind == "stint":
        st = stint_clip(train["Stint"])
        ste = stint_clip(test["Stint"])
        return st, ste, 6, "Stint_clip5"
    elif kind == "compound_stint":
        cats = sorted(set(train["Compound"].astype(str).unique()) |
                      set(test["Compound"].astype(str).unique()))
        mp = {c: i for i, c in enumerate(cats)}
        c_tr = train["Compound"].astype(str).map(mp).astype(int).values
        c_te = test["Compound"].astype(str).map(mp).astype(int).values
        s_tr = stint_clip(train["Stint"])
        s_te = stint_clip(test["Stint"])
        st = c_tr * 6 + s_tr
        ste = c_te * 6 + s_te
        return st, ste, len(cats) * 6, f"Compound×Stint ({len(cats)*6})"
    elif kind == "year_compound":
        years = sorted(set(train["Year"].astype(int).unique()) |
                       set(test["Year"].astype(int).unique()))
        ymp = {y: i for i, y in enumerate(years)}
        cats = sorted(set(train["Compound"].astype(str).unique()) |
                      set(test["Compound"].astype(str).unique()))
        cmp = {c: i for i, c in enumerate(cats)}
        y_tr = train["Year"].astype(int).map(ymp).astype(int).values
        y_te = test["Year"].astype(int).map(ymp).astype(int).values
        c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
        c_te = test["Compound"].astype(str).map(cmp).astype(int).values
        st = y_tr * len(cats) + c_tr
        ste = y_te * len(cats) + c_te
        return st, ste, len(years) * len(cats), \
               f"Year×Compound ({len(years)*len(cats)})"
    raise ValueError(kind)


def main():
    t_total = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy"
                            )[:, 1].astype(np.float64)

    # Build K=21 = POOL_KEEP (16) + TOP_3_D9 (3) + FM_A + FM_B
    print("Loading K=21 PRIMARY pool (Strat OOFs)…")
    base_oofs, base_tests, names = [], [], []
    for label, fname in POOL_KEEP + TOP_3_D9:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        base_oofs.append(oo); base_tests.append(te); names.append(label)
    for label, src_oof, src_test in [
        ("FM_A", "oof_d9f_FM_A_strat.npy", "test_d9f_FM_A_strat.npy"),
        ("FM_B", "oof_d9f_FM_B_strat.npy", "test_d9f_FM_B_strat.npy"),
    ]:
        oo = np.load(ART / src_oof)[:, 1].astype(np.float64)
        te = np.load(ART / src_test)[:, 1].astype(np.float64)
        base_oofs.append(oo); base_tests.append(te); names.append(label)
    K = len(names)
    print(f"  K={K} bases loaded")
    P_oof = np.column_stack(base_oofs)
    P_test = np.column_stack(base_tests)
    F_oof = expand(P_oof); F_test = expand(P_test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Global-LR baseline = PRIMARY behavior.
    meta_global_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc_global = float(roc_auc_score(y, meta_global_oof))
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    test_global = lr_full.predict_proba(F_test)[:, 1]
    rho_global, _ = spearmanr(test_global, primary_test)
    print(f"\n=== Global LR meta (PRIMARY behavior) ===")
    print(f"  Strat OOF: {auc_global:.5f}  ρ vs PRIMARY: {rho_global:.6f}")

    # Hierarchical EB sweep.
    taus = [50, 200, 1000, 5000, 20000, 100000]
    results = {}
    for kind in ["compound", "stint", "compound_stint", "year_compound"]:
        seg_tr, seg_te, n_seg, name = make_segments(train, test, kind)
        sizes = np.bincount(seg_tr, minlength=n_seg)
        sized = sizes[sizes > 0]
        print(f"\n--- Segments: {name} (n={n_seg}, "
              f"populated={len(sized)}, "
              f"min/median/max rows: {sized.min()}/{int(np.median(sized))}/"
              f"{sized.max()}) ---")
        oofs_tau, tests_tau = hier_oof(F_oof, F_test, y, seg_tr, seg_te,
                                        n_seg, taus, splits)
        for tau in taus:
            auc = float(roc_auc_score(y, oofs_tau[tau]))
            rho, _ = spearmanr(tests_tau[tau], primary_test)
            d_global = (auc - auc_global) * 1e4
            d_lb = predicted_lb_delta(d_global, rho)
            print(f"  τ={tau:>6}: OOF {auc:.5f}  Δglobal {d_global:+.2f}bp  "
                  f"ρ vs PRIMARY {rho:.6f}  pred-LB Δ {d_lb:+.2f}bp")
            results[(kind, tau)] = dict(
                segment=name, tau=tau, oof=auc,
                delta_global_bp=float(d_global),
                rho_vs_primary=float(rho),
                pred_lb_delta_bp=float(d_lb))

    # Pick best variant per pred-LB delta with ρ ≥ 0.99 gate.
    valid = [(k, v) for k, v in results.items() if v["rho_vs_primary"] >= 0.99]
    if valid:
        best_key, best_v = max(valid, key=lambda kv: kv[1]["pred_lb_delta_bp"])
        print(f"\n=== Best variant (ρ ≥ 0.99 gate) ===")
        print(f"  Segments: {best_v['segment']}, τ={best_v['tau']}")
        print(f"  Strat OOF: {best_v['oof']:.5f}  Δ vs global "
              f"{best_v['delta_global_bp']:+.2f}bp")
        print(f"  ρ vs PRIMARY: {best_v['rho_vs_primary']:.6f}")
        print(f"  pred-LB Δ: {best_v['pred_lb_delta_bp']:+.2f}bp")
        # Save best test prediction
        kind, tau = best_key
        seg_tr, seg_te, n_seg, _ = make_segments(train, test, kind)
        oofs_tau, tests_tau = hier_oof(F_oof, F_test, y, seg_tr, seg_te,
                                        n_seg, [tau], splits)
        np.save(ART / "oof_d13_path_b_hier_meta_strat.npy",
                np.column_stack([1 - oofs_tau[tau], oofs_tau[tau]]))
        np.save(ART / "test_d13_path_b_hier_meta_strat.npy",
                np.column_stack([1 - tests_tau[tau], tests_tau[tau]]))
        sub = sample_sub.copy(); sub[TARGET] = tests_tau[tau]
        sub.to_csv("submissions/submission_d13_path_b_hier_meta.csv",
                   index=False)
        # G3 rare-class flip ratio
        rare_thr = float(np.quantile(primary_test, 0.99))
        primary_pos = primary_test >= rare_thr
        new_pos = tests_tau[tau] >= rare_thr
        flips_to_pos = int(np.sum(~primary_pos & new_pos))
        flips_to_neg = int(np.sum(primary_pos & ~new_pos))
        ratio = (min(flips_to_pos, flips_to_neg) /
                 max(flips_to_pos, flips_to_neg)) if max(flips_to_pos,
                                                          flips_to_neg) > 0 else 1.0
        print(f"  rare-class flips: + → −  {flips_to_neg}, "
              f"− → +  {flips_to_pos}, ratio {ratio:.3f}")

    final = dict(
        global_meta=dict(auc=auc_global,
                         rho_vs_primary=float(rho_global)),
        sweep={f"{k[0]}_tau{k[1]}": v for k, v in results.items()},
        wall_s=time.time() - t_total,
    )
    (ART / "d13_path_b_hier_meta.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13_path_b_hier_meta.json  "
          f"(wall {time.time()-t_total:.0f}s)")


def predicted_lb_delta(d_global_bp, rho):
    """Predict LB delta given OOF delta and ρ vs PRIMARY."""
    if rho >= RHO_TIE:
        return d_global_bp
    if rho >= 0.995:
        return d_global_bp - 1.0  # 1bp penalty for diff prediction
    if rho >= 0.99:
        return d_global_bp - 2.5
    return d_global_bp - 4.0


if __name__ == "__main__":
    main()
