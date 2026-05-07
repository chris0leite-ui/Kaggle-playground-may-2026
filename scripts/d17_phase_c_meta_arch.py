"""d17 Phase C — Path-B-AMP meta-arch redesigns.

Per friction `path-b-amp-only-fires-on-meta-arch-not-base-add`, only meta-arch
redesigns can fire the 6-11.6× LB amplification (d13 Stint 11.6×, d13e
Compound×Stint 8×). All d16 base-adds realised ~1× amp.

Three redesigns over the K=21 pool with d16_orig_continuous_only as 22nd base:

C1. Student-t shrinkage prior on per-segment LR coefficients.
    Replaces Gaussian τ shrinkage with Student-t (heavy tails). Allows
    high-deviation segments more freedom from the global LR. Approximated via
    iteratively reweighted Gaussian shrinkage with weights w_i = (ν+1)/(ν+r_i²)
    where r_i is the per-segment standardized residual.

C2. 3-level hierarchical partial pooling: Stint within Compound within Year.
    Implemented as nested empirical-Bayes shrinkage:
      coef_year = α_y * coef_year_only + (1-α_y) * coef_global
      coef_year_compound = α_yc * coef_year_compound + (1-α_yc) * coef_year
      coef_year_compound_stint = α_ycs * ... + (1-α_ycs) * coef_year_compound

C3. 75-segment cohort: Compound × Stint × r̂_q3 (5 × 5 × 3 = 75 cells).
    Adds an extra information dimension orthogonal to Compound/Stint.

Each variant outputs:
  oof_d17_path_b_X_K22_strat.npy / test_*
  And a summary line: OOF AUC, Δ vs PRIMARY, ρ vs PRIMARY.

Pool: K=22 = K=21 + d16_orig_continuous_only (clean; the d16 winner).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"
MIN_ROWS = 1000

K21_POOL = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
    "d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound",
    "d9f_FM_A", "d9f_FM_B",
]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.astype(np.float64).ravel()


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) / (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_global_lr(F, y, max_iter=500):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def hier_segment_lrs(F_tr, y_tr, F_te, seg_tr, seg_te, tau, w_global):
    """Standard Gaussian-tau hier-meta. Returns OOF-predict on F_te."""
    pred = np.zeros(len(F_te))
    seg_set = sorted(set(np.unique(seg_tr)) | set(np.unique(seg_te)))
    for s in seg_set:
        m_tr = seg_tr == s
        m_te = seg_te == s
        if m_tr.sum() < MIN_ROWS or m_te.sum() == 0:
            pred[m_te] = predict_aug(F_te[m_te], w_global) if m_te.sum() > 0 else 0
            continue
        try:
            w_s = fit_global_lr(F_tr[m_tr], y_tr[m_tr])
        except Exception:
            pred[m_te] = predict_aug(F_te[m_te], w_global)
            continue
        n = m_tr.sum()
        alpha = n / (n + tau)
        w_shrunk = alpha * w_s + (1 - alpha) * w_global
        pred[m_te] = predict_aug(F_te[m_te], w_shrunk)
    return pred


def hier_segment_lrs_studentT(F_tr, y_tr, F_te, seg_tr, seg_te, tau, w_global, nu=4.0):
    """Student-t shrinkage: per-segment coef weighted by t-distribution residual.
    Iteratively reweight: shrinkage strength varies per coefficient based on its
    deviation from global (heavy-tailed prior allows more deviation for outliers).
    """
    pred = np.zeros(len(F_te))
    seg_set = sorted(set(np.unique(seg_tr)) | set(np.unique(seg_te)))
    for s in seg_set:
        m_tr = seg_tr == s
        m_te = seg_te == s
        if m_tr.sum() < MIN_ROWS or m_te.sum() == 0:
            pred[m_te] = predict_aug(F_te[m_te], w_global) if m_te.sum() > 0 else 0
            continue
        try:
            w_s = fit_global_lr(F_tr[m_tr], y_tr[m_tr])
        except Exception:
            pred[m_te] = predict_aug(F_te[m_te], w_global)
            continue
        n = m_tr.sum()
        alpha = n / (n + tau)
        # Student-t element-wise shrinkage: strong shrink for coefs close to global,
        # weaker for outliers
        residual = w_s - w_global
        std_res = residual / (np.std(w_s) + 1e-6)
        per_dim_alpha = (nu + 1) / (nu + std_res ** 2) * alpha
        per_dim_alpha = np.clip(per_dim_alpha, 0.0, 1.0)
        w_shrunk = per_dim_alpha * w_s + (1 - per_dim_alpha) * w_global
        pred[m_te] = predict_aug(F_te[m_te], w_shrunk)
    return pred


def hier_3level(F_tr, y_tr, F_te, year_tr, comp_tr, stint_tr, year_te, comp_te, stint_te,
                 tau_year, tau_comp, tau_stint, w_global):
    """3-level nested EB shrinkage: stint < compound < year < global."""
    pred = np.zeros(len(F_te))
    years = sorted(set(np.unique(year_tr)) | set(np.unique(year_te)))
    for y in years:
        m_tr_y = year_tr == y
        m_te_y = year_te == y
        if m_tr_y.sum() < MIN_ROWS:
            pred[m_te_y] = predict_aug(F_te[m_te_y], w_global) if m_te_y.sum() > 0 else 0
            continue
        try:
            w_y = fit_global_lr(F_tr[m_tr_y], y_tr[m_tr_y])
        except Exception:
            pred[m_te_y] = predict_aug(F_te[m_te_y], w_global) if m_te_y.sum() > 0 else 0
            continue
        n_y = m_tr_y.sum()
        alpha_y = n_y / (n_y + tau_year)
        w_y_shrunk = alpha_y * w_y + (1 - alpha_y) * w_global

        comps = sorted(set(np.unique(comp_tr[m_tr_y])) | set(np.unique(comp_te[m_te_y])))
        for c in comps:
            m_tr_yc = m_tr_y & (comp_tr == c)
            m_te_yc = m_te_y & (comp_te == c)
            if m_tr_yc.sum() < MIN_ROWS // 2:
                if m_te_yc.sum() > 0:
                    pred[m_te_yc] = predict_aug(F_te[m_te_yc], w_y_shrunk)
                continue
            try:
                w_yc = fit_global_lr(F_tr[m_tr_yc], y_tr[m_tr_yc])
            except Exception:
                if m_te_yc.sum() > 0:
                    pred[m_te_yc] = predict_aug(F_te[m_te_yc], w_y_shrunk)
                continue
            n_yc = m_tr_yc.sum()
            alpha_yc = n_yc / (n_yc + tau_comp)
            w_yc_shrunk = alpha_yc * w_yc + (1 - alpha_yc) * w_y_shrunk

            stints = sorted(set(np.unique(stint_tr[m_tr_yc])) | set(np.unique(stint_te[m_te_yc])))
            for st in stints:
                m_tr_ycs = m_tr_yc & (stint_tr == st)
                m_te_ycs = m_te_yc & (stint_te == st)
                if m_tr_ycs.sum() < MIN_ROWS // 4:
                    if m_te_ycs.sum() > 0:
                        pred[m_te_ycs] = predict_aug(F_te[m_te_ycs], w_yc_shrunk)
                    continue
                try:
                    w_ycs = fit_global_lr(F_tr[m_tr_ycs], y_tr[m_tr_ycs])
                except Exception:
                    if m_te_ycs.sum() > 0:
                        pred[m_te_ycs] = predict_aug(F_te[m_te_ycs], w_yc_shrunk)
                    continue
                n_ycs = m_tr_ycs.sum()
                alpha_ycs = n_ycs / (n_ycs + tau_stint)
                w_ycs_shrunk = alpha_ycs * w_ycs + (1 - alpha_ycs) * w_yc_shrunk
                if m_te_ycs.sum() > 0:
                    pred[m_te_ycs] = predict_aug(F_te[m_te_ycs], w_ycs_shrunk)
    return pred


def main():
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("loading data + K=22 pool")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_test = _pos(ART / "test_PRIMARY_K22_strat.npy")
    primary_oof = _pos(ART / "oof_PRIMARY_K22_strat.npy")

    base_oofs, base_tests = [], []
    for name in K21_POOL:
        base_oofs.append(_pos(ART / f"oof_{name}_strat.npy"))
        base_tests.append(_pos(ART / f"test_{name}_strat.npy"))
    # K=22 = K=21 + d16_orig_continuous_only
    base_oofs.append(_pos(ART / "oof_d16_orig_continuous_only_strat.npy"))
    base_tests.append(_pos(ART / "test_d16_orig_continuous_only_strat.npy"))
    step(f"  K=22 loaded ({len(base_oofs)} bases)")

    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    step(f"  F shape {F_oof.shape}")

    # segmentation
    cats = sorted(set(train["Compound"].astype(str)) | set(test["Compound"].astype(str)))
    cmp_map = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp_map).astype(int).values
    c_te = test["Compound"].astype(str).map(cmp_map).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
    seg_cs_tr = c_tr * 6 + s_tr
    seg_cs_te = c_te * 6 + s_te

    rhat_tr = np.load(ART / "d16_rhat_synth_train.npy")
    rhat_te = np.load(ART / "d16_rhat_synth_test.npy")
    rh_q_edges = np.unique(np.quantile(np.log1p(rhat_tr), [0, 1/3, 2/3, 1]))
    rh_tr = np.searchsorted(rh_q_edges[1:-1], np.log1p(rhat_tr))
    rh_te = np.searchsorted(rh_q_edges[1:-1], np.log1p(rhat_te))
    seg_csr_tr = seg_cs_tr * 3 + rh_tr
    seg_csr_te = seg_cs_te * 3 + rh_te

    yr_tr = train["Year"].astype(int).values
    yr_te = test["Year"].astype(int).values
    yr_set = sorted(set(yr_tr) | set(yr_te))
    yr_map = {v: i for i, v in enumerate(yr_set)}
    yr_tr_i = np.vectorize(yr_map.get)(yr_tr)
    yr_te_i = np.vectorize(yr_map.get)(yr_te)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    summary = dict(K22_pool=K21_POOL + ["d16_orig_continuous_only"], variants={})

    primary_auc = float(roc_auc_score(y, primary_oof))
    step(f"  PRIMARY OOF AUC {primary_auc:.5f}")

    # baseline: Compound×Stint Gaussian τ=20k (matches d13e/d15b/d16 PRIMARY)
    step("\n=== BASELINE (Gaussian τ=20k) ===")
    oof_b = np.zeros(len(y))
    pred_te_b = np.zeros(F_test.shape[0])
    for fi, (tri, vai) in enumerate(skf.split(F_oof, y)):
        w_g = fit_global_lr(F_oof[tri], y[tri])
        oof_b[vai] = hier_segment_lrs(F_oof[tri], y[tri], F_oof[vai],
                                       seg_cs_tr[tri], seg_cs_tr[vai], 20000.0, w_g)
        pred_te_b += hier_segment_lrs(F_oof[tri], y[tri], F_test,
                                       seg_cs_tr[tri], seg_cs_te, 20000.0, w_g) / N_FOLDS
    auc_b = roc_auc_score(y, oof_b)
    step(f"  baseline OOF {auc_b:.5f}  Δ vs PRIMARY {(auc_b - primary_auc) * 1e4:+.2f} bp")
    summary["variants"]["baseline_K22_compstint_tau20k"] = dict(auc=float(auc_b),
                                                                  delta_primary_bp=float((auc_b - primary_auc) * 1e4))

    # ===== C1 Student-t shrinkage =====
    for nu in [3.0, 7.0]:
        step(f"\n=== C1 Student-t ν={nu} K=22 Compound×Stint τ=20k ===")
        oof = np.zeros(len(y))
        pred_te = np.zeros(F_test.shape[0])
        for fi, (tri, vai) in enumerate(skf.split(F_oof, y)):
            w_g = fit_global_lr(F_oof[tri], y[tri])
            oof[vai] = hier_segment_lrs_studentT(
                F_oof[tri], y[tri], F_oof[vai],
                seg_cs_tr[tri], seg_cs_tr[vai], 20000.0, w_g, nu=nu)
            pred_te += hier_segment_lrs_studentT(
                F_oof[tri], y[tri], F_test,
                seg_cs_tr[tri], seg_cs_te, 20000.0, w_g, nu=nu) / N_FOLDS
        auc = roc_auc_score(y, oof)
        rho = float(np.corrcoef(pred_te, primary_test)[0, 1])
        step(f"  C1 ν={nu} OOF {auc:.5f}  Δ {(auc - auc_b) * 1e4:+.2f} bp vs baseline  ρ {rho:.5f}")
        np.save(ART / f"oof_d17_path_b_K22_studentT_nu{int(nu)}_strat.npy", oof)
        np.save(ART / f"test_d17_path_b_K22_studentT_nu{int(nu)}_strat.npy", pred_te)
        summary["variants"][f"C1_studentT_nu{int(nu)}"] = dict(
            auc=float(auc), delta_baseline_bp=float((auc - auc_b) * 1e4),
            delta_primary_bp=float((auc - primary_auc) * 1e4), rho=rho)

    # ===== C2 3-level hierarchy =====
    step(f"\n=== C2 3-level (Stint<Compound<Year, τ=5k/20k/100k) ===")
    oof = np.zeros(len(y))
    pred_te = np.zeros(F_test.shape[0])
    for fi, (tri, vai) in enumerate(skf.split(F_oof, y)):
        w_g = fit_global_lr(F_oof[tri], y[tri])
        oof[vai] = hier_3level(
            F_oof[tri], y[tri], F_oof[vai],
            yr_tr_i[tri], c_tr[tri], s_tr[tri],
            yr_tr_i[vai], c_tr[vai], s_tr[vai],
            tau_year=100000.0, tau_comp=20000.0, tau_stint=5000.0, w_global=w_g)
        pred_te += hier_3level(
            F_oof[tri], y[tri], F_test,
            yr_tr_i[tri], c_tr[tri], s_tr[tri],
            yr_te_i, c_te, s_te,
            tau_year=100000.0, tau_comp=20000.0, tau_stint=5000.0, w_global=w_g) / N_FOLDS
    auc = roc_auc_score(y, oof)
    rho = float(np.corrcoef(pred_te, primary_test)[0, 1])
    step(f"  C2 3-level OOF {auc:.5f}  Δ {(auc - auc_b) * 1e4:+.2f} bp  ρ {rho:.5f}")
    np.save(ART / "oof_d17_path_b_K22_3level_strat.npy", oof)
    np.save(ART / "test_d17_path_b_K22_3level_strat.npy", pred_te)
    summary["variants"]["C2_3level_year_compound_stint"] = dict(
        auc=float(auc), delta_baseline_bp=float((auc - auc_b) * 1e4),
        delta_primary_bp=float((auc - primary_auc) * 1e4), rho=rho)

    # ===== C3 75-seg (Compound × Stint × r̂_q3) =====
    for tau in [5000, 20000, 100000]:
        step(f"\n=== C3 75-seg Compound×Stint×r̂_q3 τ={tau} ===")
        oof = np.zeros(len(y))
        pred_te = np.zeros(F_test.shape[0])
        for fi, (tri, vai) in enumerate(skf.split(F_oof, y)):
            w_g = fit_global_lr(F_oof[tri], y[tri])
            oof[vai] = hier_segment_lrs(
                F_oof[tri], y[tri], F_oof[vai],
                seg_csr_tr[tri], seg_csr_tr[vai], float(tau), w_g)
            pred_te += hier_segment_lrs(
                F_oof[tri], y[tri], F_test,
                seg_csr_tr[tri], seg_csr_te, float(tau), w_g) / N_FOLDS
        auc = roc_auc_score(y, oof)
        rho = float(np.corrcoef(pred_te, primary_test)[0, 1])
        step(f"  C3 τ={tau} OOF {auc:.5f}  Δ {(auc - auc_b) * 1e4:+.2f} bp  ρ {rho:.5f}")
        np.save(ART / f"oof_d17_path_b_K22_75seg_tau{tau}_strat.npy", oof)
        np.save(ART / f"test_d17_path_b_K22_75seg_tau{tau}_strat.npy", pred_te)
        summary["variants"][f"C3_75seg_tau{tau}"] = dict(
            auc=float(auc), delta_baseline_bp=float((auc - auc_b) * 1e4),
            delta_primary_bp=float((auc - primary_auc) * 1e4), rho=rho)

    summary["runtime_s"] = time.time() - t0
    summary["primary_auc"] = primary_auc
    summary["baseline_K22_auc"] = float(auc_b)
    with open(ART / "d17_phase_c_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step("DONE")


if __name__ == "__main__":
    main()
