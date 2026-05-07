"""scripts/t1_3_segmentation_crosses.py — T1#3: Path-B alternative
segmentation crosses on K=10.

Per HANDOVER Day-18 next-step recommendation. Tests whether Path-B
amp persists at K=10 (predicted yes — saturation is in info space,
not pool size) and whether non-Compound×Stint segmentations carry
distinct signal.

Three segmentations × τ ∈ {5000, 20000, 100000}:
  S1: Year × Compound          (4 × 5 = 20 nominal cells)
  S2: Compound × TyreLife_q5   (5 × 5 = 25 cells)
  S3: Driver_freq_q4 × Stint   (4 × 5 = 20 cells; label-free)

For each (S, τ), train Path-B segment-local LR with shrinkage to
global; record per-fold OOF AUC, ρ vs d18 PRIMARY, ρ vs d13e
(canonical Compound × Stint τ=20k). Predicted EV per Rule 19
family priors: meta_arch_redesign p=0.30, (1, 4, 8) bp.

Reference: d13e Compound × Stint τ=20k OOF 0.95083 / LB 0.95049
historically gave 8× LB-amp. d18 K=24 PRIMARY OOF 0.95385.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MIN_ROWS = 1000
MAX_ITER = 500
TAUS = [5000, 20000, 100000]

K10_BASES = [
    "d17_h1d_yekenot_full", "p1_single_cb_v3_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P):
    n = len(P)
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    rk = np.column_stack([np.argsort(np.argsort(c)) / n for c in P.T])
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _fit_lr_aug(F, y):
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def _pred_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def _build_segmentations(train, test):
    """Return dict[name] -> (seg_train, seg_test, n_seg)."""
    out = {}

    # S1: Year × Compound. Encode in train+test to share IDs.
    years_tr = train["Year"].astype(int).values
    years_te = test["Year"].astype(int).values
    year_levels = sorted(set(years_tr) | set(years_te))
    y_id = {y: i for i, y in enumerate(year_levels)}
    cmp_levels = sorted(set(train["Compound"].astype(str).unique()) |
                        set(test["Compound"].astype(str).unique()))
    c_id = {c: i for i, c in enumerate(cmp_levels)}
    n_year = len(year_levels)
    n_cmp = len(cmp_levels)
    s1_tr = (np.array([y_id[y] for y in years_tr]) * n_cmp +
             np.array([c_id[c] for c in train["Compound"].astype(str).values]))
    s1_te = (np.array([y_id[y] for y in years_te]) * n_cmp +
             np.array([c_id[c] for c in test["Compound"].astype(str).values]))
    out["S1_year_x_compound"] = (s1_tr, s1_te, n_year * n_cmp)

    # S2: Compound × TyreLife_q5. Quintiles fit on train only.
    tyre_q_edges = np.quantile(train["TyreLife"].values, np.linspace(0, 1, 6))
    tyre_q_edges[0] = -np.inf; tyre_q_edges[-1] = np.inf
    tyre_q_tr = np.searchsorted(tyre_q_edges[1:-1],
                                 train["TyreLife"].values, side="right")
    tyre_q_te = np.searchsorted(tyre_q_edges[1:-1],
                                 test["TyreLife"].values, side="right")
    s2_tr = (np.array([c_id[c] for c in train["Compound"].astype(str).values]) * 5
             + tyre_q_tr)
    s2_te = (np.array([c_id[c] for c in test["Compound"].astype(str).values]) * 5
             + tyre_q_te)
    out["S2_compound_x_tyrelife_q5"] = (s2_tr, s2_te, n_cmp * 5)

    # S3: Driver_freq_q4 × Stint. Label-free.
    drv_counts = train["Driver"].value_counts()
    drv_freq_train = train["Driver"].map(drv_counts).fillna(0).values.astype(np.float64)
    drv_freq_test = test["Driver"].map(drv_counts).fillna(0).values.astype(np.float64)
    fr_q_edges = np.quantile(drv_freq_train, np.linspace(0, 1, 5))
    fr_q_edges[0] = -np.inf; fr_q_edges[-1] = np.inf
    fr_q_tr = np.searchsorted(fr_q_edges[1:-1], drv_freq_train, side="right")
    fr_q_te = np.searchsorted(fr_q_edges[1:-1], drv_freq_test, side="right")
    stint_tr = np.clip(train["Stint"].astype(int).values, 0, 4)
    stint_te = np.clip(test["Stint"].astype(int).values, 0, 4)
    s3_tr = fr_q_tr * 5 + stint_tr
    s3_te = fr_q_te * 5 + stint_te
    out["S3_drvfreq_q4_x_stint"] = (s3_tr, s3_te, 4 * 5)

    return out


def run_path_b(F_oof, F_test, y, seg_train, seg_test, n_seg, splits, name):
    print(f"\n--- Path-B segmentation: {name} (n_seg={n_seg}) ---")
    sizes = np.bincount(seg_train, minlength=n_seg)
    populated = int(np.sum(sizes >= MIN_ROWS))
    print(f"  ≥{MIN_ROWS} rows in {populated}/{n_seg} segments; "
          f"sizes min/med/max = "
          f"{sizes[sizes>0].min() if sizes.max()>0 else 0}/"
          f"{int(np.median(sizes[sizes>0])) if sizes.max()>0 else 0}/"
          f"{sizes.max()}")

    oofs = {tau: np.zeros(len(y)) for tau in TAUS}
    for fold, (tr, va) in enumerate(splits):
        t0 = time.time()
        w_global = _fit_lr_aug(F_oof[tr], y[tr])
        n_dim = len(w_global)
        W_local = np.zeros((n_seg, n_dim))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_local[s] = _fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            mask[s] = True
        for tau in TAUS:
            n_local = counts.astype(np.float64)
            alpha = n_local / (n_local + tau)
            W_shrunk = (alpha[:, None] * W_local +
                        (1 - alpha[:, None]) * w_global[None, :])
            for s in np.unique(seg_train[va]):
                idx = np.where(seg_train[va] == s)[0]
                w = W_shrunk[s] if mask[s] else w_global
                oofs[tau][va[idx]] = _pred_aug(F_oof[va[idx]], w)
        print(f"  fold {fold+1}/{len(splits)}: {time.time()-t0:.1f}s "
              f"({int(mask.sum())}/{n_seg} segs fit)")

    # Full-train fit for test predictions
    print(f"  full-train fit ...")
    t0 = time.time()
    w_global_full = _fit_lr_aug(F_oof, y)
    W_local_full = np.zeros((n_seg, len(w_global_full)))
    counts_full = np.zeros(n_seg, dtype=np.int64)
    mask_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        counts_full[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_local_full[s] = _fit_lr_aug(F_oof[idx], y[idx])
        mask_full[s] = True
    test_preds = {}
    for tau in TAUS:
        n_local = counts_full.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local_full +
                    (1 - alpha[:, None]) * w_global_full[None, :])
        tp = np.zeros(F_test.shape[0])
        for s in np.unique(seg_test):
            idx = np.where(seg_test == s)[0]
            w = W_shrunk[s] if mask_full[s] else w_global_full
            tp[idx] = _pred_aug(F_test[idx], w)
        test_preds[tau] = tp
    print(f"  full-train wall: {time.time()-t0:.1f}s")
    return oofs, test_preds, populated


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # Reference: d13e canonical and d18 PRIMARY
    d18_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    d18_test = _pos(ART / "test_d17_K24_d18pool_h1d_strat.npy")
    d13e_oof = _pos(ART / "oof_d13e_compound_stint_tau20000_strat.npy")
    d13e_test = _pos(ART / "test_d13e_compound_stint_tau20000_strat.npy")
    k10_oof = _pos(ART / "oof_K10_lr_meta_strat.npy")
    k10_test = _pos(ART / "test_K10_lr_meta_strat.npy")
    auc_d18 = roc_auc_score(y, d18_oof)
    auc_d13e = roc_auc_score(y, d13e_oof)
    auc_k10 = roc_auc_score(y, k10_oof)
    print(f"References:")
    print(f"  d18 K=24 PRIMARY:                    AUC {auc_d18:.5f}")
    print(f"  d13e Compound×Stint τ=20k (canonical Path-B): AUC {auc_d13e:.5f}")
    print(f"  K=10 LR-meta (T2):                   AUC {auc_k10:.5f}")

    # K=10 features
    P_oof = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in K10_BASES])
    P_test = np.column_stack([_pos(ART / f"test_{b}_strat.npy") for b in K10_BASES])
    F_oof = _expand(P_oof)
    F_test = _expand(P_test)
    print(f"K=10 features: F shape {F_oof.shape}")

    # Segmentations
    segs = _build_segmentations(train, test)
    print(f"Segmentations: {list(segs.keys())}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    summary = {
        "auc_d18_primary": round(float(auc_d18), 6),
        "auc_d13e_compound_stint": round(float(auc_d13e), 6),
        "auc_k10_global_lr": round(float(auc_k10), 6),
        "segmentations": {},
    }

    for name, (seg_tr, seg_te, n_seg) in segs.items():
        oofs, test_preds, populated = run_path_b(
            F_oof, F_test, y, seg_tr, seg_te, n_seg, splits, name
        )
        seg_results = {"populated_segments": populated, "n_seg": n_seg, "taus": {}}
        for tau in TAUS:
            oof_tau = oofs[tau]
            tp_tau = test_preds[tau]
            auc = roc_auc_score(y, oof_tau)
            d_d18 = (auc - auc_d18) * 1e4
            d_k10 = (auc - auc_k10) * 1e4
            d_d13e = (auc - auc_d13e) * 1e4
            rho_d18, _ = spearmanr(tp_tau, d18_test)
            rho_d13e, _ = spearmanr(tp_tau, d13e_test)
            rho_k10, _ = spearmanr(tp_tau, k10_test)
            print(f"\n  {name} τ={tau}:")
            print(f"    Strat OOF: {auc:.5f}  Δ vs d18: {d_d18:+.2f}bp  "
                  f"Δ vs k10: {d_k10:+.2f}bp  Δ vs d13e: {d_d13e:+.2f}bp")
            print(f"    ρ_test vs d18:  {rho_d18:.5f}")
            print(f"    ρ_test vs d13e: {rho_d13e:.5f}")
            print(f"    ρ_test vs k10:  {rho_k10:.5f}")
            seg_results["taus"][str(tau)] = dict(
                oof_auc=round(float(auc), 6),
                delta_bp_vs_d18=round(float(d_d18), 3),
                delta_bp_vs_k10=round(float(d_k10), 3),
                delta_bp_vs_d13e=round(float(d_d13e), 3),
                rho_test_vs_d18=round(float(rho_d18), 6),
                rho_test_vs_d13e=round(float(rho_d13e), 6),
                rho_test_vs_k10=round(float(rho_k10), 6),
            )
            # save artifacts (namespaced)
            short = name.split("_")[0]   # S1/S2/S3
            np.save(ART / f"oof_t1_3_{short}_K10_tau{tau}_strat.npy",
                    np.column_stack([1 - oof_tau, oof_tau]))
            np.save(ART / f"test_t1_3_{short}_K10_tau{tau}_strat.npy",
                    np.column_stack([1 - tp_tau, tp_tau]))
        summary["segmentations"][name] = seg_results

    json_path = ART / "t1_3_segmentation_crosses.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
