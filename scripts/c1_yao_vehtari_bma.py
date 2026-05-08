"""scripts/c1_yao_vehtari_bma.py — Yao/Vehtari covariance-modelled BMA on K=27.

The remaining untested meta-arch on the only live Path-B amp axis
(Compound × Stint). Tests four meta variants on K=27 = K=21 + v4 +
h1d + d16 + d18 + E2 + F2, against the current PRIMARY (Path-B
plain shrinkage τ=100k OOF 0.95432).

Variants:
  V0 — Plain LR-meta (global, no segmentation).
  V1 — Path-B current (per-segment plain shrinkage W = αW_loc + (1-α)w_glo).
  V2 — Plain BMA: w_k ∝ exp(-N * BCE_k) on (raw, rank, logit) blocks.
  V3 — Yao/Vehtari covariance-modulated Path-B: shrinkage matrix uses
       inter-base correlation Σ. Per-segment fit:
            W_loc_s = argmin ||X_s w - y_s||² + τ * (w - w_glo)' Σ^-1 (w - w_glo)
       Solving: w = w_glo + (X_s'X_s + τΣ^-1)^-1 X_s'(y - X_s w_glo).
       Shrinks weights along low-eigenvalue (highly-correlated) directions
       MORE than plain Path-B; preserves dominant orthogonal directions.

Outputs:
  scripts/artifacts/oof_c1_v3_yv_bma_<seg>_strat.npy + test
  scripts/artifacts/c1_yao_vehtari_bma_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MIN_ROWS = 1000

POOL_KEEP = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
    "d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound",
    "d9f_FM_A", "d9f_FM_B",
]
EXTRAS_K27 = ["p1_single_cb_v4_gpu", "d17_h1d_yekenot_full",
              "d16_orig_continuous_only", "d18_chain_decomp",
              "d18_e2_preimage_knn", "d18_f2_constraint"]
ALL_BASES = POOL_KEEP + EXTRAS_K27


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def fit_lr_aug(F, y, max_iter=500):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def fit_yv_aug(F, y, w_global, Sigma_inv_aug, tau, max_iter=200):
    """Yao/Vehtari covariance-modulated ridge LR.
    Solves: argmin BCE + τ/2 * (w - w_glo)' Σ^-1 (w - w_glo)
    via Newton-Raphson with Hessian regularization H + τΣ^-1.
    """
    n, d = F.shape
    F_aug = np.column_stack([np.ones(n), F])
    w = w_global.copy()
    for _ in range(max_iter):
        eta = np.clip(F_aug @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = F_aug.T @ (p - y) / n + tau * Sigma_inv_aug @ (w - w_global) / n
        W = p * (1.0 - p)
        # Hessian: F'WF/n + τ Σ^-1 / n
        H = F_aug.T @ (W[:, None] * F_aug) / n + tau * Sigma_inv_aug / n
        try:
            step = np.linalg.solve(H + 1e-6 * np.eye(d + 1), grad)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.linalg.norm(step) < 1e-6:
            return w_new
        w = w_new
    return w


def load_pos(name):
    def _pos(p):
        a = np.load(p)
        return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)
    return (_pos(ART / f"oof_{name}_strat.npy"),
            _pos(ART / f"test_{name}_strat.npy"))


def main():
    t0 = time.time()
    print("=== C1 Yao/Vehtari covariance-modelled BMA on K=27 ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"K=27 bases: {len(ALL_BASES)}")

    base_oofs, base_tests = [], []
    for name in ALL_BASES:
        oo, te = load_pos(name)
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    n_train, n_test, d_feat = len(y), len(test), F_oof.shape[1]
    print(f"F shape oof={F_oof.shape} test={F_test.shape}")

    primary_test_path = ART / "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"
    primary_test_arr = np.load(primary_test_path)
    primary_test = (primary_test_arr[:, 1] if primary_test_arr.ndim == 2
                    else primary_test_arr.ravel()).astype(np.float64)

    # Segmentation: Compound × Stint (only live amp axis)
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
    splits = list(skf.split(np.zeros(n_train), y))

    # ==================================================================
    # V0 — Plain LR-meta (global, no segmentation)
    # ==================================================================
    print("\n--- V0: plain global LR-meta ---")
    v0_oof = np.zeros(n_train)
    for fold, (tr, va) in enumerate(splits):
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        v0_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    v0_auc = float(roc_auc_score(y, v0_oof))
    print(f"  V0 OOF: {v0_auc:.5f}  ({time.time()-t0:.0f}s)")

    # ==================================================================
    # V2 — Plain BMA: w_k ∝ exp(-N * BCE_k)
    # ==================================================================
    print("\n--- V2: plain BMA per (raw|rank|logit) block ---")
    bce = np.array([log_loss(y, np.clip(F_oof[:, j], 1e-7, 1 - 1e-7))
                    if j < len(ALL_BASES) else
                    log_loss(y, np.clip(F_oof[:, j], 1e-7, 1 - 1e-7))
                    for j in range(d_feat)])
    # softmax over -N * BCE within each block (raw / rank / logit)
    # Skip rank/logit blocks (not probability scale); just BMA on raw.
    n_bases = len(ALL_BASES)
    bce_raw = bce[:n_bases]
    w_bma_raw = np.exp(-(bce_raw - bce_raw.min()) * np.sqrt(n_train))
    w_bma_raw /= w_bma_raw.sum()
    v2_oof = F_oof[:, :n_bases] @ w_bma_raw
    v2_oof = np.clip(v2_oof, 1e-7, 1 - 1e-7)
    v2_auc = float(roc_auc_score(y, v2_oof))
    print(f"  V2 BMA OOF: {v2_auc:.5f}")
    print(f"  top-5 BMA weights: {sorted(zip(ALL_BASES, w_bma_raw), key=lambda x: -x[1])[:5]}")

    # ==================================================================
    # V1 — Path-B plain shrinkage (re-implement for direct comparison)
    # ==================================================================
    print("\n--- V1: Path-B plain shrinkage τ=100k ---")
    tau = 100000
    v1_oof = np.zeros(n_train)
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
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local +
                    (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg_train[va]):
            idx = np.where(seg_train[va] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            v1_oof[va[idx]] = predict_aug(F_oof[va[idx]], w)
        print(f"  V1 fold {fold}: {time.time()-t_fold:.0f}s")
    v1_auc = float(roc_auc_score(y, v1_oof))
    print(f"  V1 OOF: {v1_auc:.5f}")

    # ==================================================================
    # V3 — Yao/Vehtari covariance-modulated Path-B
    # ==================================================================
    print("\n--- V3: Yao/Vehtari covariance-modulated Path-B ---")
    # Build Σ_aug: covariance of [intercept-col-of-1s, F] across rows.
    # Use full-train F, normalised to zero-mean for the non-intercept part.
    F_centered = F_oof - F_oof.mean(axis=0, keepdims=True)
    Sigma_F = (F_centered.T @ F_centered) / n_train
    # Augment for intercept: [[1, 0...], [0, Σ_F]]
    Sigma_aug = np.zeros((d_feat + 1, d_feat + 1))
    Sigma_aug[0, 0] = 1.0
    Sigma_aug[1:, 1:] = Sigma_F + 1e-4 * np.eye(d_feat)  # ridge for invertibility
    Sigma_inv_aug = np.linalg.inv(Sigma_aug)
    print(f"  Σ_aug cond: {np.linalg.cond(Sigma_aug):.2e}")
    print(f"  Σ_F top eigenvalues: {sorted(np.linalg.eigvalsh(Sigma_F))[-5:]}")

    taus_yv = [10000, 50000, 200000]
    v3_oofs = {tau: np.zeros(n_train) for tau in taus_yv}
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = fit_lr_aug(F_oof[tr], y[tr])
        for tau in taus_yv:
            for s in np.unique(seg_train[va]):
                idx_tr = np.where(seg_train[tr] == s)[0]
                idx_va = np.where(seg_train[va] == s)[0]
                if (len(idx_tr) < MIN_ROWS
                        or len(np.unique(y[tr][idx_tr])) < 2):
                    v3_oofs[tau][va[idx_va]] = predict_aug(
                        F_oof[va[idx_va]], w_global)
                    continue
                w_yv = fit_yv_aug(
                    F_oof[tr][idx_tr], y[tr][idx_tr],
                    w_global, Sigma_inv_aug, tau)
                v3_oofs[tau][va[idx_va]] = predict_aug(F_oof[va[idx_va]], w_yv)
        print(f"  V3 fold {fold}: {time.time()-t_fold:.0f}s")

    v3_aucs = {tau: float(roc_auc_score(y, v3_oofs[tau])) for tau in taus_yv}
    for tau, auc in v3_aucs.items():
        print(f"  V3 τ={tau}: OOF {auc:.5f}  Δ vs V1 = {(auc-v1_auc)*1e4:+.2f} bp")

    # ==================================================================
    # Save artifacts + summary
    # ==================================================================
    summary = dict(
        K=len(ALL_BASES),
        n_seg=n_seg,
        v0_global_lr_oof=v0_auc,
        v1_path_b_tau100k_oof=v1_auc,
        v2_plain_bma_oof=v2_auc,
        v3_yao_vehtari_oof={str(t): a for t, a in v3_aucs.items()},
        delta_v3_vs_v1_bp={str(t): float((v3_aucs[t]-v1_auc)*1e4) for t in taus_yv},
        wall_total_s=time.time() - t0,
    )
    (ART / "c1_yao_vehtari_bma_results.json").write_text(json.dumps(summary, indent=2))
    # Save best-V3 OOF
    best_tau = max(v3_aucs, key=v3_aucs.get)
    np.save(ART / f"oof_c1_v3_yv_bma_compound_stint_tau{best_tau}_strat.npy",
            np.column_stack([1 - v3_oofs[best_tau], v3_oofs[best_tau]]))
    print(f"\n=== Total wall: {time.time()-t0:.0f}s ===")
    print(f"  V0 plain LR-meta:           {v0_auc:.5f}")
    print(f"  V1 Path-B plain τ=100k:     {v1_auc:.5f}")
    print(f"  V2 plain BMA:               {v2_auc:.5f}")
    print(f"  V3 Yao/Vehtari best τ={best_tau}:  {v3_aucs[best_tau]:.5f}  "
          f"Δ vs V1 = {(v3_aucs[best_tau]-v1_auc)*1e4:+.2f} bp")


if __name__ == "__main__":
    main()
