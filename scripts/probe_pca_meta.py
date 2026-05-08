"""scripts/probe_pca_meta.py — PCA-meta probe (4 variants).

PI ask: try a PCA of the K=27 ensemble and similar ideas. Anchor =
K=10+1 plain LR-meta (~0.94850).

Why this is a meaningful probe given A29/A30:
    - A30 says the 3-D logit subspace is the information ceiling under
      the LR-meta family. PCA-LR with K=27 components must equal raw
      LR-on-K=27 modulo regularization (linear invariance), so plain
      PCA-LR cannot break that ceiling.
    - The four variants below probe DIFFERENT mechanisms:
      A) PCA-truncate + LR  : tests "is L2 the binding constraint that
         shrinks low-variance directions away?" Truncating then
         re-fitting LR with full coefficient on a smaller basis tests
         whether the bottom PCs hold any signal.
      B) GB-meta on PCs     : the only architecturally untested avenue
         per A30. PCA decorrelates features for the boosted tree.
      C) Per-PC Path-B C×S  : routes per (Compound, Stint) on
         orthogonal directions. The current Path-B routes on
         correlated bases.
      D) PC residuals as aux: strip top-3 PCs from the K=10 logit pool,
         feed the residuals as auxiliary features to plain LR. Direct
         empirical test of the 3-D-ceiling claim (A30).

Fold-safety:
    Standardize and SVD-fit on the meta-train rows of each fold; project
    meta-val rows through the train basis. PCA of base OOFs is honest
    when each base's OOF was held-out for that row (Strat-KF seed=42),
    which matches our meta split.

Cost target: ~5-10 min CPU.
Outputs: scripts/artifacts/probe_pca_meta.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS, MAX_ITER = 42, 5, 500
MIN_ROWS = 1000

K27_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
    "d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound",
    "d9f_FM_A", "d9f_FM_B",
    "p1_single_cb_v4_gpu", "d17_h1d_yekenot_full",
    "d16_orig_continuous_only", "d18_chain_decomp",
    "d18_e2_preimage_knn", "d18_f2_constraint",
]
K10_FWD = [
    "d17_h1d_yekenot_full", "p1_single_cb_v4_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]
K_LIST = [3, 5, 10, 15, 27]


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def to_logit(P: np.ndarray) -> np.ndarray:
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.log(Pc / (1 - Pc))


def fold_pca(L_tr: np.ndarray, L_va: np.ndarray):
    """Standardize on train, SVD on train, project both."""
    mu = L_tr.mean(axis=0)
    sigma = L_tr.std(axis=0) + 1e-12
    Z_tr = (L_tr - mu) / sigma
    Z_va = (L_va - mu) / sigma
    # Economy SVD; columns of V are the principal directions
    _, S, Vt = np.linalg.svd(Z_tr, full_matrices=False)
    PCs_tr = Z_tr @ Vt.T
    PCs_va = Z_va @ Vt.T
    return PCs_tr, PCs_va, S, Vt, mu, sigma


def fit_lr_on_features(F_tr, y_tr, F_va, C=1.0):
    lr = LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F_tr, y_tr)
    return lr.predict_proba(F_va)[:, 1]


def fit_lgbm(F_tr, y_tr, F_va, params=None):
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=15, max_depth=4, feature_fraction=0.9,
             bagging_fraction=0.9, bagging_freq=5, verbose=-1,
             num_threads=4)
    if params:
        p.update(params)
    dtr = lgb.Dataset(F_tr, label=y_tr)
    booster = lgb.train(p, dtr, num_boost_round=300)
    return booster.predict(F_va)


def variant_A_pca_lr(L, y, splits, K_list):
    """Variant A — PCA-truncate + LR."""
    out = {}
    for k in K_list:
        oof = np.zeros(len(y))
        for tr, va in splits:
            PCs_tr, PCs_va, _, _, _, _ = fold_pca(L[tr], L[va])
            oof[va] = fit_lr_on_features(PCs_tr[:, :k], y[tr], PCs_va[:, :k])
        out[k] = float(roc_auc_score(y, oof))
    return out


def variant_B_pca_gb(L, y, splits, K_list):
    """Variant B — GB-meta on top-K PCs."""
    out = {}
    for k in K_list:
        oof = np.zeros(len(y))
        for tr, va in splits:
            PCs_tr, PCs_va, _, _, _, _ = fold_pca(L[tr], L[va])
            oof[va] = fit_lgbm(PCs_tr[:, :k], y[tr], PCs_va[:, :k])
        out[k] = float(roc_auc_score(y, oof))
    return out


def variant_C_pca_pathb(L, y, splits, seg, n_seg, k=10, tau=100000.0):
    """Variant C — Path-B C×S on top-k PCs."""
    oof = np.zeros(len(y))
    for tr_idx, va_idx in splits:
        PCs_tr_full, PCs_va_full, _, _, _, _ = fold_pca(L[tr_idx], L[va_idx])
        F_tr = PCs_tr_full[:, :k]
        F_va = PCs_va_full[:, :k]
        # Global LR on PCs
        lr_g = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr_g.fit(F_tr, y[tr_idx])
        w_global = np.concatenate([lr_g.intercept_, lr_g.coef_.ravel()])
        # Per-segment LR
        seg_tr = seg[tr_idx]
        seg_va = seg[va_idx]
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_tr == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr_idx][idx])) < 2:
                continue
            lr_s = LogisticRegression(C=1.0, max_iter=MAX_ITER,
                                       solver="lbfgs")
            lr_s.fit(F_tr[idx], y[tr_idx][idx])
            W_local[s] = np.concatenate([lr_s.intercept_,
                                          lr_s.coef_.ravel()])
            mask[s] = True
        n_local = counts.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local +
                    (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg_va):
            idx = np.where(seg_va == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            F_aug = np.column_stack([np.ones(len(idx)), F_va[idx]])
            oof[va_idx[idx]] = 1.0 / (1.0 + np.exp(
                -np.clip(F_aug @ w, -30, 30)))
    return float(roc_auc_score(y, oof))


def variant_D_residuals(L_K10, L_K27, y, splits, top_strip=3):
    """Variant D — strip top-`top_strip` PCs of K=27 logits, feed
    residual logits as 24 extra features alongside K=10 [P, rank, logit]
    expansion. Direct empirical test of the 3-D-ceiling claim.
    """
    # K10 logit -> use the K=10 base [P, rank, logit] expansion
    # alongside K=27 residual logits.
    P_K10 = 1.0 / (1.0 + np.exp(-L_K10))  # back to probability for expand
    F_K10 = expand(P_K10)
    oof = np.zeros(len(y))
    for tr, va in splits:
        # PCA on K=27 logits, strip top-S
        _, _, _, Vt, mu, sigma = fold_pca(L_K27[tr], L_K27[va])
        Z_tr = (L_K27[tr] - mu) / sigma
        Z_va = (L_K27[va] - mu) / sigma
        # residual = X - X projected onto top-S
        Vts = Vt[:top_strip]  # top-S directions in feature space
        proj_tr = (Z_tr @ Vts.T) @ Vts
        proj_va = (Z_va @ Vts.T) @ Vts
        R_tr = Z_tr - proj_tr
        R_va = Z_va - proj_va
        F_tr = np.hstack([F_K10[tr], R_tr])
        F_va = np.hstack([F_K10[va], R_va])
        oof[va] = fit_lr_on_features(F_tr, y[tr], F_va)
    return float(roc_auc_score(y, oof))


def baseline_K10_plain_lr(P_K10, y, splits):
    F = expand(P_K10)
    oof = np.zeros(len(y))
    for tr, va in splits:
        oof[va] = fit_lr_on_features(F[tr], y[tr], F[va])
    return float(roc_auc_score(y, oof))


def baseline_K27_plain_lr(P_K27, y, splits):
    F = expand(P_K27)
    oof = np.zeros(len(y))
    for tr, va in splits:
        oof[va] = fit_lr_on_features(F[tr], y[tr], F[va])
    return float(roc_auc_score(y, oof))


def main():
    t0 = time.time()
    print("Loading K=27 OOFs ...")
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    P_K27 = np.column_stack([_pos(ART / f"oof_{n}_strat.npy")
                              for n in K27_BASES])
    L_K27 = to_logit(P_K27)
    print(f"  K=27 OOF shape {P_K27.shape}; logit range "
          f"[{L_K27.min():.2f}, {L_K27.max():.2f}]")

    # K=10 forward-greedy slice (note: indices in K27_BASES order)
    k10_idx = [K27_BASES.index(n) for n in K10_FWD]
    P_K10 = P_K27[:, k10_idx]
    L_K10 = L_K27[:, k10_idx]

    # 5-fold StratKF (matches base CV split)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Compound × Stint segmentation for Path-B
    train["Compound"] = train["Compound"].astype(str)
    cats = sorted(train["Compound"].unique())
    cmp_map = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].map(cmp_map).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_train = c_tr * 6 + s_tr
    print(f"  segments: {n_seg} ({len(cats)} compounds × 6 stints)")

    # Anchor baselines
    print("\n=== Baselines ===")
    auc_k10_plain = baseline_K10_plain_lr(P_K10, y, splits)
    print(f"  K=10+1 plain LR-meta:  {auc_k10_plain:.5f}  (anchor)")
    auc_k27_plain = baseline_K27_plain_lr(P_K27, y, splits)
    print(f"  K=27+1 plain LR-meta:  {auc_k27_plain:.5f}  "
          f"(Δ vs K=10 = {(auc_k27_plain - auc_k10_plain)*1e4:+.2f} bp)")

    # ============ Variant A: PCA-truncate + LR ============
    print("\n=== Variant A — PCA-truncate + LR (regularizer test) ===")
    A = variant_A_pca_lr(L_K27, y, splits, K_LIST)
    for k in K_LIST:
        d = (A[k] - auc_k10_plain) * 1e4
        print(f"  PCA(K=27)→top-{k:>2d} + LR : {A[k]:.5f}  "
              f"(Δ vs K=10 = {d:+.2f} bp)")

    # ============ Variant B: GB-meta on PCs ============
    print("\n=== Variant B — GB-meta on PCs (EXP-NEW) ===")
    B = variant_B_pca_gb(L_K27, y, splits, K_LIST)
    for k in K_LIST:
        d = (B[k] - auc_k10_plain) * 1e4
        print(f"  PCA(K=27)→top-{k:>2d} + LightGBM : {B[k]:.5f}  "
              f"(Δ vs K=10 = {d:+.2f} bp)")
    # Also: GB-meta on raw K=27 logits (no PCA) for comparison
    print("  (control) LightGBM on raw K=27 [P, rank, logit] expansion:")
    F_K27 = expand(P_K27)
    oof_gb_raw = np.zeros(len(y))
    for tr, va in splits:
        oof_gb_raw[va] = fit_lgbm(F_K27[tr], y[tr], F_K27[va])
    auc_gb_raw = float(roc_auc_score(y, oof_gb_raw))
    print(f"  LightGBM on K=27 raw expansion: {auc_gb_raw:.5f}  "
          f"(Δ vs K=10 = {(auc_gb_raw - auc_k10_plain)*1e4:+.2f} bp)")
    # And: GB on K=10 raw expansion (no PCA)
    F_K10 = expand(P_K10)
    oof_gb_k10 = np.zeros(len(y))
    for tr, va in splits:
        oof_gb_k10[va] = fit_lgbm(F_K10[tr], y[tr], F_K10[va])
    auc_gb_k10 = float(roc_auc_score(y, oof_gb_k10))
    print(f"  LightGBM on K=10 raw expansion: {auc_gb_k10:.5f}  "
          f"(Δ vs K=10 = {(auc_gb_k10 - auc_k10_plain)*1e4:+.2f} bp)")

    # ============ Variant C: per-PC Path-B C×S ============
    print("\n=== Variant C — Path-B Compound×Stint on PCs ===")
    C_results = {}
    for k in [3, 5, 10]:
        auc_c = variant_C_pca_pathb(L_K27, y, splits, seg_train, n_seg,
                                      k=k, tau=100000.0)
        C_results[k] = auc_c
        d = (auc_c - auc_k10_plain) * 1e4
        print(f"  Path-B C×S τ=100k on top-{k:>2d} PCs: {auc_c:.5f}  "
              f"(Δ vs K=10 = {d:+.2f} bp)")

    # ============ Variant D: PC residuals as auxiliary features ============
    print("\n=== Variant D — top-3 stripped K=27 residuals as aux features ===")
    D_results = {}
    for s in [3, 5]:
        auc_d = variant_D_residuals(L_K10, L_K27, y, splits, top_strip=s)
        D_results[s] = auc_d
        d = (auc_d - auc_k10_plain) * 1e4
        print(f"  K=10+1 LR + (K=27 residuals after top-{s} strip): "
              f"{auc_d:.5f}  (Δ vs K=10 = {d:+.2f} bp)")

    # ============ Save + summarize ============
    out = {
        "anchor_K10_plain_lr_oof": auc_k10_plain,
        "anchor_K27_plain_lr_oof": auc_k27_plain,
        "variant_A_pca_lr": A,
        "variant_B_pca_gb": B,
        "variant_B_control_gb_K27_raw": auc_gb_raw,
        "variant_B_control_gb_K10_raw": auc_gb_k10,
        "variant_C_pca_pathb": C_results,
        "variant_D_residuals_aux": D_results,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_pca_meta.json").write_text(json.dumps(out, indent=2))

    # ====== Verdict ======
    print("\n=== VERDICT ===")
    best_label, best_auc = "K=10+1 plain LR (anchor)", auc_k10_plain
    candidates = [
        (f"PCA-LR top-{k}", A[k]) for k in K_LIST
    ] + [
        (f"PCA-GB top-{k}", B[k]) for k in K_LIST
    ] + [
        ("GB on K=27 raw", auc_gb_raw),
        ("GB on K=10 raw", auc_gb_k10),
    ] + [
        (f"PCA-PathB C×S top-{k}", C_results[k]) for k in [3, 5, 10]
    ] + [
        (f"K10 + K27 resid (strip {s})", D_results[s]) for s in [3, 5]
    ]
    for lbl, a in candidates:
        d = (a - auc_k10_plain) * 1e4
        if a > best_auc:
            best_label, best_auc = lbl, a
        marker = "  ← LIFT" if d >= 0.5 else ("  ← null" if abs(d) < 0.5
                                                else "")
        print(f"  {lbl:<35s} {a:.5f}  Δ {d:+6.2f} bp{marker}")
    print(f"\nBest variant: {best_label}  ({best_auc:.5f})  "
          f"vs anchor K=10+1 plain LR {auc_k10_plain:.5f}")

    print(f"\nWrote {ART / 'probe_pca_meta.json'}. Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
