"""scripts/path_b_K5_rf_yekenot.py — Path-B C×S τ=100k on K=5 = K=4 + RF.

Mirrors d17_path_b_K23_v4_h1d.py exactly (Compound × Stint, MIN_ROWS=1000)
on the K=4 forward-greedy PRIMARY pool extended by the RF-yekenot base
from probe_forest_sweep.py Angle A.

Rationale: across 4 independent RF runs (Angle A, Kitchen-sink, Optuna
seed 42 / 7) the K=4+1 LR-meta lift is +0.24-0.27 bp with std 0.013 bp.
Hyperparameter optimization caps at the same +0.25 bp ceiling. The
remaining lever is whether per-segment shrinkage (Path-B C×S τ=100k)
amplifies the +0.25 bp OOF signal to LB. Historical amp ratios on
similarly-orthogonal new bases:
  - d15b DAE base (ρ=0.948): realized 1.4× amp (+0.715 bp OOF → +1.0 bp LB)
  - d13e Compound × Stint: 8× amp on FM-class
  - d13 Stint amp: 11.6×
At the 1.4× floor on +0.25 bp OOF, predicted LB Δ +0.35 bp; sits inside
the public-LB sample-noise band (±12 bp on 20% draw) but on the positive
side. PRIMARY (K=4 + Path-B C×S τ=100k) is at LB 0.95351; goal is to
move it positively.

Outputs:
  scripts/artifacts/oof_path_b_K5_rf_yekenot_tau{5k,20k,100k}_strat.npy
  scripts/artifacts/test_path_b_K5_rf_yekenot_tau{5k,20k,100k}_strat.npy
  scripts/artifacts/path_b_K5_rf_yekenot_results.json
  submissions/submission_path_b_K5_rf_yekenot_tau{5k,20k,100k}.csv
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
PRIMARY_LB = 0.95351  # K=4 + Path-B C×S τ=100k

# K=4 forward-greedy bases + RF-yekenot (Angle A)
POOL = [
    ("d17_h1d_yekenot", "d17_h1d_yekenot_full"),
    ("p1_single_cb_v4", "p1_single_cb_v4_gpu"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("d16_orig_continuous", "d16_orig_continuous_only"),
    ("rf_yekenot", "rf_yekenot"),
]


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def fit_lr_aug(F: np.ndarray, y: np.ndarray, max_iter: int = MAX_ITER):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    print("=== Path-B C×S τ=100k on K=5 = K=4 + RF-yekenot ===")
    print(f"  PRIMARY (K=4 + Path-B C×S τ=100k) LB = {PRIMARY_LB}")
    print(f"  pool ({len(POOL)} bases):")
    base_oofs, base_tests, names = [], [], []
    for label, fname in POOL:
        oo = _pos(ART / f"oof_{fname}_strat.npy")
        te = _pos(ART / f"test_{fname}_strat.npy")
        base_oofs.append(oo)
        base_tests.append(te)
        names.append(label)
        print(f"    {label:<25s} {fname}  oof_mean={oo.mean():.4f} "
              f"test_mean={te.mean():.4f}")

    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"  F_oof {F_oof.shape}  F_test {F_test.shape}")

    # Compound × Stint segmentation
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
    populated = int(np.sum(sizes >= MIN_ROWS))
    print(f"  Compound×Stint: n_seg={n_seg}  ≥{MIN_ROWS} rows: {populated}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Global K=5 LR baseline (sanity)
    print("\n--- Global K=5 LR baseline ---")
    meta_global = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global[va] = lr.predict_proba(F_oof[va])[:, 1]
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s")
    auc_global = float(roc_auc_score(y, meta_global))
    print(f"  Global K=5 LR-meta OOF: {auc_global:.5f}")

    # Hier-meta sweep — focus τ=100k (PRIMARY's setting); also 20k, 5k for hedge
    taus = [5000, 20000, 100000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}

    print("\n--- Compound × Stint hier-meta on K=5 ---")
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
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s "
              f"({int(mask.sum())}/{n_seg} segments fit; "
              f"{skipped} skipped)")

    # Full-train fit for test predictions
    print("\n--- Full-train test predictions ---")
    t_full = time.time()
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
    print(f"  full-train wall: {time.time()-t_full:.1f}s")

    # PRIMARY test reference: K=4 forward-greedy + Path-B C×S τ=100k.
    primary_artifact = ART / "test_K4_fwd_pathb.npy"
    if primary_artifact.exists():
        primary_test_pos = _pos(primary_artifact)
        primary_label = primary_artifact.name
    else:
        # Refit K=4 Path-B for diff
        print("\n--- Building K=4 Path-B C×S τ=100k reference for diff ---")
        F_oof_k4 = expand(np.column_stack(base_oofs[:-1]))
        F_test_k4 = expand(np.column_stack(base_tests[:-1]))
        w_g_k4 = fit_lr_aug(F_oof_k4, y)
        W_l_k4 = np.zeros((n_seg, len(w_g_k4)))
        c_k4 = np.zeros(n_seg, dtype=np.int64)
        m_k4 = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train == s)[0]
            c_k4[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
                continue
            W_l_k4[s] = fit_lr_aug(F_oof_k4[idx], y[idx])
            m_k4[s] = True
        n_loc = c_k4.astype(np.float64)
        alpha = n_loc / (n_loc + 100000)
        W_sh = alpha[:, None] * W_l_k4 + (1 - alpha[:, None]) * w_g_k4[None, :]
        primary_test_pos = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx = np.where(seg_test == s)[0]
            w = W_sh[s] if m_k4[s] else w_g_k4
            primary_test_pos[idx] = predict_aug(F_test_k4[idx], w)
        primary_label = "K=4 Path-B C×S τ=100k (rebuilt)"

    rare_thr = float(np.quantile(primary_test_pos, 0.99))
    primary_pos_mask = primary_test_pos >= rare_thr

    print(f"\n=== Path-B K=5 sweep (vs PRIMARY={primary_label}, "
          f"PRIMARY LB={PRIMARY_LB}) ===")
    final = dict(
        names=names,
        global_k5_lr_oof=auc_global,
        primary_lb=PRIMARY_LB,
        primary_label=primary_label,
        taus={},
    )
    Path("submissions").mkdir(exist_ok=True)
    for tau in taus:
        oof = oofs[tau]
        tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho_primary, _ = spearmanr(tp, primary_test_pos)
        new_pos = tp >= rare_thr
        flips_to_neg = int(np.sum(primary_pos_mask & ~new_pos))
        flips_to_pos = int(np.sum(~primary_pos_mask & new_pos))
        ratio = (min(flips_to_neg, flips_to_pos) /
                 max(flips_to_neg, flips_to_pos)
                 if max(flips_to_neg, flips_to_pos) > 0 else 1.0)
        d_oof_global = (auc - auc_global) * 1e4
        # ρ band → predicted LB Δ vs PRIMARY (uses path-B amp prior)
        if rho_primary >= 0.99996:
            band = "TIE"
        elif rho_primary >= 0.999:
            band = "TIGHT (ρ≥0.999)"
        elif rho_primary >= 0.995:
            band = "MID (0.995≤ρ<0.999)"
        else:
            band = "LOOSE (ρ<0.995)"
        print(f"\n  τ={tau}:")
        print(f"    Strat OOF: {auc:.5f}  Δ vs K=5 global: {d_oof_global:+.2f}bp")
        print(f"    ρ vs PRIMARY-test: {rho_primary:.6f}  ({band})")
        print(f"    flips: +→− {flips_to_neg}, −→+ {flips_to_pos}, "
              f"ratio {ratio:.3f}")

        np.save(ART / f"oof_path_b_K5_rf_yekenot_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]).astype(np.float32))
        np.save(ART / f"test_path_b_K5_rf_yekenot_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]).astype(np.float32))
        sub = sample_sub.copy()
        sub[TARGET] = tp
        sub.to_csv(
            f"submissions/submission_path_b_K5_rf_yekenot_tau{tau}.csv",
            index=False,
        )
        print(f"    saved oof/test_path_b_K5_rf_yekenot_tau{tau}_strat.npy")
        print(f"    saved submissions/submission_path_b_K5_rf_yekenot_tau{tau}.csv")
        final["taus"][str(tau)] = dict(
            oof=auc,
            delta_oof_global_bp=float(d_oof_global),
            rho_vs_primary=float(rho_primary),
            band=band,
            flips_to_neg=flips_to_neg,
            flips_to_pos=flips_to_pos,
            flip_ratio=float(ratio),
        )

    final["wall_s"] = time.time() - t0
    (ART / "path_b_K5_rf_yekenot_results.json").write_text(
        json.dumps(final, indent=2)
    )
    print(f"\n→ scripts/artifacts/path_b_K5_rf_yekenot_results.json"
          f"  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
