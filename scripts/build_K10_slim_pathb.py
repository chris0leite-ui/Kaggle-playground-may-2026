"""Build K=10 = K=4 forward-greedy + all six slim nearest-neighbour bases
under per-segment shrinkage (Path-B C x Stint) at tau=100k. NO K=27
super-base.

This is the slim-nearest-neighbour-only complement to the K=11 stack.
Historical K=9-class (slim-kNN-only) submission landed at LB 0.95375,
about 1 bp below K=11. Its value is as the diversity leg of a 70/30
rank-blend with K=11 — that blend lifted LB from 0.95385 to 0.95386
on 2026-05-12.

Test rank correlation against K=11 expected to be 0.999X (slim) — well
below the K=11 vs K=8 0.999901 reading, since K=10 ablates the K=27
super-base entirely.

Outputs:
  artifacts/K10_slim_pathb_tau100000_oof.npy
  artifacts/K10_slim_pathb_tau100000_test.npy
  artifacts/K10_slim_pathb_tau100000.json
  artifacts/submission_K10_slim_pathb_tau100000.csv
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
DATA = Path("data")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MIN_ROWS = 1000
MAX_ITER = 500
TAU = 100_000


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def fit_lr_aug(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def lr_meta_oof(Xm: np.ndarray, y: np.ndarray) -> np.ndarray:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    om = np.zeros(len(y))
    for tr, va in skf.split(Xm, y):
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        m.fit(Xm[tr], y[tr])
        om[va] = m.predict_proba(Xm[va])[:, 1]
    return om


def run_pathb(base_oofs, base_tests, train, test, y, tau=TAU):
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
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
    splits = list(skf.split(np.zeros(len(y)), y))
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t0 = time.time()
        w_g = fit_lr_aug(F_oof[tr], y[tr])
        W_l = np.zeros((n_seg, len(w_g)))
        cnt = np.zeros(n_seg, dtype=np.int64)
        msk = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            cnt[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_l[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            msk[s] = True
        n_local = cnt.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_sh = alpha[:, None] * W_l + (1 - alpha[:, None]) * w_g[None, :]
        for s in np.unique(seg_train[va]):
            idx_v = np.where(seg_train[va] == s)[0]
            w = W_sh[s] if msk[s] else w_g
            oof[va[idx_v]] = predict_aug(F_oof[va[idx_v]], w)
        print(f"  fold {fold+1}: {time.time()-t0:.1f}s", flush=True)

    w_g_full = fit_lr_aug(F_oof, y)
    W_l_full = np.zeros((n_seg, len(w_g_full)))
    cnt_full = np.zeros(n_seg, dtype=np.int64)
    msk_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        cnt_full[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_l_full[s] = fit_lr_aug(F_oof[idx], y[idx])
        msk_full[s] = True
    alpha = cnt_full.astype(np.float64) / (cnt_full.astype(np.float64) + tau)
    W_sh = alpha[:, None] * W_l_full + (1 - alpha[:, None]) * w_g_full[None, :]
    test_pred = np.zeros(len(test))
    for s in np.unique(seg_test):
        idx_t = np.where(seg_test == s)[0]
        w = W_sh[s] if msk_full[s] else w_g_full
        test_pred[idx_t] = predict_aug(F_test[idx_t], w)
    return oof, test_pred


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values

    K4_FILES = [
        ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy",     "test_d17_h1d_yekenot_full_strat.npy"),
        ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy",      "test_p1_single_cb_v4_gpu_strat.npy"),
        ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy",             "test_f1_hgbc_deep_strat.npy"),
        ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ]
    EXTRAS = [
        ("qAT", "dgp_v3_qAT_K1_oof.npy",          "dgp_v3_qAT_K1_test.npy"),
        ("qAV", "dgp_v3_qAV_K1_7feat_oof.npy",    "dgp_v3_qAV_K1_7feat_test.npy"),
        ("qAO", "dgp_v3_qAO_knn_multi_oof.npy",   "dgp_v3_qAO_knn_multi_test.npy"),
        ("qAA", "dgp_v3_qAA_stint_imputed_oof.npy", "dgp_v3_qAA_stint_imputed_test.npy"),
        ("qAF", "dgp_v3_qAF_d16plus_oof.npy",     "dgp_v3_qAF_d16plus_test.npy"),
        ("qAK", "dgp_v3_qAK_knn3_oof.npy",        "dgp_v3_qAK_knn3_test.npy"),
    ]
    all_oof = [_pos(ART / o) for _, o, _ in K4_FILES + EXTRAS]
    all_test = [_pos(ART / t) for _, _, t in K4_FILES + EXTRAS]
    names = [n for n, _, _ in K4_FILES + EXTRAS]
    print(f"Loaded K={len(names)} bases: {names}", flush=True)

    plain_oof = lr_meta_oof(expand(np.column_stack(all_oof)), y)
    plain_auc = float(roc_auc_score(y, plain_oof))
    print(f"K=10 plain LR-meta OOF: {plain_auc:.5f}", flush=True)

    # References from disk
    K11_oof_path = ART / "K11_full_pathb_tau100000_oof.npy"
    K11_test_path = ART / "K11_full_pathb_tau100000_test.npy"
    K11_pathb_oof = _pos(K11_oof_path) if K11_oof_path.exists() else None
    K11_pathb_test = _pos(K11_test_path) if K11_test_path.exists() else None

    print(f"\n--- Running Path-B C x Stint tau={TAU} on K=10 (no K=27) ---", flush=True)
    oof, test_pred = run_pathb(all_oof, all_test, train, test, y, tau=TAU)
    auc = float(roc_auc_score(y, oof))
    print(f"\nK=10 + Path-B OOF AUC:   {auc:.5f}", flush=True)

    if K11_pathb_oof is not None:
        K11_auc = float(roc_auc_score(y, K11_pathb_oof))
        rho_oof = float(spearmanr(oof, K11_pathb_oof).statistic)
        rho_test = float(spearmanr(test_pred, K11_pathb_test).statistic)
        print(f"K=11 + Path-B reference: {K11_auc:.5f}", flush=True)
        print(f"  delta vs K=11:         {(auc - K11_auc) * 1e4:+.3f} bp", flush=True)
        print(f"rho_oof  vs K=11:        {rho_oof:.6f}", flush=True)
        print(f"rho_test vs K=11:        {rho_test:.6f}", flush=True)

    np.save(ART / f"K10_slim_pathb_tau{TAU}_oof.npy", oof)
    np.save(ART / f"K10_slim_pathb_tau{TAU}_test.npy", test_pred)
    sub = pd.DataFrame({"id": test["id"], "PitNextLap": test_pred})
    csv_path = ART / f"submission_K10_slim_pathb_tau{TAU}.csv"
    sub.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path.name}  ({len(sub)} rows)", flush=True)

    summary = {
        "bases": names,
        "tau": TAU,
        "K10_plain_oof": plain_auc,
        "K10_pathb_oof": auc,
        "K11_pathb_oof_ref": K11_auc if K11_pathb_oof is not None else None,
        "delta_vs_K11_bp": (auc - K11_auc) * 1e4 if K11_pathb_oof is not None else None,
        "rho_oof_vs_K11": rho_oof if K11_pathb_oof is not None else None,
        "rho_test_vs_K11": rho_test if K11_pathb_oof is not None else None,
        "csv": csv_path.name,
        "elapsed_sec": time.time() - t0,
    }
    (ART / f"K10_slim_pathb_tau{TAU}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
