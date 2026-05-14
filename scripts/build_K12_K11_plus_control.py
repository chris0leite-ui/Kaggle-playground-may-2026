"""Build K=12 = K=11 + control_logloss (rich-feature LightGBM) under
per-segment shrinkage (Path-B C x Stint) at tau=100k.

Tonight's first overnight submit. The control variant of the
loss-diversity probe (plain log-loss LightGBM on 68 engineered features
via make_features_static + cross-validated target encodings) measured
a +17.957 bp lift at the K=11+1 gate. Rank correlation against K=11
cross-validation is 0.919 (versus the typical 0.997+ for K-add bases) -
the most diverse new base we've seen. The magnitude is suspicious so
this submit is the empirical reality-check.

Approach: 2-input restack. Treat K=11 path-b output and control variant
as two bases, run Path-B on top. This is faster than rebuilding K=12
from the 12 raw bases (would require a fresh LR-meta on 12*3=36 features).
The K=11+1 LR-meta gate already validated the approach is non-degenerate.

Outputs:
  artifacts/K12_K11plus_control_pathb_tau100000_oof.npy
  artifacts/K12_K11plus_control_pathb_tau100000_test.npy
  artifacts/K12_K11plus_control_pathb_tau100000.json
  artifacts/submission_K12_K11plus_control_pathb_tau100000.csv
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


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def fit_lr_aug(F, y):
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def lr_meta_oof(Xm, y):
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

    # Inputs: K=11 (full slim+K27+Path-B) and control_logloss (rich-feature LGBM).
    K11_oof = _pos(ART / "K11_full_pathb_tau100000_oof.npy")
    K11_test = _pos(ART / "K11_full_pathb_tau100000_test.npy")
    new_oof = _pos(ART / "loss_div_control_logloss_oof.npy")
    new_test = _pos(ART / "loss_div_control_logloss_test.npy")

    K11_auc = float(roc_auc_score(y, K11_oof))
    new_auc = float(roc_auc_score(y, new_oof))
    print(f"K=11 OOF AUC:                   {K11_auc:.5f}", flush=True)
    print(f"control_logloss OOF AUC:        {new_auc:.5f}", flush=True)
    rho_oof = float(spearmanr(K11_oof, new_oof).statistic)
    rho_test = float(spearmanr(K11_test, new_test).statistic)
    print(f"rho_oof  K=11 vs control:       {rho_oof:.6f}", flush=True)
    print(f"rho_test K=11 vs control:       {rho_test:.6f}", flush=True)

    # Plain LR-meta on the 2-input stack
    Xm_oof = expand(np.column_stack([K11_oof, new_oof]))
    om_plain = lr_meta_oof(Xm_oof, y)
    plain_auc = float(roc_auc_score(y, om_plain))
    delta_plain = (plain_auc - K11_auc) * 1e4
    print(f"\nK=12 plain LR-meta OOF:         {plain_auc:.5f}  delta vs K=11 {delta_plain:+.3f} bp",
          flush=True)

    # Split-stability sanity check: re-run LR-meta with random_state=43
    skf43 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=43)
    om43 = np.zeros(len(y))
    for tr, va in skf43.split(Xm_oof, y):
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        m.fit(Xm_oof[tr], y[tr])
        om43[va] = m.predict_proba(Xm_oof[va])[:, 1]
    auc43 = float(roc_auc_score(y, om43))
    delta43 = (auc43 - K11_auc) * 1e4
    print(f"K=12 plain LR-meta OOF (split=43): {auc43:.5f}  delta vs K=11 {delta43:+.3f} bp",
          flush=True)
    if abs(delta43 - delta_plain) > 5.0:
        print(f"  WARNING: lift varies >5 bp between split seeds ({delta_plain:+.3f} vs {delta43:+.3f})",
              flush=True)
        print(f"  This is a fold-split-artifact signal; transfer to LB may be much smaller.",
              flush=True)
    else:
        print(f"  split-stable: lift within {abs(delta43 - delta_plain):.3f} bp across seeds",
              flush=True)

    # Path-B on K=12 (2 base inputs)
    print(f"\n--- Path-B C x Stint tau={TAU} on K=12 (K=11 + control) ---", flush=True)
    oof_pb, test_pb = run_pathb([K11_oof, new_oof],
                                 [K11_test, new_test], train, test, y, tau=TAU)
    pb_auc = float(roc_auc_score(y, oof_pb))
    delta_pb = (pb_auc - K11_auc) * 1e4
    print(f"K=12 + Path-B OOF:              {pb_auc:.5f}  delta vs K=11 {delta_pb:+.3f} bp",
          flush=True)

    # Pre-submit diagnostics
    rho_oof_pb = float(spearmanr(oof_pb, K11_oof).statistic)
    rho_test_pb = float(spearmanr(test_pb, K11_test).statistic)
    print(f"rho_oof  K=12+pathb vs K=11:    {rho_oof_pb:.6f}", flush=True)
    print(f"rho_test K=12+pathb vs K=11:    {rho_test_pb:.6f}", flush=True)
    verdict = "ABORT_TIE" if rho_test_pb > 0.9999 else (
        "MARGINAL" if rho_test_pb > 0.999 else "OK_TO_SUBMIT")
    print(f"Verdict: {verdict}", flush=True)

    np.save(ART / f"K12_K11plus_control_pathb_tau{TAU}_oof.npy", oof_pb)
    np.save(ART / f"K12_K11plus_control_pathb_tau{TAU}_test.npy", test_pb)

    sub = pd.DataFrame({"id": test["id"], "PitNextLap": test_pb})
    csv_path = ART / f"submission_K12_K11plus_control_pathb_tau{TAU}.csv"
    sub.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path.name} ({len(sub)} rows)", flush=True)

    summary = {
        "K11_oof_auc": K11_auc,
        "control_oof_auc": new_auc,
        "rho_oof_K11_control": rho_oof,
        "rho_test_K11_control": rho_test,
        "K12_plain_lrmeta_oof_seed42": plain_auc,
        "K12_plain_lrmeta_oof_seed43": auc43,
        "delta_vs_K11_seed42_bp": delta_plain,
        "delta_vs_K11_seed43_bp": delta43,
        "split_stability_delta_bp": abs(delta43 - delta_plain),
        "K12_pathb_oof": pb_auc,
        "delta_vs_K11_pathb_bp": delta_pb,
        "rho_oof_K12pathb_K11": rho_oof_pb,
        "rho_test_K12pathb_K11": rho_test_pb,
        "verdict": verdict,
        "csv": csv_path.name,
        "elapsed_sec": time.time() - t0,
    }
    (ART / f"K12_K11plus_control_pathb_tau{TAU}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
