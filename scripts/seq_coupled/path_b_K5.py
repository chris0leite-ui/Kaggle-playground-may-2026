"""Path-B Compound × Stint amp test for K=5 (K=4 + V4 kNN-aug base).

Compares against the K=4 PRIMARY (Path-B C×S τ=100k OOF 0.95403) to
quantify the actual production-quality lift, since the current PRIMARY
uses Path-B not plain LR.

Three OOF outputs:
  K=4 plain LR-meta             — anchor
  K=4 + Path-B C×S τ=100k       — current PRIMARY (re-derive for sanity)
  K=5 + Path-B C×S τ=100k       — candidate

Output: scripts/artifacts/probe_K5_path_b.json
       scripts/artifacts/oof_K5_pathb.npy
       scripts/artifacts/test_K5_pathb.npy
       submissions/submission_K5_kNNaugbase_pathb.csv (file built but
                                                       NOT submitted)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path('scripts/artifacts')
SUB = Path('submissions')
SUB.mkdir(exist_ok=True, parents=True)
SEED, N_FOLDS = 42, 5
TAU = 100_000
MIN_ROWS = 1_000
MAX_ITER = 500


def pos(p):
    a = np.load(p)
    return a[:, 1] if a.ndim == 2 else a


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1-1e-9) / (1 - np.clip(P, 1e-9, 1-1e-9)))
    return np.hstack([P, rk, logit]).astype(np.float64)


def fit_lr_aug(F, y):
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver='lbfgs')
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def run_path_b(F_oof, F_test, seg_tr, seg_te, y, splits, n_seg, tau):
    """5-fold OOF + full-train test predictions for Path-B C×S."""
    oof = np.zeros(len(y))
    for k, (tri, vai) in enumerate(splits):
        w_global = fit_lr_aug(F_oof[tri], y[tri])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_tr[tri] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tri][idx])) < 2:
                continue
            W_local[s] = fit_lr_aug(F_oof[tri][idx], y[tri][idx])
            mask[s] = True
        n_local = counts.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local +
                    (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg_tr[vai]):
            idx = np.where(seg_tr[vai] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            oof[vai[idx]] = predict_aug(F_oof[vai[idx]], w)
    # full-train for test
    w_global_full = fit_lr_aug(F_oof, y)
    W_local_full = np.zeros((n_seg, len(w_global_full)))
    counts_full = np.zeros(n_seg, dtype=np.int64)
    mask_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_tr == s)[0]
        counts_full[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_local_full[s] = fit_lr_aug(F_oof[idx], y[idx])
        mask_full[s] = True
    n_local = counts_full.astype(np.float64)
    alpha = n_local / (n_local + tau)
    W_shrunk = (alpha[:, None] * W_local_full +
                (1 - alpha[:, None]) * w_global_full[None, :])
    test_pred = np.zeros(F_test.shape[0])
    for s in np.unique(seg_te):
        idx = np.where(seg_te == s)[0]
        w = W_shrunk[s] if mask_full[s] else w_global_full
        test_pred[idx] = predict_aug(F_test[idx], w)
    return oof, test_pred


def main():
    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)

    bases_K4_oof = {
        'h1d':  pos(ART / 'oof_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'oof_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'oof_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'oof_d16_orig_continuous_only_strat.npy'),
    }
    bases_K4_test = {
        'h1d':  pos(ART / 'test_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'test_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'test_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'test_d16_orig_continuous_only_strat.npy'),
    }
    v4_oof  = pos(ART / 'oof_knn_aug_base.npy')
    v4_test = pos(ART / 'test_knn_aug_base.npy')

    P_K4_oof  = np.column_stack([bases_K4_oof[k]  for k in ('h1d','p1cb','hgbc','d16o')])
    P_K4_test = np.column_stack([bases_K4_test[k] for k in ('h1d','p1cb','hgbc','d16o')])
    P_K5_oof  = np.column_stack([P_K4_oof,  v4_oof[:,  None]])
    P_K5_test = np.column_stack([P_K4_test, v4_test[:, None]])

    F_K4_oof, F_K4_test = expand(P_K4_oof), expand(P_K4_test)
    F_K5_oof, F_K5_test = expand(P_K5_oof), expand(P_K5_test)

    # Compound × Stint segmentation (matches d13e)
    cats = sorted(set(tr['Compound'].astype(str)) | set(te['Compound'].astype(str)))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = tr['Compound'].astype(str).map(cmp).astype(int).values
    c_te = te['Compound'].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(tr['Stint'].astype(int).values, 0, 5)
    s_te = np.clip(te['Stint'].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_tr = c_tr * 6 + s_tr
    seg_te = c_te * 6 + s_te

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    print(f'\n=== K=4 + Path-B C×S τ={TAU} (re-derive PRIMARY anchor) ===')
    oof_K4, test_K4 = run_path_b(F_K4_oof, F_K4_test, seg_tr, seg_te,
                                  y, splits, n_seg, TAU)
    auc_K4 = float(roc_auc_score(y, oof_K4))
    print(f'  OOF AUC = {auc_K4:.5f}')

    print(f'\n=== K=5 (K=4 + V4) + Path-B C×S τ={TAU} ===')
    oof_K5, test_K5 = run_path_b(F_K5_oof, F_K5_test, seg_tr, seg_te,
                                  y, splits, n_seg, TAU)
    auc_K5 = float(roc_auc_score(y, oof_K5))
    delta_bp = (auc_K5 - auc_K4) * 1e4
    print(f'  OOF AUC = {auc_K5:.5f}  Δ = {delta_bp:+.2f} bp')

    # Pre-submit diff
    rho_oof  = float(spearmanr(oof_K5, oof_K4).statistic)
    rho_test = float(spearmanr(test_K5, test_K4).statistic)
    print(f'\nρ_oof_K5_vs_K4   = {rho_oof:.6f}')
    print(f'ρ_test_K5_vs_K4  = {rho_test:.6f}')

    # Save artifacts
    np.save(ART / 'oof_K5_pathb.npy',  np.column_stack([1-oof_K5, oof_K5]).astype(np.float32))
    np.save(ART / 'test_K5_pathb.npy', np.column_stack([1-test_K5, test_K5]).astype(np.float32))

    # Build submission file (NOT submitted)
    sub = pd.DataFrame({'id': te['id'].values, 'PitNextLap': test_K5})
    sub.to_csv(SUB / 'submission_K5_kNNaugbase_pathb.csv', index=False)
    print(f'\nSubmission file written to '
          f'{SUB / "submission_K5_kNNaugbase_pathb.csv"} (NOT submitted)')

    out = {
        'oof_K4_pathb_auc':       auc_K4,
        'oof_K5_pathb_auc':       auc_K5,
        'delta_bp':               delta_bp,
        'rho_oof_K5_vs_K4':       rho_oof,
        'rho_test_K5_vs_K4':      rho_test,
        'verdict_oof':            ('PASS' if delta_bp >= 0.5 else
                                   'WEAK' if delta_bp >= 0 else 'FAIL'),
        'rule27_safe':            rho_test < 0.999,
    }
    (ART / 'probe_K5_path_b.json').write_text(json.dumps(out, indent=2))
    print('\n=== Path-B K=5 SUMMARY ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
