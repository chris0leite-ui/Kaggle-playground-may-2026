"""Fit LR-meta and RankNet-meta on K=4 + sequence-coupled features.

OOF discipline (5-fold StratifiedKFold, seed=42, the project's pinned split):
  Each fold k:
    1. Build the meta input X_tr (K=4 [P, rank, logit] = 12 row-local feats
       + 24 sequence-coupled feats from build_features.py = 36 total).
    2. Fit a meta on rows in folds≠k.
    3. Predict for rows in fold k → fold-k OOF.
  Final OOF = concatenation. Compare to:
    a. K=4 PRIMARY OOF (Path-B Compound × Stint τ=100k) at 0.95403.
    b. K=4 plain LR-meta OOF (row-local-only 12 features) — this is the
       "rank-lock anchor".

If seq-coupled OOF beats the plain LR-meta by ≥+0.5 bp, the rank-lock
ceiling has cracked at the meta layer.

Two losses tested:
  - LR (binary cross-entropy) — sklearn LogisticRegression with C=1.
  - RankNet pairwise — minimise Σ_{(i,j)|y_i=1,y_j=0} -log σ(s_i - s_j).
    Direct AUC objective, perfect Q6 alignment with row-AUC metric.
    Implemented in numpy with subsampled pair batches (2M pairs).

Output:
    scripts/artifacts/probe_seq_coupled_meta.json
    scripts/artifacts/oof_seq_coupled_lr.npy   — final LR OOF
    scripts/artifacts/oof_seq_coupled_ranknet.npy — final RankNet OOF
    scripts/artifacts/test_seq_coupled_lr.npy
    scripts/artifacts/test_seq_coupled_ranknet.npy
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
from sklearn.preprocessing import StandardScaler

ART = Path('scripts/artifacts')
SEED = 42
N_FOLDS = 5
RANKNET_PAIRS_PER_FOLD = 2_000_000      # subsample for tractability
RANKNET_LR = 0.05
RANKNET_EPOCHS = 200
RANKNET_L2 = 1e-4


def pos(p):
    a = np.load(p)
    return a[:, 1] if a.ndim == 2 else a


def expand_K4(oofs: dict[str, np.ndarray]) -> np.ndarray:
    """[P, rank, logit] expansion of the 4 base OOFs → 12-feature row-local."""
    parts = []
    for v in oofs.values():
        parts.append(v)
        parts.append(rankdata(v) / len(v))
        parts.append(np.log(np.clip(v, 1e-7, 1 - 1e-7)
                            / np.clip(1 - v, 1e-7, 1 - 1e-7)))
    return np.column_stack(parts).astype(np.float32)


def fit_lr(X_tr, y_tr, X_va):
    sc = StandardScaler().fit(X_tr)
    Xt = sc.transform(X_tr)
    Xv = sc.transform(X_va)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    lr.fit(Xt, y_tr)
    return lr.predict_proba(Xv)[:, 1]


def fit_ranknet(X_tr, y_tr, X_va, *, rng: np.random.Generator):
    """Logistic RankNet on pairwise differences, minibatch SGD with momentum.

    Model: score(x) = w·x + b. We don't need a bias for AUC since AUC is
    translation-invariant. Loss on pair (i, j) with y_i=1, y_j=0 is
    -log σ(s_i - s_j) = log(1 + exp(s_j - s_i)).
    """
    sc = StandardScaler().fit(X_tr)
    Xt = sc.transform(X_tr).astype(np.float32)
    Xv = sc.transform(X_va).astype(np.float32)
    n_feat = Xt.shape[1]
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]

    w = np.zeros(n_feat, dtype=np.float32)
    v = np.zeros_like(w)              # momentum buffer
    momentum = 0.9
    lr = RANKNET_LR
    batch = 200_000

    for ep in range(RANKNET_EPOCHS):
        # subsample pairs for this epoch
        pi = rng.choice(pos_idx, RANKNET_PAIRS_PER_FOLD, replace=True)
        ni = rng.choice(neg_idx, RANKNET_PAIRS_PER_FOLD, replace=True)
        # iterate minibatches
        for st in range(0, RANKNET_PAIRS_PER_FOLD, batch):
            ed = min(st + batch, RANKNET_PAIRS_PER_FOLD)
            xp = Xt[pi[st:ed]]
            xn = Xt[ni[st:ed]]
            d = xp - xn                      # (b, f)
            s = d @ w                        # (b,)
            sigm = 1.0 / (1.0 + np.exp(-s))  # σ(s)
            # gradient of -log σ(s) wrt w is -(1 - sigm) * d
            grad = -((1.0 - sigm)[:, None] * d).mean(axis=0)
            grad += RANKNET_L2 * w
            v = momentum * v - lr * grad
            w = w + v
        # decay LR slowly
        if ep == 100:
            lr *= 0.5
    return Xv @ w


def main():
    print('--- loading data ---')
    tr = pd.read_csv('data/train.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)

    oofs = {
        'h1d':  pos(ART / 'oof_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'oof_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'oof_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'oof_d16_orig_continuous_only_strat.npy'),
    }
    tests = {
        'h1d':  pos(ART / 'test_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'test_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'test_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'test_d16_orig_continuous_only_strat.npy'),
    }
    k4_oof = pos(ART / 'oof_K4_fwd_pathb.npy')

    X_local_tr = expand_K4(oofs)               # (n_train, 12)
    X_local_te = expand_K4(tests)              # (n_test,  12)
    X_seq_tr   = np.load(ART / 'seq_coupled_X_train.npy')   # (n_train, 24)
    X_seq_te   = np.load(ART / 'seq_coupled_X_test.npy')    # (n_test,  24)

    X_full_tr = np.concatenate([X_local_tr, X_seq_tr], axis=1)
    X_full_te = np.concatenate([X_local_te, X_seq_te], axis=1)
    print(f'X_full_tr: {X_full_tr.shape}')

    # Anchors:
    auc_k4_primary = float(roc_auc_score(y, k4_oof))
    print(f'\nANCHOR: K=4 PRIMARY (Path-B C×S τ=100k) OOF AUC = {auc_k4_primary:.5f}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)

    # Plain LR meta on row-local 12 features (the rank-lock anchor)
    print('\n=== fitting plain LR meta on K=4 row-local [P,rank,logit] (12 feat) ===')
    oof_lr_local = np.zeros(n, dtype=np.float32)
    test_lr_local = np.zeros(len(X_local_te), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
        t0 = time.time()
        oof_lr_local[va_idx] = fit_lr(X_local_tr[tr_idx], y[tr_idx], X_local_tr[va_idx])
        # test: fit again on full fold's training rows for averaging
        test_pred = fit_lr(X_local_tr[tr_idx], y[tr_idx], X_local_te)
        test_lr_local += test_pred / N_FOLDS
        print(f'  fold {k}: oof_auc_so_far '
              f'{roc_auc_score(y[va_idx], oof_lr_local[va_idx]):.5f}  '
              f'({time.time()-t0:.1f}s)')
    auc_lr_local = float(roc_auc_score(y, oof_lr_local))
    print(f'  → AUC = {auc_lr_local:.5f}  Δ vs PRIMARY = {(auc_lr_local-auc_k4_primary)*1e4:+.2f} bp')

    # LR meta on row-local + sequence-coupled (36 features)
    print('\n=== fitting LR meta on K=4 + sequence-coupled (36 feat) ===')
    oof_lr_full = np.zeros(n, dtype=np.float32)
    test_lr_full = np.zeros(len(X_full_te), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
        t0 = time.time()
        oof_lr_full[va_idx] = fit_lr(X_full_tr[tr_idx], y[tr_idx], X_full_tr[va_idx])
        test_pred = fit_lr(X_full_tr[tr_idx], y[tr_idx], X_full_te)
        test_lr_full += test_pred / N_FOLDS
        print(f'  fold {k}: oof_auc_so_far '
              f'{roc_auc_score(y[va_idx], oof_lr_full[va_idx]):.5f}  '
              f'({time.time()-t0:.1f}s)')
    auc_lr_full = float(roc_auc_score(y, oof_lr_full))
    print(f'  → AUC = {auc_lr_full:.5f}  Δ vs PRIMARY = {(auc_lr_full-auc_k4_primary)*1e4:+.2f} bp  '
          f'Δ vs LR-row-local = {(auc_lr_full-auc_lr_local)*1e4:+.2f} bp')

    # RankNet meta on K=4 + sequence-coupled
    print('\n=== fitting RankNet meta on K=4 + sequence-coupled (36 feat) ===')
    print(f'  pairs/fold: {RANKNET_PAIRS_PER_FOLD:,}; epochs: {RANKNET_EPOCHS}; '
          f'lr: {RANKNET_LR}; momentum: 0.9')
    oof_rn_full = np.zeros(n, dtype=np.float32)
    test_rn_full = np.zeros(len(X_full_te), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
        t0 = time.time()
        oof_rn_full[va_idx] = fit_ranknet(X_full_tr[tr_idx], y[tr_idx], X_full_tr[va_idx], rng=rng)
        test_pred = fit_ranknet(X_full_tr[tr_idx], y[tr_idx], X_full_te, rng=rng)
        test_rn_full += test_pred / N_FOLDS
        print(f'  fold {k}: oof_auc_so_far '
              f'{roc_auc_score(y[va_idx], oof_rn_full[va_idx]):.5f}  '
              f'({time.time()-t0:.1f}s)')
    auc_rn_full = float(roc_auc_score(y, oof_rn_full))
    print(f'  → AUC = {auc_rn_full:.5f}  Δ vs PRIMARY = {(auc_rn_full-auc_k4_primary)*1e4:+.2f} bp  '
          f'Δ vs LR-row-local = {(auc_rn_full-auc_lr_local)*1e4:+.2f} bp')

    # Save artifacts
    np.save(ART / 'oof_seq_coupled_lr_local.npy',  oof_lr_local)
    np.save(ART / 'test_seq_coupled_lr_local.npy', test_lr_local)
    np.save(ART / 'oof_seq_coupled_lr.npy',        oof_lr_full)
    np.save(ART / 'test_seq_coupled_lr.npy',       test_lr_full)
    np.save(ART / 'oof_seq_coupled_ranknet.npy',   oof_rn_full)
    np.save(ART / 'test_seq_coupled_ranknet.npy',  test_rn_full)

    out = {
        'anchor_K4_PRIMARY_oof_auc':    auc_k4_primary,
        'lr_row_local_oof_auc':         auc_lr_local,
        'lr_full_oof_auc':              auc_lr_full,
        'ranknet_full_oof_auc':         auc_rn_full,
        'delta_lr_full_vs_primary_bp':  (auc_lr_full - auc_k4_primary) * 1e4,
        'delta_lr_full_vs_lr_local_bp': (auc_lr_full - auc_lr_local) * 1e4,
        'delta_ranknet_vs_primary_bp':  (auc_rn_full - auc_k4_primary) * 1e4,
        'delta_ranknet_vs_lr_local_bp': (auc_rn_full - auc_lr_local) * 1e4,
        'verdict_lr_full': ('PASS' if (auc_lr_full - auc_lr_local) * 1e4 >= 0.5
                            else 'WEAK' if (auc_lr_full - auc_lr_local) * 1e4 >= 0
                            else 'FAIL'),
        'verdict_ranknet':  ('PASS' if (auc_rn_full - auc_lr_local) * 1e4 >= 0.5
                             else 'WEAK' if (auc_rn_full - auc_lr_local) * 1e4 >= 0
                             else 'FAIL'),
    }
    (ART / 'probe_seq_coupled_meta.json').write_text(json.dumps(out, indent=2))
    print('\n=== SUMMARY ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
