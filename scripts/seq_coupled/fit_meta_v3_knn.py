"""V3: kNN-target-mean correction at the META layer.

V1 / V2 result: sequence-coupled features + LR + RankNet + LGBM-meta
all WEAK or NULL (≤+0.03 bp). The rank-lock at K=4 is at the
**conditional target-correlation** level — features that don't
add new partial correlation with y CONDITIONAL on K=4 get absorbed
even when their feature-space orthogonality is high (R² 0.487 with
look-ahead).

V3 angle: a TRANSDUCTIVE feature that the K=4 bases by construction
cannot replicate — the empirical y-rate among K-nearest-neighbours
in feature space. The discrepancy `y_knn_mean - K=4_PRIMARY_pred`
flags rows where the local feature-density disagrees with the model.
This is fundamentally different from per-row prediction and the d16
GRU sequence base (which infers a row prediction from sequence
context, not from local-feature density).

Algorithm (per fold k):
  1. Build feature matrix X_feat from train rows in folds≠k:
     {LapNumber, TyreLife, RaceProgress, Stint, PitStop, lap-time
     residual against (Race, Year) median, Compound one-hot}.
     Standardise.
  2. Fit BallTree (or annoy/HNSW) on standardised X_feat[folds≠k].
  3. For each row R in fold k: query K=20 nearest neighbours in
     X_feat (training rows only). Compute y_knn_mean(R) = mean of
     y over those neighbours.
  4. Compute discrepancy `disc(R) = y_knn_mean(R) - K=4_PRIMARY[R]`.
  5. Add disc + y_knn_mean as 2 extra meta features alongside the
     12 row-local K=4 [P, rank, logit].
  6. Fit LR meta; report 5-fold OOF AUC.

For test rows: query against the FULL training set (all 5 folds
combined since we use OOF discipline for the meta inputs, not for
the kNN reference set itself).

Cost ~25 min (KNN setup ~15 min, 5-fold OOF meta ~10 min).

Q6: aligned. The kNN-target-mean is a regression-style auxiliary
feature for the LR meta predicting binary AUC.
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
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

ART = Path('scripts/artifacts')
SEED, N_FOLDS = 42, 5
K_NEIGHBOURS = 20


def pos(p):
    a = np.load(p)
    return a[:, 1] if a.ndim == 2 else a


def expand_K4(oofs):
    parts = []
    for v in oofs.values():
        parts.append(v)
        parts.append(rankdata(v) / len(v))
        parts.append(np.log(np.clip(v, 1e-7, 1-1e-7) /
                            np.clip(1-v, 1e-7, 1-1e-7)))
    return np.column_stack(parts).astype(np.float32)


def build_kNN_features(df: pd.DataFrame) -> np.ndarray:
    """Standardised feature matrix for distance kNN."""
    cmp_dum = pd.get_dummies(df['Compound'].astype(str), prefix='cmp').astype(np.float32)
    F = np.column_stack([
        df['LapNumber'].astype(np.float32).to_numpy(),
        df['TyreLife'].astype(np.float32).to_numpy(),
        df['RaceProgress'].astype(np.float32).to_numpy(),
        df['Stint'].astype(np.float32).to_numpy(),
        df['PitStop'].astype(np.float32).to_numpy(),
        cmp_dum.to_numpy(),
    ])
    sc = StandardScaler().fit(F)
    return sc.transform(F).astype(np.float32)


def fit_lr(X_tr, y_tr, X_va):
    sc = StandardScaler().fit(X_tr)
    Xt, Xv = sc.transform(X_tr), sc.transform(X_va)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    lr.fit(Xt, y_tr)
    return lr.predict_proba(Xv)[:, 1]


def main():
    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)
    n_test = len(te)
    print(f'n_train={n:,}, n_test={n_test:,}')

    X_kNN_tr = build_kNN_features(tr[['LapNumber', 'TyreLife', 'RaceProgress',
                                       'Stint', 'PitStop', 'Compound']])
    X_kNN_te = build_kNN_features(te[['LapNumber', 'TyreLife', 'RaceProgress',
                                       'Stint', 'PitStop', 'Compound']])
    print(f'X_kNN_tr: {X_kNN_tr.shape}')

    oofs = {
        'h1d':  pos(ART / 'oof_d17_h1d_yekenot_full_strat.npy'),
        'p1cb': pos(ART / 'oof_p1_single_cb_v4_gpu_strat.npy'),
        'hgbc': pos(ART / 'oof_f1_hgbc_deep_strat.npy'),
        'd16o': pos(ART / 'oof_d16_orig_continuous_only_strat.npy'),
    }
    k4_oof = pos(ART / 'oof_K4_fwd_pathb.npy')
    X_local = expand_K4(oofs)

    auc_anchor = float(roc_auc_score(y, k4_oof))
    print(f'\nANCHOR K=4 PRIMARY OOF AUC: {auc_anchor:.5f}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    print(f'\n=== Computing kNN target-mean (K={K_NEIGHBOURS}) per fold ===')
    y_knn = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()
        # tree on training rows only
        tree = BallTree(X_kNN_tr[tri], leaf_size=40)
        # query val rows for K nearest neighbours
        _, idx = tree.query(X_kNN_tr[vai], k=K_NEIGHBOURS)
        y_knn[vai] = y[tri][idx].mean(axis=1).astype(np.float32)
        print(f'  fold {k}: tree built + queried in {time.time()-t0:.1f}s; '
              f'y_knn[vai] mean={y_knn[vai].mean():.4f}, '
              f'std={y_knn[vai].std():.4f}')

    # For test: query full training set
    print(f'\n--- kNN for test against full train set ---')
    t0 = time.time()
    tree_all = BallTree(X_kNN_tr, leaf_size=40)
    _, idx_te = tree_all.query(X_kNN_te, k=K_NEIGHBOURS)
    y_knn_te = y[idx_te].mean(axis=1).astype(np.float32)
    print(f'  test query: {time.time()-t0:.1f}s; '
          f'y_knn_te mean={y_knn_te.mean():.4f}, std={y_knn_te.std():.4f}')

    # Discrepancy feature
    disc_tr = (y_knn - k4_oof).astype(np.float32)
    print(f'\nDiscrepancy (y_knn - K4): mean={disc_tr.mean():.5f}, '
          f'std={disc_tr.std():.5f}')
    print(f'Correlation with y: {np.corrcoef(disc_tr, y)[0,1]:.5f}')

    # Build meta inputs (12 row-local + 2 kNN features)
    X_meta = np.concatenate([X_local,
                              y_knn[:, None],
                              disc_tr[:, None]], axis=1).astype(np.float32)
    print(f'X_meta shape: {X_meta.shape}')

    print(f'\n=== Fitting LR meta on K=4 + kNN features (14 feat) ===')
    oof_lr_knn = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()
        oof_lr_knn[vai] = fit_lr(X_meta[tri], y[tri], X_meta[vai])
        print(f'  fold {k}: {roc_auc_score(y[vai], oof_lr_knn[vai]):.5f} '
              f'({time.time()-t0:.1f}s)')
    auc_lr_knn = float(roc_auc_score(y, oof_lr_knn))
    print(f'  → AUC = {auc_lr_knn:.5f}  Δ vs PRIMARY = {(auc_lr_knn-auc_anchor)*1e4:+.2f} bp')

    # Bare kNN AUC
    auc_y_knn_bare = float(roc_auc_score(y, y_knn))
    print(f'\nBare y_knn AUC: {auc_y_knn_bare:.5f}')

    out = {
        'anchor_K4_PRIMARY_oof_auc': auc_anchor,
        'lr_with_knn_oof_auc':       auc_lr_knn,
        'bare_y_knn_oof_auc':        auc_y_knn_bare,
        'delta_lr_knn_vs_primary_bp': (auc_lr_knn - auc_anchor) * 1e4,
        'discrepancy_corr_with_y':   float(np.corrcoef(disc_tr, y)[0, 1]),
        'verdict': ('PASS' if (auc_lr_knn - auc_anchor) * 1e4 >= 0.5
                    else 'WEAK' if (auc_lr_knn - auc_anchor) * 1e4 >= -0.2
                    else 'FAIL'),
    }
    (ART / 'probe_seq_coupled_meta_v3.json').write_text(json.dumps(out, indent=2))
    print('\n=== V3 SUMMARY ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
