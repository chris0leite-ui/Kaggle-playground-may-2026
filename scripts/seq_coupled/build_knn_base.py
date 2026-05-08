"""V4 (fallback if V3 fails): a NEW BASE that uses transductive
kNN-target-mean as an INPUT FEATURE alongside raw row features.

Difference from V3 (kNN-target-mean as META feature):
  V3 → LR meta sees [K=4 [P,rank,logit], kNN_y_mean, disc] = 14 feat.
       LR is linear; can only weight kNN_y_mean as a global scalar.
  V4 → A new LightGBM BASE sees [raw row feats, kNN_y_mean, kNN_y_std]
       as input. The base can learn non-linear interactions between
       row features and local label density (e.g., "for high-TyreLife
       rows in MEDIUM compound, kNN_y_mean is more informative").
       The OOF of this base joins K=4 as a 5th base; standard K=4+1
       LR-meta gate.

This base:
  - Tests whether transductive label info adds NEW partial-correlation
    with y conditional on row features, when ingested non-linearly
    (LightGBM tree splits) instead of linearly (LR meta).
  - Per-fold OOF discipline: kNN_y_mean for fold-k rows is computed
    using TRAINING rows in folds≠k only (no leakage).
  - Fold-safe LightGBM: trained on folds≠k, predicted on fold k.

Output:
    scripts/artifacts/oof_knn_aug_base.npy  (n_train, 2)
    scripts/artifacts/test_knn_aug_base.npy (n_test, 2)
    scripts/artifacts/probe_knn_aug_base_results.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ART = Path('scripts/artifacts')
SEED = 42
N_FOLDS = 5
K_NEIGHBOURS = 20


def build_kNN_features(df: pd.DataFrame) -> np.ndarray:
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


def build_row_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Raw + categorical-encoded features for the LightGBM base."""
    X_blocks = [
        ('LapNumber',     df['LapNumber'].astype(np.float32).to_numpy()),
        ('TyreLife',      df['TyreLife'].astype(np.float32).to_numpy()),
        ('RaceProgress',  df['RaceProgress'].astype(np.float32).to_numpy()),
        ('Stint',         df['Stint'].astype(np.float32).to_numpy()),
        ('PitStop',       df['PitStop'].astype(np.float32).to_numpy()),
        ('LapTime',       df['LapTime (s)'].astype(np.float32).to_numpy()
                          if 'LapTime (s)' in df.columns
                          else df['LapTime'].astype(np.float32).to_numpy()),
        ('Year',          df['Year'].astype(np.float32).to_numpy()),
    ]
    cmp_codes = df['Compound'].astype(str).astype('category').cat.codes.astype(np.int32)
    drv_codes = df['Driver'].astype(str).astype('category').cat.codes.astype(np.int32)
    race_codes = df['Race'].astype(str).astype('category').cat.codes.astype(np.int32)
    X_blocks += [
        ('Compound_code', cmp_codes.to_numpy()),
        ('Driver_code',   drv_codes.to_numpy()),
        ('Race_code',     race_codes.to_numpy()),
    ]
    if 'Cumulative_Degradation' in df.columns:
        X_blocks.append(('Cumulative_Degradation',
                         df['Cumulative_Degradation'].astype(np.float32).to_numpy()))
    if 'LapTime_Delta' in df.columns:
        X_blocks.append(('LapTime_Delta',
                         df['LapTime_Delta'].astype(np.float32).to_numpy()))
    if 'Position' in df.columns:
        X_blocks.append(('Position',
                         df['Position'].astype(np.float32).to_numpy()))
    names = [n for n, _ in X_blocks]
    X = np.column_stack([b for _, b in X_blocks]).astype(np.float32)
    return X, names


def main():
    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)
    print(f'n_train={n:,}, n_test={len(te):,}')
    print(f'train columns: {[c for c in tr.columns if c != "id"]}')

    X_kNN_tr = build_kNN_features(tr)
    X_kNN_te = build_kNN_features(te)
    X_row_tr, names = build_row_features(tr)
    X_row_te, _     = build_row_features(te)
    print(f'X_kNN_tr: {X_kNN_tr.shape}')
    print(f'X_row_tr: {X_row_tr.shape}, features: {names}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    print(f'\n=== Computing kNN target-mean + std (K={K_NEIGHBOURS}) per fold ===')
    y_knn_mean = np.zeros(n, dtype=np.float32)
    y_knn_std  = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()
        tree = BallTree(X_kNN_tr[tri], leaf_size=40)
        _, idx = tree.query(X_kNN_tr[vai], k=K_NEIGHBOURS)
        nb_y = y[tri][idx]
        y_knn_mean[vai] = nb_y.mean(axis=1).astype(np.float32)
        y_knn_std[vai]  = nb_y.std(axis=1).astype(np.float32)
        print(f'  fold {k}: tree built + queried in {time.time()-t0:.1f}s; '
              f'y_knn_mean[vai] mean={y_knn_mean[vai].mean():.4f}, '
              f'std_mean={y_knn_std[vai].mean():.4f}')

    # For test: query against full training set
    print('\n--- kNN for test against full train set ---')
    t0 = time.time()
    tree_all = BallTree(X_kNN_tr, leaf_size=40)
    _, idx_te = tree_all.query(X_kNN_te, k=K_NEIGHBOURS)
    nb_y_te = y[idx_te]
    y_knn_mean_te = nb_y_te.mean(axis=1).astype(np.float32)
    y_knn_std_te  = nb_y_te.std(axis=1).astype(np.float32)
    print(f'  test query: {time.time()-t0:.1f}s')

    # Build augmented input for the base
    X_aug_tr = np.column_stack([X_row_tr, y_knn_mean, y_knn_std]).astype(np.float32)
    X_aug_te = np.column_stack([X_row_te, y_knn_mean_te, y_knn_std_te]).astype(np.float32)
    aug_names = names + ['kNN_y_mean', 'kNN_y_std']
    cat_idx = [aug_names.index(c) for c in ('Compound_code', 'Driver_code', 'Race_code')]
    print(f'X_aug_tr: {X_aug_tr.shape}; cat indices: {cat_idx}')

    print('\n=== Training kNN-augmented LightGBM base (5-fold OOF) ===')
    oof = np.zeros(n, dtype=np.float32)
    test_pred = np.zeros(len(te), dtype=np.float32)
    base_params = dict(
        objective='binary', metric='auc',
        num_leaves=63, max_depth=-1,
        learning_rate=0.03, n_estimators=2000,
        reg_alpha=0.5, reg_lambda=2.0,
        bagging_fraction=0.8, bagging_freq=5,
        feature_fraction=0.85, min_data_in_leaf=80,
        verbose=-1, seed=SEED,
    )
    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()
        m = lgb.LGBMClassifier(**base_params)
        m.fit(X_aug_tr[tri], y[tri],
              eval_set=[(X_aug_tr[vai], y[vai])],
              categorical_feature=cat_idx,
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[vai] = m.predict_proba(X_aug_tr[vai])[:, 1]
        test_pred += m.predict_proba(X_aug_te)[:, 1] / N_FOLDS
        print(f'  fold {k}: AUC={roc_auc_score(y[vai], oof[vai]):.5f}  '
              f'best_iter={m.booster_.best_iteration}  ({time.time()-t0:.1f}s)')
    auc_oof = float(roc_auc_score(y, oof))
    print(f'\nkNN-augmented base 5-fold OOF AUC: {auc_oof:.5f}')

    # Save 2-column [neg, pos] format
    np.save(ART / 'oof_knn_aug_base.npy',  np.column_stack([1 - oof, oof]).astype(np.float32))
    np.save(ART / 'test_knn_aug_base.npy', np.column_stack([1 - test_pred, test_pred]).astype(np.float32))
    out = {
        'oof_auc_standalone': auc_oof,
        'k_neighbours':       K_NEIGHBOURS,
        'feature_names':      aug_names,
    }
    (ART / 'probe_knn_aug_base_results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
