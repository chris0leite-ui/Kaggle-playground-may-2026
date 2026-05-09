"""V5: V4-mechanism extended with multi-cell target encoding.

V4 won by ingesting kNN-target-mean (K=20 in standardised feature space)
through tree splits inside a new LightGBM base. +0.20 bp OOF, +0.80 bp LB.

V5 hypothesis: more transductive label-density sources, ingested through
the same tree-splits mechanism, lift further. Specifically, add
fold-safe per-cell target-encoded features at multiple granularities:
  - Compound × Stint               (most granular cell, ~24 cells)
  - Compound × Stint × RaceProgress-bin (finer; ~120 cells)
  - Driver × Compound              (cross-cut by driver; ~hundreds)
  - Driver × Stint                 (driver-specific stint pattern)
  - Race × Compound                (race-specific compound use)

All target encodings are computed PER FOLD using only training rows
in folds≠k, with prior-mean smoothing (alpha=50) so small cells fall
back to the global rate.

Output:
    scripts/artifacts/oof_v5_base.npy  (n_train, 2)
    scripts/artifacts/test_v5_base.npy (n_test,  2)
    scripts/artifacts/probe_v5_base_results.json

If V5 standalone OOF AUC > V4's 0.94163, we swap V4 → V5 in the K=5
pool. If V5 OOF significantly higher than V4 at K=5+1 LR-meta gate,
we have a Path-B candidate for submission.
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
SEED, N_FOLDS, K_NN = 42, 5, 20
TE_ALPHA = 50.0    # prior smoothing


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


def race_progress_bin(rp: np.ndarray) -> np.ndarray:
    """Bin RaceProgress into 10 equal-width bins on [0, 1]."""
    return np.clip(np.floor(rp * 10), 0, 9).astype(np.int32)


def cell_keys(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build categorical cell keys for target-encoding."""
    cmp = df['Compound'].astype(str).to_numpy()
    drv = df['Driver'].astype(str).to_numpy()
    rac = df['Race'].astype(str).to_numpy()
    st  = df['Stint'].astype(str).to_numpy()
    rpb = race_progress_bin(df['RaceProgress'].astype(np.float32).to_numpy())
    rpb_str = rpb.astype(str)
    return {
        'cmp_x_stint':           np.char.add(np.char.add(cmp, '|'), st),
        'cmp_x_stint_x_rp':      np.char.add(np.char.add(np.char.add(np.char.add(
                                  cmp, '|'), st), '|'), rpb_str),
        'drv_x_cmp':             np.char.add(np.char.add(drv, '|'), cmp),
        'drv_x_stint':           np.char.add(np.char.add(drv, '|'), st),
        'rac_x_cmp':             np.char.add(np.char.add(rac, '|'), cmp),
    }


def smoothed_te(train_y: np.ndarray, train_keys: np.ndarray,
                target_keys: np.ndarray, alpha: float) -> np.ndarray:
    """Bayesian-smoothed target encoding: cell_mean shrunk to global mean.

    p_cell = (sum_y + alpha * global) / (count + alpha).
    """
    global_mean = train_y.mean()
    df = pd.DataFrame({'k': train_keys, 'y': train_y})
    g = df.groupby('k', sort=False)['y'].agg(['sum', 'count'])
    g['p'] = (g['sum'] + alpha * global_mean) / (g['count'] + alpha)
    out = np.full(len(target_keys), global_mean, dtype=np.float32)
    mp = g['p'].to_dict()
    for i, k in enumerate(target_keys):
        if k in mp:
            out[i] = mp[k]
    return out


def main():
    t0 = time.time()
    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)
    print(f'n_train={n:,}, n_test={len(te):,}')

    # Build kNN distance feature matrices (same as V4)
    X_kNN_tr = build_kNN_features(tr)
    X_kNN_te = build_kNN_features(te)

    # Build cell keys for both train and test
    keys_tr = cell_keys(tr)
    keys_te = cell_keys(te)
    cell_names = list(keys_tr.keys())
    print(f'TE cell schemes: {cell_names}')

    # Row-feature block for the LightGBM base (same as V4's build_row_features)
    cmp_codes_tr = tr['Compound'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()
    drv_codes_tr = tr['Driver'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()
    race_codes_tr = tr['Race'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()
    cmp_codes_te = te['Compound'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()
    drv_codes_te = te['Driver'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()
    race_codes_te = te['Race'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()

    def row_feats(df, c, d, r):
        cols = [
            df['LapNumber'].astype(np.float32).to_numpy(),
            df['TyreLife'].astype(np.float32).to_numpy(),
            df['RaceProgress'].astype(np.float32).to_numpy(),
            df['Stint'].astype(np.float32).to_numpy(),
            df['PitStop'].astype(np.float32).to_numpy(),
            df['LapTime (s)'].astype(np.float32).to_numpy() if 'LapTime (s)' in df.columns else df['LapTime'].astype(np.float32).to_numpy(),
            df['Year'].astype(np.float32).to_numpy(),
            c, d, r,
        ]
        names = ['LapNumber', 'TyreLife', 'RaceProgress', 'Stint',
                 'PitStop', 'LapTime', 'Year',
                 'Compound_code', 'Driver_code', 'Race_code']
        for opt in ('Cumulative_Degradation', 'LapTime_Delta', 'Position'):
            if opt in df.columns:
                cols.append(df[opt].astype(np.float32).to_numpy())
                names.append(opt)
        return np.column_stack(cols).astype(np.float32), names

    X_row_tr, names = row_feats(tr, cmp_codes_tr, drv_codes_tr, race_codes_tr)
    X_row_te, _     = row_feats(te, cmp_codes_te, drv_codes_te, race_codes_te)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    # Compute kNN target-mean & target-std per fold (same as V4)
    print(f'\n=== kNN target-mean (K={K_NN}) per fold ===')
    y_knn_mean = np.zeros(n, dtype=np.float32)
    y_knn_std  = np.zeros(n, dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        ts = time.time()
        tree = BallTree(X_kNN_tr[tri], leaf_size=40)
        _, idx = tree.query(X_kNN_tr[vai], k=K_NN)
        nb_y = y[tri][idx]
        y_knn_mean[vai] = nb_y.mean(axis=1).astype(np.float32)
        y_knn_std[vai]  = nb_y.std(axis=1).astype(np.float32)
        print(f'  fold {k}: {time.time()-ts:.1f}s')
    print('--- test query ---')
    ts = time.time()
    tree_all = BallTree(X_kNN_tr, leaf_size=40)
    _, idx_te = tree_all.query(X_kNN_te, k=K_NN)
    y_knn_mean_te = y[idx_te].mean(axis=1).astype(np.float32)
    y_knn_std_te  = y[idx_te].std(axis=1).astype(np.float32)
    print(f'  test: {time.time()-ts:.1f}s')

    # Compute fold-safe target encoding for each cell scheme
    print('\n=== fold-safe target encoding per fold ===')
    n_te_feats = len(cell_names)
    te_tr = np.zeros((n, n_te_feats), dtype=np.float32)
    for k, (tri, vai) in enumerate(splits):
        for fi, scheme in enumerate(cell_names):
            te_tr[vai, fi] = smoothed_te(y[tri], keys_tr[scheme][tri],
                                          keys_tr[scheme][vai], TE_ALPHA)
        print(f'  fold {k} TE done')
    # For test: use full training rows
    te_te = np.zeros((len(te), n_te_feats), dtype=np.float32)
    for fi, scheme in enumerate(cell_names):
        te_te[:, fi] = smoothed_te(y, keys_tr[scheme], keys_te[scheme], TE_ALPHA)

    # Build augmented base input
    aug_names = (names + ['kNN_y_mean', 'kNN_y_std']
                 + [f'te_{c}' for c in cell_names])
    X_aug_tr = np.column_stack([X_row_tr, y_knn_mean, y_knn_std, te_tr]).astype(np.float32)
    X_aug_te = np.column_stack([X_row_te, y_knn_mean_te, y_knn_std_te, te_te]).astype(np.float32)
    cat_idx = [aug_names.index(c) for c in ('Compound_code', 'Driver_code', 'Race_code')]
    print(f'\nX_aug_tr: {X_aug_tr.shape}; cat indices: {cat_idx}')
    print(f'feature names: {aug_names}')

    print('\n=== Training V5 LightGBM base (5-fold OOF) ===')
    oof = np.zeros(n, dtype=np.float32)
    test_pred = np.zeros(len(te), dtype=np.float32)
    base_params = dict(
        objective='binary', metric='auc',
        num_leaves=63, max_depth=-1,
        learning_rate=0.03, n_estimators=2500,
        reg_alpha=0.5, reg_lambda=2.0,
        bagging_fraction=0.8, bagging_freq=5,
        feature_fraction=0.85, min_data_in_leaf=80,
        verbose=-1, seed=SEED,
    )
    for k, (tri, vai) in enumerate(splits):
        ts = time.time()
        m = lgb.LGBMClassifier(**base_params)
        m.fit(X_aug_tr[tri], y[tri],
              eval_set=[(X_aug_tr[vai], y[vai])],
              categorical_feature=cat_idx,
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[vai] = m.predict_proba(X_aug_tr[vai])[:, 1]
        test_pred += m.predict_proba(X_aug_te)[:, 1] / N_FOLDS
        print(f'  fold {k}: AUC={roc_auc_score(y[vai], oof[vai]):.5f}  '
              f'best_iter={m.booster_.best_iteration}  ({time.time()-ts:.1f}s)')
    auc_oof = float(roc_auc_score(y, oof))
    print(f'\nV5 standalone 5-fold OOF AUC: {auc_oof:.5f}')

    # Compare to V4
    v4 = np.load(ART / 'oof_knn_aug_base.npy')
    v4_pos = v4[:, 1] if v4.ndim == 2 else v4
    auc_v4 = float(roc_auc_score(y, v4_pos))
    print(f'V4 standalone OOF AUC:        {auc_v4:.5f}  (Δ V5 vs V4 = {(auc_oof-auc_v4)*1e4:+.2f} bp)')

    np.save(ART / 'oof_v5_base.npy',
            np.column_stack([1-oof, oof]).astype(np.float32))
    np.save(ART / 'test_v5_base.npy',
            np.column_stack([1-test_pred, test_pred]).astype(np.float32))
    out = {
        'v5_oof_auc_standalone':      auc_oof,
        'v4_oof_auc_standalone':      auc_v4,
        'delta_vs_v4_bp':             (auc_oof - auc_v4) * 1e4,
        'feature_names':              aug_names,
        'wall_secs':                  time.time() - t0,
    }
    (ART / 'probe_v5_base_results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
