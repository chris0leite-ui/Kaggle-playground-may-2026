"""V6: learned-embedding kNN as base input feature.

V4 used kNN(K=20) in raw STANDARDISED feature space. V5 added per-cell
target encodings and was absorbed by V4 (ρ 0.989). The transductive
signal V4 captures is shaped by the raw-feature similarity metric.

V6 hypothesis: a TASK-LEARNED similarity metric finds neighbours whose
labels actually correlate with the row's PitNextLap, beyond what raw
feature distance captures. Specifically:

  1. Per fold k: train a small NN encoder on training rows in folds≠k
     to predict PitNextLap. Architecture: 14 input → 64 hidden → 32
     hidden → 16 hidden (the EMBEDDING) → 1 binary output.
  2. Extract the 16-dim PENULTIMATE-LAYER embedding for ALL rows
     (train + test).
  3. Build BallTree on training-fold embeddings; query val + test rows
     for K=20 nearest neighbours in EMBEDDING space.
  4. Compute kNN-target-mean and kNN-target-std as features.
  5. Train V6 LightGBM base on (row features + embedding-kNN-mean +
     embedding-kNN-std + V4-style raw-feature-kNN-mean + std). The
     base sees BOTH similarity metrics; tree splits route between them.

Why this might break further into the K=4 ceiling: the NN encoder
learns a similarity that prioritises features with PitNextLap signal.
Two rows with similar raw features but different RaceProgress regimes
would have similar raw-kNN but different embedding-kNN distance.

Output:
    scripts/artifacts/oof_v6_base.npy  (n_train, 2)
    scripts/artifacts/test_v6_base.npy (n_test,  2)
    scripts/artifacts/probe_v6_base_results.json
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
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

ART = Path('scripts/artifacts')
SEED, N_FOLDS, K_NN = 42, 5, 20
EMBED_DIM = 16
HIDDEN_LAYERS = (64, 32, 16)         # last layer = embedding
MLP_MAX_ITER = 60                    # small to keep wall short
MLP_BATCH = 4096
MLP_LR = 1e-3


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


def extract_embedding(mlp: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """Forward-pass X through all but the final layer to get the embedding.

    sklearn's MLPClassifier exposes its weights via .coefs_ and .intercepts_.
    For HIDDEN_LAYERS = (64, 32, 16), there are 4 weight matrices:
      W0 (input → 64), W1 (64 → 32), W2 (32 → 16), W3 (16 → 1).
    The penultimate-layer embedding is the activation after W2 (16-dim).
    """
    h = X
    # All hidden activations are ReLU per default.
    for i in range(len(mlp.coefs_) - 1):     # all but the output layer
        h = np.maximum(0.0, h @ mlp.coefs_[i] + mlp.intercepts_[i])
    return h.astype(np.float32)


def build_row_features(df: pd.DataFrame):
    """Same as V4/V5 row-feature block."""
    cmp_codes = df['Compound'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()
    drv_codes = df['Driver'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()
    race_codes = df['Race'].astype(str).astype('category').cat.codes.astype(np.int32).to_numpy()
    cols = [
        df['LapNumber'].astype(np.float32).to_numpy(),
        df['TyreLife'].astype(np.float32).to_numpy(),
        df['RaceProgress'].astype(np.float32).to_numpy(),
        df['Stint'].astype(np.float32).to_numpy(),
        df['PitStop'].astype(np.float32).to_numpy(),
        df['LapTime (s)'].astype(np.float32).to_numpy() if 'LapTime (s)' in df.columns else df['LapTime'].astype(np.float32).to_numpy(),
        df['Year'].astype(np.float32).to_numpy(),
        cmp_codes, drv_codes, race_codes,
    ]
    names = ['LapNumber', 'TyreLife', 'RaceProgress', 'Stint',
             'PitStop', 'LapTime', 'Year',
             'Compound_code', 'Driver_code', 'Race_code']
    for opt in ('Cumulative_Degradation', 'LapTime_Delta', 'Position'):
        if opt in df.columns:
            cols.append(df[opt].astype(np.float32).to_numpy())
            names.append(opt)
    return np.column_stack(cols).astype(np.float32), names


def main():
    t0 = time.time()
    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    y = tr['PitNextLap'].astype(np.int8).values
    n = len(tr)
    print(f'n_train={n:,}, n_test={len(te):,}')

    # NN encoder input: standardised dense features
    X_kNN_tr = build_kNN_features(tr)             # 11 feat
    X_kNN_te = build_kNN_features(te)
    print(f'X_kNN: {X_kNN_tr.shape}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y))

    print(f'\n=== Per-fold MLP encoder + embedding kNN ===')
    emb_kNN_mean_tr = np.zeros(n, dtype=np.float32)
    emb_kNN_std_tr  = np.zeros(n, dtype=np.float32)
    raw_kNN_mean_tr = np.zeros(n, dtype=np.float32)
    raw_kNN_std_tr  = np.zeros(n, dtype=np.float32)
    emb_kNN_mean_te_per = np.zeros((len(te), N_FOLDS), dtype=np.float32)
    emb_kNN_std_te_per  = np.zeros((len(te), N_FOLDS), dtype=np.float32)

    for k, (tri, vai) in enumerate(splits):
        ts = time.time()
        # Train MLP on training fold rows
        mlp = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation='relu',
            solver='adam',
            learning_rate_init=MLP_LR,
            batch_size=MLP_BATCH,
            max_iter=MLP_MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=SEED + k,
            verbose=False,
        )
        mlp.fit(X_kNN_tr[tri], y[tri])
        # Holdout AUC of MLP itself (sanity)
        val_proba = mlp.predict_proba(X_kNN_tr[vai])[:, 1]
        mlp_auc = float(roc_auc_score(y[vai], val_proba))
        # Extract embedding for all rows
        emb_tr = extract_embedding(mlp, X_kNN_tr)
        emb_te = extract_embedding(mlp, X_kNN_te)
        # Build BallTree on training-fold embeddings
        tree_emb = BallTree(emb_tr[tri], leaf_size=40)
        _, idx_emb = tree_emb.query(emb_tr[vai], k=K_NN)
        nb_y = y[tri][idx_emb]
        emb_kNN_mean_tr[vai] = nb_y.mean(axis=1).astype(np.float32)
        emb_kNN_std_tr[vai]  = nb_y.std(axis=1).astype(np.float32)
        # Test: query against training-fold embeddings (per-fold averaging later)
        _, idx_te = tree_emb.query(emb_te, k=K_NN)
        emb_kNN_mean_te_per[:, k] = y[tri][idx_te].mean(axis=1).astype(np.float32)
        emb_kNN_std_te_per[:, k]  = y[tri][idx_te].std(axis=1).astype(np.float32)
        # Raw-feature kNN (for comparison + as additional feature)
        tree_raw = BallTree(X_kNN_tr[tri], leaf_size=40)
        _, idx_raw = tree_raw.query(X_kNN_tr[vai], k=K_NN)
        nb_y_raw = y[tri][idx_raw]
        raw_kNN_mean_tr[vai] = nb_y_raw.mean(axis=1).astype(np.float32)
        raw_kNN_std_tr[vai]  = nb_y_raw.std(axis=1).astype(np.float32)
        print(f'  fold {k}: mlp_auc={mlp_auc:.5f}  '
              f'emb_kNN[vai].mean={emb_kNN_mean_tr[vai].mean():.4f}  '
              f'wall={time.time()-ts:.1f}s')

    # Average test embeddings across folds
    emb_kNN_mean_te = emb_kNN_mean_te_per.mean(axis=1)
    emb_kNN_std_te  = emb_kNN_std_te_per.mean(axis=1)

    # For raw kNN test: load V4's already-saved test predictions implicitly?
    # No — recompute on full train BallTree.
    print('\n--- raw-feature kNN test query ---')
    ts = time.time()
    tree_full_raw = BallTree(X_kNN_tr, leaf_size=40)
    _, idx_raw_te = tree_full_raw.query(X_kNN_te, k=K_NN)
    raw_kNN_mean_te = y[idx_raw_te].mean(axis=1).astype(np.float32)
    raw_kNN_std_te  = y[idx_raw_te].std(axis=1).astype(np.float32)
    print(f'  test: {time.time()-ts:.1f}s')

    # Build V6 base input: row + emb_kNN + raw_kNN
    X_row_tr, names = build_row_features(tr)
    X_row_te, _     = build_row_features(te)
    aug_names = names + ['emb_kNN_mean', 'emb_kNN_std',
                         'raw_kNN_mean', 'raw_kNN_std']
    X_aug_tr = np.column_stack([X_row_tr,
                                emb_kNN_mean_tr, emb_kNN_std_tr,
                                raw_kNN_mean_tr, raw_kNN_std_tr]).astype(np.float32)
    X_aug_te = np.column_stack([X_row_te,
                                emb_kNN_mean_te, emb_kNN_std_te,
                                raw_kNN_mean_te, raw_kNN_std_te]).astype(np.float32)
    cat_idx = [aug_names.index(c) for c in ('Compound_code', 'Driver_code', 'Race_code')]
    print(f'\nX_aug_tr: {X_aug_tr.shape}; cat indices: {cat_idx}')

    print('\n=== Training V6 LightGBM base (5-fold OOF) ===')
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
    print(f'\nV6 standalone 5-fold OOF AUC: {auc_oof:.5f}')

    v4 = np.load(ART / 'oof_knn_aug_base.npy')
    v4_pos = v4[:, 1] if v4.ndim == 2 else v4
    auc_v4 = float(roc_auc_score(y, v4_pos))
    print(f'V4 standalone OOF AUC:        {auc_v4:.5f}  (Δ = {(auc_oof-auc_v4)*1e4:+.2f} bp)')

    np.save(ART / 'oof_v6_base.npy',
            np.column_stack([1-oof, oof]).astype(np.float32))
    np.save(ART / 'test_v6_base.npy',
            np.column_stack([1-test_pred, test_pred]).astype(np.float32))
    out = {
        'v6_oof_auc_standalone':      auc_oof,
        'v4_oof_auc_standalone':      auc_v4,
        'delta_vs_v4_bp':             (auc_oof - auc_v4) * 1e4,
        'feature_names':              aug_names,
        'embedding_dim':              EMBED_DIM,
        'mlp_layers':                 list(HIDDEN_LAYERS),
        'wall_secs':                  time.time() - t0,
    }
    (ART / 'probe_v6_base_results.json').write_text(json.dumps(out, indent=2))
    print('\n=== V6 SUMMARY ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
