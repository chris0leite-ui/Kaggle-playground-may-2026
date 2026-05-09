"""Phase 4 — anomaly + ghost-driver clustering scan (numpy-light).

Quick probe (≤2 min CPU) to identify:

  1. Per-numeric-column low-frequency outlier values (e.g. TyreLife=60.5
     was a single such row in 627k synth rows). Are these CTGAN
     extrapolation artifacts? Class rate at these rows?

  2. Ghost-driver feature centroids: cluster D-prefix and 3-letter abbrev
     drivers by their mean features; do clusters map to specific
     (Race × Year × Compound) regimes?

  3. Per-row neighborhood density via kth-nearest-neighbor distance
     in 7 KS-low feature space (sklearn NearestNeighbors).

Outputs anomaly + neighborhood features as a separate base
candidate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
SEED = 42

NAME = "p4_anomaly"


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    full = pd.concat([train.drop(columns=["PitNextLap"]), test], ignore_index=True)
    n_train = len(train)
    print(f"train {train.shape} test {test.shape}", flush=True)

    NUM_COLS = [
        "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change", "Year",
    ]

    # ---- 1. Anomaly value frequency ----
    print("\n--- Outlier value scan (n_unique vs occurrence_count) ---",
          flush=True)
    outlier_features = {}
    for c in NUM_COLS:
        v = full[c].to_numpy()
        unique, counts = np.unique(v, return_counts=True)
        rare_mask = counts == 1
        n_rare = int(rare_mask.sum())
        outlier_features[c] = {
            "n_unique": int(len(unique)),
            "n_singleton_values": n_rare,
            "max_count": int(counts.max()),
            "min_count": int(counts.min()),
            "fraction_singleton": float(n_rare / len(unique)),
        }
        if n_rare > 0 and n_rare < 30:
            print(f"  {c:25s} n_unique={len(unique):6d} singletons={n_rare}",
                  flush=True)

    # Class rate at "anomaly" rows: rows with any singleton-frequency value
    # in a numeric column (excl LapTime which is naturally near-unique)
    print("\n--- Class rate at anomalous rows ---", flush=True)
    discrete_cols = ["LapNumber", "Stint", "TyreLife", "Position",
                     "Position_Change", "Year"]
    is_anomaly = np.zeros(len(full), dtype=bool)
    for c in discrete_cols:
        v = full[c].to_numpy()
        unique, counts = np.unique(v, return_counts=True)
        rare_set = set(unique[counts < 5].tolist())
        if rare_set:
            mask = np.isin(v, list(rare_set))
            print(f"  {c}: {mask.sum()} rows with values in <5-occurrences set",
                  flush=True)
            is_anomaly |= mask

    train_anomaly = is_anomaly[:n_train]
    if train_anomaly.sum() > 0:
        rate_normal = float(train["PitNextLap"].iloc[~train_anomaly].mean())
        rate_anomaly = float(train["PitNextLap"].iloc[train_anomaly].mean())
        print(f"  rate_normal: {rate_normal:.4f}; rate_anomaly: {rate_anomaly:.4f}; "
              f"diff: {rate_anomaly-rate_normal:+.4f}", flush=True)

    # ---- 2. Per-row neighborhood density ----
    print(f"\n--- kth-NN distance in 7 KS-low feature space "
          f"({time.time()-ts:.0f}s) ---", flush=True)
    KS_LOW = ["TyreLife", "Position", "LapTime (s)",
              "Cumulative_Degradation", "RaceProgress",
              "LapTime_Delta", "LapNumber"]
    sc = StandardScaler()
    X = sc.fit_transform(full[KS_LOW].fillna(0).to_numpy())
    # Use sklearn's NearestNeighbors with random subsample for memory
    from sklearn.neighbors import NearestNeighbors
    K_NN = 20
    print(f"  fitting NearestNeighbors on {len(X)} rows, k={K_NN}...",
          flush=True)
    # Use ball_tree or kd_tree; on 627k × 7 features, this is fast
    nn = NearestNeighbors(n_neighbors=K_NN+1, algorithm="auto", n_jobs=2)
    nn.fit(X)
    print(f"  fit done, querying... [{time.time()-ts:.0f}s]", flush=True)
    # Query distances in CHUNKS to avoid memory
    chunk = 50_000
    dist_k = np.zeros(len(X), dtype=np.float32)
    dist_mean = np.zeros(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        j = min(i + chunk, len(X))
        d, _ = nn.kneighbors(X[i:j], n_neighbors=K_NN+1, return_distance=True)
        # Skip self (first neighbor at distance 0)
        dist_k[i:j] = d[:, K_NN]  # k-th nearest
        dist_mean[i:j] = d[:, 1:K_NN+1].mean(axis=1)  # mean of k neighbors
        if i % 200000 == 0:
            print(f"    chunk {i}-{j} [{time.time()-ts:.0f}s]", flush=True)

    # ---- 3. Train base with anomaly features ----
    print(f"\n--- Training LGBM with anomaly features "
          f"[{time.time()-ts:.0f}s] ---", flush=True)
    full = full.copy()
    full["dist_kNN_20"] = dist_k
    full["dist_kNN_mean"] = dist_mean
    full["is_anomaly_value"] = is_anomaly.astype("int8")

    feat_cols = [
        # standard 14 (excl id, target)
        "Driver", "Compound", "Race", "Year", "PitStop",
        "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
        # anomaly additions
        "dist_kNN_20", "dist_kNN_mean", "is_anomaly_value",
    ]
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        cats = pd.Categorical(full[c]).categories
        full[c] = pd.Categorical(full[c], categories=cats).codes.astype("int32")

    train_feat = full.iloc[:n_train].copy()
    train_feat["PitNextLap"] = train["PitNextLap"].to_numpy()
    test_feat = full.iloc[n_train:].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = train_feat["PitNextLap"].to_numpy()
    oof = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(len(test_feat), dtype=np.float32)
    fold_aucs = []
    PARAMS = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, n_estimators=2000,
        num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=4, seed=SEED, verbose=-1,
        n_jobs=2,
    )

    for fold, (tr, va) in enumerate(skf.split(np.zeros(n_train), y)):
        fts = time.time()
        Xtr = train_feat.iloc[tr][feat_cols]
        Xva = train_feat.iloc[va][feat_cols]
        Xte = test_feat[feat_cols]
        ytr = y[tr]; yva = y[va]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(
            Xtr, ytr, eval_set=[(Xva, yva)],
            categorical_feature=cat_cols + ["Year"],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va] = m.predict_proba(Xva)[:, 1]
        test_preds += m.predict_proba(Xte)[:, 1] / 5
        fold_aucs.append(float(roc_auc_score(yva, oof[va])))
        print(f"  fold {fold} AUC {fold_aucs[-1]:.5f}  "
              f"[{time.time()-fts:.0f}s, {time.time()-ts:.0f}s]", flush=True)

    overall = float(roc_auc_score(y, oof))
    print(f"\nOverall OOF AUC {overall:.5f}", flush=True)
    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_preds, test_preds])
    np.save(ART / f"oof_{NAME}_strat.npy", oof2)
    np.save(ART / f"test_{NAME}_strat.npy", test2)
    summary = {
        "name": NAME,
        "overall_oof_auc": overall,
        "fold_aucs": fold_aucs,
        "fold_std": float(np.std(fold_aucs)),
        "outlier_features": outlier_features,
        "anomaly_class_rate_normal": rate_normal if train_anomaly.sum() > 0 else None,
        "anomaly_class_rate_anomaly": rate_anomaly if train_anomaly.sum() > 0 else None,
        "anomaly_n_train_rows": int(train_anomaly.sum()),
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {NAME} artifacts.", flush=True)


if __name__ == "__main__":
    main()
