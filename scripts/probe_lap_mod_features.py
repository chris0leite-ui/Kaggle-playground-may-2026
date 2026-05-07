"""scripts/probe_lap_mod_features.py — LapNumber-mod-K and id-mod-K features.

Synthetic-data finding (probe_id_order_audit): target rate spans
~566 bp by `LapNumber % 10`, ~568 bp by `id % 1000`. These modular
patterns are typical synthetic-generator artifacts that GBDTs miss
(GBDTs split by threshold, not modulo).

Build a LightGBM base with explicit modular features:
  - LapNumber_mod_3, _5, _7, _10
  - id_mod_5, _7, _13, _100, _1000
  - LapNumber_lapquot_10  = LapNumber // 10
+ existing numerics & categoricals.

Standalone OOF + min-meta gate via probe_min_meta.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))

    feat_num = ["TyreLife", "RaceProgress", "LapTime_Delta",
                "Cumulative_Degradation", "Position", "LapTime (s)",
                "Stint", "Year", "Position_Change", "LapNumber"]
    feat_cat = ["Driver", "Compound", "Race"]

    for df in [train, test]:
        for mod in [3, 5, 7, 10]:
            df[f"LapNumber_mod_{mod}"] = (df["LapNumber"].astype(int) % mod).astype(np.int8)
        df["LapNumber_quot_10"] = (df["LapNumber"].astype(int) // 10).astype(np.int16)
        for mod in [5, 7, 13, 100, 1000]:
            df[f"id_mod_{mod}"] = (df["id"].astype(int) % mod).astype(np.int16)
        df["id_quot_1000"] = (df["id"].astype(int) // 1000).astype(np.int16)

    mod_feat_cols = ([f"LapNumber_mod_{m}" for m in [3, 5, 7, 10]] +
                     ["LapNumber_quot_10"] +
                     [f"id_mod_{m}" for m in [5, 7, 13, 100, 1000]] +
                     ["id_quot_1000"])
    feat_cols = feat_num + feat_cat + mod_feat_cols
    cat_cols = feat_cat + [f"LapNumber_mod_{m}" for m in [3, 5, 7, 10]] + \
               [f"id_mod_{m}" for m in [5, 7, 13]]

    X = train[feat_cols].copy()
    X_test = test[feat_cols].copy()
    for c in feat_cat:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    print(f"feat shape: train {X.shape}, test {X_test.shape}")
    print(f"mod-features added: {mod_feat_cols}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  min_data_in_leaf=200, verbose=-1, seed=SEED)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        test_pred += m.predict(X_test) / N_FOLDS
        s = float(roc_auc_score(y[va], oof[va]))
        print(f"  fold {k}: AUC {s:.5f} best_iter {m.best_iteration} "
              f"wall {time.time()-t:.1f}s")
        if k == 0:
            # Feature importance for the first fold
            imp = pd.DataFrame({"feature": X.columns,
                                "imp_split": m.feature_importance(importance_type="split"),
                                "imp_gain": m.feature_importance(importance_type="gain")})
            imp = imp.sort_values("imp_gain", ascending=False)
            print("  top-15 features by gain:")
            print(imp.head(15).to_string(index=False))

    auc = float(roc_auc_score(y, oof))
    rho, _ = spearmanr(test_pred, primary_test)
    print(f"\n=== lap_mod_features base ===")
    print(f"  std OOF: {auc:.5f}  Δ vs PRIMARY {(auc-auc_primary)*1e4:+.2f} bp")
    print(f"  ρ vs PRIMARY: {rho:.6f}")

    np.save(ART / "oof_lap_mod_features_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_lap_mod_features_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))
    sub = sample_sub.copy(); sub[TARGET] = test_pred
    sub.to_csv("submissions/submission_lap_mod_features.csv", index=False)
    summary = dict(std_oof=auc, delta_vs_primary_bp=(auc - auc_primary)*1e4,
                   rho_vs_primary=float(rho),
                   wall_s=time.time() - t0)
    (ART / "probe_lap_mod_features.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/probe_lap_mod_features.json (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
