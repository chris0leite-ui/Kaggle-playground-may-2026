"""d15c — ExtraTreesClassifier as K=22 stack-add candidate.

Branch C of Day-15 4-branch parallel probe. Cheap-diversity slot per
NVIDIA Grandmaster Playbook: Extra Trees splits at random thresholds,
underfitting the within-fold leakage that GBDTs eat. Predicted ρ vs
LGBM bases ~0.93.

Mechanism family: extra_trees_ensemble (FAMILY_PRIORS P=0.35,
band 2/6/12 bp). Standalone OOF expected ~0.92-0.93 (mid-pack but
high-diversity per the prior). Pass criteria:
  - min-meta Δbp ≥ +0.05 AND ρ vs PRIMARY < 0.997 (PASS)
  - min-meta Δbp > 0 (WEAK_PASS)
  - ρ ≥ 0.998 (TIE)
  - min-meta Δbp < 0 (FAIL)

Save: oof_d15c_extra_trees_strat.npy / test_d15c_extra_trees_strat.npy
(2-col [1-p, p]) per canonical convention.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
NAME = "d15c_extra_trees"


def label_encode_inplace(X: pd.DataFrame, X_test: pd.DataFrame, cols):
    """Fit label encoding on union of train+test for given categorical cols."""
    for c in cols:
        if c not in X.columns:
            continue
        uniq = pd.concat([X[c], X_test[c]], ignore_index=True
                         ).astype(str).unique()
        mp = {v: i for i, v in enumerate(sorted(uniq))}
        X[c] = X[c].astype(str).map(mp).astype(np.int32)
        X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)


def main():
    t_total = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"Train {train.shape}  test {test.shape}")

    y = train[TARGET].astype(int).values
    print(f"prior y={y.mean():.4f}")

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    X_test = test.drop(columns=[ID_COL], errors="ignore").copy()

    cat_cols = ["Driver", "Compound", "Race", "Year"]
    label_encode_inplace(X, X_test, cat_cols)

    # Cast all numeric columns to float32 for memory + speed; sklearn will
    # accept the int32 label-encoded categoricals as-is.
    for c in X.columns:
        if X[c].dtype == np.int32:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype(np.float32)

    # Fill any NaN from numeric coercion / source data
    X = X.fillna(-1).astype({c: np.float32 for c in X.columns
                             if X[c].dtype != np.int32})
    X_test = X_test.fillna(-1).astype({c: np.float32 for c in X_test.columns
                                       if X_test[c].dtype != np.int32})

    feat_names = list(X.columns)
    print(f"n_features={len(feat_names)}  cols={feat_names}")

    X_arr = X.values
    X_test_arr = X_test.values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float64)
    test_avg = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []
    walls = []

    n_estimators = 4000
    print(f"\nExtraTreesClassifier(n_estimators={n_estimators}, "
          f"max_features='sqrt', min_samples_leaf=20, n_jobs=-1)")

    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=SEED,
            bootstrap=False,  # default; ET signature
        )
        clf.fit(X_arr[tr], y[tr])
        oof[va] = clf.predict_proba(X_arr[va])[:, 1]
        test_avg += clf.predict_proba(X_test_arr)[:, 1] / N_FOLDS

        s = float(roc_auc_score(y[va], oof[va]))
        wall = time.time() - t0
        fold_aucs.append(s)
        walls.append(wall)
        print(f"  f{k}: AUC={s:.5f}  wall={wall:.1f}s", flush=True)

    auc_oof = float(roc_auc_score(y, oof))
    print(f"\n=== Standalone result ===")
    print(f"OOF AUC: {auc_oof:.5f}  per-fold std={np.std(fold_aucs):.5f}")
    print(f"Total wall: {time.time()-t_total:.1f}s")

    np.save(ART / f"oof_{NAME}_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / f"test_{NAME}_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))

    results = {
        "oof_auc": auc_oof,
        "fold_aucs": fold_aucs,
        "fold_walls_s": walls,
        "runtime_s": time.time() - t_total,
        "n_estimators": n_estimators,
        "n_features": len(feat_names),
        "feature_names": feat_names,
    }
    (ART / f"{NAME}_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved oof_{NAME}_strat.npy / test_{NAME}_strat.npy / "
          f"{NAME}_results.json")


if __name__ == "__main__":
    main()
