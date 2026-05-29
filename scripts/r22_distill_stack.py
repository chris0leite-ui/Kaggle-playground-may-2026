"""Distillation + Stack pipeline — public CSVs as features through a clone model.

Phase 1: Train a 'raunakdey clone' on test features → raunakdey test predictions.
         5-fold CV on test rows. LightGBM regression.
Phase 2: Apply clone to training rows → pseudo_raunakdey_train_OOF.
Phase 3: Train meta stacker: target = train_label,
         features = (K=20 PathB OOF, pseudo_raunakdey_OOF).
         5-fold CV; report OOF AUC.
Phase 4: Apply meta to test: meta(K20_test, raunakdey_test) → final CSV.

Validates that the clone reconstructs raunakdey before applying meta-stack.
"""

import os, sys, time, json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT = "/home/user/Kaggle-playground-may-2026"
OUT = f"{ROOT}/submissions"
ART = f"{ROOT}/scripts/artifacts"

t0 = time.time()
print("=== Phase 0: Load data ===")
tr = pd.read_csv(f"{ROOT}/data/train.csv")
te = pd.read_csv(f"{ROOT}/data/test.csv")
print(f"  train n={len(tr)}, test n={len(te)}")

# Anchor (raunakdey test predictions) — id-aligned with test
anchor = pd.read_csv(f"{ROOT}/submissions/public_scrape/raunakdey07_f1-pit-stops-blender-0-95454/submission.csv")
anchor = anchor.set_index("id").reindex(te["id"].values)["PitNextLap"].values
assert len(anchor) == len(te), f"anchor len mismatch {len(anchor)} vs {len(te)}"
print(f"  anchor: mean={anchor.mean():.4f} min={anchor.min():.4f} max={anchor.max():.4f}")

# K=20 PathB DriverClass×Stint OOF (train) + test predictions
K20_oof = np.load(f"{ART}/oof_K20_pathb_driverclass_stint_tau100000.npy")
K20_test_csv = pd.read_csv(f"{ROOT}/submissions/submission_K20_pathb_driverclass_stint_tau100000.csv")
K20_test = K20_test_csv.set_index("id").reindex(te["id"].values).iloc[:, 0].values
print(f"  K20 OOF shape: {K20_oof.shape}, K20 test shape: {K20_test.shape}")
assert len(K20_oof) == len(tr), f"K20 OOF len mismatch {len(K20_oof)} vs {len(tr)}"

# Feature engineering (consistent between train and test)
CATS = ["Driver", "Compound", "Race"]
NUMS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change"]

# Encode categoricals — fit on train+test union to share codes
combined_cat = pd.concat([tr[CATS], te[CATS]], axis=0, ignore_index=True)
codes = {}
for c in CATS:
    codes[c] = pd.Categorical(combined_cat[c]).codes
    tr[c + "_code"] = codes[c][:len(tr)]
    te[c + "_code"] = codes[c][len(tr):]
FEATS = [c + "_code" for c in CATS] + NUMS
print(f"  features: {FEATS}")

X_train = tr[FEATS].values
X_test  = te[FEATS].values
y_train = tr["PitNextLap"].values
print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

print(f"\n=== Phase 1: Distill raunakdey clone (5-fold CV on test) — t={time.time()-t0:.1f}s ===")
# Train clone on test features → anchor (raunakdey test preds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
clone_val_preds = np.zeros(len(te), dtype=float)
clone_train_preds = np.zeros((5, len(tr)), dtype=float)  # apply each fold's clone to train, then average
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_test)):
    dtrain = lgb.Dataset(X_test[tr_idx], anchor[tr_idx], categorical_feature=[0, 1, 2])
    dval   = lgb.Dataset(X_test[vl_idx], anchor[vl_idx], categorical_feature=[0, 1, 2], reference=dtrain)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbosity": -1,
    }
    bst = lgb.train(params, dtrain, num_boost_round=400,
                    valid_sets=[dval], callbacks=[lgb.early_stopping(25, verbose=False)])
    clone_val_preds[vl_idx] = bst.predict(X_test[vl_idx])
    clone_train_preds[fold] = bst.predict(X_train)
    print(f"  fold {fold}: best_iter={bst.best_iteration}, "
          f"val rmse={((clone_val_preds[vl_idx] - anchor[vl_idx])**2).mean()**0.5:.5f}")

clone_r2 = 1.0 - ((clone_val_preds - anchor)**2).mean() / anchor.var()
clone_rho = np.corrcoef(rankdata(clone_val_preds), rankdata(anchor))[0, 1]
print(f"\n  Clone R²  vs raunakdey on test: {clone_r2:.4f}")
print(f"  Clone ρ   vs raunakdey on test: {clone_rho:.5f}")

pseudo_raunakdey_train = clone_train_preds.mean(axis=0)  # average 5 fold predictions
print(f"  pseudo_raunakdey_train: mean={pseudo_raunakdey_train.mean():.4f} "
      f"min={pseudo_raunakdey_train.min():.4f} max={pseudo_raunakdey_train.max():.4f}")
# Sanity vs K20 OOF
rho_pseudo_K20 = np.corrcoef(rankdata(pseudo_raunakdey_train), rankdata(K20_oof))[0, 1]
print(f"  ρ pseudo_raunakdey_train vs K20_OOF = {rho_pseudo_K20:.5f}")

print(f"\n=== Phase 2: K20 OOF AUC baseline — t={time.time()-t0:.1f}s ===")
auc_K20 = roc_auc_score(y_train, K20_oof)
auc_pseudo = roc_auc_score(y_train, pseudo_raunakdey_train)
print(f"  K=20 OOF AUC:                {auc_K20:.5f}")
print(f"  pseudo_raunakdey_train AUC:  {auc_pseudo:.5f}")
print(f"  Simple 0.9/0.1 OOF AUC:      {roc_auc_score(y_train, 0.9*rankdata(pseudo_raunakdey_train) + 0.1*rankdata(K20_oof)):.5f}")

print(f"\n=== Phase 3: Train meta stacker — t={time.time()-t0:.1f}s ===")
# Meta features: K20_oof + pseudo_raunakdey_train
META_FEATS = np.column_stack([K20_oof, pseudo_raunakdey_train])
# Also include interaction features
META_FEATS = np.column_stack([
    K20_oof,
    pseudo_raunakdey_train,
    K20_oof * pseudo_raunakdey_train,            # interaction
    np.abs(K20_oof - pseudo_raunakdey_train),     # disagreement
    (K20_oof + pseudo_raunakdey_train) / 2.0,    # mean
])
META_NAMES = ["K20", "pseudo_anchor", "interact", "disagree", "mean"]
print(f"  meta features: {META_NAMES}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof = np.zeros(len(tr), dtype=float)
meta_test = np.zeros((5, len(te)), dtype=float)
META_TEST = np.column_stack([
    K20_test,
    anchor,
    K20_test * anchor,
    np.abs(K20_test - anchor),
    (K20_test + anchor) / 2.0,
])

for fold, (tr_idx, vl_idx) in enumerate(skf.split(META_FEATS, y_train)):
    dtrain = lgb.Dataset(META_FEATS[tr_idx], y_train[tr_idx])
    dval   = lgb.Dataset(META_FEATS[vl_idx], y_train[vl_idx], reference=dtrain)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 200,
        "verbosity": -1,
    }
    bst = lgb.train(params, dtrain, num_boost_round=500,
                    valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
    meta_oof[vl_idx] = bst.predict(META_FEATS[vl_idx])
    meta_test[fold] = bst.predict(META_TEST)
    print(f"  fold {fold}: best_iter={bst.best_iteration}  val AUC={roc_auc_score(y_train[vl_idx], meta_oof[vl_idx]):.5f}")

meta_auc_oof = roc_auc_score(y_train, meta_oof)
print(f"\n  Meta OOF AUC: {meta_auc_oof:.5f}")
print(f"  K20 OOF AUC:   {auc_K20:.5f}  (delta {meta_auc_oof - auc_K20:+.5f})")

print(f"\n=== Phase 4: Generate final test predictions — t={time.time()-t0:.1f}s ===")
meta_test_avg = meta_test.mean(axis=0)
final = pd.DataFrame({"id": te["id"].values, "PitNextLap": meta_test_avg})
out_path = f"{OUT}/submission_d30_R22S1_distill_stack_metaLGBM.csv"
final.to_csv(out_path, index=False)

# Also compare to V8 (baseline post-hoc 0.9/0.1 blend)
V8 = pd.read_csv(f"{OUT}/submission_d29_R22V8_raunakdey90_C5_10.csv").set_index("id").reindex(te["id"].values).iloc[:, 0].values
rho_meta_V8 = np.corrcoef(rankdata(meta_test_avg), rankdata(V8))[0, 1]
print(f"  S1 ρ vs V8 = {rho_meta_V8:.5f}")
print(f"  S1 file: {out_path}")

# Save artifacts for inspection
np.save(f"{ART}/pseudo_raunakdey_train.npy", pseudo_raunakdey_train)
np.save(f"{ART}/clone_val_preds_test.npy", clone_val_preds)
np.save(f"{ART}/meta_oof_distill_stack.npy", meta_oof)
np.save(f"{ART}/meta_test_distill_stack.npy", meta_test_avg)

stats = {
    "clone_r2_vs_raunakdey": float(clone_r2),
    "clone_rho_vs_raunakdey": float(clone_rho),
    "K20_oof_auc": float(auc_K20),
    "pseudo_raunakdey_auc": float(auc_pseudo),
    "meta_oof_auc": float(meta_auc_oof),
    "rho_S1_vs_V8": float(rho_meta_V8),
}
with open(f"{ART}/distill_stack_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nDone in {time.time()-t0:.1f}s")
print(json.dumps(stats, indent=2))
