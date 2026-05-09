"""qAR — yekenot-flavored LightGBM + qAK kNN features (combined base).

Hypothesis: qAK's 6 kNN features achieve +0.717 bp at K=4+1 alone.
A base combining qAK features with the full yekenot 38-feature recipe
ingests both signals at the tree-split level. Tree splits CAN exploit
interactions between yekenot's count-encoded categoricals and the
kNN features that the LR-meta cannot.

Predicted: K=4+1 lift could be higher than qAK alone if interactions
matter; or similar if they don't. Either way, lower ρ to PRIMARY.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
sys.path.insert(0, str(ROOT / "scripts/dgp_v3"))
from qAA_stint_imputed_base import build_stint_features, encode_categoricals

SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"


def t(label, ts):
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main():
    ts = time.time()
    out = {}
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    # Build stint features (qAA-style)
    train_fe = build_stint_features(train)
    test_fe = build_stint_features(test)
    t("stint feats", ts)

    # Load qAK kNN OOF + test predictions, AND raw kNN feature columns
    # We need the raw features, not just qAK predictions. qAK saved
    # raw features inside its training; reload data and rebuild
    sys.path.insert(0, str(ROOT / "scripts/dgp_v3"))
    from qAK_orig_kNN_tight import cell_knn_with_fallback, CELL_LEVELS, CONT_COLS

    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv").rename(
        columns={"LapTime (s)": "LapTime"}).drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)
    train_rn = train.rename(columns={"LapTime (s)": "LapTime"})
    test_rn = test.rename(columns={"LapTime (s)": "LapTime"})

    print("\nBuild kNN features for train...")
    tr_mean, tr_std, tr_max, tr_min, tr_d, tr_lvl = cell_knn_with_fallback(
        orig, train_rn, CONT_COLS, CELL_LEVELS, k=3)
    print("Build kNN features for test...")
    te_mean, te_std, te_max, te_min, te_d, te_lvl = cell_knn_with_fallback(
        orig, test_rn, CONT_COLS, CELL_LEVELS, k=3)
    t("kNN feats built", ts)

    train_fe["knn3_mean"] = tr_mean
    train_fe["knn3_std"] = tr_std
    train_fe["knn3_max"] = tr_max
    train_fe["knn3_min"] = tr_min
    train_fe["knn3_d_med"] = tr_d
    train_fe["knn3_level"] = tr_lvl.astype(np.float32)
    test_fe["knn3_mean"] = te_mean
    test_fe["knn3_std"] = te_std
    test_fe["knn3_max"] = te_max
    test_fe["knn3_min"] = te_min
    test_fe["knn3_d_med"] = te_d
    test_fe["knn3_level"] = te_lvl.astype(np.float32)

    base_num = ["LapNumber", "TyreLife", "Position", "LapTime (s)",
                "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
                "Position_Change", "PitStop", "Stint", "Year"]
    base_cat = ["Driver", "Compound", "Race"]
    stint_num = ["stint_imputed", "CumulativeTimeStint", "prev_lap_delta_stint",
                 "prev_lap_delta_drv", "stint_lap_idx", "stint_size",
                 "stint_lap_frac", "compound_changes", "position_at_stint_start",
                 "position_change_in_stint"]
    stint_cat = ["prev_compound"]
    knn_num = ["knn3_mean", "knn3_std", "knn3_max", "knn3_min", "knn3_d_med",
               "knn3_level"]
    feat_cols = base_num + base_cat + stint_num + stint_cat + knn_num
    cat_cols = base_cat + stint_cat
    print(f"  total features: {len(feat_cols)}")

    train_enc, test_enc = encode_categoricals(train_fe, test_fe, cat_cols)
    X = train_enc[feat_cols].values
    y = train_enc[TARGET].values
    X_test = test_enc[feat_cols].values
    cat_idx = [feat_cols.index(c) for c in cat_cols]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    lgb_params = dict(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        t1 = time.time()
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X[tr], y[tr], categorical_feature=cat_idx,
              eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        a = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(a)
        print(f"  fold {fold+1} AUC = {a:.5f}  ({time.time()-t1:.0f}s)  best_iter={m.best_iteration_}", flush=True)

    auc = float(roc_auc_score(y, oof))
    print(f"\n  qAR standalone OOF = {auc:.5f}", flush=True)
    out["oof_auc"] = auc
    out["fold_aucs"] = fold_aucs

    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")
    if primary_oof.ndim == 2: primary_oof = primary_oof[:, 1]
    if primary_test.ndim == 2: primary_test = primary_test[:, 1]
    rho_oof = float(spearmanr(oof, primary_oof).correlation)
    rho_test = float(spearmanr(test_pred, primary_test).correlation)
    print(f"  rho_oof: {rho_oof:.5f}  rho_test: {rho_test:.5f}", flush=True)
    out["rho_oof_vs_primary"] = rho_oof
    out["rho_test_vs_primary"] = rho_test

    BASES = [("d17_h1d_yekenot_full", "oof_d17_h1d_yekenot_full_strat.npy"),
             ("p1_single_cb_v4_gpu", "oof_p1_single_cb_v4_gpu_strat.npy"),
             ("f1_hgbc_deep", "oof_f1_hgbc_deep_strat.npy"),
             ("d16_orig_continuous_only", "oof_d16_orig_continuous_only_strat.npy")]
    base_oofs = []
    for nm, fn in BASES:
        o = np.load(ART / fn)
        if o.ndim == 2: o = o[:, 1]
        base_oofs.append(o)

    def expand(p_list):
        cols = []
        for p in p_list:
            p = np.clip(p, 1e-6, 1 - 1e-6)
            cols += [p, pd.Series(p).rank().values / len(p), np.log(p / (1 - p))]
        return np.column_stack(cols)

    def lr_meta_oof(Xm, y_):
        skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        om = np.zeros(len(y_))
        for tr, va in skf2.split(Xm, y_):
            mm = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            mm.fit(Xm[tr], y_[tr])
            om[va] = mm.predict_proba(Xm[va])[:, 1]
        return om

    Xm_K4 = expand(base_oofs)
    Xm_K5 = expand(base_oofs + [oof])
    auc_K4 = float(roc_auc_score(y, lr_meta_oof(Xm_K4, y)))
    auc_K5 = float(roc_auc_score(y, lr_meta_oof(Xm_K5, y)))
    delta = (auc_K5 - auc_K4) * 1e4
    print(f"\n  K=4+1 lift: {delta:+.3f} bp (qAK was +0.717)", flush=True)
    out["k4plus1_lift_bp"] = delta

    # Also test with qAK + qAR
    qAK_oof = np.load(ART / "dgp_v3_qAK_knn3_oof.npy")
    Xm_K6 = expand(base_oofs + [qAK_oof, oof])
    auc_K6 = float(roc_auc_score(y, lr_meta_oof(Xm_K6, y)))
    delta6 = (auc_K6 - auc_K4) * 1e4
    print(f"  K=4 + qAK + qAR: OOF={auc_K6:.5f} Δ={delta6:+.3f} bp", flush=True)
    out["k4plus2_qAK_qAR_lift_bp"] = delta6

    np.save(ART / "dgp_v3_qAR_yekenot_kNN_oof.npy", oof)
    np.save(ART / "dgp_v3_qAR_yekenot_kNN_test.npy", test_pred)
    fp = ART / "dgp_v3_qAR_yekenot_kNN.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
