"""qAJ — qAA stint features + qAH orig-driver-rate features in ONE base.

qAA standalone OOF 0.94495, K=4+1 +0.143 bp (WEAK).
qAH standalone OOF 0.54597, K=4+1 +0.028 bp (WEAK; fails because
ghost rows fall back to global rate, killing AUC for 93% of test).

Combining: stint features handle ALL rows; orig-driver features carry
real-driver-specific signal that K=4 fails to fully exploit (real-driver
AUC 0.946 vs ghost AUC 0.958). The combined base should fit ghost rows
via stint features and real rows via stint+orig-driver — capturing the
distinction that K=4 misses.

Distribution shift caveat: train has 38% real drivers, test only 7%.
Real-driver lift translates to ~1/5 of train-time AUC contribution.
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
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv")
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)

    train_fe = build_stint_features(train)
    test_fe = build_stint_features(test)
    t("stint feats built", ts)

    # Build orig-driver TE features
    K_SMOOTH = 30.0
    global_rate = float(orig[TARGET].mean())

    def build_te(orig_df, keys, k_smooth=K_SMOOTH):
        g = orig_df.groupby(keys, observed=True)[TARGET].agg(["sum", "count"])
        g.columns = ["sum", "count"]
        g["rate"] = (g["sum"] + k_smooth * global_rate) / (g["count"] + k_smooth)
        return g[["rate", "count"]]

    te_d = build_te(orig, ["Driver"])
    te_dy = build_te(orig, ["Driver", "Year"])
    te_dc = build_te(orig, ["Driver", "Compound"])
    te_dyc = build_te(orig, ["Driver", "Year", "Compound"])
    real_drivers = set(orig.Driver.unique())

    def attach(df, keys, te, col_rate, col_cnt):
        merged = df[keys].merge(te, left_on=keys, right_index=True, how="left")
        df[col_rate] = merged["rate"].fillna(global_rate).astype(np.float32)
        df[col_cnt] = merged["count"].fillna(0).astype(np.float32)
        return df

    for df in (train_fe, test_fe):
        attach(df, ["Driver"], te_d, "orig_drv_rate", "orig_drv_n")
        attach(df, ["Driver", "Year"], te_dy, "orig_dy_rate", "orig_dy_n")
        attach(df, ["Driver", "Compound"], te_dc, "orig_dc_rate", "orig_dc_n")
        attach(df, ["Driver", "Year", "Compound"], te_dyc, "orig_dyc_rate", "orig_dyc_n")
        df["driver_is_real"] = df["Driver"].isin(real_drivers).astype(np.int8)

    t("orig-driver feats attached", ts)

    base_num = ["LapNumber", "TyreLife", "Position", "LapTime (s)",
                "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
                "Position_Change", "PitStop", "Stint", "Year"]
    base_cat = ["Driver", "Compound", "Race"]
    stint_num = ["stint_imputed", "CumulativeTimeStint", "prev_lap_delta_stint",
                 "prev_lap_delta_drv", "stint_lap_idx", "stint_size",
                 "stint_lap_frac", "compound_changes", "position_at_stint_start",
                 "position_change_in_stint"]
    stint_cat = ["prev_compound"]
    orig_drv_num = ["orig_drv_rate", "orig_drv_n", "orig_dy_rate", "orig_dy_n",
                    "orig_dc_rate", "orig_dc_n", "orig_dyc_rate", "orig_dyc_n",
                    "driver_is_real"]
    feat_cols = base_num + base_cat + stint_num + stint_cat + orig_drv_num
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
    print(f"\n  qAJ standalone OOF AUC = {auc:.5f}", flush=True)
    print(f"  (qAA was 0.94495 — delta {(auc-0.94495)*1e4:+.2f} bp)", flush=True)

    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")
    if primary_oof.ndim == 2: primary_oof = primary_oof[:, 1]
    if primary_test.ndim == 2: primary_test = primary_test[:, 1]
    rho_oof = float(spearmanr(oof, primary_oof).correlation)
    rho_test = float(spearmanr(test_pred, primary_test).correlation)
    print(f"  rho_oof: {rho_oof:.5f}  rho_test: {rho_test:.5f}", flush=True)
    out.update(oof_auc=auc, fold_aucs=fold_aucs,
               rho_oof_vs_primary=rho_oof, rho_test_vs_primary=rho_test)

    BASES = [
        ("d17_h1d_yekenot_full", "oof_d17_h1d_yekenot_full_strat.npy"),
        ("p1_single_cb_v4_gpu", "oof_p1_single_cb_v4_gpu_strat.npy"),
        ("f1_hgbc_deep", "oof_f1_hgbc_deep_strat.npy"),
        ("d16_orig_continuous_only", "oof_d16_orig_continuous_only_strat.npy"),
    ]
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
    print(f"\n  K=4 LR-meta: {auc_K4:.5f}", flush=True)
    print(f"  K=5 (K=4 + qAJ) LR-meta: {auc_K5:.5f}", flush=True)
    print(f"  K=4+1 lift: {delta:+.3f} bp", flush=True)
    out["k4plus1_lift_bp"] = delta

    # Cross-check: K=4 + qAA vs K=4 + qAJ
    qaa_oof = np.load(ART / "dgp_v3_qAA_stint_imputed_oof.npy")
    Xm_K5_qAA = expand(base_oofs + [qaa_oof])
    auc_K5_qAA = float(roc_auc_score(y, lr_meta_oof(Xm_K5_qAA, y)))
    print(f"  reference K=5 + qAA: {auc_K5_qAA:.5f}  (Δ_qAJ_vs_qAA: {(auc_K5-auc_K5_qAA)*1e4:+.3f} bp)", flush=True)

    # Real vs ghost diagnostic
    real_mask = train_enc["driver_is_real"].values == 1
    auc_real_K4 = float(roc_auc_score(y[real_mask], primary_oof[real_mask]))
    auc_real_qAJ = float(roc_auc_score(y[real_mask], oof[real_mask]))
    auc_ghost_K4 = float(roc_auc_score(y[~real_mask], primary_oof[~real_mask]))
    auc_ghost_qAJ = float(roc_auc_score(y[~real_mask], oof[~real_mask]))
    print(f"\n  real-driver:  K=4 {auc_real_K4:.5f} → qAJ {auc_real_qAJ:.5f}  (Δ {(auc_real_qAJ-auc_real_K4)*1e4:+.2f} bp)", flush=True)
    print(f"  ghost-driver: K=4 {auc_ghost_K4:.5f} → qAJ {auc_ghost_qAJ:.5f}  (Δ {(auc_ghost_qAJ-auc_ghost_K4)*1e4:+.2f} bp)", flush=True)

    np.save(ART / "dgp_v3_qAJ_stint_orig_drv_oof.npy", oof)
    np.save(ART / "dgp_v3_qAJ_stint_orig_drv_test.npy", test_pred)
    fp = ART / "dgp_v3_qAJ_stint_orig_drv.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
