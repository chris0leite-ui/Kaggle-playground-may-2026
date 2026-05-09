"""Sprint E — joint base combining qAA stint_imputed + qAB orig-derived features.

qAA standalone OOF: 0.94495, ρ_test=0.965, K=4+1 +0.143 bp WEAK.
qAB standalone: TBD (orig-only features are structurally orthogonal to all
4 K=4 bases since only d16 has any orig signal).

Hypothesis: stint_imputed sequence features (qAA) + orig hierarchical TE
+ kNN + density (qAB) are PARTIALLY orthogonal. A joint base ingesting
both is the V4-lesson realisation: features at the BASE level (where
tree splits exploit non-linear interactions) rather than at the meta.

Total feature count: 14 base raw + 11 stint + 15 orig = ~40 features.

Output: standalone OOF, ρ vs PRIMARY, K=4+1 + K=5 (K=4 + qAA) + K=5
(K=4 + qAB) gates for ablation.
"""
from __future__ import annotations

import json
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

import sys
sys.path.insert(0, str(ROOT / "scripts/dgp_v3"))
from qAA_stint_imputed_base import build_stint_features, encode_categoricals
from qAB_orig_cell_label_vote import hierarchical_te, cell_knn_vote

SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def cell_gauss_logp(orig_, df_, cont_cols, cell_key):
    out_lp = np.full(len(df_), -1e6, dtype=np.float32)
    orig_grp = orig_.groupby(cell_key, observed=True).indices
    df_grp = df_.groupby(cell_key, observed=True).indices
    for cell, q_idx in df_grp.items():
        if cell not in orig_grp:
            continue
        o_idx = orig_grp[cell]
        if len(o_idx) < 10:
            continue
        mu = orig_[cont_cols].values[o_idx].mean(axis=0)
        cov = np.cov(orig_[cont_cols].values[o_idx].T) + 1e-3 * np.eye(len(cont_cols))
        Xq = df_[cont_cols].values[q_idx]
        try:
            inv = np.linalg.inv(cov)
            sign, lod = np.linalg.slogdet(cov)
            if sign <= 0:
                continue
            diff = Xq - mu
            m = np.einsum("ij,jk,ik->i", diff, inv, diff)
            logp = -0.5 * (m + lod + len(cont_cols) * np.log(2*np.pi))
            out_lp[q_idx] = logp.astype(np.float32)
        except np.linalg.LinAlgError:
            continue
    return out_lp


def main():
    ts = time.time()
    out: dict = {}

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv").rename(
        columns={"LapTime (s)": "LapTime"}).drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)
    t(f"train {train.shape} test {test.shape} orig {orig.shape}", ts)

    # Build stint_imputed features (qAA-style)
    train_fe = build_stint_features(train)
    test_fe = build_stint_features(test)
    t(f"stint feats built", ts)

    # Build orig-derived features (qAB-style) — operate on orig.LapTime, train/test.LapTime (s)
    # Standardize column names
    train_o = train.rename(columns={"LapTime (s)": "LapTime"})
    test_o = test.rename(columns={"LapTime (s)": "LapTime"})

    h_train = hierarchical_te(orig, train_o)
    h_test = hierarchical_te(orig, test_o)
    cont_cols = ["LapTime", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]
    knn_train_pn, knn_train_d = cell_knn_vote(orig, train_o, cont_cols,
                                              ["Year", "Compound", "PitStop"], k=20)
    knn_test_pn, knn_test_d = cell_knn_vote(orig, test_o, cont_cols,
                                            ["Year", "Compound", "PitStop"], k=20)
    knn_train_pn = np.where(np.isnan(knn_train_pn), h_train["orig_pn_L3"].values, knn_train_pn)
    knn_test_pn = np.where(np.isnan(knn_test_pn), h_test["orig_pn_L3"].values, knn_test_pn)
    knn_train_d = np.where(np.isnan(knn_train_d), 0.0, knn_train_d)
    knn_test_d = np.where(np.isnan(knn_test_d), 0.0, knn_test_d)
    lp_train = cell_gauss_logp(orig, train_o, cont_cols, ["Year", "Compound", "PitStop"])
    lp_test = cell_gauss_logp(orig, test_o, cont_cols, ["Year", "Compound", "PitStop"])
    t(f"orig feats built", ts)

    # Add orig features to train_fe/test_fe
    for col in h_train.columns:
        train_fe[col] = h_train[col].values
        test_fe[col] = h_test[col].values
    train_fe["orig_knn_pn"] = knn_train_pn
    train_fe["orig_knn_d_med"] = knn_train_d
    train_fe["cell_log_density"] = lp_train
    test_fe["orig_knn_pn"] = knn_test_pn
    test_fe["orig_knn_d_med"] = knn_test_d
    test_fe["cell_log_density"] = lp_test

    # Feature columns
    base_num = ["LapNumber", "TyreLife", "Position", "LapTime (s)",
                "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
                "Position_Change", "PitStop", "Stint", "Year"]
    base_cat = ["Driver", "Compound", "Race"]
    stint_num = ["stint_imputed", "CumulativeTimeStint", "prev_lap_delta_stint",
                 "prev_lap_delta_drv", "stint_lap_idx", "stint_size",
                 "stint_lap_frac", "compound_changes", "position_at_stint_start",
                 "position_change_in_stint"]
    stint_cat = ["prev_compound"]
    orig_num = list(h_train.columns) + ["orig_knn_pn", "orig_knn_d_med", "cell_log_density"]

    feat_cols = base_num + base_cat + stint_num + stint_cat + orig_num
    cat_cols = base_cat + stint_cat
    out["feat_cols"] = feat_cols
    out["n_features"] = len(feat_cols)
    print(f"  total features: {len(feat_cols)}", flush=True)

    train_enc, test_enc = encode_categoricals(train_fe, test_fe, cat_cols)
    X = train_enc[feat_cols].values
    y = train_enc[TARGET].values
    X_test = test_enc[feat_cols].values
    cat_idx = [feat_cols.index(c) for c in cat_cols]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
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
        val_pred = m.predict_proba(X[va])[:, 1]
        oof[va] = val_pred
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        a = float(roc_auc_score(y[va], val_pred))
        fold_aucs.append(a)
        print(f"  fold {fold+1} AUC = {a:.5f}  ({time.time()-t1:.0f}s)  best_iter={m.best_iteration_}", flush=True)

    auc = float(roc_auc_score(y, oof))
    print(f"\n=== qAC joint OOF AUC = {auc:.5f} ===", flush=True)
    out["fold_aucs"] = fold_aucs
    out["oof_auc"] = auc
    out["fold_std"] = float(np.std(fold_aucs))

    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")
    if primary_oof.ndim == 2: primary_oof = primary_oof[:, 1]
    if primary_test.ndim == 2: primary_test = primary_test[:, 1]
    rho_oof = float(spearmanr(oof, primary_oof).correlation)
    rho_test = float(spearmanr(test_pred, primary_test).correlation)
    print(f"  rho_oof: {rho_oof:.5f}  rho_test: {rho_test:.5f}", flush=True)
    out["rho_oof_vs_primary"] = rho_oof
    out["rho_test_vs_primary"] = rho_test

    # K=4+1 gate
    BASES = [
        ("d17_h1d_yekenot_full", "oof_d17_h1d_yekenot_full_strat.npy", "test_d17_h1d_yekenot_full_strat.npy"),
        ("p1_single_cb_v4_gpu", "oof_p1_single_cb_v4_gpu_strat.npy", "test_p1_single_cb_v4_gpu_strat.npy"),
        ("f1_hgbc_deep", "oof_f1_hgbc_deep_strat.npy", "test_f1_hgbc_deep_strat.npy"),
        ("d16_orig_continuous_only", "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ]
    base_oofs, base_tests = [], []
    for name, oof_f, test_f in BASES:
        o = np.load(ART / oof_f); te = np.load(ART / test_f)
        if o.ndim == 2: o = o[:, 1]
        if te.ndim == 2: te = te[:, 1]
        base_oofs.append(o); base_tests.append(te)

    def expand(p_list):
        cols = []
        for p in p_list:
            p = np.clip(p, 1e-6, 1 - 1e-6)
            cols += [p, pd.Series(p).rank().values / len(p), np.log(p / (1 - p))]
        return np.column_stack(cols)

    Xm_K4 = expand(base_oofs)
    Xm_K5 = expand(base_oofs + [oof])
    def lr_meta_oof(Xm, y_):
        skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        om = np.zeros(len(y_))
        for tr, va in skf2.split(Xm, y_):
            mlr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            mlr.fit(Xm[tr], y_[tr])
            om[va] = mlr.predict_proba(Xm[va])[:, 1]
        return om

    auc_K4 = float(roc_auc_score(y, lr_meta_oof(Xm_K4, y)))
    auc_K5 = float(roc_auc_score(y, lr_meta_oof(Xm_K5, y)))
    delta_bp = (auc_K5 - auc_K4) * 1e4
    print(f"\n  K=4 LR-meta: {auc_K4:.5f}", flush=True)
    print(f"  K=5 (K=4+qAC) LR-meta: {auc_K5:.5f}", flush=True)
    print(f"  K=4+1 lift: {delta_bp:+.3f} bp", flush=True)
    print(f"  GATE: {'PASS' if delta_bp >= 0.5 else ('WEAK' if delta_bp > -0.3 else 'FAIL')}", flush=True)
    out["k4_lr_meta_oof"] = auc_K4
    out["k5_lr_meta_oof"] = auc_K5
    out["k4plus1_lift_bp"] = delta_bp
    out["gate_verdict"] = "PASS" if delta_bp >= 0.5 else ("WEAK" if delta_bp > -0.3 else "FAIL")

    np.save(ART / "dgp_v3_qAC_joint_oof.npy", oof)
    np.save(ART / "dgp_v3_qAC_joint_test.npy", test_pred)
    fp = ART / "dgp_v3_qAC_joint.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
