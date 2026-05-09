"""Sprint F — d16++ refit: LightGBM trained on ORIG with stint_imputed features.

The current d16_orig_continuous_only base trains a LightGBM on orig
(101k rows, 7 KS-low continuous features) and predicts on synth. Standalone
synth-train AUC ~0.91483, ρ to PRIMARY 0.85, K=4+1 lift unknown but it's
in the K=4 pool so it carries some weight.

V4 lesson (state/current.md 2026-05-09 AM): kNN-target-mean ingested as
a tree feature INSIDE a base produced +0.8 bp on LB at ρ=0.99989.
Same feature added at meta extracted +0.01 bp. Tree-internal split
exploitation is structurally different.

This probe tests the V4 pattern with stint_imputed features:
- Train LightGBM on ORIG (not synth) with the 7 KS-low features +
  stint_imputed-derived features computed FROM ORIG.
- Predict on synth.
- Compare standalone OOF + ρ vs current d16; gate at K=4+1.

Hypothesis: the d16 base trained with stint_imputed expands its
representational capacity into the recovered stint-identity axis. Even
if absorbed at the meta, the base's tree splits route by stint_imputed
× TyreLife × Compound differently than the meta can reconstruct from
predictions alone.

Output: scripts/artifacts/dgp_v3_qAF_d16plus_oof.npy (orig 5-fold OOF)
        scripts/artifacts/dgp_v3_qAF_d16plus_test.npy (synth-test prediction)
        scripts/artifacts/dgp_v3_qAF_d16plus.json
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
from qAA_stint_imputed_base import build_stint_features

SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main():
    ts = time.time()
    out: dict = {}

    # Load orig (training source) and synth (target prediction)
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv")
    orig = orig.drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    t(f"orig {orig.shape} train {train.shape} test {test.shape}", ts)

    # Build stint_imputed features on all three
    orig_fe = build_stint_features(orig)
    train_fe = build_stint_features(train)
    test_fe = build_stint_features(test)
    t("stint feats built on orig + synth", ts)

    # Continuous + cell-key feature set (similar to qZ but with stint features)
    cont_feats = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                  "RaceProgress", "Position", "TyreLife", "Position_Change"]
    stint_feats = ["stint_imputed", "CumulativeTimeStint", "prev_lap_delta_stint",
                   "prev_lap_delta_drv", "stint_lap_idx", "stint_size",
                   "stint_lap_frac", "compound_changes", "position_at_stint_start",
                   "position_change_in_stint"]
    # Cell-key categoricals (encoded via Compound + Race + binarised)
    cat_feats = ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber",
                 "prev_compound"]

    # Ensure all columns exist, encode categoricals
    enc = {}
    for c in ["Compound", "Race", "prev_compound"]:
        all_vals = pd.concat([orig_fe[c], train_fe[c], test_fe[c]],
                             ignore_index=True)
        enc[c] = pd.Categorical(all_vals).categories

    def encode(df):
        df = df.copy()
        for c in ["Compound", "Race", "prev_compound"]:
            df[c] = pd.Categorical(df[c], categories=enc[c]).codes.astype(np.int32)
        return df

    orig_e = encode(orig_fe)
    train_e = encode(train_fe)
    test_e = encode(test_fe)

    feat_cols = cont_feats + stint_feats + cat_feats
    cat_idx = [feat_cols.index(c) for c in cat_feats]
    print(f"  total features: {len(feat_cols)}", flush=True)

    X_orig = orig_e[feat_cols].values
    y_orig = orig_e[TARGET].values.astype(np.int32)
    X_train_synth = train_e[feat_cols].values
    X_test_synth = test_e[feat_cols].values

    # 5-fold OOF on orig (training source); apply each fold to synth
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_orig = np.zeros(len(X_orig), dtype=np.float64)
    train_synth_pred = np.zeros(len(X_train_synth), dtype=np.float64)
    test_synth_pred = np.zeros(len(X_test_synth), dtype=np.float64)
    fold_aucs = []

    lgb_params = dict(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    for fold, (tr, va) in enumerate(skf.split(X_orig, y_orig)):
        t1 = time.time()
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_orig[tr], y_orig[tr], categorical_feature=cat_idx,
              eval_set=[(X_orig[va], y_orig[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        val_pred = m.predict_proba(X_orig[va])[:, 1]
        oof_orig[va] = val_pred
        train_synth_pred += m.predict_proba(X_train_synth)[:, 1] / N_FOLDS
        test_synth_pred += m.predict_proba(X_test_synth)[:, 1] / N_FOLDS
        a = float(roc_auc_score(y_orig[va], val_pred))
        fold_aucs.append(a)
        print(f"  fold {fold+1} AUC = {a:.5f}  ({time.time()-t1:.0f}s)  best_iter={m.best_iteration_}", flush=True)

    auc_orig = float(roc_auc_score(y_orig, oof_orig))
    print(f"\n  d16++ orig 5-fold OOF AUC = {auc_orig:.5f}", flush=True)
    out["orig_5fold_oof_auc"] = auc_orig
    out["fold_aucs"] = fold_aucs

    # Apply to synth-train
    y_synth = train_e[TARGET].values
    auc_synth = float(roc_auc_score(y_synth, train_synth_pred))
    print(f"  applied to synth-train AUC = {auc_synth:.5f}", flush=True)
    print(f"  (current d16_orig_continuous_only standalone AUC ~0.915; qZ d16++ 0.93985)", flush=True)
    out["synth_train_auc"] = auc_synth

    # rho vs PRIMARY (test)
    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")
    if primary_oof.ndim == 2: primary_oof = primary_oof[:, 1]
    if primary_test.ndim == 2: primary_test = primary_test[:, 1]
    rho_test_synth = float(spearmanr(test_synth_pred, primary_test).correlation)
    rho_oof_synth = float(spearmanr(train_synth_pred, primary_oof).correlation)
    print(f"  rho_oof_synth: {rho_oof_synth:.5f}  rho_test_synth: {rho_test_synth:.5f}", flush=True)
    out["rho_test_vs_primary"] = rho_test_synth
    out["rho_oof_vs_primary"] = rho_oof_synth

    # K=4+1 LR-meta gate using train_synth_pred as the OOF input (since the
    # qAF model is trained on orig, NOT on synth labels — predictions on
    # synth-train are out-of-sample naturally; no fold-leakage)
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

    def lr_meta_oof(Xm, y_):
        skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        om = np.zeros(len(y_))
        for tr, va in skf2.split(Xm, y_):
            m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            m.fit(Xm[tr], y_[tr])
            om[va] = m.predict_proba(Xm[va])[:, 1]
        return om

    Xm_K4 = expand(base_oofs)
    Xm_K5 = expand(base_oofs + [train_synth_pred])
    auc_K4 = float(roc_auc_score(y_synth, lr_meta_oof(Xm_K4, y_synth)))
    auc_K5 = float(roc_auc_score(y_synth, lr_meta_oof(Xm_K5, y_synth)))
    delta_bp = (auc_K5 - auc_K4) * 1e4
    print(f"\n  K=4 LR-meta: {auc_K4:.5f}", flush=True)
    print(f"  K=5 (K=4 + d16++_with_stint) LR-meta: {auc_K5:.5f}", flush=True)
    print(f"  K=4+1 lift: {delta_bp:+.3f} bp", flush=True)
    print(f"  GATE: {'PASS' if delta_bp >= 0.5 else ('WEAK' if delta_bp > -0.3 else 'FAIL')}", flush=True)
    out["k4plus1_lift_bp"] = delta_bp

    # Also test SWAP: replace d16_orig in K=4 with this new base
    base_oofs_swap = base_oofs[:3] + [train_synth_pred]
    base_tests_swap = base_tests[:3] + [test_synth_pred]
    Xm_K4swap = expand(base_oofs_swap)
    auc_K4swap = float(roc_auc_score(y_synth, lr_meta_oof(Xm_K4swap, y_synth)))
    delta_swap = (auc_K4swap - auc_K4) * 1e4
    print(f"\n  SWAP: K=4 with d16++_stint replacing d16_orig: {auc_K4swap:.5f}", flush=True)
    print(f"  SWAP delta: {delta_swap:+.3f} bp", flush=True)
    out["swap_d16_lift_bp"] = delta_swap

    np.save(ART / "dgp_v3_qAF_d16plus_oof.npy", train_synth_pred)
    np.save(ART / "dgp_v3_qAF_d16plus_test.npy", test_synth_pred)
    fp = ART / "dgp_v3_qAF_d16plus.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
