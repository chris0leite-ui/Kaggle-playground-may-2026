"""qAH — orig-driver-rate features (real-driver PitNextLap signal from orig).

The residual analysis (this session) showed K=4 PRIMARY:
  - real drivers (n=168k):  AUC 0.94616  (1.2 bp WORSE than ghosts)
  - ghost drivers (n=270k): AUC 0.95834
  - PS=1 in non-2023 years: AUC 0.87-0.89

Real drivers are HARDER than fabricated ones — they carry F1-racing
behavioural patterns the synth doesn't fully randomise. The 31 real
drivers have data in orig with ground-truth labels.

Hypothesis: import orig-derived driver-specific PitNextLap rates as
features. Synth-trained models don't have access to orig labels at the
driver level. This is genuinely new information.

Features (per row):
  H1  orig_driver_rate          per-Driver PitNextLap rate from orig
  H2  orig_driver_pn_rate_year  per-(Driver, Year) rate
  H3  orig_driver_pn_rate_cmp   per-(Driver, Compound) rate
  H4  orig_driver_pn_rate_year_cmp  per-(Driver, Year, Compound) rate
  H5  orig_driver_n_rows        sample count per driver in orig (sparsity)
  H6  driver_is_real            bool: Driver in orig
  H7  orig_pn_rate_global       global PitNextLap rate from orig
       (anchor for ghost drivers)
  H8  same as orig_pn_rate_global − orig_driver_rate (residual)

For ghost drivers (not in orig), H1-H4 fall back to orig_pn_rate_global.

All features are computed using ALL of orig (not fold-restricted) since
orig is external (not in CV stream) per Rule 24's intent.

Output: standalone OOF, ρ vs PRIMARY, K=4+1 gate, K=5 with qAA gate.
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
    t(f"orig {orig.shape} train {train.shape} test {test.shape}", ts)

    # Compute per-driver rates from orig with empirical-Bayes shrinkage
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
    print(f"  real drivers: {len(real_drivers)}")
    print(f"  TE per-Driver counts: min={te_d['count'].min()}, p10={te_d['count'].quantile(0.10):.0f}, p90={te_d['count'].quantile(0.90):.0f}")

    def attach(df, keys, te, col_rate, col_cnt):
        merged = df[keys].merge(te, left_on=keys, right_index=True, how="left")
        df[col_rate] = merged["rate"].fillna(global_rate).astype(np.float32)
        df[col_cnt] = merged["count"].fillna(0).astype(np.float32)
        return df

    for df in (train, test):
        attach(df, ["Driver"], te_d, "orig_driver_rate", "orig_driver_n")
        attach(df, ["Driver", "Year"], te_dy, "orig_dyr_rate", "orig_dyr_n")
        attach(df, ["Driver", "Compound"], te_dc, "orig_dcr_rate", "orig_dcr_n")
        attach(df, ["Driver", "Year", "Compound"], te_dyc, "orig_dycr_rate", "orig_dycr_n")
        df["driver_is_real"] = df["Driver"].isin(real_drivers).astype(np.int8)
        df["orig_global_rate"] = global_rate
        df["orig_driver_resid"] = df["orig_driver_rate"] - global_rate

    t("features attached", ts)
    print(f"  driver_is_real frac in train: {train.driver_is_real.mean():.3f}")
    print(f"  driver_is_real frac in test:  {test.driver_is_real.mean():.3f}")

    feat_cols = ["orig_driver_rate", "orig_driver_n",
                 "orig_dyr_rate", "orig_dyr_n",
                 "orig_dcr_rate", "orig_dcr_n",
                 "orig_dycr_rate", "orig_dycr_n",
                 "driver_is_real", "orig_global_rate", "orig_driver_resid"]

    X = train[feat_cols].values.astype(np.float32)
    y = train[TARGET].values.astype(np.int32)
    X_test = test[feat_cols].values.astype(np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    lgb_params = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        t1 = time.time()
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        a = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(a)
        print(f"  fold {fold+1} AUC = {a:.5f}  ({time.time()-t1:.0f}s)  best_iter={m.best_iteration_}", flush=True)

    auc = float(roc_auc_score(y, oof))
    print(f"\n  qAH standalone OOF AUC = {auc:.5f}", flush=True)
    out["fold_aucs"] = fold_aucs
    out["oof_auc"] = auc

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
    print(f"  K=5 (K=4 + qAH) LR-meta: {auc_K5:.5f}", flush=True)
    print(f"  K=4+1 lift: {delta:+.3f} bp", flush=True)
    out["k4plus1_lift_bp"] = delta

    # K=5 with qAA also (for joint + qAA + qAH)
    qaa_oof_path = ART / "dgp_v3_qAA_stint_imputed_oof.npy"
    if qaa_oof_path.exists():
        qaa_oof = np.load(qaa_oof_path)
        Xm_K6 = expand(base_oofs + [qaa_oof, oof])
        auc_K6 = float(roc_auc_score(y, lr_meta_oof(Xm_K6, y)))
        delta_K6 = (auc_K6 - auc_K4) * 1e4
        print(f"  K=4 + qAA + qAH LR-meta: {auc_K6:.5f}  Δ={delta_K6:+.3f} bp", flush=True)
        out["k4plus2_qAA_qAH_lift_bp"] = delta_K6

    # Per-driver-cluster diagnostic
    real_mask = train["driver_is_real"].values == 1
    auc_real_K4 = float(roc_auc_score(y[real_mask], primary_oof[real_mask])) if y[real_mask].sum() > 0 else np.nan
    auc_real_qAH = float(roc_auc_score(y[real_mask], oof[real_mask])) if y[real_mask].sum() > 0 else np.nan
    auc_ghost_K4 = float(roc_auc_score(y[~real_mask], primary_oof[~real_mask])) if y[~real_mask].sum() > 0 else np.nan
    auc_ghost_qAH = float(roc_auc_score(y[~real_mask], oof[~real_mask])) if y[~real_mask].sum() > 0 else np.nan
    print(f"\n  real-driver: K=4 AUC {auc_real_K4:.5f} → qAH {auc_real_qAH:.5f}", flush=True)
    print(f"  ghost-driver: K=4 AUC {auc_ghost_K4:.5f} → qAH {auc_ghost_qAH:.5f}", flush=True)
    out["auc_real_K4"] = auc_real_K4
    out["auc_real_qAH"] = auc_real_qAH
    out["auc_ghost_K4"] = auc_ghost_K4
    out["auc_ghost_qAH"] = auc_ghost_qAH

    np.save(ART / "dgp_v3_qAH_orig_driver_oof.npy", oof)
    np.save(ART / "dgp_v3_qAH_orig_driver_test.npy", test_pred)
    fp = ART / "dgp_v3_qAH_orig_driver.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
