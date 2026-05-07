"""d18 F5 — Class-conditional GMM log-ratio (E4 done right, simpler density).

Two GMMs on orig: one fit on orig[y=0] rows, one on orig[y=1] rows.
For each synth row, compute log_ratio = log p(x|y=1) - log p(x|y=0).
This is the orig-DGP class-conditional log Bayes factor.

Plus: fit a separate pair of GMMs on synth (split by y) → log_ratio_synth.
The DIFFERENCE log_ratio_orig - log_ratio_synth quantifies how the
synthesizer's class-conditional generation drifts from orig's.

Features (5 per row):
  ll_y0_orig         : log p_orig(x | y=0)
  ll_y1_orig         : log p_orig(x | y=1)
  log_ratio_orig     : ll_y1_orig - ll_y0_orig
  log_ratio_synth    : ll_y1_synth - ll_y0_synth (synth-fitted GMMs)
  log_ratio_drift    : log_ratio_orig - log_ratio_synth

Density model: GaussianMixture(n_components=8, full covariance) per class
on the 7 KS-low marginal-aligned features.

Then 5-fold LGBM on (raw 14 features + 5 class-cond features).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"
LAPTIME = "LapTime (s)"
N_COMP = 8

KS_LOW_FEATS = ["TyreLife", "Position", LAPTIME, "Cumulative_Degradation",
                "RaceProgress", "LapTime_Delta", "LapNumber"]
NUM_FEATS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             LAPTIME, "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change"]


def fit_gmm(X, n_comp=N_COMP, label=""):
    g = GaussianMixture(n_components=n_comp, covariance_type="full",
                        max_iter=200, random_state=SEED, reg_covar=1e-3)
    g.fit(X)
    print(f"  GMM[{label}]  fit on {len(X)} rows  conv={g.converged_}  "
          f"iter={g.n_iter_}  ll={g.score(X):.3f}")
    return g


def main():
    t0 = time.time()
    print("[F5 class-cond GMM]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()

    # Build feature matrices
    def to_mat(df):
        X = df[KS_LOW_FEATS].astype(float).values
        np.nan_to_num(X, copy=False, nan=0.0)
        return X

    Xo = to_mat(orig); yo = orig[TARGET].astype(int).values
    Xtr = to_mat(tr);  ytr = tr[TARGET].astype(int).values
    Xte = to_mat(te)
    # Standardize using orig stats
    mu = Xo.mean(axis=0); sd = Xo.std(axis=0) + 1e-8
    Xo_s = (Xo - mu) / sd
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd

    print(f"\n[fit GMMs on orig (n={len(Xo)})]")
    g_o0 = fit_gmm(Xo_s[yo == 0], label="orig y=0")
    g_o1 = fit_gmm(Xo_s[yo == 1], label="orig y=1")

    print(f"\n[fit GMMs on synth-train (n={len(Xtr)})]")
    g_s0 = fit_gmm(Xtr_s[ytr == 0], label="synth y=0")
    g_s1 = fit_gmm(Xtr_s[ytr == 1], label="synth y=1")

    print(f"\n[score chains on train + test]")
    ll_o0_tr = g_o0.score_samples(Xtr_s); ll_o1_tr = g_o1.score_samples(Xtr_s)
    ll_s0_tr = g_s0.score_samples(Xtr_s); ll_s1_tr = g_s1.score_samples(Xtr_s)
    ll_o0_te = g_o0.score_samples(Xte_s); ll_o1_te = g_o1.score_samples(Xte_s)
    ll_s0_te = g_s0.score_samples(Xte_s); ll_s1_te = g_s1.score_samples(Xte_s)

    def feats(ll_o0, ll_o1, ll_s0, ll_s1):
        ratio_orig = ll_o1 - ll_o0
        ratio_synth = ll_s1 - ll_s0
        drift = ratio_orig - ratio_synth
        return pd.DataFrame(dict(
            cc_ll_y0_orig=ll_o0.astype(np.float32),
            cc_ll_y1_orig=ll_o1.astype(np.float32),
            cc_log_ratio_orig=ratio_orig.astype(np.float32),
            cc_log_ratio_synth=ratio_synth.astype(np.float32),
            cc_log_ratio_drift=drift.astype(np.float32),
        ))

    tr_F = feats(ll_o0_tr, ll_o1_tr, ll_s0_tr, ll_s1_tr)
    te_F = feats(ll_o0_te, ll_o1_te, ll_s0_te, ll_s1_te)

    # Per-feature standalone AUC (informative but not used)
    for c in tr_F.columns:
        v = tr_F[c].values
        a = roc_auc_score(ytr, v) if v.std() > 1e-9 else 0.5
        print(f"  {c}  std AUC {a:.5f}")

    # KS y=0 vs y=1 of each feature
    from scipy.stats import ks_2samp
    pos = ytr == 1; neg = ytr == 0
    print()
    for c in tr_F.columns:
        v = tr_F[c].values
        ks, _ = ks_2samp(v[pos], v[neg])
        print(f"  {c}  KS y=0 vs y=1 = {ks:.4f}")

    # Downstream LGBM (raw + 5 class-cond features)
    cmps = sorted(set(tr["Compound"].astype(str)) | set(te["Compound"].astype(str)))
    cm = {c: i for i, c in enumerate(cmps)}
    races = sorted(set(tr["Race"].astype(str)) | set(te["Race"].astype(str)))
    rm = {r: i for i, r in enumerate(races)}
    raw_cols = ["Compound", "Race"] + NUM_FEATS
    trX = tr[raw_cols].copy()
    teX = te[raw_cols].copy()
    trX["Compound"] = tr["Compound"].astype(str).map(cm).astype(int)
    teX["Compound"] = te["Compound"].astype(str).map(cm).astype(int)
    trX["Race"] = tr["Race"].astype(str).map(rm).astype(int)
    teX["Race"] = te["Race"].astype(str).map(rm).astype(int)
    trX = pd.concat([trX.reset_index(drop=True), tr_F.reset_index(drop=True)], axis=1)
    teX = pd.concat([teX.reset_index(drop=True), te_F.reset_index(drop=True)], axis=1)

    print(f"\n[downstream LGBM raw + 5 class-cond features]")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(ytr)); test_avg = np.zeros(len(te))
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=127, min_data_in_leaf=80, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbosity=-1, seed=SEED)
    cat_idx = [trX.columns.get_loc(c) for c in ["Compound", "Race"]]
    for fi, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(ytr)), ytr), 1):
        ds_tr = lgb.Dataset(trX.iloc[tr_i], label=ytr[tr_i],
                            categorical_feature=cat_idx, free_raw_data=False)
        ds_va = lgb.Dataset(trX.iloc[va_i], label=ytr[va_i],
                            categorical_feature=cat_idx, reference=ds_tr,
                            free_raw_data=False)
        m = lgb.train(p, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_i] = m.predict(trX.iloc[va_i], num_iteration=m.best_iteration)
        test_avg += m.predict(teX, num_iteration=m.best_iteration) / N_FOLDS
        print(f"  fold {fi}: AUC={roc_auc_score(ytr[va_i], oof[va_i]):.5f}")

    auc = float(roc_auc_score(ytr, oof))
    print(f"  OOF AUC = {auc:.5f}")
    np.save(ART / "oof_d18_f5_class_cond_gmm_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d18_f5_class_cond_gmm_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))
    summary = dict(oof_auc=auc, n_components=N_COMP, ks_low_feats=KS_LOW_FEATS,
                   wall_s=time.time() - t0)
    (ART / "d18_f5_class_cond_gmm_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done F5]  wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
