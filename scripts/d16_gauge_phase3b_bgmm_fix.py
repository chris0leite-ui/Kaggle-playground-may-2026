"""d16 Phase 3b — BGMM rerun with stronger regularization (v1 crashed mid-fit).

Original BGMM(32, dirichlet_process, reg_covar=1e-3) failed with ill-defined
empirical covariance (singleton/collapsed components). v2: use reg_covar=1.0
and concentration prior 1.0 to encourage smoother components.

Reuses GMM logp arrays already saved by Phase 3 v1.

Outputs:
  d16_logp_orig_bgmm_synth_train.npy
  d16_logp_orig_bgmm_synth_test.npy
  oof_d16_logp_bgmm_strat.npy
  test_d16_logp_bgmm_strat.npy
  d16_phase3_summary.json   (full summary including v1 GMM)
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
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

NUM = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
       "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
       "RaceProgress", "Position_Change"]
CAT_OHE = ["Compound"]


def main():
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    y = tr[TARGET].astype(int).values

    union = pd.concat([orig, tr, te], axis=0, ignore_index=True)
    pieces = [union[NUM].astype(float).fillna(union[NUM].astype(float).median())]
    for c in CAT_OHE:
        d = pd.get_dummies(union[c].astype(str), prefix=c)
        pieces.append(d)
    X = pd.concat(pieces, axis=1).astype(float)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    n_orig, n_tr = len(orig), len(tr)
    Xo, Xs, Xt = X[:n_orig], X[n_orig:n_orig + n_tr], X[n_orig + n_tr:]
    step(f"  Xo={Xo.shape} Xs={Xs.shape} Xt={Xt.shape}")

    step("BGMM(16, full, reg_covar=1.0, conc_prior=1.0)")
    bgmm = BayesianGaussianMixture(
        n_components=16, covariance_type="full", max_iter=200,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0, reg_covar=1.0, random_state=SEED,
    )
    try:
        bgmm.fit(Xo)
        step(f"  converged={bgmm.converged_} n_iter={bgmm.n_iter_} "
             f"effective={int((bgmm.weights_ > 1e-3).sum())}")
        logp_tr_b = bgmm.score_samples(Xs)
        logp_te_b = bgmm.score_samples(Xt)
        np.save(ART / "d16_logp_orig_bgmm_synth_train.npy", logp_tr_b)
        np.save(ART / "d16_logp_orig_bgmm_synth_test.npy", logp_te_b)

        # single-feat LGBM gate
        feat_tr = logp_tr_b.reshape(-1, 1)
        feat_te = logp_te_b.reshape(-1, 1)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros(n_tr)
        pred_te = np.zeros(len(Xt))
        for fi, (tri, vai) in enumerate(skf.split(feat_tr, y)):
            m = lgb.train(
                dict(objective="binary", metric="auc", learning_rate=0.05,
                     num_leaves=15, min_data_in_leaf=500, feature_fraction=1.0,
                     verbose=-1, n_jobs=4, seed=SEED),
                lgb.Dataset(feat_tr[tri], y[tri]),
                num_boost_round=200,
                valid_sets=[lgb.Dataset(feat_tr[vai], y[vai])],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            oof[vai] = m.predict(feat_tr[vai])
            pred_te += m.predict(feat_te) / N_FOLDS
        auc_b = roc_auc_score(y, oof)
        np.save(ART / "oof_d16_logp_bgmm_strat.npy", oof)
        np.save(ART / "test_d16_logp_bgmm_strat.npy", pred_te)
        step(f"  P3.2 single-feat OOF AUC {auc_b:.5f}")
        bgmm_ok = True
    except Exception as e:
        step(f"  BGMM failed: {e}")
        bgmm_ok = False
        auc_b = None
        logp_tr_b = None

    # consolidate Phase 3 summary
    gmm_oof = np.load(ART / "oof_d16_logp_gmm_strat.npy")
    auc_g = float(roc_auc_score(y, gmm_oof))

    summary = dict(
        P3_1_gmm=dict(single_feat_auc=auc_g),
        P3_2_bgmm=dict(success=bgmm_ok, single_feat_auc=auc_b),
    )
    if bgmm_ok and logp_tr_b is not None:
        logp_tr_g = np.load(ART / "d16_logp_orig_gmm_synth_train.npy")
        rho = float(np.corrcoef(logp_tr_g, logp_tr_b)[0, 1])
        summary["P3_gmm_bgmm_logp_correlation"] = rho
    summary["runtime_s"] = time.time() - t0
    with open(ART / "d16_phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step("DONE")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
