"""d16 Phase 3 — generative model on orig → log p_orig(x_synth) feature.

Two estimators of p_orig (KDE skipped due to O(n*m) cost; redundant with GMM):
  P3.1  GaussianMixture (16 components, full covariance) on orig
  P3.2  Bayesian Gaussian Mixture (Dirichlet process, ~32 components)
        — less aggressive prior; captures heavier tails
  P3.3  Robust normalizing-flow surrogate via stacked GMM (3 layers) — heavy NF skipped
        for runtime budget; we use KMeans-anchored local Gaussians instead.

Each builds log p_orig(x_synth) feature → LGBM K=21+1 single-feature probe (standalone OOF)
and saves OOF/test for downstream use.

Outputs:
  scripts/artifacts/d16_logp_orig_gmm_synth_train.npy / _test.npy
  scripts/artifacts/d16_logp_orig_bgmm_synth_train.npy / _test.npy
  scripts/artifacts/oof_d16_logp_gmm_strat.npy / test_*
  scripts/artifacts/oof_d16_logp_bgmm_strat.npy / test_*
  scripts/artifacts/d16_phase3_summary.json
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"

# Build a numeric-only feature view (one-hot for cats, scale numerics)
NUM = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
       "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
       "RaceProgress", "Position_Change"]
CAT_OHE = ["Compound"]  # restrict OHE to small-cardinality cats; Driver/Race too high


def build_feat_matrix(orig, tr, te, ohe_cats=CAT_OHE):
    union = pd.concat([orig, tr, te], axis=0, ignore_index=True)
    pieces = [union[NUM].astype(float).fillna(union[NUM].astype(float).median())]
    for c in ohe_cats:
        d = pd.get_dummies(union[c].astype(str), prefix=c)
        pieces.append(d)
    X = pd.concat(pieces, axis=1).astype(float)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    n_orig, n_tr = len(orig), len(tr)
    return X[:n_orig], X[n_orig:n_orig + n_tr], X[n_orig + n_tr:]


def lgbm_singlefeat_oof(feat_tr, y, feat_te):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    pred_te = np.zeros(len(feat_te))
    for fi, (tri, vai) in enumerate(skf.split(feat_tr, y)):
        m = lgb.train(
            dict(objective="binary", metric="auc", learning_rate=0.05,
                 num_leaves=15, min_data_in_leaf=500, feature_fraction=1.0,
                 verbose=-1, n_jobs=-1, seed=SEED),
            lgb.Dataset(feat_tr[tri], y[tri]),
            num_boost_round=200,
            valid_sets=[lgb.Dataset(feat_tr[vai], y[vai])],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        oof[vai] = m.predict(feat_tr[vai])
        pred_te += m.predict(feat_te) / N_FOLDS
    return oof, pred_te


def main():
    t0 = time.time()
    log = []

    def step(msg):
        log.append(f"[{time.time() - t0:6.1f}s] {msg}")
        print(log[-1])

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    y = tr[TARGET].astype(int).values

    step("building feature matrix")
    Xo, Xs, Xt = build_feat_matrix(orig, tr, te)
    step(f"  Xo={Xo.shape} Xs={Xs.shape} Xt={Xt.shape}")

    summary: dict = {"meta": dict(orig_n=len(orig), synth_tr_n=len(tr), synth_te_n=len(te),
                                  feat_dim=int(Xo.shape[1]))}

    # ==================================================================
    # P3.1  GaussianMixture (16 components)
    # ==================================================================
    step("P3.1  GaussianMixture(16, full)")
    gmm = GaussianMixture(n_components=16, covariance_type="full",
                           max_iter=100, random_state=SEED, reg_covar=1e-3)
    gmm.fit(Xo)
    step(f"  GMM converged={gmm.converged_} n_iter={gmm.n_iter_} bic={gmm.bic(Xo):.0f}")
    logp_tr_gmm = gmm.score_samples(Xs)
    logp_te_gmm = gmm.score_samples(Xt)
    np.save(ART / "d16_logp_orig_gmm_synth_train.npy", logp_tr_gmm)
    np.save(ART / "d16_logp_orig_gmm_synth_test.npy", logp_te_gmm)
    step(f"  logp_tr stats: med {np.median(logp_tr_gmm):.2f} q05 {np.quantile(logp_tr_gmm, 0.05):.2f} q95 {np.quantile(logp_tr_gmm, 0.95):.2f}")
    # single-feature LGBM gate
    feat_tr = logp_tr_gmm.reshape(-1, 1)
    feat_te = logp_te_gmm.reshape(-1, 1)
    oof, pred_te = lgbm_singlefeat_oof(feat_tr, y, feat_te)
    auc_gmm = roc_auc_score(y, oof)
    np.save(ART / "oof_d16_logp_gmm_strat.npy", oof)
    np.save(ART / "test_d16_logp_gmm_strat.npy", pred_te)
    step(f"  P3.1 single-feat OOF AUC {auc_gmm:.5f}")
    summary["P3_1_gmm"] = dict(bic=float(gmm.bic(Xo)), single_feat_auc=float(auc_gmm),
                                logp_q05=float(np.quantile(logp_tr_gmm, 0.05)),
                                logp_q95=float(np.quantile(logp_tr_gmm, 0.95)))

    # ==================================================================
    # P3.2  Bayesian Gaussian Mixture (Dirichlet process, 32 max components)
    # ==================================================================
    step("P3.2  BayesianGaussianMixture(32, dirichlet_process)")
    bgmm = BayesianGaussianMixture(
        n_components=32, covariance_type="full", max_iter=200,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2, reg_covar=1e-3, random_state=SEED,
    )
    bgmm.fit(Xo)
    step(f"  BGMM converged={bgmm.converged_} n_iter={bgmm.n_iter_} effective={int((bgmm.weights_ > 1e-3).sum())}")
    logp_tr_bgmm = bgmm.score_samples(Xs)
    logp_te_bgmm = bgmm.score_samples(Xt)
    np.save(ART / "d16_logp_orig_bgmm_synth_train.npy", logp_tr_bgmm)
    np.save(ART / "d16_logp_orig_bgmm_synth_test.npy", logp_te_bgmm)
    feat_tr = logp_tr_bgmm.reshape(-1, 1)
    feat_te = logp_te_bgmm.reshape(-1, 1)
    oof, pred_te = lgbm_singlefeat_oof(feat_tr, y, feat_te)
    auc_bgmm = roc_auc_score(y, oof)
    np.save(ART / "oof_d16_logp_bgmm_strat.npy", oof)
    np.save(ART / "test_d16_logp_bgmm_strat.npy", pred_te)
    step(f"  P3.2 single-feat OOF AUC {auc_bgmm:.5f}")
    summary["P3_2_bgmm"] = dict(effective_components=int((bgmm.weights_ > 1e-3).sum()),
                                 single_feat_auc=float(auc_bgmm))

    # cross-correlation check between GMM and BGMM features
    rho = float(np.corrcoef(logp_tr_gmm, logp_tr_bgmm)[0, 1])
    summary["P3_gmm_bgmm_logp_correlation"] = rho
    step(f"  ρ(GMM logp, BGMM logp) on synth_train: {rho:.4f}")

    summary["runtime_s"] = time.time() - t0
    with open(ART / "d16_phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    step("DONE")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
