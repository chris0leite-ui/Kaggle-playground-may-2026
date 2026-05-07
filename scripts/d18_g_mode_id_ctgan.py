"""d18 G — CTGAN mode-id attribution as features.

CTGAN uses Variational Gaussian Mixture (~10 modes) per continuous feature
during mode-specific normalization. The mode-id IS the GAN's discrete
latent for each row.

For each of the 7 KS-low features, fit BayesianGaussianMixture(n=10) on
**orig only** (leakage-clean). For each synth row, assign mode-id per
feature via posterior argmax. 7 new categorical features at up to 10 levels.

Plus aggregate features:
  mode_distinct_count   number of distinct mode-ids the row touches
  mode_loglik_avg       average log-likelihood across feature mixtures

Then 5-fold LGBM on (raw 14 features + 7 mode-ids + 2 aggregates) →
PitNextLap. The 7 mode-ids are passed as categorical_feature= so LGBM
can split on them directly.

Outputs:
  scripts/artifacts/oof_d18_g_mode_id_strat.npy  (n,2)
  scripts/artifacts/test_d18_g_mode_id_strat.npy
  scripts/artifacts/d18_g_mode_id_summary.json
  data/mode_id_features_train.parquet  (gitignored, regenerable)
  data/mode_id_features_test.parquet
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
DATA_OUT = Path("data")
SEED, N_FOLDS = 42, 5
N_MODES = 10
TARGET = "PitNextLap"
LAPTIME = "LapTime (s)"

KS_LOW_FEATS = ["TyreLife", "Position", LAPTIME, "Cumulative_Degradation",
                "RaceProgress", "LapTime_Delta", "LapNumber"]

CAT_OK = ["Compound", "Race"]
NUM_FEATS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             LAPTIME, "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change"]


def fit_vgm_per_feature(orig: pd.DataFrame, feats):
    """Fit a BGMM on each continuous feature using orig values only."""
    models = {}
    for f in feats:
        v = pd.to_numeric(orig[f], errors="coerce").dropna().values.reshape(-1, 1)
        bgmm = BayesianGaussianMixture(
            n_components=N_MODES, covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0, max_iter=200,
            random_state=SEED, reg_covar=1e-3,
        ).fit(v)
        models[f] = bgmm
        # Effective components after BGMM regularisation
        eff = int((bgmm.weights_ > 1e-4).sum())
        print(f"  VGM[{f}]  conv={bgmm.converged_} iter={bgmm.n_iter_}  "
              f"effective_components={eff}/{N_MODES}  weight_max={bgmm.weights_.max():.3f}")
    return models


def assign_modes(models, df: pd.DataFrame, feats) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    n = len(df)
    log_lik_acc = np.zeros(n, dtype=np.float64)
    distinct = np.zeros(n, dtype=np.int8)
    for f in feats:
        v = pd.to_numeric(df[f], errors="coerce").astype(float).values.reshape(-1, 1)
        # NaN handling
        mask = ~np.isnan(v[:, 0])
        ids = np.full(n, -1, dtype=np.int16)
        ll_per_row = np.full(n, np.nan, dtype=np.float64)
        if mask.any():
            v_ok = v[mask]
            ids_ok = models[f].predict(v_ok)
            ids[mask] = ids_ok.astype(np.int16)
            # log-likelihood per row under the fitted mixture
            ll_per_row[mask] = models[f].score_samples(v_ok)
        col_safe = f.replace(" ", "_").replace("(", "").replace(")", "")
        out[f"mode_{col_safe}"] = ids
        log_lik_acc += np.where(np.isnan(ll_per_row), 0.0, ll_per_row)
        distinct += (ids >= 0).astype(np.int8)
    # Aggregate
    out["mode_distinct_count"] = distinct.astype(np.int16)
    out["mode_loglik_avg"] = (log_lik_acc / np.maximum(distinct, 1)).astype(np.float32)
    return out


def main():
    t0 = time.time()
    print("[G mode-id ctgan]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    print(f"  train {tr.shape}  test {te.shape}  orig {orig.shape}")

    print(f"\n[fit VGM per feature on orig]")
    models = fit_vgm_per_feature(orig, KS_LOW_FEATS)

    print(f"\n[assign modes → train, test]")
    tr_M = assign_modes(models, tr, KS_LOW_FEATS)
    te_M = assign_modes(models, te, KS_LOW_FEATS)

    # KS y=0 vs y=1 per mode-id feature
    y = tr[TARGET].astype(int).values
    pos = y == 1; neg = y == 0
    print()
    for c in tr_M.columns:
        if c.startswith("mode_") and c not in ("mode_distinct_count", "mode_loglik_avg"):
            v = tr_M[c].values
            ks, _ = ks_2samp(v[pos], v[neg])
            uniq = np.unique(v[v >= 0])
            print(f"  {c:32s}  KS y=0 vs y=1 = {ks:.4f}  used_modes={len(uniq)}")

    # Per-mode class rates (diagnostic)
    print()
    for c in tr_M.columns:
        if not c.startswith("mode_") or c in ("mode_distinct_count", "mode_loglik_avg"):
            continue
        v = tr_M[c].values
        rates = pd.DataFrame({c: v, "y": y}).groupby(c).agg(
            n=("y", "size"), rate=("y", "mean")
        )
        if len(rates) <= 12:
            print(f"  {c}: per-mode class rates")
            print(rates.to_string())

    # Save diagnostic parquets
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    tr_M.to_parquet(DATA_OUT / "mode_id_features_train.parquet")
    te_M.to_parquet(DATA_OUT / "mode_id_features_test.parquet")

    # Downstream LGBM (raw + mode-id cats + aggregates)
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

    # Convert mode-ids: -1 (NaN sentinel) → max+1 so LGBM treats as own category
    mode_cols = [c for c in tr_M.columns
                 if c.startswith("mode_") and c not in ("mode_distinct_count", "mode_loglik_avg")]
    for c in mode_cols:
        v_tr = tr_M[c].values; v_te = te_M[c].values
        # Map -1 to N_MODES (a unique extra bucket)
        v_tr = np.where(v_tr < 0, N_MODES, v_tr)
        v_te = np.where(v_te < 0, N_MODES, v_te)
        trX[c] = v_tr.astype(int)
        teX[c] = v_te.astype(int)
    trX["mode_distinct_count"] = tr_M["mode_distinct_count"].astype(int).values
    teX["mode_distinct_count"] = te_M["mode_distinct_count"].astype(int).values
    trX["mode_loglik_avg"] = tr_M["mode_loglik_avg"].values
    teX["mode_loglik_avg"] = te_M["mode_loglik_avg"].values

    print(f"\n[downstream LGBM raw + 7 mode-ids (cat) + 2 aggregates]")
    cat_idx = [trX.columns.get_loc(c) for c in ["Compound", "Race"] + mode_cols]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y)); test_avg = np.zeros(len(te))
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=127, min_data_in_leaf=80, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbosity=-1, seed=SEED)
    for fi, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        ds_tr = lgb.Dataset(trX.iloc[tr_i], label=y[tr_i],
                            categorical_feature=cat_idx, free_raw_data=False)
        ds_va = lgb.Dataset(trX.iloc[va_i], label=y[va_i],
                            categorical_feature=cat_idx, reference=ds_tr,
                            free_raw_data=False)
        m = lgb.train(p, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_i] = m.predict(trX.iloc[va_i], num_iteration=m.best_iteration)
        test_avg += m.predict(teX, num_iteration=m.best_iteration) / N_FOLDS
        print(f"  fold {fi}: AUC={roc_auc_score(y[va_i], oof[va_i]):.5f}  "
              f"best_iter={m.best_iteration}")

    auc = float(roc_auc_score(y, oof))
    print(f"\n  OOF AUC = {auc:.5f}")

    np.save(ART / "oof_d18_g_mode_id_strat.npy", np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d18_g_mode_id_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))
    summary = dict(oof_auc=auc, ks_low_feats=KS_LOW_FEATS, n_modes=N_MODES,
                   wall_s=time.time() - t0)
    (ART / "d18_g_mode_id_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done G]  wall {time.time()-t0:.0f}s  OOF {auc:.5f}")


if __name__ == "__main__":
    main()
