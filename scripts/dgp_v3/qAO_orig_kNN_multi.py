"""qAO — multi-K orig kNN (K=3, K=5, K=10) hierarchical fallback.

qAK with K=3 alone gave K=4+1 +0.717 bp PASS. Test whether multi-K
adds more signal at different scales:
  K=3:  per-row identity (high variance)
  K=5:  small-neighbourhood smoothing
  K=10: per-cell aggregate (smooth)

Each K-value contributes mean/std/max/min/d_med = 5 features.
Plus the level_used indicator. Total: 3*5 + 1 = 16 features.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"

CELL_LEVELS = [
    ("L6", ["Year", "Compound", "PitStop", "Race", "Stint", "LapNumber"]),
    ("L5", ["Year", "Compound", "PitStop", "Race", "Stint"]),
    ("L4", ["Year", "Compound", "PitStop", "Race"]),
    ("L3", ["Year", "Compound", "PitStop"]),
]
CONT_COLS = ["LapTime", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]
K_VALUES = [3, 5, 10]


def t(label, ts):
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def cell_knn_multi_k(orig, df, cont_cols, cell_levels, k_values):
    """For each row, find max(K) NN orig rows in the most specific cell
    that has ≥max(K) orig rows. Compute features per K-value.
    Returns: dict {f"k{k}_{stat}": np.array} for stat in mean/std/max/min/d_med
             plus "level" array.
    """
    sc = StandardScaler().fit(orig[cont_cols].values)
    Xo = sc.transform(orig[cont_cols].values)
    Xq = sc.transform(df[cont_cols].values)
    yo_all = orig[TARGET].values

    max_k = max(k_values)
    out_d_full = np.full((len(df), max_k), np.nan, dtype=np.float32)
    out_y_full = np.full((len(df), max_k), np.nan, dtype=np.float32)
    out_level = np.full(len(df), -1, dtype=np.int32)

    for level_idx, (level_name, keys) in enumerate(cell_levels):
        unfilled = np.isnan(out_d_full[:, 0])
        n_unfilled = unfilled.sum()
        if n_unfilled == 0:
            break
        print(f"    {level_name}: {n_unfilled} unfilled", flush=True)

        orig_grp = orig.groupby(keys, observed=True).indices
        df_local = df.loc[unfilled, keys].copy()
        df_local["_qidx"] = np.where(unfilled)[0]

        for cell, sub_df in df_local.groupby(keys, observed=True):
            if cell not in orig_grp:
                continue
            o_idx = orig_grp[cell]
            if len(o_idx) < max_k:
                continue
            q_idx = sub_df["_qidx"].values
            Xo_c = Xo[o_idx]
            Xq_c = Xq[q_idx]
            yo_c = yo_all[o_idx]
            nn = NearestNeighbors(n_neighbors=max_k, n_jobs=1).fit(Xo_c)
            d, ii = nn.kneighbors(Xq_c)
            out_d_full[q_idx] = d.astype(np.float32)
            out_y_full[q_idx] = yo_c[ii].astype(np.float32)
            out_level[q_idx] = level_idx

    # Build features per K
    features = {}
    global_rate = float(orig[TARGET].mean())
    for k in k_values:
        d_k = out_d_full[:, :k]
        y_k = out_y_full[:, :k]
        valid = ~np.isnan(d_k[:, 0])
        # Distance-weighted mean
        w = np.where(valid[:, None], 1.0 / (d_k + 1e-3), 0.0)
        wsum = w.sum(axis=1)
        wsum_safe = np.where(wsum > 0, wsum, 1.0)
        mean = np.where(valid, (y_k * w).sum(axis=1) / wsum_safe, global_rate)
        std = np.where(valid, np.nanstd(y_k, axis=1), 0.0)
        mx = np.where(valid, np.nanmax(y_k, axis=1), global_rate)
        mn = np.where(valid, np.nanmin(y_k, axis=1), global_rate)
        d_med = np.where(valid, np.nanmedian(d_k, axis=1), 0.0)
        features[f"k{k}_mean"] = mean.astype(np.float32)
        features[f"k{k}_std"] = std.astype(np.float32)
        features[f"k{k}_max"] = mx.astype(np.float32)
        features[f"k{k}_min"] = mn.astype(np.float32)
        features[f"k{k}_d_med"] = d_med.astype(np.float32)
    features["level"] = out_level.astype(np.float32)
    return features


def main():
    ts = time.time()
    out = {}
    train = pd.read_csv(DATA / "train.csv").rename(columns={"LapTime (s)": "LapTime"})
    test = pd.read_csv(DATA / "test.csv").rename(columns={"LapTime (s)": "LapTime"})
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv").rename(
        columns={"LapTime (s)": "LapTime"}).drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)
    t(f"orig {orig.shape} train {train.shape} test {test.shape}", ts)

    print("\nBuilding multi-K kNN features for train...")
    feats_tr = cell_knn_multi_k(orig, train, CONT_COLS, CELL_LEVELS, K_VALUES)
    t("train multi-K kNN done", ts)

    print("\nBuilding multi-K kNN features for test...")
    feats_te = cell_knn_multi_k(orig, test, CONT_COLS, CELL_LEVELS, K_VALUES)
    t("test multi-K kNN done", ts)

    feat_names = sorted(feats_tr.keys())
    X = np.column_stack([feats_tr[c] for c in feat_names]).astype(np.float32)
    X_test = np.column_stack([feats_te[c] for c in feat_names]).astype(np.float32)
    y = train[TARGET].values
    print(f"  total features: {X.shape[1]} ({feat_names})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    lgb_params = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=80, reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        a = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(a)
        print(f"  fold {fold+1} AUC = {a:.5f}  best_iter={m.best_iteration_}", flush=True)

    auc = float(roc_auc_score(y, oof))
    print(f"\n  qAO standalone OOF = {auc:.5f}", flush=True)
    print(f"  (qAK was 0.87320 — delta {(auc-0.87320)*1e4:+.2f} bp)", flush=True)
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

    np.save(ART / "dgp_v3_qAO_knn_multi_oof.npy", oof)
    np.save(ART / "dgp_v3_qAO_knn_multi_test.npy", test_pred)
    fp = ART / "dgp_v3_qAO_knn_multi.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
