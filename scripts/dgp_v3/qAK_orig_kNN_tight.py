"""qAK — orig kNN with TIGHT K (K=3) at the 6-axis cell, hierarchical fallback.

qAB used K=20 within (Y, C, PS) cell — too smooth, captures per-cell
aggregate. Tighter K=3 within (Y, C, PS, R, S, LapN) captures per-row
identity (the orig source-row analog).

Hypothesis: at the meta layer, K=3 votes carry the per-row VARIANCE
that K=20 averages out. Different logit direction.

Procedure: For each train+test row, find K=3 nearest orig rows that
share its full 6-axis cell. If <3 orig rows in that cell, fall back to
5-axis cell, then 4, then 3. Compute distance-weighted PitNextLap mean
+ std + min + max as 4 features.
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


def t(label, ts):
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def cell_knn_with_fallback(orig, df, cont_cols, cell_levels, k=3):
    """For each row in df, find K NN orig rows within the most specific
    cell key that has at least K orig rows. Fallback through levels.
    """
    sc = StandardScaler().fit(orig[cont_cols].values)
    Xo = sc.transform(orig[cont_cols].values)
    Xq = sc.transform(df[cont_cols].values)

    out_mean = np.full(len(df), np.nan, dtype=np.float32)
    out_std = np.full(len(df), np.nan, dtype=np.float32)
    out_max = np.full(len(df), np.nan, dtype=np.float32)
    out_min = np.full(len(df), np.nan, dtype=np.float32)
    out_d_med = np.full(len(df), np.nan, dtype=np.float32)
    out_level_used = np.full(len(df), -1, dtype=np.int32)

    yo_all = orig[TARGET].values

    for level_idx, (level_name, keys) in enumerate(cell_levels):
        unfilled = np.isnan(out_mean)
        n_unfilled = unfilled.sum()
        if n_unfilled == 0:
            break
        print(f"    {level_name}: {n_unfilled} rows still unassigned", flush=True)

        # Group orig and query df by keys
        orig_grp = orig.groupby(keys, observed=True).indices
        df_local = df.loc[unfilled, keys + ([] if False else [])]
        df_local["_qidx"] = np.where(unfilled)[0]
        # Inner: per cell, do kNN
        for cell, sub_df in df_local.groupby(keys, observed=True):
            if cell not in orig_grp:
                continue
            o_idx = orig_grp[cell]
            if len(o_idx) < k:
                continue
            q_idx = sub_df["_qidx"].values
            Xo_c = Xo[o_idx]
            Xq_c = Xq[q_idx]
            yo_c = yo_all[o_idx]
            kk = min(k, len(o_idx))
            nn = NearestNeighbors(n_neighbors=kk, n_jobs=1).fit(Xo_c)
            d, ii = nn.kneighbors(Xq_c)
            w = 1.0 / (d + 1e-3)
            wn = w / w.sum(axis=1, keepdims=True)
            yvals = yo_c[ii]
            out_mean[q_idx] = (yvals * wn).sum(axis=1).astype(np.float32)
            out_std[q_idx] = yvals.std(axis=1).astype(np.float32)
            out_max[q_idx] = yvals.max(axis=1).astype(np.float32)
            out_min[q_idx] = yvals.min(axis=1).astype(np.float32)
            out_d_med[q_idx] = np.median(d, axis=1).astype(np.float32)
            out_level_used[q_idx] = level_idx

    # Fill remainder with global rate
    global_rate = float(orig[TARGET].mean())
    rem = np.isnan(out_mean)
    out_mean[rem] = global_rate
    out_std[rem] = 0.0
    out_max[rem] = global_rate
    out_min[rem] = global_rate
    out_d_med[rem] = 0.0
    out_level_used[rem] = len(cell_levels)

    return out_mean, out_std, out_max, out_min, out_d_med, out_level_used


def main():
    ts = time.time()
    out = {}
    train = pd.read_csv(DATA / "train.csv").rename(columns={"LapTime (s)": "LapTime"})
    test = pd.read_csv(DATA / "test.csv").rename(columns={"LapTime (s)": "LapTime"})
    orig = pd.read_csv(DATA / "original/f1_strategy_dataset_v4.csv").rename(
        columns={"LapTime (s)": "LapTime"}).drop(columns=["Normalized_TyreLife"]).dropna().reset_index(drop=True)
    t(f"orig {orig.shape} train {train.shape} test {test.shape}", ts)

    print("\nBuilding kNN features for train (K=3, hierarchical fallback)...")
    tr_mean, tr_std, tr_max, tr_min, tr_d, tr_lvl = cell_knn_with_fallback(
        orig, train, CONT_COLS, CELL_LEVELS, k=3)
    t("train kNN done", ts)

    print("\nBuilding kNN features for test...")
    te_mean, te_std, te_max, te_min, te_d, te_lvl = cell_knn_with_fallback(
        orig, test, CONT_COLS, CELL_LEVELS, k=3)
    t("test kNN done", ts)

    print(f"  level usage train: {pd.Series(tr_lvl).value_counts().sort_index().to_dict()}")
    print(f"  level usage test:  {pd.Series(te_lvl).value_counts().sort_index().to_dict()}")

    # Build LightGBM with these 6 features alone
    feat_cols = ["knn3_mean", "knn3_std", "knn3_max", "knn3_min", "knn3_d_med", "knn3_level"]
    X = np.column_stack([tr_mean, tr_std, tr_max, tr_min, tr_d, tr_lvl]).astype(np.float32)
    y = train[TARGET].values
    X_test = np.column_stack([te_mean, te_std, te_max, te_min, te_d, te_lvl]).astype(np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    lgb_params = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
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
    print(f"\n  qAK standalone OOF = {auc:.5f}", flush=True)
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
    print(f"\n  K=4+1 lift: {delta:+.3f} bp", flush=True)
    out["k4plus1_lift_bp"] = delta

    np.save(ART / "dgp_v3_qAK_knn3_oof.npy", oof)
    np.save(ART / "dgp_v3_qAK_knn3_test.npy", test_pred)
    fp = ART / "dgp_v3_qAK_knn3.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)


if __name__ == "__main__":
    main()
