"""scripts/probe_fe_combo.py — combined FE probe (wide-TE + quantile bins + test rank).

Synthetic-data philosophy: brute-force statistical FE on existing
columns; don't rely on physical meaning. Build 3 candidate bases:

  1. wide_te_lgbm  — LightGBM with multi-way TE features (3-way + 4-way
                     interactions: Driver_Compound_Stint, Driver_Year_Compound,
                     Race_LapBin_Stint, Driver_Race_Compound).
  2. qbin_fm_lgbm  — LightGBM with quantile-bin features for every
                     numeric (5/10/20 quantile of TyreLife, RaceProgress,
                     LapTime_Delta, Cumulative_Degradation, Position).
  3. test_rank_lgbm — LightGBM with test-side rank features (rank
                     within test only) joined as a column.

Each gets standalone OOF + min-meta gate via probe_min_meta.py
afterward.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def smoothed_te(train_y, train_key, apply_key, alpha=80):
    gm = float(train_y.mean())
    df = pd.DataFrame({"k": train_key, "y": train_y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    g["te"] = (g["sum"] + alpha * gm) / (g["count"] + alpha)
    return pd.Series(apply_key).map(g["te"]).fillna(gm).to_numpy()


def oof_te_train(y, key, alpha=80, n_inner=5, seed=42):
    out = np.zeros(len(y), dtype=np.float32)
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for tr, va in kf.split(np.zeros(len(y))):
        out[va] = smoothed_te(y[tr], key[tr], key[va], alpha)
    return out


def lgbm_5fold(X, y, X_test, cat_cols, name="probe"):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    fold_aucs = []
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  min_data_in_leaf=200, verbose=-1, seed=SEED)
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        test_pred += m.predict(X_test) / N_FOLDS
        s = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(s)
        print(f"  [{name}] fold {k}: AUC {s:.5f} wall {time.time()-t:.1f}s")
    return oof, test_pred, float(roc_auc_score(y, oof)), fold_aucs


def add_lapbin(df, edges=None, n_q=10):
    if edges is None:
        edges = np.quantile(df["RaceProgress"].values, np.linspace(0, 1, n_q + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
    bins = np.clip(np.digitize(df["RaceProgress"].values, edges, right=True) - 1,
                   0, n_q - 1).astype(np.int32)
    return bins, edges


def main():
    t_total = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    auc_primary = float(roc_auc_score(y, primary_oof))

    base_num = ["TyreLife", "RaceProgress", "LapTime_Delta",
                "Cumulative_Degradation", "Position", "LapTime (s)",
                "Stint", "Year", "Position_Change"]
    base_cat = ["Driver", "Compound", "Race"]

    # LapBin (target-independent)
    train["LapBin"], edges = add_lapbin(train)
    test["LapBin"], _ = add_lapbin(test, edges=edges)

    summary = {}

    # --------------------------------------------------------------
    # Probe A: wide-TE (3-way + 4-way) → LightGBM
    # --------------------------------------------------------------
    print("\n=== Probe A: wide-TE LightGBM ===")
    te_keys_3 = [
        ("Driver_Compound_Stint",
         lambda d: d["Driver"].astype(str) + "|" + d["Compound"].astype(str) +
                   "|" + d["Stint"].astype(str)),
        ("Driver_Year_Compound",
         lambda d: d["Driver"].astype(str) + "|" + d["Year"].astype(str) +
                   "|" + d["Compound"].astype(str)),
        ("Race_LapBin_Stint",
         lambda d: d["Race"].astype(str) + "|" + d["LapBin"].astype(str) +
                   "|" + d["Stint"].astype(str)),
        ("Driver_Race_Compound",
         lambda d: d["Driver"].astype(str) + "|" + d["Race"].astype(str) +
                   "|" + d["Compound"].astype(str)),
    ]
    Xa = train[base_num + base_cat].copy()
    Xa_test = test[base_num + base_cat].copy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    # Per-fold TE
    for nm, fn in te_keys_3:
        train_k = fn(train).values
        test_k = fn(test).values
        # Per-fold inner-OOF for outer-train; per-fold test
        col_oof = np.zeros(len(y))
        col_test = np.zeros(len(test))
        for tr, va in splits:
            col_oof[tr] += oof_te_train(y[tr], train_k[tr]) / (N_FOLDS - 1)
            col_oof[va] = smoothed_te(y[tr], train_k[tr], train_k[va])
            col_test += smoothed_te(y[tr], train_k[tr], test_k) / N_FOLDS
        Xa[f"te_{nm}"] = col_oof
        Xa_test[f"te_{nm}"] = col_test
    for c in base_cat:
        Xa[c] = Xa[c].astype("category")
        Xa_test[c] = Xa_test[c].astype("category")
    oof_a, test_a, auc_a, _ = lgbm_5fold(Xa, y, Xa_test, base_cat, "wide_te")
    np.save(ART / "oof_wide_te_lgbm_strat.npy", np.column_stack([1-oof_a, oof_a]))
    np.save(ART / "test_wide_te_lgbm_strat.npy", np.column_stack([1-test_a, test_a]))
    summary["wide_te_lgbm"] = dict(std_oof=auc_a,
                                    delta_vs_primary_bp=(auc_a - auc_primary)*1e4)
    print(f"  std OOF {auc_a:.5f}  Δ vs PRIMARY {(auc_a-auc_primary)*1e4:+.2f} bp")

    # --------------------------------------------------------------
    # Probe B: quantile-bin features → LightGBM
    # --------------------------------------------------------------
    print("\n=== Probe B: quantile-bin LightGBM ===")
    Xb = train[base_num + base_cat].copy()
    Xb_test = test[base_num + base_cat].copy()
    qcols = ["TyreLife", "RaceProgress", "LapTime_Delta",
             "Cumulative_Degradation", "Position"]
    for q_n in [5, 10, 20]:
        for c in qcols:
            edges_c = np.quantile(train[c].values, np.linspace(0, 1, q_n + 1))
            edges_c[0] -= 1e-9; edges_c[-1] += 1e-9
            Xb[f"qbin_{c}_q{q_n}"] = np.clip(
                np.digitize(train[c].values, edges_c, right=True) - 1,
                0, q_n - 1).astype(np.int32)
            Xb_test[f"qbin_{c}_q{q_n}"] = np.clip(
                np.digitize(test[c].values, edges_c, right=True) - 1,
                0, q_n - 1).astype(np.int32)
    for c in base_cat:
        Xb[c] = Xb[c].astype("category")
        Xb_test[c] = Xb_test[c].astype("category")
    oof_b, test_b, auc_b, _ = lgbm_5fold(Xb, y, Xb_test, base_cat, "qbin")
    np.save(ART / "oof_qbin_lgbm_strat.npy", np.column_stack([1-oof_b, oof_b]))
    np.save(ART / "test_qbin_lgbm_strat.npy", np.column_stack([1-test_b, test_b]))
    summary["qbin_lgbm"] = dict(std_oof=auc_b,
                                 delta_vs_primary_bp=(auc_b - auc_primary)*1e4)
    print(f"  std OOF {auc_b:.5f}  Δ vs PRIMARY {(auc_b-auc_primary)*1e4:+.2f} bp")

    # --------------------------------------------------------------
    # Probe C: test-side rank features → LightGBM (synth lens)
    # --------------------------------------------------------------
    print("\n=== Probe C: test-side rank features LightGBM ===")
    Xc = train[base_num + base_cat].copy()
    Xc_test = test[base_num + base_cat].copy()
    # Test-side ranks: rank computed on TEST only, mapped back
    for c in qcols:
        # train: use rank within train (separate from test signal)
        Xc[f"rank_self_{c}"] = (rankdata(train[c].values) / len(train)).astype(np.float32)
        Xc_test[f"rank_self_{c}"] = (rankdata(test[c].values) / len(test)).astype(np.float32)
        # combined rank using train+test pool — gives test-side info
        comb = np.concatenate([train[c].values, test[c].values])
        comb_rank = rankdata(comb) / len(comb)
        Xc[f"rank_combined_{c}"] = comb_rank[:len(train)].astype(np.float32)
        Xc_test[f"rank_combined_{c}"] = comb_rank[len(train):].astype(np.float32)
    for c in base_cat:
        Xc[c] = Xc[c].astype("category")
        Xc_test[c] = Xc_test[c].astype("category")
    oof_c, test_c, auc_c, _ = lgbm_5fold(Xc, y, Xc_test, base_cat, "test_rank")
    np.save(ART / "oof_test_rank_lgbm_strat.npy",
            np.column_stack([1-oof_c, oof_c]))
    np.save(ART / "test_test_rank_lgbm_strat.npy",
            np.column_stack([1-test_c, test_c]))
    summary["test_rank_lgbm"] = dict(std_oof=auc_c,
                                      delta_vs_primary_bp=(auc_c - auc_primary)*1e4)
    print(f"  std OOF {auc_c:.5f}  Δ vs PRIMARY {(auc_c-auc_primary)*1e4:+.2f} bp")

    summary["wall_s"] = time.time() - t_total
    out = ART / "probe_fe_combo.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out}  (wall {time.time()-t_total:.0f}s)")


if __name__ == "__main__":
    main()
