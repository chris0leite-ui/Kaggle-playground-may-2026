"""H4 — cumdeg_per_lap base + min-meta gate.

Phase C found Cumulative_Degradation has near-zero correlation with TyreLife
within each Compound (HARD ρ=-0.08, INTER ρ=-0.26).  They encode independent
signals despite naming overlap.  H4: explicit ratio `cumdeg_per_lap = Cum_Deg /
max(TyreLife, 1)` should add new info to all bases.

Three bases trained:
  H4a — LGBM with cumdeg_per_lap added (raw features intact)
  H4b — LGBM with cumdeg_per_lap + Compound × cumdeg_per_lap interaction
        target encoding (TE-style smoothed mean)
  H4c — sparse-LR (interaction-free) on cumdeg_per_lap quintiles

Min-meta gate: K=23.
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED = 42

K22 = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B", "d9h_FM_aug12",
]


def expand(M: np.ndarray) -> np.ndarray:
    n = len(M)
    rk = np.column_stack([rankdata(c) / n for c in M.T])
    logit = np.log(np.clip(M, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(M, 1e-9, 1 - 1e-9)))
    return np.hstack([M, rk, logit])


def gate_min_meta(label, primary_oof, primary_test, base_oof, base_test, y):
    Z = np.column_stack([primary_oof, base_oof])
    Zt = np.column_stack([primary_test, base_test])
    F_oof = expand(Z); F_te = expand(Zt)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    moo = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(F_oof[tr], y[tr])
        moo[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = roc_auc_score(y, moo)
    rho = spearmanr(base_oof, primary_oof).correlation
    return auc, rho, moo


def gate_K23(label, P_oof, P_test, base_oof, base_test, y):
    Z = np.column_stack([P_oof, base_oof])
    Zt = np.column_stack([P_test, base_test])
    F_oof = expand(Z); F_te = expand(Zt)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    moo = np.zeros(len(y))
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(F_oof[tr], y[tr])
        moo[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000)
    lr_full.fit(F_oof, y)
    test = lr_full.predict_proba(F_te)[:, 1]
    return roc_auc_score(y, moo), moo, test


def train_lgbm(X_tr, X_te, y, cats, num_boost=600):
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=4, seed=SEED, verbose=-1)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test = np.zeros(len(X_te), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        dtr = lgb.Dataset(X_tr.iloc[tr], y[tr], categorical_feature=cats)
        dva = lgb.Dataset(X_tr.iloc[va], y[va], categorical_feature=cats)
        m = lgb.train(params, dtr, num_boost_round=num_boost,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        oof[va] = m.predict(X_tr.iloc[va])
        test += m.predict(X_te) / 5
    return oof, test


def main() -> None:
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).to_numpy()

    # --- Build the new feature ---
    for df in (train, test):
        df["cumdeg_per_lap"] = (df["Cumulative_Degradation"]
                                  / df["TyreLife"].clip(lower=1.0))
        # quintile bins (frozen on train) for sparse-LR
    qs = train["cumdeg_per_lap"].quantile([0.2, 0.4, 0.6, 0.8]).values
    train["cumdeg_per_lap_q5"] = np.digitize(train["cumdeg_per_lap"], qs)
    test["cumdeg_per_lap_q5"] = np.digitize(test["cumdeg_per_lap"], qs)

    print("cumdeg_per_lap stats:")
    print(train["cumdeg_per_lap"].describe())
    print("\nMean PitNextLap by Compound × cumdeg_per_lap_q5:")
    print((train.groupby(["Compound", "cumdeg_per_lap_q5"])["PitNextLap"]
                 .agg(["count", "mean"])).round(3))

    base_feats = [
        "LapNumber", "Stint", "TyreLife", "LapTime (s)", "LapTime_Delta",
        "Cumulative_Degradation", "RaceProgress", "Position",
        "Position_Change", "PitStop", "Year",
    ]
    cats = ["Driver", "Compound", "Race"]

    # --- H4a: LGBM + cumdeg_per_lap ---
    feats_a = base_feats + ["cumdeg_per_lap"] + cats
    X_tr_a = train[feats_a].copy()
    X_te_a = test[feats_a].copy()
    for c in cats:
        X_tr_a[c] = X_tr_a[c].astype("category")
        X_te_a[c] = X_te_a[c].astype("category")
    print("\nTraining H4a...")
    oof_a, test_a = train_lgbm(X_tr_a, X_te_a, y, cats)
    print(f"  H4a OOF AUC = {roc_auc_score(y, oof_a):.5f}")
    np.save(ART / "oof_H4a_cumdeg_strat.npy", oof_a.astype(np.float32))
    np.save(ART / "test_H4a_cumdeg_strat.npy", test_a.astype(np.float32))

    # --- H4c: sparse-LR on hashed (Compound × cumdeg_per_lap_q5 × Stint × Year) ---
    print("\nBuilding sparse-LR feature hash...")
    def hash_feats(df):
        rows = []
        for r in df.itertuples():
            rows.append({
                f"compXq5={r.Compound}|{r.cumdeg_per_lap_q5}": 1,
                f"stintXq5={r.Stint}|{r.cumdeg_per_lap_q5}": 1,
                f"yearXq5={r.Year}|{r.cumdeg_per_lap_q5}": 1,
                f"compXstintXq5={r.Compound}|{r.Stint}|{r.cumdeg_per_lap_q5}": 1,
                f"q5_only={r.cumdeg_per_lap_q5}": 1,
                f"comp={r.Compound}": 1,
                f"stint={r.Stint}": 1,
            })
        return rows
    fh = FeatureHasher(n_features=2 ** 16, input_type="dict")
    X_tr_c = fh.transform(hash_feats(train[["Compound", "Stint", "Year",
                                              "cumdeg_per_lap_q5"]]))
    X_te_c = fh.transform(hash_feats(test[["Compound", "Stint", "Year",
                                             "cumdeg_per_lap_q5"]]))

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof_c = np.zeros(len(y))
    test_c = np.zeros(X_te_c.shape[0])
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=300, solver="liblinear")
        lr.fit(X_tr_c[tr], y[tr])
        oof_c[va] = lr.predict_proba(X_tr_c[va])[:, 1]
        test_c += lr.predict_proba(X_te_c)[:, 1] / 5
    print(f"  H4c OOF AUC = {roc_auc_score(y, oof_c):.5f}")
    np.save(ART / "oof_H4c_cumdegLR_strat.npy", oof_c.astype(np.float32))
    np.save(ART / "test_H4c_cumdegLR_strat.npy", test_c.astype(np.float32))

    # --- min-meta gates ---
    primary_oof = np.load(ART / "oof_PRIMARY_K22_strat.npy").astype(np.float64)
    primary_test = np.load(ART / "test_PRIMARY_K22_strat.npy").astype(np.float64)
    primary_auc = roc_auc_score(y, primary_oof)

    P_oof = []; P_te = []
    for fn in K22:
        oo = np.load(ART / f"oof_{fn}_strat.npy")
        if oo.ndim == 2: oo = oo[:, 1]
        P_oof.append(oo)
        tt = np.load(ART / f"test_{fn}_strat.npy")
        if tt.ndim == 2: tt = tt[:, 1]
        P_te.append(tt)
    P_oof = np.column_stack(P_oof).astype(np.float64)
    P_te = np.column_stack(P_te).astype(np.float64)

    for label, oo, tt in [("H4a_cumdeg_GBDT", oof_a, test_a),
                           ("H4c_cumdegLR_sparse", oof_c, test_c)]:
        mm_auc, rho, _ = gate_min_meta(label, primary_oof, primary_test,
                                        oo, tt, y)
        delta = (mm_auc - primary_auc) * 1e4
        print(f"\n--- min-meta gate {label} ---")
        print(f"  base OOF AUC     = {roc_auc_score(y, oo):.5f}")
        print(f"  ρ vs PRIMARY     = {rho:.5f}")
        print(f"  min-meta OOF AUC = {mm_auc:.5f}  Δ = {delta:+.2f} bp"
              f"  ({'PASS' if delta >= 0.5 else 'FAIL'})")

        k23_auc, k23_oof, k23_test = gate_K23(label, P_oof, P_te, oo, tt, y)
        d23 = (k23_auc - primary_auc) * 1e4
        print(f"  K=23 add OOF AUC = {k23_auc:.5f}  Δ = {d23:+.2f} bp")
        np.save(ART / f"oof_K23_add_{label}_strat.npy",
                k23_oof.astype(np.float32))
        np.save(ART / f"test_K23_add_{label}_strat.npy",
                k23_test.astype(np.float32))

    print(f"\nwall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
