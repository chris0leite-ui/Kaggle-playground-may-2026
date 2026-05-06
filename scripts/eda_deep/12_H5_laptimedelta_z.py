"""H5 — LapTime_Delta race-z-score base + min-meta gate.

Phase F single-feature LR proved LapTime_Delta has +922 bp Strat→GroupKF gap
(highest-leakage feature in the set).  Hypothesis: replacing raw LapTime_Delta
with `(x - μ_g) / σ_g` per group g=(Race, Year, Compound) drains the leakage,
and a base trained on the rescored feature transfers better to held-out groups.

Two bases trained:
  H5_lgbm_zr — LGBM with raw features but LapTime_Delta replaced by z-score
  H5_lgbm_zr_extra — same + retain raw LapTime_Delta + add z-version (parity test)

Min-meta gate: K=23 = K=22 + new base.  Strat OOF target: ≥ PRIMARY + 0.5 bp.
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED = 42


def add_zscore(train: pd.DataFrame, test: pd.DataFrame,
               group_cols: list[str], target_col: str,
               new_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Race-Year-Compound z-score, fit on train and applied to both."""
    grp = train.groupby(group_cols)[target_col]
    mu = grp.transform("mean")
    sd = grp.transform("std").clip(lower=1e-6)
    train[new_col] = ((train[target_col] - mu) / sd).fillna(0.0)
    # for test, look up train stats
    stats = (train.groupby(group_cols)[target_col]
                  .agg(["mean", "std"])
                  .reset_index())
    stats.columns = group_cols + ["__mu", "__sd"]
    stats["__sd"] = stats["__sd"].clip(lower=1e-6)
    merged = test.merge(stats, on=group_cols, how="left")
    merged["__mu"] = merged["__mu"].fillna(train[target_col].mean())
    merged["__sd"] = merged["__sd"].fillna(train[target_col].std() + 1e-6)
    test[new_col] = ((merged[target_col].values - merged["__mu"].values)
                     / merged["__sd"].values)
    test[new_col] = np.nan_to_num(test[new_col], nan=0.0)
    return train, test


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expand(M: np.ndarray) -> np.ndarray:
    n = len(M)
    rk = np.column_stack([rankdata(c) / n for c in M.T])
    logit = np.log(np.clip(M, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(M, 1e-9, 1 - 1e-9)))
    return np.hstack([M, rk, logit])


def train_lgbm(X_tr, X_te, y, cats, params=None, num_boost=600):
    """5-fold OOF + test predictions."""
    if params is None:
        params = dict(objective="binary", metric="auc", learning_rate=0.05,
                      num_leaves=63, min_data_in_leaf=200,
                      feature_fraction=0.9, bagging_fraction=0.9,
                      bagging_freq=4, seed=SEED, verbose=-1)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test = np.zeros(len(X_te), dtype=np.float64)
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        dtr = lgb.Dataset(X_tr.iloc[tr], y[tr], categorical_feature=cats)
        dva = lgb.Dataset(X_tr.iloc[va], y[va], categorical_feature=cats)
        model = lgb.train(params, dtr, num_boost_round=num_boost,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        oof[va] = model.predict(X_tr.iloc[va])
        test += model.predict(X_te) / 5
    return oof, test


def main() -> None:
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).to_numpy()

    # Build z-score features for the 2 highest-leakage features
    train, test = add_zscore(train.copy(), test.copy(),
                             ["Race", "Year", "Compound"],
                             "LapTime_Delta", "LapTime_Delta_zr")
    train, test = add_zscore(train, test,
                             ["Race", "Year", "Compound"],
                             "Position", "Position_zr")
    train, test = add_zscore(train, test,
                             ["Race", "Year", "Compound"],
                             "LapTime (s)", "LapTime_zr")

    print("z-score features built. Sample stats:")
    print(train[["LapTime_Delta", "LapTime_Delta_zr",
                 "Position", "Position_zr"]].describe())

    base_feats = [
        "LapNumber", "Stint", "TyreLife", "LapTime (s)",
        "Cumulative_Degradation", "RaceProgress",
        "Position_Change", "PitStop", "Year",
    ]
    cats = ["Driver", "Compound", "Race"]

    # H5a: replace raw LapTime_Delta + Position with z-score versions
    feats_a = base_feats + ["LapTime_Delta_zr", "Position_zr"] + cats
    X_tr_a = train[feats_a].copy()
    X_te_a = test[feats_a].copy()
    for c in cats:
        X_tr_a[c] = X_tr_a[c].astype("category")
        X_te_a[c] = X_te_a[c].astype("category")

    print("\nTraining H5a (z-score replacement)...")
    oof_a, test_a = train_lgbm(X_tr_a, X_te_a, y, cats)
    auc_a = roc_auc_score(y, oof_a)
    print(f"  H5a OOF AUC = {auc_a:.5f}")

    # H5b: keep raw + add z (parity)
    feats_b = base_feats + ["LapTime_Delta", "Position",
                            "LapTime_Delta_zr", "Position_zr",
                            "LapTime_zr"] + cats
    X_tr_b = train[feats_b].copy()
    X_te_b = test[feats_b].copy()
    for c in cats:
        X_tr_b[c] = X_tr_b[c].astype("category")
        X_te_b[c] = X_te_b[c].astype("category")

    print("\nTraining H5b (raw + z parity)...")
    oof_b, test_b = train_lgbm(X_tr_b, X_te_b, y, cats)
    auc_b = roc_auc_score(y, oof_b)
    print(f"  H5b OOF AUC = {auc_b:.5f}")

    # save
    np.save(ART / "oof_H5a_zreplace_strat.npy", oof_a.astype(np.float32))
    np.save(ART / "test_H5a_zreplace_strat.npy", test_a.astype(np.float32))
    np.save(ART / "oof_H5b_zaugment_strat.npy", oof_b.astype(np.float32))
    np.save(ART / "test_H5b_zaugment_strat.npy", test_b.astype(np.float32))

    # ----- min-meta gate -----
    primary_oof = np.load(ART / "oof_PRIMARY_K22_strat.npy").astype(np.float64)
    primary_test = np.load(ART / "test_PRIMARY_K22_strat.npy").astype(np.float64)
    primary_auc = roc_auc_score(y, primary_oof)

    # K=23 add: PRIMARY + new base.  Use full K=22 expand + new base expand.
    from scipy.stats import spearmanr

    def gate(label, base_oof, base_test):
        from scripts.eda_deep_path import K22_BASES  # noqa
    # reuse K22 bases inline:
    K22 = [
        "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
        "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
        "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
        "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
        "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
        "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B", "d9h_FM_aug12",
    ]
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

    for label, oof_x, test_x in [("H5a_zreplace", oof_a, test_a),
                                  ("H5b_zaugment", oof_b, test_b)]:
        # Min-meta with PRIMARY + base
        Z_oof = np.column_stack([primary_oof, oof_x])
        Z_te = np.column_stack([primary_test, test_x])
        F_oof = expand(Z_oof); F_te = expand(Z_te)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        meta_oof = np.zeros(len(y))
        for tr, va in skf.split(np.zeros(len(y)), y):
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(F_oof[tr], y[tr])
            meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
        mm = roc_auc_score(y, meta_oof)
        rho = spearmanr(oof_x, primary_oof).correlation
        delta = (mm - primary_auc) * 1e4
        print(f"\n--- min-meta gate {label} ---")
        print(f"  base OOF AUC      = {roc_auc_score(y, oof_x):.5f}")
        print(f"  ρ vs PRIMARY OOF  = {rho:.5f}")
        print(f"  min-meta OOF AUC  = {mm:.5f}")
        print(f"  Δ vs PRIMARY      = {delta:+.2f} bp  "
              f"({'PASS' if delta >= 0.5 else 'FAIL'} at +0.5bp)")

        # Full K=23 add eval (PRIMARY components + new base)
        K23_oof = np.column_stack([P_oof, oof_x])
        K23_te = np.column_stack([P_te, test_x])
        F_oof23 = expand(K23_oof); F_te23 = expand(K23_te)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        meta23_oof = np.zeros(len(y))
        for tr, va in skf.split(np.zeros(len(y)), y):
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(F_oof23[tr], y[tr])
            meta23_oof[va] = lr.predict_proba(F_oof23[va])[:, 1]
        lr_full = LogisticRegression(C=1.0, max_iter=2000)
        lr_full.fit(F_oof23, y)
        meta23_te = lr_full.predict_proba(F_te23)[:, 1]
        mm23 = roc_auc_score(y, meta23_oof)
        delta23 = (mm23 - primary_auc) * 1e4
        print(f"  K=23 add OOF AUC  = {mm23:.5f}  Δ = {delta23:+.2f} bp")
        np.save(ART / f"oof_K23_add_{label}_strat.npy", meta23_oof.astype(np.float32))
        np.save(ART / f"test_K23_add_{label}_strat.npy", meta23_te.astype(np.float32))

    print(f"\nwall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
