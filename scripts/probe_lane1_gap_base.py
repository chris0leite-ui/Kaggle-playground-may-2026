"""scripts/probe_lane1_gap_base.py — gap features at the BASE level.

Lane 1's P1.1 added gap features to the K=4 LR META. That tested whether
the LR projection could exploit gap directly. It NULL'd at +0.02 bp.

This probe tests a different question: when a LightGBM base trained on
raw 14 features + gap features is added to K=4, does the K=4+1 gate
fire? If yes, the gap signal exists but the meta-level LR couldn't
linearly extract it (would suggest a non-additive base transform).

Cost: ~10 min CPU.

Construction:
  - Build base features: 14 raw columns + 6 gap features (gap_to_next_obs,
    gap_to_prev_obs, stint_lap_idx, is_last_in_stint, stint_density,
    stint_size). Combined-frame, label-free.
  - Train LightGBM 5-fold StratifiedKFold on PitNextLap.
  - Compute standalone OOF AUC.
  - K=4+1 gate via plain LR meta on [P, rank, logit] expansion.

Outputs:
  scripts/artifacts/probe_lane1_gap_base.json
  scripts/artifacts/oof_lane1_gap_base_lgbm_strat.npy
  scripts/artifacts/test_lane1_gap_base_lgbm_strat.npy
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 500

K4_FWD = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def main():
    t0 = time.time()
    print("Loading data + K=4 base OOFs ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    n_tr = len(train)
    df = pd.concat([train.assign(_split="tr"),
                    test.assign(_split="te")],
                   ignore_index=True, sort=False)
    df["row_id"] = np.arange(len(df))
    keys = ["Race", "Driver", "Year", "Stint"]
    df = df.sort_values(keys + ["LapNumber"], kind="stable").reset_index(drop=True)
    g = df.groupby(keys, sort=False)
    df["next_lap"] = g["LapNumber"].shift(-1)
    df["prev_lap"] = g["LapNumber"].shift(1)
    df["gap_to_next_obs"] = (df["next_lap"] - df["LapNumber"]).fillna(-1).astype(float)
    df["gap_to_prev_obs"] = (df["LapNumber"] - df["prev_lap"]).fillna(-1).astype(float)
    df["stint_size"] = g["LapNumber"].transform("size")
    df["stint_lap_idx"] = g["LapNumber"].rank("dense").astype(int) - 1
    df["is_last_in_stint"] = (df["next_lap"].isna()).astype(int)
    df["stint_min_lap"] = g["LapNumber"].transform("min")
    df["stint_max_lap"] = g["LapNumber"].transform("max")
    df["stint_span"] = df["stint_max_lap"] - df["stint_min_lap"] + 1
    df["stint_density"] = df["stint_size"] / df["stint_span"].clip(lower=1)
    df = df.sort_values("row_id").reset_index(drop=True)

    raw_feats = ["LapNumber", "Stint", "TyreLife", "Position",
                 "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                 "RaceProgress", "Position_Change", "PitStop"]
    raw_feats = [f for f in raw_feats if f in df.columns]
    cat_feats = ["Driver", "Compound", "Race", "Year"]
    for c in cat_feats:
        df[c] = df[c].astype("category").cat.codes.astype(int)
    gap_feats = ["gap_to_next_obs", "gap_to_prev_obs", "stint_lap_idx",
                 "is_last_in_stint", "stint_density", "stint_size"]
    all_feats = raw_feats + cat_feats + gap_feats
    print(f"  base features: {len(all_feats)} = {len(raw_feats)} raw + "
          f"{len(cat_feats)} categorical + {len(gap_feats)} gap")

    tr_x = df.iloc[:n_tr].reset_index(drop=True)
    te_x = df.iloc[n_tr:].reset_index(drop=True)
    X = tr_x[all_feats].astype(float).values
    X_test = te_x[all_feats].astype(float).values

    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED).split(X, y))

    LGB = dict(
        objective="binary", metric="auc", learning_rate=0.05,
        num_leaves=63, min_data_in_leaf=80, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
    )

    # ---- Variant A: WITHOUT gap features (baseline LGBM on 14 raw)
    print("\n--- Variant A: LGBM on raw 14 features only (baseline)")
    feats_no_gap = [f for f in all_feats if f not in gap_feats]
    X_a = tr_x[feats_no_gap].astype(float).values
    Xt_a = te_x[feats_no_gap].astype(float).values
    oof_a = np.zeros(len(y))
    test_a = np.zeros(X_test.shape[0])
    for fold, (tr, va) in enumerate(splits):
        ds_tr = lgb.Dataset(X_a[tr], label=y[tr])
        ds_va = lgb.Dataset(X_a[va], label=y[va], reference=ds_tr)
        booster = lgb.train(
            LGB, ds_tr, num_boost_round=600, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        oof_a[va] = booster.predict(X_a[va])
        test_a += booster.predict(Xt_a) / N_FOLDS
    auc_a = float(roc_auc_score(y, oof_a))
    print(f"  Standalone LGBM (raw only) OOF: {auc_a:.5f}")

    # ---- Variant B: WITH gap features
    print("\n--- Variant B: LGBM on raw 14 + gap features")
    oof_b = np.zeros(len(y))
    test_b = np.zeros(X_test.shape[0])
    for fold, (tr, va) in enumerate(splits):
        ds_tr = lgb.Dataset(X[tr], label=y[tr])
        ds_va = lgb.Dataset(X[va], label=y[va], reference=ds_tr)
        booster = lgb.train(
            LGB, ds_tr, num_boost_round=600, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        oof_b[va] = booster.predict(X[va])
        test_b += booster.predict(X_test) / N_FOLDS
    auc_b = float(roc_auc_score(y, oof_b))
    delta_standalone = (auc_b - auc_a) * 1e4
    print(f"  Standalone LGBM (raw + gap) OOF: {auc_b:.5f}  (Δ {delta_standalone:+.2f} bp vs baseline)")

    # ---- K=4 + 1 gate for variant B
    base_oofs = [_pos(ART / f"oof_{n}_strat.npy") for n in K4_FWD]
    base_tests = [_pos(ART / f"test_{n}_strat.npy") for n in K4_FWD]
    P4 = np.column_stack(base_oofs)
    P4_test = np.column_stack(base_tests)
    F4 = _expand(P4)
    F4_test = _expand(P4_test)

    def fit_lr(F, F_test):
        op = np.zeros(len(y))
        tp = np.zeros(F_test.shape[0])
        for tr, va in splits:
            lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            op[va] = lr.predict_proba(F[va])[:, 1]
            tp += lr.predict_proba(F_test)[:, 1] / N_FOLDS
        return op, tp

    F4plus_b = _expand(np.hstack([P4, oof_b.reshape(-1, 1)]))
    F4plus_b_test = _expand(np.hstack([P4_test, test_b.reshape(-1, 1)]))
    F4plus_a = _expand(np.hstack([P4, oof_a.reshape(-1, 1)]))
    F4plus_a_test = _expand(np.hstack([P4_test, test_a.reshape(-1, 1)]))

    oof_K4, _ = fit_lr(F4, F4_test)
    oof_K4_a, _ = fit_lr(F4plus_a, F4plus_a_test)
    oof_K4_b, test_K4_b = fit_lr(F4plus_b, F4plus_b_test)

    auc_K4 = float(roc_auc_score(y, oof_K4))
    auc_K4_a = float(roc_auc_score(y, oof_K4_a))
    auc_K4_b = float(roc_auc_score(y, oof_K4_b))
    delta_a = (auc_K4_a - auc_K4) * 1e4
    delta_b = (auc_K4_b - auc_K4) * 1e4
    delta_b_vs_a = (auc_K4_b - auc_K4_a) * 1e4

    print(f"\n--- K=4 + 1 gate")
    print(f"  K=4 plain LR meta            : {auc_K4:.5f}")
    print(f"  K=4 + LGBM(raw)              : {auc_K4_a:.5f}  (Δ {delta_a:+.3f} bp)")
    print(f"  K=4 + LGBM(raw + gap)        : {auc_K4_b:.5f}  (Δ {delta_b:+.3f} bp)")
    print(f"  Δ from gap features alone    : {delta_b_vs_a:+.3f} bp")

    rho_b = float(spearmanr(oof_b, oof_a)[0])
    print(f"  ρ(LGBM_with_gap, LGBM_no_gap): {rho_b:.5f}")

    np.save(ART / "oof_lane1_gap_base_lgbm_strat.npy", oof_b)
    np.save(ART / "test_lane1_gap_base_lgbm_strat.npy",
            np.column_stack([1 - test_b, test_b]))

    out = {
        "K4_bases": K4_FWD,
        "feat_count_raw_only": len(feats_no_gap),
        "feat_count_with_gap": len(all_feats),
        "standalone_lgbm_raw_oof": auc_a,
        "standalone_lgbm_with_gap_oof": auc_b,
        "delta_standalone_bp": float(delta_standalone),
        "K4_LR_plain_oof": auc_K4,
        "K4_plus_lgbm_raw_oof": auc_K4_a,
        "K4_plus_lgbm_with_gap_oof": auc_K4_b,
        "delta_K4_plus_raw_bp": float(delta_a),
        "delta_K4_plus_with_gap_bp": float(delta_b),
        "delta_gap_alone_bp": float(delta_b_vs_a),
        "rho_with_gap_vs_raw": rho_b,
        "verdict_K4_plus_gap": ("PASS" if delta_b >= 0.5
                                else "AMBIG" if delta_b >= -0.1
                                else "NULL"),
        "verdict_gap_marginal_to_raw": ("PASS" if delta_b_vs_a >= 0.5
                                         else "AMBIG" if delta_b_vs_a >= -0.1
                                         else "NULL"),
        "wall_s": time.time() - t0,
    }
    (ART / "probe_lane1_gap_base.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {ART/'probe_lane1_gap_base.json'}. Wall {out['wall_s']:.1f}s")
    print(f"Verdict K=4+gap-base: {out['verdict_K4_plus_gap']}")
    print(f"Verdict gap-marginal-to-raw: {out['verdict_gap_marginal_to_raw']}")


if __name__ == "__main__":
    main()
