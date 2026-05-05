"""D3-C — Stint-2 specialist + ensemble switch.

Per-segment diagnostic showed Stint 2 is the lift surface:
  - 30% of data (130k rows)
  - M5h OOF AUC on Stint=2 = 0.916 (-341bp from agg 0.95043)
  - Strategic-decision zone: post-first-pit, planning second pit

Specialist design:
  - LGBM on raw features + 3 sequence-FE features (cum_pits_this_race,
    laps_since_last_pit, rolling_target_rate(window=5)).
  - Trained on Stint=2 train rows ONLY.
  - 5-fold StratifiedKFold within the Stint=2 subset (seed=42 to match
    baseline; new seed produces independent split, OK because we use
    the specialist OOF only on Stint=2 rows where M5h OOF was also
    out-of-fold against the OUTER split — these are independent).

Blend strategy:
  - For Stint=2 train rows: use specialist OOF (replaces M5h OOF).
  - For other Stint train rows: keep M5h OOF.
  - Compute aggregate AUC of the blend.

If blend AUC > M5h Strat 0.95043 by ≥5bp → slot-7 candidate.

Test inference: same switch — specialist for Stint=2 test rows,
M5h for others.

R1: Strat-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
M5H_AGG_S = 0.95043
ROLL_W = 5


def make_lgb_params() -> dict:
    return dict(objective="binary", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                min_data_in_leaf=200, verbose=-1, seed=SEED)


def add_seqfe_global(combined: pd.DataFrame, target_col_for_roll: np.ndarray):
    """Compute cum_pits_this_race + laps_since_last_pit + rolling_target_rate
    over (Race, Driver, sorted by LapNumber) — global pass.

    target_col_for_roll: same length as combined; train rows have actual
    target, test rows have NaN. Rolling uses shift(1) to avoid self-leak.
    Caller is responsible for the val-leakage discipline.
    """
    df = combined.copy()
    df["_y_for_roll"] = target_col_for_roll
    sort_idx = df.sort_values(["Race", "Driver", "LapNumber"]).index
    df_s = df.loc[sort_idx]
    grp = df_s.groupby(["Race", "Driver"], sort=False)
    df_s["cum_pits_this_race"] = grp["PitStop"].cumsum()
    df_s["_last_pit_marker"] = df_s["LapNumber"].where(df_s["PitStop"] == 1)
    df_s["_last_pit_lap"] = grp["_last_pit_marker"].ffill()
    df_s["laps_since_last_pit"] = (df_s["LapNumber"] - df_s["_last_pit_lap"]).fillna(
        df_s["LapNumber"]
    )
    shifted = grp["_y_for_roll"].shift(1)
    df_s = df_s.assign(_y_shifted=shifted)
    rolled = (
        df_s.groupby(["Race", "Driver"], sort=False)["_y_shifted"]
            .rolling(ROLL_W, min_periods=1).mean()
            .reset_index(level=[0, 1], drop=True)
    )
    df_s["rolling_target_rate"] = rolled
    df_s = df_s.drop(columns=["_last_pit_marker", "_last_pit_lap",
                              "_y_for_roll", "_y_shifted"])
    return df_s.loc[df.index]


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    n_train = len(train)

    y = train[TARGET].astype(int).values
    global_mean = float(y.mean())

    test_with_target = test.copy(); test_with_target[TARGET] = np.nan
    combined = pd.concat([train, test_with_target], axis=0, ignore_index=True)

    # Stint=2 mask in train and test
    s2_train_mask = (train["Stint"] == 2).values
    s2_test_mask = (test["Stint"] == 2).values
    s2_train_idx = np.where(s2_train_mask)[0]
    s2_test_idx = np.where(s2_test_mask)[0]
    print(f"Stint=2 train: {len(s2_train_idx)} ({100*len(s2_train_idx)/n_train:.1f}%)")
    print(f"Stint=2 test: {len(s2_test_idx)} ({100*len(s2_test_idx)/len(test):.1f}%)")

    # M5h OOF (LB-proxy anchor) for non-S2 + comparison on S2
    m5h_oof = np.load(ART / "oof_m5h_strat.npy")[:, 1].astype(np.float64)
    m5h_test = np.load(ART / "test_m5h_strat.npy")[:, 1].astype(np.float64)
    auc_m5h_s2 = float(roc_auc_score(y[s2_train_mask], m5h_oof[s2_train_mask]))
    print(f"M5h OOF AUC on Stint=2 only: {auc_m5h_s2:.5f}")

    # Build specialist features. 5-fold on Stint=2 train; per outer fold,
    # hide outer-val target before computing rolling_target_rate.
    base_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    cat_cols = train[base_cols].select_dtypes(include=["object", "string"]).columns.tolist()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(s2_train_idx)), y[s2_train_idx]))

    s2_oof = np.zeros(len(s2_train_idx), dtype=np.float32)
    s2_test_proba = np.zeros(len(s2_test_idx), dtype=np.float32)
    fold_scores = []

    for k_fold, (tr_local, va_local) in enumerate(splits):
        # Map to global indices (over combined)
        tr_global = s2_train_idx[tr_local]
        va_global = s2_train_idx[va_local]

        # Build per-fold seq-FE: hide outer-val target before rolling
        target_col = np.full(len(combined), np.nan)
        target_col[:n_train] = y.astype(float)
        # Hide outer-val
        target_col[va_global] = np.nan
        df_with_seqfe = add_seqfe_global(combined, target_col)
        df_with_seqfe["rolling_target_rate"] = df_with_seqfe["rolling_target_rate"].fillna(global_mean)

        # Build feature matrices
        for c in cat_cols:
            df_with_seqfe[c] = df_with_seqfe[c].astype("category")
        feat_cols = [c for c in df_with_seqfe.columns
                     if c not in (TARGET, ID_COL)]
        X_full = df_with_seqfe[feat_cols].copy()

        X_tr = X_full.iloc[tr_global]
        X_va = X_full.iloc[va_global]
        X_te = X_full.iloc[n_train + s2_test_idx]
        cat_features = cat_cols

        dtrain = lgb.Dataset(X_tr, y[tr_global], categorical_feature=cat_features)
        dval = lgb.Dataset(X_va, y[va_global], categorical_feature=cat_features)
        model = lgb.train(make_lgb_params(), dtrain, num_boost_round=2000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100),
                                     lgb.log_evaluation(0)])
        p_va = model.predict(X_va)
        s2_oof[va_local] = p_va
        s2_test_proba += model.predict(X_te) / N_FOLDS
        s = float(roc_auc_score(y[va_global], p_va))
        fold_scores.append(s)
        print(f"  S2-fold {k_fold}: AUC={s:.5f} (best_iter={model.best_iteration})")

    auc_s2 = float(roc_auc_score(y[s2_train_mask], s2_oof))
    print(f"\nStint-2 specialist OOF AUC (within S2): {auc_s2:.5f}")
    print(f"  vs M5h on S2:                          {auc_m5h_s2:.5f}")
    print(f"  Δ specialist vs M5h on S2:             {(auc_s2-auc_m5h_s2)*1e4:+.1f}bp")

    # === Blend: specialist on S2, M5h on rest ===
    blend_oof = m5h_oof.copy()
    blend_oof[s2_train_mask] = s2_oof
    auc_blend = float(roc_auc_score(y, blend_oof))
    print(f"\nBlend OOF (M5h ∪ S2-specialist on S2): {auc_blend:.5f}  "
          f"Δ M5h={(auc_blend-M5H_AGG_S)*1e4:+.1f}bp")

    # === Variant: rank-blend (rank M5h on S2 vs rank specialist on S2,
    #                          take a weighted combination) ===
    # Simple convex blend on S2: 0.5 * specialist + 0.5 * M5h
    for w in [0.3, 0.5, 0.7, 1.0]:
        blend_v = m5h_oof.copy()
        blend_v[s2_train_mask] = w * s2_oof + (1 - w) * m5h_oof[s2_train_mask]
        auc_v = float(roc_auc_score(y, blend_v))
        print(f"  w_specialist={w:.1f} blend OOF: {auc_v:.5f}  "
              f"Δ M5h={(auc_v-M5H_AGG_S)*1e4:+.1f}bp")

    # Save artifacts (full-train OOF aligned to train.csv row order)
    full_oof = m5h_oof.copy()
    full_oof[s2_train_mask] = s2_oof
    full_test = m5h_test.copy()
    full_test[s2_test_idx] = s2_test_proba

    np.save(ART / "oof_d3c_stint2_specialist_strat.npy",
            np.column_stack([1 - full_oof, full_oof]))
    np.save(ART / "test_d3c_stint2_specialist_strat.npy",
            np.column_stack([1 - full_test, full_test]))

    # Submission with the best blend (will be selected after audit)
    sub = sample_sub.copy()
    sub[TARGET] = full_test
    sub.to_csv("submissions/submission_d3c_stint2_blend.csv", index=False)

    res = dict(
        specialist_oof_within_s2=auc_s2,
        m5h_oof_within_s2=auc_m5h_s2,
        delta_specialist_vs_m5h_on_s2_bp=(auc_s2 - auc_m5h_s2) * 1e4,
        blend_oof_replace=auc_blend,
        delta_blend_vs_m5h_bp=(auc_blend - M5H_AGG_S) * 1e4,
        n_train_s2=int(len(s2_train_idx)),
        n_test_s2=int(len(s2_test_idx)),
        fold_scores=fold_scores,
    )
    (ART / "d3c_stint2_specialist_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n→ scripts/artifacts/d3c_stint2_specialist_results.json")


if __name__ == "__main__":
    main()
