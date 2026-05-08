"""D3-B — Sequence-FE base (HANDOVER Step 2, 2026-05-04 Day-3).

Exploits the (Race, Driver) lap-sequence structure. Strategy critique
flagged this lever: 97.4% of test rows have a same-(Race, Driver)
in-test successor → the lap-sequence space is unscouted.

Three features over (Race, Driver) groups, sorted by LapNumber:

  1. cumulative_pitstops_this_race
     groupby(Race, Driver).PitStop.cumsum()
     LEAK-FREE: uses observed PitStop column (present in train AND test),
     no target dependence.

  2. laps_since_last_pitstop
     For each row: LapNumber − (LapNumber where PitStop=1, ffill within
     group; 0 if no prior pit).
     LEAK-FREE: uses observed PitStop only.

  3. rolling_target_rate(window=5)
     groupby(Race, Driver).PitNextLap.shift(1).rolling(5, min_periods=1).mean()
     LEAKAGE-RISK: uses target. Discipline: per outer fold, set
     outer-val PitNextLap to NaN before computing → rolling for outer-val
     rows uses only outer-train priors. Test rows always have NaN target,
     so their rolling uses train-side priors only.

Single LGBM (same params as baseline_two_anchor) on baseline + these 3
features. Two-anchor (Strat + GroupKF Race).

Gate: standalone Strat OOF ≥ 0.946 → add to M5h pool as M5k base.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET = "PitNextLap"
ID_COL = "id"
ROLL_W = 5


def make_lgb_params() -> dict:
    return dict(objective="binary", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                min_data_in_leaf=200, verbose=-1, seed=SEED)


def add_leakfree_seqfe(df: pd.DataFrame) -> pd.DataFrame:
    """cumulative_pitstops_this_race + laps_since_last_pitstop.

    Both leak-free: uses observed PitStop column only.
    """
    df = df.copy()
    sort_idx = df.sort_values(["Race", "Driver", "LapNumber"]).index
    df_s = df.loc[sort_idx].copy()
    grp = df_s.groupby(["Race", "Driver"], sort=False)

    df_s["cum_pits_this_race"] = grp["PitStop"].cumsum()

    # laps_since_last_pit: LapNumber - (LapNumber of most-recent prior PitStop=1
    # in group, ffilled). Sentinel = LapNumber if no prior pit (laps from start).
    df_s["_last_pit_marker"] = df_s["LapNumber"].where(df_s["PitStop"] == 1)
    df_s["_last_pit_lap"] = grp["_last_pit_marker"].ffill()
    df_s["laps_since_last_pit"] = (df_s["LapNumber"] - df_s["_last_pit_lap"]).fillna(
        df_s["LapNumber"]
    )
    df_s = df_s.drop(columns=["_last_pit_marker", "_last_pit_lap"])

    return df_s.loc[df.index]  # restore original ordering


def add_rolling_target_rate(combined: pd.DataFrame, val_mask: np.ndarray,
                            global_mean: float) -> np.ndarray:
    """Rolling target rate with per-fold leakage discipline.

    Args:
      combined: train+test concat with PitNextLap (NaN for test).
      val_mask: boolean mask over combined; True for outer-val ROWS to hide.
      global_mean: fillna value for early-group rows.

    Returns:
      rolling_target_rate array aligned to combined's index order.
    """
    df = combined.copy()
    target_col = df[TARGET].astype(float).values.copy()
    target_col[val_mask] = np.nan
    df["_y_for_roll"] = target_col

    sort_idx = df.sort_values(["Race", "Driver", "LapNumber"]).index
    df_s = df.loc[sort_idx]
    grp = df_s.groupby(["Race", "Driver"], sort=False)
    # shift(1) so current-row target never enters its own rolling window
    shifted = grp["_y_for_roll"].shift(1)
    df_s = df_s.assign(_y_shifted=shifted)
    rolled = (
        df_s.groupby(["Race", "Driver"], sort=False)["_y_shifted"]
            .rolling(ROLL_W, min_periods=1).mean()
            .reset_index(level=[0, 1], drop=True)
    )
    df_s["rolling_target_rate"] = rolled
    df_s["rolling_target_rate"] = df_s["rolling_target_rate"].fillna(global_mean)
    out = df_s["rolling_target_rate"].loc[df.index].values
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    n_train = len(train)
    print(f"train shape: {train.shape}, test shape: {test.shape}")

    # Combine train+test for sequence-FE (target unknown for test → NaN)
    test_with_target = test.copy()
    test_with_target[TARGET] = np.nan
    combined = pd.concat([train, test_with_target], axis=0, ignore_index=True)
    print(f"combined shape: {combined.shape}")

    # Leak-free features (computed once globally)
    combined = add_leakfree_seqfe(combined)
    print("Added cum_pits_this_race + laps_since_last_pit (leak-free).")

    y = train[TARGET].astype(int).values
    global_mean = float(y.mean())
    print(f"global PitNextLap mean: {global_mean:.4f}")

    # Sanity: feature value-counts
    print("cum_pits_this_race describe (combined):")
    print(combined["cum_pits_this_race"].describe())
    print("laps_since_last_pit describe (combined):")
    print(combined["laps_since_last_pit"].describe())

    # Build feature matrix
    base_cols = [c for c in combined.columns if c not in (TARGET, ID_COL)]
    cat_cols_obj = combined[base_cols].select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"cat_cols_obj: {cat_cols_obj}")

    # Cast categorical
    for c in cat_cols_obj:
        combined[c] = combined[c].astype("category")

    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(combined))

    def run_anchor(name: str, splits):
        oof = np.zeros(n_train, dtype=np.float32)
        test_proba = np.zeros(len(test), dtype=np.float32)
        fold_scores = []
        for k_fold, (tr_idx, va_idx) in enumerate(splits):
            # Compute rolling_target_rate with outer-val target hidden
            val_mask = np.zeros(len(combined), dtype=bool)
            val_mask[va_idx] = True   # va_idx is over train rows
            rtr = add_rolling_target_rate(combined, val_mask, global_mean)
            X_full = combined[base_cols].copy()
            X_full["rolling_target_rate"] = rtr

            X_tr = X_full.iloc[tr_idx]
            X_va = X_full.iloc[va_idx]
            X_te = X_full.iloc[test_idx]
            cat_features = cat_cols_obj
            dtrain = lgb.Dataset(X_tr, y[tr_idx], categorical_feature=cat_features)
            dval = lgb.Dataset(X_va, y[va_idx], categorical_feature=cat_features)
            model = lgb.train(make_lgb_params(), dtrain, num_boost_round=2000,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(100),
                                         lgb.log_evaluation(0)])
            p_va = model.predict(X_va)
            oof[va_idx] = p_va
            test_proba += model.predict(X_te) / N_FOLDS
            s = float(roc_auc_score(y[va_idx], p_va))
            fold_scores.append(s)
            print(f"  [{name}] fold {k_fold}: AUC={s:.5f} (best_iter={model.best_iteration})")
        return (oof, test_proba, float(roc_auc_score(y, oof)),
                fold_scores, float(np.std(fold_scores)))

    print("=== Anchor A: StratifiedKFold(5) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(n_train), y))
    oof_a, test_a, auc_a, folds_a, std_a = run_anchor("STRAT", splits_a)

    print("=== Anchor B: GroupKFold(Race) ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(n_train), y, train["Race"].values))
    oof_b, test_b, auc_b, folds_b, std_b = run_anchor("GROUP", splits_b)

    base_a, base_b = 0.94075, 0.92059
    delta_a = (auc_a - base_a) * 1e4
    delta_b = (auc_b - base_b) * 1e4
    print(f"\nOOF_A (Strat): {auc_a:.5f}  std={std_a:.5f}  Δ baseline={delta_a:+.1f}bp")
    print(f"OOF_B (GroupKF): {auc_b:.5f}  std={std_b:.5f}  Δ baseline={delta_b:+.1f}bp")
    print(f"Gate (HANDOVER Step 2): standalone Strat ≥ 0.946 → "
          f"{'PASS' if auc_a >= 0.946 else 'FAIL'} (Strat={auc_a:.5f})")

    save_oof("d3b_seqfe_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=std_a, fold_scores=folds_a,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_a, roll_window=ROLL_W,
                  features_added=["cum_pits_this_race",
                                  "laps_since_last_pit",
                                  "rolling_target_rate"]))
    save_oof("d3b_seqfe_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=std_b, fold_scores=folds_b,
                  cv="GroupKFold(Race)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_b, roll_window=ROLL_W))

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_d3b_seqfe.csv", index=False)


if __name__ == "__main__":
    main()
