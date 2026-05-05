"""D3-A — Unified OOF target encoding with the missed Day-1 levers.

Extends d2a_target_encoding.py with the two TE keys flagged by Day-2
strategy critique (analyticaobscura Source 1 #2) but never executed:
  - Driver_Compound (≈4k pairs)
  - Race_LapBin (≈26×10 = 260 pairs; LapBin = qcut(RaceProgress, 10))

Single keys: Driver, Race, Compound (same as d2a)
2-way keys (4): Driver_Race, Race_Compound (same as d2a)
                + Driver_Compound, Race_LapBin (NEW — orthogonal-signal play
                per HANDOVER 2026-05-04 mid-session).

OOF discipline: per outer fold, compute TE using ONLY the outer-train
portion (with global-mean smoothing α=80). The OOF TE values for outer-train
are computed via inner 5-fold CV. Test TE is fit on all train.

LapBin edges are computed once, globally, on RaceProgress quantiles.
RaceProgress is a deterministic feature (no target dependence), so the
binning itself does not leak; only the TE *values* per bin require fold
discipline (inherited from d2a recipe).

Two-anchor 5-fold (StratKFold + GroupKFold Race), same as d2a.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET = "PitNextLap"
ID_COL = "id"
ALPHA = 80.0
LAP_BIN_Q = 10

TE_SINGLES = ["Driver", "Race", "Compound"]
TE_INTERACTIONS = [
    ("Driver", "Race"),
    ("Race", "Compound"),
    ("Driver", "Compound"),  # NEW
    ("Race", "LapBin"),      # NEW
]


def make_lgb_params() -> dict:
    return dict(objective="binary", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                min_data_in_leaf=200, verbose=-1, seed=SEED)


def add_lapbin(df: pd.DataFrame, edges: np.ndarray | None = None):
    """Discretize RaceProgress into quantile bins. Returns (df_with_LapBin, edges)."""
    df = df.copy()
    if edges is None:
        # Use train RaceProgress to compute quantile edges
        edges = np.quantile(df["RaceProgress"].values, np.linspace(0, 1, LAP_BIN_Q + 1))
        edges[0] -= 1e-9   # guard against boundary issues
        edges[-1] += 1e-9
    bins = np.digitize(df["RaceProgress"].values, edges, right=True) - 1
    bins = np.clip(bins, 0, LAP_BIN_Q - 1).astype(np.int32)
    df["LapBin"] = bins
    return df, edges


def build_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction keys as new string columns."""
    df = df.copy()
    for a, b in TE_INTERACTIONS:
        df[f"{a}_{b}"] = df[a].astype(str) + "|" + df[b].astype(str)
    return df


def smoothed_te(train_y: np.ndarray, train_key: np.ndarray,
                apply_key: np.ndarray, alpha: float) -> np.ndarray:
    global_mean = float(train_y.mean())
    df = pd.DataFrame({"k": train_key, "y": train_y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    g["te"] = (g["sum"] + alpha * global_mean) / (g["count"] + alpha)
    return pd.Series(apply_key).map(g["te"]).fillna(global_mean).to_numpy()


def oof_te_train(y: np.ndarray, key: np.ndarray, alpha: float,
                 n_inner: int = 5, seed: int = 42) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float32)
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for tr, va in kf.split(np.zeros(len(y))):
        out[va] = smoothed_te(y[tr], key[tr], key[va], alpha)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    # LapBin from train RaceProgress, applied to both
    train, edges = add_lapbin(train, edges=None)
    test, _ = add_lapbin(test, edges=edges)
    print(f"LapBin edges (RaceProgress quantiles, q={LAP_BIN_Q}): {edges}")

    train = build_keys(train)
    test = build_keys(test)

    y = train[TARGET].astype(int).values
    all_te_keys = TE_SINGLES + [f"{a}_{b}" for a, b in TE_INTERACTIONS]
    print(f"TE keys ({len(all_te_keys)}): {all_te_keys}")
    print("cardinalities:")
    for k in all_te_keys:
        print(f"  {k}: train={train[k].nunique()}, test={test[k].nunique()}")

    base_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    X = train[base_cols].copy()
    X_test = test[[c for c in base_cols if c in test.columns]].copy()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    def add_te_for_split(tr_idx, va_idx, X_train_full, X_test_full):
        Xtr = X_train_full.copy()
        Xte_aug = X_test_full.copy()
        for k in all_te_keys:
            full_train_key = train[k].values
            inner_oof = oof_te_train(y[tr_idx], full_train_key[tr_idx],
                                     ALPHA, n_inner=5, seed=SEED)
            te_va = smoothed_te(y[tr_idx], full_train_key[tr_idx],
                                full_train_key[va_idx], ALPHA)
            te_test = smoothed_te(y[tr_idx], full_train_key[tr_idx],
                                  test[k].values, ALPHA)
            te_col = np.zeros(len(Xtr), dtype=np.float64)
            te_col[tr_idx] = inner_oof
            te_col[va_idx] = te_va
            Xtr[f"te_{k}"] = te_col
            Xte_aug[f"te_{k}"] = te_test
        return Xtr, Xte_aug

    def run_anchor(name: str, splits):
        oof = np.zeros(len(y), dtype=np.float32)
        test_proba = np.zeros(len(test), dtype=np.float32)
        fold_scores = []
        for k_fold, (tr_idx, va_idx) in enumerate(splits):
            Xtr_aug, Xte_aug = add_te_for_split(tr_idx, va_idx, X, X_test)
            cat_features = cat_cols  # raw categoricals only; TE are numeric
            dtrain = lgb.Dataset(Xtr_aug.iloc[tr_idx], y[tr_idx],
                                 categorical_feature=cat_features)
            dval = lgb.Dataset(Xtr_aug.iloc[va_idx], y[va_idx],
                               categorical_feature=cat_features)
            model = lgb.train(make_lgb_params(), dtrain, num_boost_round=2000,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(100),
                                         lgb.log_evaluation(0)])
            p_va = model.predict(Xtr_aug.iloc[va_idx])
            oof[va_idx] = p_va
            test_proba += model.predict(Xte_aug) / N_FOLDS
            s = float(roc_auc_score(y[va_idx], p_va))
            fold_scores.append(s)
            print(f"  [{name}] fold {k_fold}: AUC={s:.5f} "
                  f"(best_iter={model.best_iteration})")
        return (oof, test_proba, float(roc_auc_score(y, oof)),
                fold_scores, float(np.std(fold_scores)))

    print("=== Anchor A: StratifiedKFold(5) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y)), y))
    oof_a, test_a, auc_a, folds_a, std_a = run_anchor("STRAT", splits_a)

    print("=== Anchor B: GroupKFold(Race) ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, train["Race"].values))
    oof_b, test_b, auc_b, folds_b, std_b = run_anchor("GROUP", splits_b)

    base_a, base_b = 0.94075, 0.92059
    d2a_a, d2a_b = 0.93670, 0.91628   # d2a_te reference (orthogonality benchmark)
    delta_a = (auc_a - base_a) * 1e4
    delta_b = (auc_b - base_b) * 1e4
    delta_d2a_a = (auc_a - d2a_a) * 1e4
    delta_d2a_b = (auc_b - d2a_b) * 1e4
    print(f"\nOOF_A: {auc_a:.5f}  std={std_a:.5f}  "
          f"Δ vs baseline={delta_a:+.1f}bp  Δ vs d2a_te={delta_d2a_a:+.1f}bp")
    print(f"OOF_B: {auc_b:.5f}  std={std_b:.5f}  "
          f"Δ vs baseline={delta_b:+.1f}bp  Δ vs d2a_te={delta_d2a_b:+.1f}bp")

    save_oof("d3a_te_unified_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=std_a, fold_scores=folds_a,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_a, delta_vs_d2a_bp=delta_d2a_a,
                  alpha=ALPHA, te_keys=all_te_keys, lap_bin_q=LAP_BIN_Q,
                  lap_bin_edges=edges.tolist()))
    save_oof("d3a_te_unified_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=std_b, fold_scores=folds_b,
                  cv="GroupKFold(Race)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_b, delta_vs_d2a_bp=delta_d2a_b,
                  alpha=ALPHA, te_keys=all_te_keys))

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_d3a_te_unified.csv", index=False)


if __name__ == "__main__":
    main()
