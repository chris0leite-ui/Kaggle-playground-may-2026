"""D2-A — OOF target encoding (analyticaobscura recipe, α=80).

Adds target-encoded versions of high-cardinality categoricals to
the baseline feature set:
  - Driver (887 levels)
  - Driver_Race (≈14k pairs)
  - Race (26)
  - Race_Compound (130)
  - Compound (5)  [diagnostic only]

OOF discipline: per outer fold, compute TE using ONLY the
outer-train portion (with global-mean smoothing α=80). The OOF
TE values for outer-train are computed via inner 5-fold CV. For
test predictions, TE is fit on all train.

Two-anchor 5-fold (StratKFold + GroupKFold Race).
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
ALPHA = 80.0  # smoothing per analyticaobscura

# Categorical columns to TE-encode
TE_KEYS = ["Driver", "Race", "Compound"]
TE_INTERACTIONS = [("Driver", "Race"), ("Race", "Compound")]


def make_lgb_params() -> dict:
    return dict(objective="binary", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                min_data_in_leaf=200, verbose=-1, seed=SEED)


def build_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction keys as new string columns."""
    df = df.copy()
    for a, b in TE_INTERACTIONS:
        df[f"{a}_{b}"] = df[a].astype(str) + "|" + df[b].astype(str)
    return df


def smoothed_te(train_y: np.ndarray, train_key: np.ndarray,
                apply_key: np.ndarray, alpha: float) -> np.ndarray:
    """Compute smoothed target encoding from train_key→train_y, apply to apply_key."""
    global_mean = float(train_y.mean())
    df = pd.DataFrame({"k": train_key, "y": train_y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    g["te"] = (g["sum"] + alpha * global_mean) / (g["count"] + alpha)
    return pd.Series(apply_key).map(g["te"]).fillna(global_mean).to_numpy()


def oof_te_train(y: np.ndarray, key: np.ndarray, alpha: float,
                 n_inner: int = 5, seed: int = 42) -> np.ndarray:
    """OOF target encoding for train rows (inner KFold)."""
    out = np.zeros(len(y), dtype=np.float32)
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for tr, va in kf.split(np.zeros(len(y))):
        out[va] = smoothed_te(y[tr], key[tr], key[va], alpha)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    train = build_keys(train)
    test = build_keys(test)

    y = train[TARGET].astype(int).values
    all_te_keys = TE_KEYS + [f"{a}_{b}" for a, b in TE_INTERACTIONS]
    print(f"TE keys: {all_te_keys}")
    print(f"cardinalities: {[(k, train[k].nunique()) for k in all_te_keys]}")

    # Build base feature frames (drop target, id; keep raw cats AND TE features)
    base_cols = [c for c in train.columns
                 if c not in (TARGET, ID_COL)]
    X = train[base_cols].copy()
    X_test = test[[c for c in base_cols if c in test.columns]].copy()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    def add_te_for_split(tr_idx, va_idx, X_train_full, X_test_full):
        """Compute TE features per split. Returns augmented frames."""
        Xtr = X_train_full.copy()
        Xva = X_train_full.iloc[va_idx].copy()
        Xte_aug = X_test_full.copy()
        # Per outer fold:
        # - outer-train (tr_idx) computes inner-OOF TE on itself
        # - outer-val (va_idx) gets TE from full outer-train
        # - test gets TE from full outer-train
        for k in all_te_keys:
            full_train_key = train[k].values
            inner_oof = oof_te_train(y[tr_idx], full_train_key[tr_idx],
                                     ALPHA, n_inner=5, seed=SEED)
            te_va = smoothed_te(y[tr_idx], full_train_key[tr_idx],
                                full_train_key[va_idx], ALPHA)
            te_test = smoothed_te(y[tr_idx], full_train_key[tr_idx],
                                  test[k].values, ALPHA)
            Xtr.loc[Xtr.index[tr_idx], f"te_{k}"] = inner_oof
            # outer-val: assign TE_va to those indices
            Xtr.loc[Xtr.index[va_idx], f"te_{k}"] = te_va
            # test: per-fold TE; we'll average across outer folds at the end
            # We store on Xte_aug as a fold-specific column we'll average later
            Xte_aug[f"te_{k}"] = te_test
        return Xtr, Xte_aug

    def run_anchor(name: str, splits):
        oof = np.zeros(len(y), dtype=np.float32)
        # Per-fold test preds; average at end
        test_proba = np.zeros(len(test), dtype=np.float32)
        fold_scores = []
        for k_fold, (tr_idx, va_idx) in enumerate(splits):
            Xtr_aug, Xte_aug = add_te_for_split(tr_idx, va_idx, X, X_test)
            cat_features = (cat_cols)  # raw categoricals only; TE are numeric
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
            print(f"  [{name}] fold {k_fold}: AUC={s:.5f} (best_iter={model.best_iteration})")
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
    delta_a = (auc_a - base_a) * 1e4
    delta_b = (auc_b - base_b) * 1e4
    print(f"\nOOF_A: {auc_a:.5f}  std={std_a:.5f}  ΔA={delta_a:+.1f}bp")
    print(f"OOF_B: {auc_b:.5f}  std={std_b:.5f}  ΔB={delta_b:+.1f}bp")

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_d2a_target_encoding.csv", index=False)
    save_oof("d2a_te_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=std_a, fold_scores=folds_a,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_a, alpha=ALPHA,
                  te_keys=all_te_keys))
    save_oof("d2a_te_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=std_b, fold_scores=folds_b,
                  cv="GroupKFold(Race)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_b, alpha=ALPHA))

    import datetime as dt
    out = Path(f"audit/{dt.date.today().isoformat()}-d2a-target-encoding.md")
    out.write_text(
        f"# D2-A — OOF target encoding ({dt.date.today()})\n\n"
        f"TE keys: {all_te_keys}; α={ALPHA}; inner 5-fold KFold(seed={SEED}).\n"
        f"Raw categoricals KEPT alongside TE (per analyticaobscura).\n\n"
        f"## Results\n\n"
        f"| anchor | OOF AUC | fold_std | per-fold | Δ vs baseline_two_anchor |\n"
        f"|---|---:|---:|---|---:|\n"
        f"| A — StratKFold | **{auc_a:.5f}** | {std_a:.5f} | "
        f"{[f'{x:.4f}' for x in folds_a]} | {delta_a:+.1f}bp |\n"
        f"| B — GroupKFold(Race) | {auc_b:.5f} | {std_b:.5f} | "
        f"{[f'{x:.4f}' for x in folds_b]} | {delta_b:+.1f}bp |\n\n"
    )
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
