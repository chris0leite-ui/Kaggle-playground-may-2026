"""D2-C — concat external (aadigupta1601) to train, refit LGBM.

Same hyperparams as baseline_two_anchor.py. Drop Normalized_TyreLife
(host explicitly forbids reintroducing it). Two-anchor 5-fold OOF.

Note: external rows are train-only — they augment training but the
OOF is computed on s6e5 train rows only (no external rows in val
folds, otherwise OOF doesn't reflect the real test distribution).
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
FORBIDDEN = ["Normalized_TyreLife"]  # host removed


def make_lgb_params() -> dict:
    return dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=200,
        verbose=-1,
        seed=SEED,
    )


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    ext = pd.read_csv("data/external/f1_strategy_dataset_v4.csv")
    ext = ext.drop(columns=[c for c in FORBIDDEN if c in ext.columns])

    # Align columns
    common_cols = [c for c in train.columns
                   if c in ext.columns and c != ID_COL]
    print(f"common cols (train ∩ ext): {len(common_cols)} — {common_cols}")
    ext = ext[common_cols].copy()
    ext[ID_COL] = -1  # sentinel id for external rows
    ext = ext[train.columns]  # reorder to match

    train_aug = pd.concat([train, ext], ignore_index=True)
    print(f"train: {train.shape};  ext: {ext.shape};  augmented: {train_aug.shape}")

    y_aug = train_aug[TARGET].astype(int).values
    X_aug = train_aug.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X_aug.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X_aug[c] = X_aug[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    is_external = train_aug[ID_COL] == -1
    train_only_idx = np.where(~is_external)[0]  # indices into train_aug

    def run_anchor(name: str, splits_on_train_only):
        oof = np.zeros(len(train), dtype=np.float32)
        test_proba = np.zeros(len(X_test), dtype=np.float32)
        fold_scores = []
        for k, (tr_train_only, va_train_only) in enumerate(splits_on_train_only):
            # tr / va are indices into train. Build train-set = (train[tr]) + ALL external.
            tr_global = np.concatenate([
                train_only_idx[tr_train_only],
                np.where(is_external)[0],
            ])
            va_global = train_only_idx[va_train_only]
            dtrain = lgb.Dataset(X_aug.iloc[tr_global], y_aug[tr_global],
                                 categorical_feature=cat_cols)
            dval = lgb.Dataset(X_aug.iloc[va_global], y_aug[va_global],
                               categorical_feature=cat_cols)
            model = lgb.train(make_lgb_params(), dtrain, num_boost_round=2000,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(100),
                                         lgb.log_evaluation(0)])
            p_va = model.predict(X_aug.iloc[va_global])
            # Map back to train index space
            train_va_idx = va_train_only
            oof[train_va_idx] = p_va
            test_proba += model.predict(X_test) / N_FOLDS
            s = float(roc_auc_score(y_aug[va_global], p_va))
            fold_scores.append(s)
            print(f"  [{name}] fold {k}: AUC={s:.5f} (best_iter={model.best_iteration})")
        oof_auc = float(roc_auc_score(train[TARGET].astype(int).values, oof))
        return oof, test_proba, oof_auc, fold_scores, float(np.std(fold_scores))

    y_train = train[TARGET].astype(int).values
    print("=== Anchor A: StratifiedKFold(5, seed=42) on TRAIN ROWS ONLY ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y_train)), y_train))
    oof_a, test_a, auc_a, folds_a, std_a = run_anchor("STRAT", splits_a)

    print("=== Anchor B: GroupKFold(5) on Race ===")
    groups = train["Race"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y_train)), y_train, groups))
    oof_b, test_b, auc_b, folds_b, std_b = run_anchor("GROUP", splits_b)

    print()
    print(f"OOF_A (StratKFold): {auc_a:.5f}  fold_std={std_a:.5f}")
    print(f"OOF_B (GroupKF Race): {auc_b:.5f}  fold_std={std_b:.5f}")
    base_a = 0.94075
    base_b = 0.92059
    delta_a_bp = (auc_a - base_a) * 1e4
    delta_b_bp = (auc_b - base_b) * 1e4
    print(f"vs baseline_two_anchor: ΔA={delta_a_bp:+.1f}bp, ΔB={delta_b_bp:+.1f}bp")

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_d2c_concat_original.csv", index=False)
    save_oof("d2c_concat_original_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=std_a, fold_scores=folds_a,
                  cv="StratifiedKFold(5, seed=42)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_a_bp))
    save_oof("d2c_concat_original_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=std_b, fold_scores=folds_b,
                  cv="GroupKFold(5) on Race", metric="roc_auc",
                  delta_vs_baseline_bp=delta_b_bp))

    import datetime as dt
    out = Path(f"audit/{dt.date.today().isoformat()}-d2c-concat-original.md")
    out.write_text(
        f"# D2-C — concat-original-data baseline ({dt.date.today()})\n\n"
        f"External: aadigupta1601 dataset (101,371 rows, "
        f"`Normalized_TyreLife` dropped per host rule). Concatenated to "
        f"s6e5 train; OOF computed on s6e5 train rows only.\n\n"
        f"## Results\n\n"
        f"| anchor | OOF AUC | fold_std | per-fold | Δ vs baseline_two_anchor |\n"
        f"|---|---:|---:|---|---:|\n"
        f"| A — StratKFold | **{auc_a:.5f}** | {std_a:.5f} | "
        f"{[f'{x:.4f}' for x in folds_a]} | {delta_a_bp:+.1f}bp |\n"
        f"| B — GroupKFold(Race) | {auc_b:.5f} | {std_b:.5f} | "
        f"{[f'{x:.4f}' for x in folds_b]} | {delta_b_bp:+.1f}bp |\n\n"
    )
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
