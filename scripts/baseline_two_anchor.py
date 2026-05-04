"""U1 probe — two-anchor 5-fold baseline LGBM.

Anchor A: StratifiedKFold(5, seed=42)        — public-notebook norm
Anchor B: GroupKFold(5) on `Race` (26 races) — leakage-honest holdout

Same LGBM hyperparams as baseline_lgbm.py. Each fold trains on
~80% of data, predicts held-out 20% (OOF) and the full test set.
Test predictions averaged across folds. Submission CSV built from
the StratifiedKFold anchor (matches public-notebook norm so our
LB gap is directly comparable).

R1 verdict: |OOF_A − OOF_B| > 50bp ⇒ leakage flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import ART, N_FOLDS, SEED, save_oof

TARGET = "PitNextLap"
ID_COL = "id"


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


def run_anchor(name: str, splits, X, y, X_test, cat_cols):
    """Run one CV scheme; return (oof_proba, test_proba, fold_scores)."""
    oof = np.zeros(len(y), dtype=np.float32)
    test_proba = np.zeros(len(X_test), dtype=np.float32)
    fold_scores = []
    for k, (tr, va) in enumerate(splits):
        dtrain = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dval = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
        model = lgb.train(
            make_lgb_params(), dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        p_va = model.predict(X.iloc[va])
        oof[va] = p_va
        test_proba += model.predict(X_test) / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        fold_scores.append(s)
        print(f"  [{name}] fold {k}: AUC={s:.5f}  (best_iter={model.best_iteration})")
    oof_auc = float(roc_auc_score(y, oof))
    fold_std = float(np.std(fold_scores))
    return oof, test_proba, oof_auc, fold_scores, fold_std


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # Anchor A: StratifiedKFold
    print("=== Anchor A: StratifiedKFold(5, seed=42) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y)), y))
    oof_a, test_a, auc_a, folds_a, std_a = run_anchor(
        "STRAT", splits_a, X, y, X_test, cat_cols)

    # Anchor B: GroupKFold on Race
    print("=== Anchor B: GroupKFold(5) on Race ===")
    groups = train["Race"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, groups))
    oof_b, test_b, auc_b, folds_b, std_b = run_anchor(
        "GROUP", splits_b, X, y, X_test, cat_cols)

    gap = auc_a - auc_b
    print()
    print(f"OOF_A (StratKFold): {auc_a:.5f}  fold_std={std_a:.5f}")
    print(f"OOF_B (GroupKF Race): {auc_b:.5f}  fold_std={std_b:.5f}")
    print(f"Gap A−B: {gap:+.5f}  ({gap*1e4:+.1f}bp)")
    if abs(gap) > 0.005:
        verdict = ("LEAKAGE FLAG: gap > 50bp. StratKFold OOF likely "
                   "leakage-inflated. Public-LB-driven decisions risky.")
    else:
        verdict = ("CV trust OK: anchors agree within 50bp. Public-LB "
                   "is a reasonable proxy.")
    print(f"R1 verdict: {verdict}")

    # Build submission from anchor A (matches public-notebook norm)
    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_a
    sample_sub.to_csv("submissions/submission_baseline_two_anchor.csv",
                      index=False)

    # Save artifacts
    save_oof("baseline_two_anchor_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_std=std_a, fold_scores=folds_a,
                  cv="StratifiedKFold(5, seed=42)", metric="roc_auc"))
    save_oof("baseline_two_anchor_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_std=std_b, fold_scores=folds_b,
                  cv="GroupKFold(5) on Race", metric="roc_auc"))

    # Comparison report
    import datetime as dt
    out = Path(f"audit/{dt.date.today().isoformat()}-u1-two-anchor-baseline.md")
    out.write_text(
        f"# U1 — two-anchor baseline LGBM ({dt.date.today()})\n\n"
        f"Anchors:\n"
        f"- A: StratifiedKFold(5, seed=42)\n"
        f"- B: GroupKFold(5) on Race ({train['Race'].nunique()} levels)\n\n"
        f"Hyperparams: lr=0.05, num_leaves=63, "
        f"min_data_in_leaf=200, num_boost_round≤2000, ES=100\n\n"
        f"## Results\n\n"
        f"| anchor | OOF AUC | fold_std | per-fold |\n"
        f"|---|---:|---:|---|\n"
        f"| A — StratKFold | **{auc_a:.5f}** | {std_a:.5f} | "
        f"{[f'{x:.4f}' for x in folds_a]} |\n"
        f"| B — GroupKFold(Race) | **{auc_b:.5f}** | {std_b:.5f} | "
        f"{[f'{x:.4f}' for x in folds_b]} |\n"
        f"| gap A−B | {gap:+.5f} ({gap*1e4:+.1f}bp) | | |\n\n"
        f"## R1 verdict\n\n{verdict}\n"
    )
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
