"""A — Horizon-shift reformulation: predict PitInNext3Laps.

Train HGBC on the broader-window label PitInNext3Laps[t] =
OR(PitNextLap[t], PitNextLap[t+1], PitNextLap[t+2]) within
(Driver, Race) sequences. The positive class density shifts from
~20% to ~50%, exposing weaker stint-late patterns. Predicted prob
is used as a base in M5c stack (LR meta will learn its mapping
back to PitNextLap).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
NEW_TARGET = "PitInNext3Laps"
BASE_S, BASE_G = 0.94075, 0.92059


def make_hgbc():
    return HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
        categorical_features="from_dtype",
    )


def build_horizon_target(train: pd.DataFrame) -> np.ndarray:
    """Compute PitInNext3Laps[t] within each (Driver, Race), original row order preserved."""
    df = train.reset_index().rename(columns={"index": "_orig_idx"})
    df = df.sort_values(["Race", "Driver", "LapNumber"], kind="stable")
    df["_pn1"] = df.groupby(["Race", "Driver"])[TARGET].shift(-1).fillna(0).astype(int)
    df["_pn2"] = df.groupby(["Race", "Driver"])[TARGET].shift(-2).fillna(0).astype(int)
    df[NEW_TARGET] = ((df[TARGET] == 1) | (df["_pn1"] == 1) | (df["_pn2"] == 1)).astype(int)
    df = df.sort_values("_orig_idx", kind="stable")
    return df[NEW_TARGET].values


def encode_for_hgbc(X, X_test):
    HIGH_CARD = ["Driver"]
    LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True).astype(str).unique()
            mapping = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mapping).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mapping).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")
    return X, X_test


def run_anchor(name, splits, X, y_new, y_orig, X_test):
    """Train on y_new (horizon target), evaluate on y_orig (PitNextLap)."""
    oof = np.zeros(len(y_orig), dtype=np.float32)
    tp = np.zeros(len(X_test), dtype=np.float32)
    fs_horizon, fs_orig, walls = [], [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        m = make_hgbc()
        m.fit(X.iloc[tr], y_new[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        tp += m.predict_proba(X_test)[:, 1] / N_FOLDS
        wall = time.time() - t0
        s_h = float(roc_auc_score(y_new[va], p))  # AUC on horizon target
        s_o = float(roc_auc_score(y_orig[va], p))  # AUC vs original target (what we care about)
        fs_horizon.append(s_h); fs_orig.append(s_o); walls.append(wall)
        print(f"  [{name}] f{k}: AUC_horizon={s_h:.5f} AUC_orig={s_o:.5f} wall={wall:.1f}s")
    return oof, tp, fs_horizon, fs_orig, walls


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    y_orig = train[TARGET].astype(int).values
    y_new = build_horizon_target(train)
    print(f"horizon target ({NEW_TARGET}) prior: {y_new.mean():.4f}  "
          f"(orig {TARGET} prior: {y_orig.mean():.4f})")

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    X, X_test = encode_for_hgbc(X, X_test)

    print("=== Anchor A: StratifiedKFold(5, seed=42) — strat on PitNextLap ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y_orig)), y_orig))
    oof_a, test_a, fh_a, fo_a, w_a = run_anchor("STRAT", splits_a, X, y_new, y_orig, X_test)
    auc_a_orig = float(roc_auc_score(y_orig, oof_a))
    auc_a_new = float(roc_auc_score(y_new, oof_a))

    print("=== Anchor B: GroupKFold(5) on Race ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y_orig)), y_orig, train["Race"].values))
    oof_b, test_b, fh_b, fo_b, w_b = run_anchor("GROUP", splits_b, X, y_new, y_orig, X_test)
    auc_b_orig = float(roc_auc_score(y_orig, oof_b))
    auc_b_new = float(roc_auc_score(y_new, oof_b))

    da = (auc_a_orig - BASE_S) * 1e4
    db = (auc_b_orig - BASE_G) * 1e4
    total = time.time() - t0
    print(f"\nStrat OOF on ORIG target: {auc_a_orig:.5f}  Δ baseline={da:+.1f}bp")
    print(f"GroupKF OOF on ORIG target: {auc_b_orig:.5f}  Δ baseline={db:+.1f}bp")
    print(f"(horizon-target AUC: Strat {auc_a_new:.5f}, GroupKF {auc_b_new:.5f})")
    print(f"total wall: {total:.0f}s")

    save_oof("a_horizon_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a_orig, oof_score_horizon=auc_a_new,
                  fold_scores_orig=fo_a, fold_scores_horizon=fh_a,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=da, target_used="PitInNext3Laps"))
    save_oof("a_horizon_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b_orig, oof_score_horizon=auc_b_new,
                  fold_scores_orig=fo_b, fold_scores_horizon=fh_b,
                  cv="GroupKFold(Race)", metric="roc_auc",
                  delta_vs_baseline_bp=db, target_used="PitInNext3Laps"))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_a
    sub.to_csv("submissions/submission_a_horizon.csv", index=False)

    body = (
        f"# A — Horizon-shift reformulation (PitInNext3Laps) (2026-05-04)\n\n"
        f"Train HGBC on PitInNext3Laps[t] = OR(PN[t], PN[t+1], PN[t+2]); "
        f"evaluate raw output as proxy for PitNextLap. Horizon prior {y_new.mean():.3f} "
        f"vs orig {y_orig.mean():.3f}.\n\n"
        f"## Two-anchor results (AUC vs ORIGINAL target)\n\n"
        f"| anchor | OOF AUC orig | OOF AUC horizon | Δ vs baseline |\n"
        f"|---|---:|---:|---:|\n"
        f"| Strat | **{auc_a_orig:.5f}** | {auc_a_new:.5f} | {da:+.1f}bp |\n"
        f"| GroupKF | **{auc_b_orig:.5f}** | {auc_b_new:.5f} | {db:+.1f}bp |\n\n"
        f"Wall: {total:.0f}s. Held for M5c stack refit.\n"
    )
    Path("audit/2026-05-04-a-horizon-shift.md").write_text(body)
    print("audit written")


if __name__ == "__main__":
    main()
