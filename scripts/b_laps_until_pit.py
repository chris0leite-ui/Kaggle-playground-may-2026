"""B — Regression reformulation: predict LapsUntilPit.

Train HGBR on log(LapsUntilPit + 1) within (Driver, Race). LapsUntilPit
is laps remaining until next PitNextLap=1; if no future pit, use
race-end lap (i.e. retired or final-stint). Output is converted to
a monotonic [0, 1] feature for the M5c stack via 1/(1+predicted_laps).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
NEW_TARGET = "LapsUntilPit"
BASE_S, BASE_G = 0.94075, 0.92059
LAPS_CAP = 30  # cap for regression target stability


def make_hgbr():
    return HistGradientBoostingRegressor(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
        categorical_features="from_dtype",
    )


def build_laps_until_pit(train: pd.DataFrame) -> np.ndarray:
    """For each row, count laps until next PitNextLap=1 within (Driver, Race).
    If no future pit, use (last_lap_of_race_for_driver - LapNumber). Capped at LAPS_CAP.
    Returns log(LapsUntilPit + 1).
    """
    df = train.reset_index().rename(columns={"index": "_orig_idx"})
    df = df.sort_values(["Race", "Driver", "LapNumber"], kind="stable").reset_index(drop=True)

    laps_until_pit = np.full(len(df), -1, dtype=np.int32)
    laps_remaining = np.full(len(df), -1, dtype=np.int32)
    # Compute last_lap per (Race, Driver)
    last_lap = df.groupby(["Race", "Driver"])["LapNumber"].transform("max").values
    laps_remaining = (last_lap - df["LapNumber"].values).astype(np.int32)

    # For each (Race, Driver) group, walk laps and find next pit
    pn = df[TARGET].values
    grp_id = df.groupby(["Race", "Driver"]).ngroup().values
    n = len(df)
    next_pit_lap = np.full(n, -1, dtype=np.int32)
    # Process in reverse within each group: track most-recent future pit
    last_pit_idx_seen = -1
    last_grp = -1
    for i in range(n - 1, -1, -1):
        if grp_id[i] != last_grp:
            last_pit_idx_seen = -1
            last_grp = grp_id[i]
        if pn[i] == 1:
            last_pit_idx_seen = i
        if last_pit_idx_seen != -1:
            next_pit_lap[i] = df["LapNumber"].iloc[last_pit_idx_seen]

    cur_lap = df["LapNumber"].values
    has_future_pit = (next_pit_lap >= 0)
    laps_until_pit = np.where(has_future_pit,
                               np.maximum(next_pit_lap - cur_lap, 0),
                               laps_remaining)
    laps_until_pit = np.clip(laps_until_pit, 0, LAPS_CAP)
    df["_lup"] = laps_until_pit
    df = df.sort_values("_orig_idx", kind="stable")
    return np.log(df["_lup"].values + 1.0).astype(np.float32)


def encode_for_hgbr(X, X_test):
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


def run_anchor(name, splits, X, y_log_lup, y_orig, X_test):
    oof_log = np.zeros(len(y_orig), dtype=np.float32)
    tp_log = np.zeros(len(X_test), dtype=np.float32)
    fs_orig, walls = [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        m = make_hgbr()
        m.fit(X.iloc[tr], y_log_lup[tr])
        oof_log[va] = m.predict(X.iloc[va])
        tp_log += m.predict(X_test) / N_FOLDS
        wall = time.time() - t0
        # Project: laps_pred = exp(log_laps_pred) - 1; pit_proba_proxy = 1/(1+laps_pred)
        proba_va = 1.0 / (1.0 + np.maximum(np.expm1(oof_log[va]), 0.0))
        s_o = float(roc_auc_score(y_orig[va], proba_va))
        fs_orig.append(s_o); walls.append(wall)
        print(f"  [{name}] f{k}: AUC_orig={s_o:.5f} wall={wall:.1f}s")
    # Convert full OOF to proba
    oof_proba = 1.0 / (1.0 + np.maximum(np.expm1(oof_log), 0.0))
    tp_proba = 1.0 / (1.0 + np.maximum(np.expm1(tp_log), 0.0))
    return oof_proba, tp_proba, fs_orig, walls


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    y_orig = train[TARGET].astype(int).values
    y_log_lup = build_laps_until_pit(train)
    print(f"log(LapsUntilPit+1): mean={y_log_lup.mean():.3f} std={y_log_lup.std():.3f} "
          f"min={y_log_lup.min():.3f} max={y_log_lup.max():.3f}")

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    X, X_test = encode_for_hgbr(X, X_test)

    print("=== Anchor A: StratKFold(5) on PitNextLap ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y_orig)), y_orig))
    oof_a, test_a, fo_a, w_a = run_anchor("STRAT", splits_a, X, y_log_lup, y_orig, X_test)
    auc_a = float(roc_auc_score(y_orig, oof_a))

    print("=== Anchor B: GroupKFold(5) on Race ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y_orig)), y_orig, train["Race"].values))
    oof_b, test_b, fo_b, w_b = run_anchor("GROUP", splits_b, X, y_log_lup, y_orig, X_test)
    auc_b = float(roc_auc_score(y_orig, oof_b))

    da = (auc_a - BASE_S) * 1e4
    db = (auc_b - BASE_G) * 1e4
    total = time.time() - t0
    print(f"\nStrat OOF (proba proxy): {auc_a:.5f}  Δ baseline={da:+.1f}bp")
    print(f"GroupKF OOF (proba proxy): {auc_b:.5f}  Δ baseline={db:+.1f}bp")

    save_oof("b_lapsuntilpit_strat",
             np.column_stack([1 - oof_a, oof_a]),
             np.column_stack([1 - test_a, test_a]),
             dict(oof_score=auc_a, fold_scores=fo_a, cv="StratKF",
                  delta_vs_baseline_bp=da,
                  target_used="log(LapsUntilPit+1) regression",
                  proba_proxy="1/(1+exp(log_laps_pred)-1)"))
    save_oof("b_lapsuntilpit_groupkf",
             np.column_stack([1 - oof_b, oof_b]),
             np.column_stack([1 - test_b, test_b]),
             dict(oof_score=auc_b, fold_scores=fo_b, cv="GroupKF(Race)",
                  delta_vs_baseline_bp=db,
                  target_used="log(LapsUntilPit+1) regression"))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_a
    sub.to_csv("submissions/submission_b_lapsuntilpit.csv", index=False)

    body = (
        f"# B — LapsUntilPit regression (2026-05-04)\n\n"
        f"HGBR on log(LapsUntilPit+1), output projected via 1/(1+laps_pred).\n\n"
        f"| anchor | OOF AUC | Δ baseline |\n|---|---:|---:|\n"
        f"| Strat | **{auc_a:.5f}** | {da:+.1f}bp |\n"
        f"| GroupKF | **{auc_b:.5f}** | {db:+.1f}bp |\n\n"
        f"Wall: {total:.0f}s.\n"
    )
    Path("audit/2026-05-04-b-laps-until-pit.md").write_text(body)
    print("audit written")


if __name__ == "__main__":
    main()
