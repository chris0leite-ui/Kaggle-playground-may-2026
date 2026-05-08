"""D5 Path B Phase 1 — pseudo-label gate + e3_hgbc MVP rebuild.

Goal: validate the pseudo-label channel on ONE fast base before
committing to a full 14-base rebuild (~3-4h CPU). The pseudo-label
thesis is HIGHEST-CEILING (30bp-class in prior comps when it lands)
but null-prone (can amplify M5q's systematic errors).

Gate (HANDOVER Risks)
  Track A — M5q confidence: M5q test proba >= 0.95 → label 1;
            M5q test proba <= 0.05 → label 0.
  Track B — multi-base vote: >= 10/13 M5h bases agree on >0.5 / <0.5.
  Final pseudo = union of A and B; rare conflicts resolved by M5q label.

MVP rebuild: e3_hgbc (anchor Strat OOF 0.94876, ~1min wall).
  Per fold (k, tr, va) of pinned StratifiedKFold(5, shuffle=True, seed=42):
    fit on train_real[tr] CONCAT all_pseudo_test (pseudo always in train)
    OOF: predict on train_real[va]  (real labels only — no leakage)
    test: average predict on full 188k test across folds
  AUC computed on real labels only (untouched).

Phase-1 decision rules (BOTH must hold to proceed to full rebuild)
  - rebuilt_oof > 0.94876 + 1bp  (1 standalone base shows pseudo signal)
  - test_rho(pseudo, orig) < 0.998 (pool rank structure meaningfully shifted)

Outputs
  scripts/artifacts/d5_pseudo_gate_stats.json
  scripts/artifacts/oof_d5_e3_hgbc_pseudo_strat.npy
  scripts/artifacts/test_d5_e3_hgbc_pseudo_strat.npy
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
E3_ANCHOR = 0.94876

M5H_TEST_NAMES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat",
    "cb_lossguide", "cb_slow-wide-bag",
]


def make_hgbc():
    return HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
        categorical_features="from_dtype",
    )


def prep_features(train: pd.DataFrame, test: pd.DataFrame):
    """Match e3_hgbc_two_anchor.py prep exactly so anchor comparison is honest."""
    HIGH_CARD = ["Driver"]
    LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in train.columns:
            uniq = pd.concat([train[c], test[c]], ignore_index=True).astype(str).unique()
            mapping = {v: i for i, v in enumerate(sorted(uniq))}
            train[c] = train[c].astype(str).map(mapping).astype(np.int32)
            test[c] = test[c].astype(str).map(mapping).astype(np.int32)
    for c in LOW_CARD:
        if c in train.columns:
            train[c] = train[c].astype("category")
            test[c] = test[c].astype("category")
    return train, test


def build_pseudo_gate(test_m5q: np.ndarray, m5h_test: np.ndarray):
    """Returns (pseudo_mask, pseudo_y, gate_stats).

    pseudo_mask: bool over 188k test rows, True if pseudo-labeled
    pseudo_y:    int8 array of labels (only valid where mask=True)
    """
    n_test = len(test_m5q)
    track_a_pos = test_m5q >= 0.95
    track_a_neg = test_m5q <= 0.05
    track_a = track_a_pos | track_a_neg

    # m5h_test shape: (n_test, 13). Vote: count of bases >0.5
    vote_pos = (m5h_test > 0.5).sum(axis=1)
    track_b_pos = vote_pos >= 10
    track_b_neg = vote_pos <= 3
    track_b = track_b_pos | track_b_neg

    pseudo_mask = track_a | track_b
    pseudo_y = np.zeros(n_test, dtype=np.int8)
    pseudo_y[track_a_pos | track_b_pos] = 1
    pseudo_y[track_a_neg | track_b_neg] = 0

    # Conflict detection (track A says X, track B says ~X)
    conflict = (
        (track_a_pos & track_b_neg) | (track_a_neg & track_b_pos)
    )
    # Resolve by M5q label (track A)
    pseudo_y[track_a_pos & track_b_neg] = 1
    pseudo_y[track_a_neg & track_b_pos] = 0

    n_pseudo = int(pseudo_mask.sum())
    n_pos = int((pseudo_y[pseudo_mask] == 1).sum())
    n_neg = int((pseudo_y[pseudo_mask] == 0).sum())
    stats = dict(
        n_test=n_test,
        n_pseudo=n_pseudo,
        n_pos=n_pos,
        n_neg=n_neg,
        pos_rate=n_pos / max(n_pseudo, 1),
        track_a_pos=int(track_a_pos.sum()),
        track_a_neg=int(track_a_neg.sum()),
        track_b_pos=int(track_b_pos.sum()),
        track_b_neg=int(track_b_neg.sum()),
        track_a_only=int((track_a & ~track_b).sum()),
        track_b_only=int((track_b & ~track_a).sum()),
        intersection=int((track_a & track_b).sum()),
        conflicts=int(conflict.sum()),
    )
    return pseudo_mask, pseudo_y, stats


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    y_real = train[TARGET].astype(int).values
    train_X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    test_X = test.drop(columns=[ID_COL], errors="ignore")
    train_X, test_X = prep_features(train_X, test_X)

    # Build pseudo gate
    test_m5q = np.load(ART / "test_m5q_strat.npy")[:, 1]
    m5h_test = np.column_stack([
        np.load(ART / f"test_{n}_strat.npy")[:, 1] for n in M5H_TEST_NAMES
    ])
    pseudo_mask, pseudo_y, stats = build_pseudo_gate(test_m5q, m5h_test)
    print(f"Pseudo gate stats:")
    for k, v in stats.items():
        print(f"  {k:<22s}: {v}")
    real_pos_rate = float(y_real.mean())
    print(f"  real_pos_rate (train) : {real_pos_rate:.4f}")
    print(f"  pseudo / test ratio   : {stats['n_pseudo']/stats['n_test']:.4f}")
    print(f"  pseudo / train ratio  : {stats['n_pseudo']/len(y_real):.4f}\n")

    (ART / "d5_pseudo_gate_stats.json").write_text(json.dumps(stats, indent=2))

    # Slice the pseudo-test rows for training augmentation
    test_pseudo_X = test_X.iloc[pseudo_mask].reset_index(drop=True)
    y_pseudo = pseudo_y[pseudo_mask]
    print(f"Pseudo augment shape: {test_pseudo_X.shape}, "
          f"y_pseudo balance={y_pseudo.mean():.4f}\n")

    # MVP rebuild — e3_hgbc with pseudo
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y_real)), y_real))

    oof = np.zeros(len(y_real), dtype=np.float32)
    tp = np.zeros(len(test), dtype=np.float32)
    fold_aucs, walls = [], []
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        # Concat real-train fold + ALL pseudo-test rows
        X_aug = pd.concat([train_X.iloc[tr], test_pseudo_X], ignore_index=True)
        y_aug = np.concatenate([y_real[tr], y_pseudo])
        m = make_hgbc()
        m.fit(X_aug, y_aug)
        # OOF on real labels only — no leakage
        oof[va] = m.predict_proba(train_X.iloc[va])[:, 1]
        tp += m.predict_proba(test_X)[:, 1] / N_FOLDS
        wall = time.time() - t_fold
        s = float(roc_auc_score(y_real[va], oof[va]))
        fold_aucs.append(s); walls.append(wall)
        print(f"  fold {k}: AUC={s:.5f}  iters={m.n_iter_}  "
              f"wall={wall:.1f}s  augN={len(y_aug)}")

    rebuilt_oof = float(roc_auc_score(y_real, oof))
    delta_anchor_bp = (rebuilt_oof - E3_ANCHOR) * 1e4

    # Diversity gate: ρ(rebuilt_test, original_e3_test)
    e3_orig_test = np.load(ART / "test_e3_hgbc_strat.npy")[:, 1]
    rho, _ = spearmanr(tp, e3_orig_test)

    print(f"\n=== Phase-1 decision metrics ===")
    print(f"e3_hgbc rebuilt Strat OOF: {rebuilt_oof:.5f}  "
          f"(anchor 0.94876)  Δ={delta_anchor_bp:+.2f}bp")
    print(f"ρ(test_rebuilt, test_orig): {rho:.5f}  "
          f"(target < 0.998)")

    gate_oof = rebuilt_oof > E3_ANCHOR + 1e-4
    gate_rho = rho < 0.998
    print(f"\nGate OOF (>+1bp):    {'PASS' if gate_oof else 'FAIL'}")
    print(f"Gate rho (<0.998):  {'PASS' if gate_rho else 'FAIL'}")
    proceed = gate_oof and gate_rho
    print(f"Proceed to full Path B rebuild: {'YES' if proceed else 'NO'}")

    np.save(ART / "oof_d5_e3_hgbc_pseudo_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d5_e3_hgbc_pseudo_strat.npy",
            np.column_stack([1 - tp, tp]))
    res = dict(
        rebuilt_oof=rebuilt_oof,
        delta_vs_anchor_bp=delta_anchor_bp,
        rho_vs_orig_test=float(rho),
        gate_oof_pass=bool(gate_oof),
        gate_rho_pass=bool(gate_rho),
        proceed=bool(proceed),
        fold_aucs=fold_aucs,
        fold_walls_s=walls,
        anchor_e3_oof=E3_ANCHOR,
        pseudo_stats=stats,
        cv="StratifiedKFold(5, shuffle=True, random_state=42)",
        metric="roc_auc",
        total_wall_s=time.time() - t0,
        notes=("Path B Phase 1 MVP. Pseudo-label channel validation on "
               "e3_hgbc only. Both gates must hold for full 14-base rebuild."),
    )
    (ART / "d5_pseudo_e3_hgbc_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nresults → scripts/artifacts/d5_pseudo_e3_hgbc_results.json")
    print(f"total wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
