"""d4_naive_bayes — GaussianNB base for stack diversity.

Generative classifier with feature-independence assumption — structurally
different from every base in the M5q pool (all are discriminative GBDT/NN).
Standalone OOF will be modest (~0.85-0.90 expected), but diversity is the
play; goal is L1 contribution in stack.

Feature handling:
  numerics  → QuantileTransformer (output=normal) → fits Gaussian likelihood.
  Year, Compound, Stint  → one-hot (low-card, full-vocab from train∪test).
  Driver, Race  → within-fold target-encoded with smoothing (α=20).
  PitStop kept as-is (binary numeric → near-Bernoulli, OK for GaussianNB).

5-fold Strat (Rule R1). CPU-light → no smoke gate needed (Rule 2 is for
heavy compute). Output: oof_d4_nb_strat.npy + test_d4_nb_strat.npy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
NUM_COLS = ["LapNumber", "TyreLife", "Position", "LapTime (s)",
            "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
            "Position_Change", "PitStop"]
LOW_CARD_CAT = ["Year", "Compound", "Stint"]
HIGH_CARD_CAT = ["Driver", "Race"]
BASE_S = 0.94075
M5Q_S = 0.95057


def target_encode(tr_y, tr_col, va_col, test_col, alpha=20.0):
    """Smoothed mean-target encoding fit on TR; applied to VA, test."""
    global_mean = float(tr_y.mean())
    tmp = pd.DataFrame({"k": tr_col, "y": tr_y})
    grp = tmp.groupby("k")["y"].agg(["sum", "count"])
    enc = (grp["sum"] + global_mean * alpha) / (grp["count"] + alpha)
    return (va_col.map(enc).fillna(global_mean).to_numpy(),
            test_col.map(enc).fillna(global_mean).to_numpy())


def build_feature_matrix(tr_df, va_df, test_df, y_tr):
    """Per-fold feature build. Returns (X_tr, X_va, X_test) numpy arrays."""
    blocks_tr, blocks_va, blocks_test = [], [], []

    # Numerics → QuantileTransformer (rank-Gauss).
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, len(tr_df)),
                             random_state=SEED, subsample=200_000)
    qt.fit(tr_df[NUM_COLS].values)
    blocks_tr.append(qt.transform(tr_df[NUM_COLS].values))
    blocks_va.append(qt.transform(va_df[NUM_COLS].values))
    blocks_test.append(qt.transform(test_df[NUM_COLS].values))

    # Low-card cats → one-hot using full-vocab from train+test union.
    for c in LOW_CARD_CAT:
        vocab = pd.Index(sorted(set(tr_df[c]) | set(va_df[c]) | set(test_df[c])))
        for v in vocab:
            blocks_tr.append((tr_df[c].values == v).astype(np.float32)[:, None])
            blocks_va.append((va_df[c].values == v).astype(np.float32)[:, None])
            blocks_test.append((test_df[c].values == v).astype(np.float32)[:, None])

    # High-card cats → smoothed within-fold target encoding.
    for c in HIGH_CARD_CAT:
        te_va, te_test = target_encode(y_tr, tr_df[c], va_df[c], test_df[c])
        # For TR fold, use a leave-one-out approximation (fit-mean is fine here
        # since GaussianNB only uses TR statistics; TR LOO not strictly needed).
        te_tr, _ = target_encode(y_tr, tr_df[c], tr_df[c], test_df[c])
        blocks_tr.append(te_tr[:, None].astype(np.float32))
        blocks_va.append(te_va[:, None].astype(np.float32))
        blocks_test.append(te_test[:, None].astype(np.float32))

    return (np.hstack(blocks_tr).astype(np.float32),
            np.hstack(blocks_va).astype(np.float32),
            np.hstack(blocks_test).astype(np.float32))


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float32)
    test_p = np.zeros(len(test), dtype=np.float64)
    scores, walls = [], []
    t0_total = time.time()
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        tr_df, va_df = train.iloc[tr], train.iloc[va]
        X_tr, X_va, X_te = build_feature_matrix(tr_df, va_df, test, y[tr])
        m = GaussianNB(var_smoothing=1e-7)
        m.fit(X_tr, y[tr])
        p_va = m.predict_proba(X_va)[:, 1]
        p_te = m.predict_proba(X_te)[:, 1]
        oof[va] = p_va
        test_p += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        scores.append(s); walls.append(time.time() - t0)
        print(f"  [nb/strat] f{k}: AUC={s:.5f} wall={walls[-1]:.1f}s")

    auc = float(roc_auc_score(y, oof))
    delta = (auc - BASE_S) * 1e4
    delta_m5q = (auc - M5Q_S) * 1e4
    print(f"\n[nb/strat] OOF={auc:.5f}  Δbase={delta:+.1f}bp  "
          f"Δm5q={delta_m5q:+.1f}bp  sd={np.std(scores):.5f}  "
          f"total={time.time()-t0_total:.1f}s")

    save_oof("d4_nb_strat",
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_p, test_p]),
             dict(oof_score=auc, fold_std=float(np.std(scores)),
                  fold_scores=scores, cv="strat", metric="roc_auc",
                  delta_vs_baseline_bp=delta, delta_vs_m5q_bp=delta_m5q,
                  num_cols=NUM_COLS, low_card_cat=LOW_CARD_CAT,
                  high_card_cat=HIGH_CARD_CAT, te_alpha=20.0,
                  fold_walls=walls))
    Path("scripts/artifacts/d4_naive_bayes.json").write_text(
        json.dumps(dict(oof=auc, delta_bp=delta, delta_m5q_bp=delta_m5q,
                        std=float(np.std(scores)), walls=walls), indent=2))


if __name__ == "__main__":
    main()
