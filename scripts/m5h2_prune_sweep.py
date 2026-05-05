"""M5h-v2 — tier-break L1 prune sweep on M5h pool.

M5h L1 ranking (Strat):
  b_lapsuntilpit       0.718
  e5_optuna_lgbm       0.670
  cb_year-cat          0.375
  baseline             0.373
  f1_hgbc_deep         0.341
  e3_hgbc              0.323
  e1_cb_sub            0.301
  cb_lossguide         0.283
  cb_slow-wide-bag     0.282
  f2_hgbc_shallow      0.237   ← potential cut
  d2a_te               0.220   ← potential cut
  m2_xgb               0.200   ← potential cut
  a_horizon            0.154   ← lowest

Hypothesis: shrinking M5h beyond the L1 tier-break (~0.20) may
tighten OOF→LB gap (M5b 7-base gap −3.5bp vs M5h 13-base gap −5.2bp).
M5i's median rule (drop below 0.301) was too aggressive; this sweep
tests finer cuts.

Sweep:
  v0 = full M5h (13)         — control
  v1 = drop a_horizon (12)   — lowest single
  v2 = drop a + m2_xgb (11)
  v3 = drop a + m2_xgb + d2a_te (10)
  v4 = drop a + m2 + d2a + f2 (9)

Pick the variant maximizing Strat OOF; if tied within 1bp, pick smaller pool.
R1: Strat-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5H_AGG_S = 0.95043
SEED, N_FOLDS = 42, 5

POOL_FULL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"),
    ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
]


def load(name, suffix):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rank, logit])


def fit_meta(F_oof, F_test, y, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc, lr_full.coef_.ravel()


def assemble(pool, suffix="strat"):
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool:
        oo, te = load(name, suffix)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    return expand(np.column_stack(Xs_oof)), expand(np.column_stack(Xs_test)), names


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    drops_by_variant = {
        "v0_full13": [],
        "v1_drop_a": ["a_horizon"],
        "v2_drop_a_m2": ["a_horizon", "m2_xgb"],
        "v3_drop_a_m2_d2a": ["a_horizon", "m2_xgb", "d2a_te"],
        "v4_drop_a_m2_d2a_f2": ["a_horizon", "m2_xgb", "d2a_te", "f2_hgbc_shallow"],
    }

    results = {}
    for v_label, drops in drops_by_variant.items():
        pool = [(lbl, name) for (lbl, name) in POOL_FULL if lbl not in drops]
        F_oof, F_test, names = assemble(pool, "strat")
        oof, test_p, auc, coef = fit_meta(F_oof, F_test, y)
        K = len(names)
        l1 = {n: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2 * K + i]))
              for i, n in enumerate(names)}
        print(f"  {v_label:<22s} K={K:>2d}  Strat={auc:.5f}  "
              f"Δ M5h={(auc - M5H_AGG_S)*1e4:+.1f}bp")
        results[v_label] = dict(K=K, strat=auc,
                                delta_m5h_bp=(auc - M5H_AGG_S) * 1e4,
                                pool=names, dropped=drops, l1=l1,
                                oof=oof, test=test_p)

    # Pick winner: highest strat AUC; tie-break smaller K
    sorted_r = sorted(results.items(), key=lambda kv: (-kv[1]["strat"], kv[1]["K"]))
    winner_label, winner = sorted_r[0]
    print(f"\n=== M5h-v2 WINNER: {winner_label} ===")
    print(f"  K={winner['K']}  Strat={winner['strat']:.5f}  "
          f"Δ M5h={winner['delta_m5h_bp']:+.1f}bp")
    print(f"  Dropped: {winner['dropped']}")
    print("  L1 per surviving base:")
    for n, v in sorted(winner["l1"].items(), key=lambda x: -x[1]):
        print(f"    {n:<22s} L1={v:.3f}")

    np.save(ART / "oof_m5h2_strat.npy",
            np.column_stack([1 - winner["oof"], winner["oof"]]))
    np.save(ART / "test_m5h2_strat.npy",
            np.column_stack([1 - winner["test"], winner["test"]]))

    sub = sample_sub.copy()
    sub[TARGET] = winner["test"]
    sub.to_csv("submissions/submission_m5h2_lr_meta.csv", index=False)

    # Persist all variants (sans heavy oof/test arrays)
    summary = {
        v_label: {k: v for k, v in res.items() if k not in ("oof", "test")}
        for v_label, res in results.items()
    }
    summary["winner"] = winner_label
    (ART / "m5h2_prune_sweep_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/m5h2_prune_sweep_results.json")
    print(f"→ submissions/submission_m5h2_lr_meta.csv (held)")


if __name__ == "__main__":
    main()
