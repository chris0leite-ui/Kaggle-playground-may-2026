"""D3-D — Per-Stint-2 LR meta (stack-on-segment).

Hypothesis: M5h's LR meta weights are globally optimized; on the
Stint=2 subset (worst large-segment AUC = 0.916), a meta fit ONLY
on Stint=2 OOFs may find better weights for that distribution.

Re-uses M5h's 13 base OOFs (raw+rank+logit expansion) but trains
the LR on Stint=2 train rows only. At test time, applies the
Stint-2 meta to Stint=2 test rows; M5h to the rest. Compute blend
OOF AUC vs M5h baseline.

If blend ≥+5bp Strat OOF → slot-7 candidate.

Inner-CV check on Stint=2: 5-fold split of S2 train rows; fit S2 meta
on 4 inner folds, eval on 5th. Inner-CV AUC is the honest
generalization estimate.

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

POOL = [
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


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    s2_train_mask = (train["Stint"] == 2).values
    s2_test_mask = (test["Stint"] == 2).values
    s2_train_idx = np.where(s2_train_mask)[0]
    s2_test_idx = np.where(s2_test_mask)[0]

    # Load all 13 base OOFs + tests (Strat anchor)
    Xs_oof, Xs_test = [], []
    for label, name in POOL:
        oo, te = load(name, "strat")
        Xs_oof.append(oo); Xs_test.append(te)
    P_oof = np.column_stack(Xs_oof)
    P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof)
    F_test = expand(P_test)

    m5h_oof = np.load(ART / "oof_m5h_strat.npy")[:, 1].astype(np.float64)
    m5h_test = np.load(ART / "test_m5h_strat.npy")[:, 1].astype(np.float64)

    auc_m5h_s2 = float(roc_auc_score(y[s2_train_mask], m5h_oof[s2_train_mask]))
    print(f"M5h OOF AUC on Stint=2: {auc_m5h_s2:.5f}")

    # === Step 1: in-sample S2 LR meta (sanity baseline; expect overfit) ===
    print("\n=== In-sample S2 LR meta (overfit upper bound) ===")
    F_s2 = F_oof[s2_train_idx]
    y_s2 = y[s2_train_idx]
    skf_s2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_s2_oof = np.zeros(len(s2_train_idx), dtype=np.float64)
    for tr, va in skf_s2.split(np.zeros(len(s2_train_idx)), y_s2):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_s2[tr], y_s2[tr])
        meta_s2_oof[va] = lr.predict_proba(F_s2[va])[:, 1]
    auc_s2_meta = float(roc_auc_score(y_s2, meta_s2_oof))
    print(f"S2 LR meta within-S2 OOF: {auc_s2_meta:.5f}  "
          f"Δ M5h-on-S2={(auc_s2_meta - auc_m5h_s2)*1e4:+.1f}bp")

    # Fit full S2 meta and apply to S2 test
    lr_full_s2 = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full_s2.fit(F_s2, y_s2)
    s2_test_proba = lr_full_s2.predict_proba(F_test[s2_test_idx])[:, 1]

    # Blend
    blend_oof = m5h_oof.copy()
    blend_oof[s2_train_mask] = meta_s2_oof
    blend_test = m5h_test.copy()
    blend_test[s2_test_idx] = s2_test_proba
    auc_blend = float(roc_auc_score(y, blend_oof))
    print(f"Blend (S2 meta on S2, M5h elsewhere): {auc_blend:.5f}  "
          f"Δ M5h={(auc_blend - M5H_AGG_S)*1e4:+.1f}bp")

    # Convex blend variants
    for w in [0.3, 0.5, 0.7, 1.0]:
        blend_v = m5h_oof.copy()
        blend_v[s2_train_mask] = w * meta_s2_oof + (1 - w) * m5h_oof[s2_train_mask]
        auc_v = float(roc_auc_score(y, blend_v))
        print(f"  w_S2-meta={w:.1f}: {auc_v:.5f}  Δ M5h={(auc_v - M5H_AGG_S)*1e4:+.1f}bp")

    # Save artifacts (best convex blend will be picked after viewing)
    np.save(ART / "oof_d3d_s2meta_strat.npy",
            np.column_stack([1 - blend_oof, blend_oof]))
    np.save(ART / "test_d3d_s2meta_strat.npy",
            np.column_stack([1 - blend_test, blend_test]))

    sub = sample_sub.copy()
    sub[TARGET] = blend_test
    sub.to_csv("submissions/submission_d3d_s2meta_blend.csv", index=False)

    res = dict(
        m5h_oof_within_s2=auc_m5h_s2,
        s2_meta_within_s2=auc_s2_meta,
        delta_s2meta_vs_m5h_on_s2_bp=(auc_s2_meta - auc_m5h_s2) * 1e4,
        blend_oof=auc_blend,
        delta_blend_vs_m5h_bp=(auc_blend - M5H_AGG_S) * 1e4,
    )
    (ART / "d3d_stint2_meta_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n→ scripts/artifacts/d3d_stint2_meta_results.json")


if __name__ == "__main__":
    main()
