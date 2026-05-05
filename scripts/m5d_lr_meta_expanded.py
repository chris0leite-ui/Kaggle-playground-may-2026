"""M5d — LR meta with M5c pool + β HGBC variants (+ optional ζ).

Pool: M5c 10-base + f1_hgbc_deep + f2_hgbc_shallow (12-base).
Optionally include zeta_catboost_deep_5fold if it ran a 5-fold pursuit.
Same LR(C=1.0) on [raw, rank, logit] recipe.
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
BASE_S, BASE_G = 0.94075, 0.92059
SEED, N_FOLDS = 42, 5

POOL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("m3_catboost", "m3_catboost"),
    ("m4_relstate", "m4_relstate"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    # Add zeta_catboost_deep below if 5-fold artifacts exist
]


def load(name, suffix):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) / (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rank, logit])


def run(suffix, base_auc):
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    Xs_oof, Xs_test, names = [], [], []
    for label, name in POOL:
        try:
            oo, te = load(name, suffix)
            Xs_oof.append(oo); Xs_test.append(te); names.append(label)
            print(f"  + {label} ({suffix}): OOF AUC {roc_auc_score(y, oo):.5f}")
        except FileNotFoundError:
            print(f"  SKIP {label} ({suffix}): file not found")

    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    print(f"  M5d OOF: {auc:.5f}  Δbase={(auc - base_auc) * 1e4:+.1f}bp  K={len(names)}")

    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc, names


def main():
    print("=== M5d — Strat ===")
    oof_s, test_s, auc_s, names_s = run("strat", BASE_S)
    print("\n=== M5d — GroupKF ===")
    oof_g, test_g, auc_g, _ = run("groupkf", BASE_G)

    np.save(ART / "oof_m5d_lr_meta_strat.npy", np.column_stack([1 - oof_s, oof_s]))
    np.save(ART / "test_m5d_lr_meta_strat.npy", np.column_stack([1 - test_s, test_s]))
    np.save(ART / "oof_m5d_lr_meta_groupkf.npy", np.column_stack([1 - oof_g, oof_g]))
    np.save(ART / "test_m5d_lr_meta_groupkf.npy", np.column_stack([1 - test_g, test_g]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv("submissions/submission_m5d_lr_meta_expanded.csv", index=False)

    res = dict(strat=dict(oof=auc_s, delta_bp=(auc_s - BASE_S) * 1e4),
               groupkf=dict(oof=auc_g, delta_bp=(auc_g - BASE_G) * 1e4),
               pool=names_s, K=len(names_s))
    (ART / "m5d_lr_meta_results.json").write_text(json.dumps(res, indent=2))

    body = (
        f"# M5d — LR meta with M5c + β HGBC variants (2026-05-04)\n\n"
        f"Pool ({len(names_s)}): {names_s}\n\n"
        f"| anchor | M5d | M5c | Δ vs M5c | Δ vs base |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| Strat | **{auc_s:.5f}** | 0.95000 | "
        f"{(auc_s - 0.95000) * 1e4:+.1f}bp | {(auc_s - BASE_S) * 1e4:+.1f}bp |\n"
        f"| GroupKF | **{auc_g:.5f}** | 0.92963 | "
        f"{(auc_g - 0.92963) * 1e4:+.1f}bp | {(auc_g - BASE_G) * 1e4:+.1f}bp |\n"
    )
    Path("audit/2026-05-04-m5d-lr-meta-expanded.md").write_text(body)
    print("audit written")


if __name__ == "__main__":
    main()
