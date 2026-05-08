"""M5h — M5f pool minus the 2 lowest-L1coef bases (m3_catboost, m4_relstate).

Per H3 sweep, top-13-by-L1coef is the only prune strategy that doesn't
lose Strat OOF vs M5f (Strat 0.95044 vs M5f 0.95042, +0.2bp). Dropping
the 2 dead-weight bases (lowest L1 sum on M5f LR meta coefs):
  m3_catboost   L1=0.112  (dominated by cb_year-cat / cb_lossguide)
  m4_relstate   L1=0.141  (dominated by e5_optuna_lgbm)

The smaller pool should shrink OOF→LB gap (per main's pool-redundancy-
gap-widen friction); +0.2bp OOF lift is within fold-noise but the
gap-tightening is the real win.

Output: oof/test_m5h_*.npy, m5h_lr_meta_results.json, audit, submission.
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
M5C_S, M5C_G = 0.95000, 0.92963
M5D_S, M5D_G = 0.95023, 0.92994
M5E_S, M5E_G = 0.95027, 0.93084
M5F_S, M5F_G = 0.95042, 0.93105
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
DROPPED_FROM_M5F = ["m3_catboost", "m4_relstate"]


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
        oo, te = load(name, suffix)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    print(f"  M5h ({suffix}): {auc:.5f}  Δbase={(auc-base_auc)*1e4:+.1f}bp  "
          f"K={len(names)}")
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc, names, lr_full.coef_.ravel()


def main():
    print(f"=== M5h — M5f pool minus {DROPPED_FROM_M5F} ===")
    print(f"=== Pool ({len(POOL)}): {[p[0] for p in POOL]} ===\n")
    print("=== M5h — Strat ===")
    oof_s, test_s, auc_s, names_s, coef_s = run("strat", BASE_S)
    print("=== M5h — GroupKF ===")
    oof_g, test_g, auc_g, _, coef_g = run("groupkf", BASE_G)

    np.save(ART / "oof_m5h_strat.npy", np.column_stack([1 - oof_s, oof_s]))
    np.save(ART / "test_m5h_strat.npy", np.column_stack([1 - test_s, test_s]))
    np.save(ART / "oof_m5h_groupkf.npy", np.column_stack([1 - oof_g, oof_g]))
    np.save(ART / "test_m5h_groupkf.npy", np.column_stack([1 - test_g, test_g]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv("submissions/submission_m5h_lr_meta_l1pruned.csv", index=False)

    feat_names = ([f"raw_{n}" for n in names_s] + [f"rank_{n}" for n in names_s]
                  + [f"logit_{n}" for n in names_s])
    res = dict(strat=dict(oof=auc_s,
                          delta_base_bp=(auc_s - BASE_S) * 1e4,
                          delta_m5f_bp=(auc_s - M5F_S) * 1e4,
                          delta_m5d_bp=(auc_s - M5D_S) * 1e4,
                          coefs={n: float(c) for n, c in zip(feat_names, coef_s)}),
               groupkf=dict(oof=auc_g,
                            delta_base_bp=(auc_g - BASE_G) * 1e4,
                            delta_m5f_bp=(auc_g - M5F_G) * 1e4),
               pool=names_s, dropped_from_m5f=DROPPED_FROM_M5F)
    (ART / "m5h_lr_meta_results.json").write_text(json.dumps(res, indent=2))

    body = (
        f"# M5h — L1coef-pruned stack (drop M5f bottom-2 by LR-meta L1) "
        f"(2026-05-04)\n\n"
        f"Pool ({len(names_s)}): {names_s}\n\n"
        f"Dropped from M5f: {DROPPED_FROM_M5F} "
        f"(L1coef = 0.112, 0.141 — bottom 2 of 15)\n\n"
        f"## Two-anchor results vs M5f\n\n"
        f"| anchor | M5h | M5f | Δ vs M5f | Δ vs M5d (LB 0.94963) |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| Strat | **{auc_s:.5f}** | {M5F_S:.5f} | "
        f"{(auc_s - M5F_S) * 1e4:+.1f}bp | {(auc_s - M5D_S) * 1e4:+.1f}bp |\n"
        f"| GroupKF | **{auc_g:.5f}** | {M5F_G:.5f} | "
        f"{(auc_g - M5F_G) * 1e4:+.1f}bp | {(auc_g - M5D_G) * 1e4:+.1f}bp |\n\n"
        f"## Hypothesis\n\n"
        f"Smaller pool (13 vs 15) should reduce OOF→LB gap-widening "
        f"(M5b 7-base gap −3.5bp vs M5d 12-base gap −6.0bp pattern). "
        f"Strat OOF lift +{(auc_s - M5F_S) * 1e4:+.1f}bp is within fold noise; "
        f"the gap-tightening is the real expected win.\n\n"
        f"Submission: submissions/submission_m5h_lr_meta_l1pruned.csv (held).\n"
    )
    Path("audit/2026-05-04-m5h-l1coef-prune.md").write_text(body)


if __name__ == "__main__":
    main()
