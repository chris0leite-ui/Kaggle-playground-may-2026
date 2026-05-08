"""M5f — LR meta combining main's M5d HGBC additions and M5e CB additions.

Pool (15-base): M5c 10-base + f1_hgbc_deep + f2_hgbc_shallow
                + cb_year-cat + cb_lossguide + cb_slow-wide-bag.

Same LR(C=1.0) on [raw, rank, logit] recipe as M5/M5b/M5c/M5d/M5e.

Reference points:
  M5c  10-base    Strat 0.95000  GroupKF 0.92963
  M5d  12-base    Strat 0.95023  GroupKF 0.92994  (main, LB 0.94963)
  M5e  13-base    Strat 0.95027  GroupKF 0.93084  (mine, held)
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
    print(f"  M5f OOF: {auc:.5f}  Δbase={(auc - base_auc) * 1e4:+.1f}bp  K={len(names)}")
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc, names, lr_full.coef_.ravel()


def main():
    print("=== M5f — Strat ===")
    oof_s, test_s, auc_s, names_s, coef_s = run("strat", BASE_S)
    print("\n=== M5f — GroupKF ===")
    oof_g, test_g, auc_g, _, coef_g = run("groupkf", BASE_G)

    np.save(ART / "oof_m5f_strat.npy", np.column_stack([1 - oof_s, oof_s]))
    np.save(ART / "test_m5f_strat.npy", np.column_stack([1 - test_s, test_s]))
    np.save(ART / "oof_m5f_groupkf.npy", np.column_stack([1 - oof_g, oof_g]))
    np.save(ART / "test_m5f_groupkf.npy", np.column_stack([1 - test_g, test_g]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv("submissions/submission_m5f_lr_meta_combined.csv", index=False)

    feat_names = ([f"raw_{n}" for n in names_s] + [f"rank_{n}" for n in names_s]
                  + [f"logit_{n}" for n in names_s])
    res = dict(strat=dict(oof=auc_s,
                          delta_base_bp=(auc_s - BASE_S) * 1e4,
                          delta_m5c_bp=(auc_s - M5C_S) * 1e4,
                          delta_m5d_bp=(auc_s - M5D_S) * 1e4,
                          delta_m5e_bp=(auc_s - M5E_S) * 1e4,
                          coefs={n: float(c) for n, c in zip(feat_names, coef_s)}),
               groupkf=dict(oof=auc_g,
                            delta_base_bp=(auc_g - BASE_G) * 1e4,
                            delta_m5c_bp=(auc_g - M5C_G) * 1e4,
                            delta_m5d_bp=(auc_g - M5D_G) * 1e4,
                            delta_m5e_bp=(auc_g - M5E_G) * 1e4),
               pool=names_s)
    (ART / "m5f_lr_meta_results.json").write_text(json.dumps(res, indent=2))

    body = (
        f"# M5f — LR meta combining M5d (HGBC) + M5e (CB) additions (2026-05-04)\n\n"
        f"Pool ({len(names_s)}): {names_s}\n\n"
        f"## Two-anchor results vs M5c / M5d / M5e\n\n"
        f"| anchor | M5f | M5c | M5d (main) | M5e (mine) | Δ vs M5d | Δ vs M5e |\n"
        f"|---|---:|---:|---:|---:|---:|---:|\n"
        f"| Strat | **{auc_s:.5f}** | {M5C_S:.5f} | {M5D_S:.5f} | {M5E_S:.5f} | "
        f"{(auc_s - M5D_S) * 1e4:+.1f}bp | {(auc_s - M5E_S) * 1e4:+.1f}bp |\n"
        f"| GroupKF | **{auc_g:.5f}** | {M5C_G:.5f} | {M5D_G:.5f} | {M5E_G:.5f} | "
        f"{(auc_g - M5D_G) * 1e4:+.1f}bp | {(auc_g - M5E_G) * 1e4:+.1f}bp |\n\n"
        f"M5d LB anchor (main): 0.94963 (Strat OOF→LB gap −6.0bp).\n"
        f"M5e: held, projected LB ~0.94992 (using M5b's −3.5bp gap).\n\n"
        f"Submission: submissions/submission_m5f_lr_meta_combined.csv (held).\n"
    )
    Path("audit/2026-05-04-m5f-lr-meta-combined.md").write_text(body)
    print("audit written")


if __name__ == "__main__":
    main()
