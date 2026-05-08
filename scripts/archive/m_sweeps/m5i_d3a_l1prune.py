"""M5i — drop d3a_te_unified into M5h pool, refit LR meta, L1-prune.

Decision protocol (HANDOVER 2026-05-04 mid-session, PI-confirmed):
  1. Add d3a_te_unified to M5h pool (13 → 14 bases).
  2. Refit LR meta on raw+rank+logit expansion (same recipe as M5h).
  3. Compute L1-coef sum per base.
  4. Drop bases with L1 sum < median (matches HANDOVER's "drop bases below
     median L1 sum" rule). Refit LR meta on the pruned pool.
  5. Compare pruned-pool Strat OOF to M5h's 0.95043. If lift, the pruned
     pool is the M5i candidate; else fall back to M5h.

Output: oof/test_m5i_*.npy, m5i_lr_meta_results.json, audit, submission.
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
M5H_S, M5H_G = 0.95043, 0.93087   # current PRIMARY
SEED, N_FOLDS = 42, 5

# M5h pool + d3a_te_unified
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
    ("d3a_te_unified", "d3a_te_unified"),  # NEW
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


def l1_sums_per_base(coef, names):
    """coef = [raw_1..raw_K, rank_1..rank_K, logit_1..logit_K]."""
    K = len(names)
    raw = np.abs(coef[:K])
    rank = np.abs(coef[K:2 * K])
    logit = np.abs(coef[2 * K:])
    return {n: float(raw[i] + rank[i] + logit[i]) for i, n in enumerate(names)}


def assemble(pool, suffix):
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool:
        oo, te = load(name, suffix)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    P_oof = np.column_stack(Xs_oof)
    P_test = np.column_stack(Xs_test)
    return expand(P_oof), expand(P_test), names


def main():
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    print(f"=== M5i — full pool ({len(POOL_FULL)}) ===")
    print(f"=== Pool: {[p[0] for p in POOL_FULL]} ===\n")

    # Stage 1: full 14-base pool, both anchors
    print("=== Stage 1: full 14-base, Strat ===")
    F_oof_s, F_test_s, names = assemble(POOL_FULL, "strat")
    meta_oof_s, meta_test_s, auc_s_full, coef_s_full = fit_meta(F_oof_s, F_test_s, y)
    print(f"  Strat: {auc_s_full:.5f}  Δ M5h={(auc_s_full-M5H_S)*1e4:+.1f}bp  K={len(names)}")
    l1_full_s = l1_sums_per_base(coef_s_full, names)
    print("  L1 per base (Strat):")
    for n, v in sorted(l1_full_s.items(), key=lambda x: -x[1]):
        print(f"    {n:<22s} L1={v:.3f}")

    print("\n=== Stage 1: full 14-base, GroupKF ===")
    F_oof_g, F_test_g, _ = assemble(POOL_FULL, "groupkf")
    meta_oof_g, meta_test_g, auc_g_full, coef_g_full = fit_meta(F_oof_g, F_test_g, y)
    print(f"  GroupKF: {auc_g_full:.5f}  Δ M5h={(auc_g_full-M5H_G)*1e4:+.1f}bp")
    l1_full_g = l1_sums_per_base(coef_g_full, names)

    # Stage 2: drop bases with L1 sum < median (Strat-driven, since Strat is LB-proxy per R1)
    median_l1 = float(np.median(list(l1_full_s.values())))
    surv = [(lbl, name) for (lbl, name) in POOL_FULL if l1_full_s[lbl] >= median_l1]
    dropped = [lbl for (lbl, _) in POOL_FULL if l1_full_s[lbl] < median_l1]
    print(f"\n=== Stage 2: L1-prune below median Strat L1={median_l1:.3f} ===")
    print(f"  Survivors ({len(surv)}): {[p[0] for p in surv]}")
    print(f"  Dropped: {dropped}")

    print("=== Stage 2: pruned pool, Strat ===")
    F_oof_s_p, F_test_s_p, names_p = assemble(surv, "strat")
    meta_oof_s_p, meta_test_s_p, auc_s_p, coef_s_p = fit_meta(F_oof_s_p, F_test_s_p, y)
    print(f"  Strat: {auc_s_p:.5f}  Δ M5h={(auc_s_p-M5H_S)*1e4:+.1f}bp  K={len(names_p)}")

    print("=== Stage 2: pruned pool, GroupKF ===")
    F_oof_g_p, F_test_g_p, _ = assemble(surv, "groupkf")
    meta_oof_g_p, meta_test_g_p, auc_g_p, coef_g_p = fit_meta(F_oof_g_p, F_test_g_p, y)
    print(f"  GroupKF: {auc_g_p:.5f}  Δ M5h={(auc_g_p-M5H_G)*1e4:+.1f}bp")

    # Pick the best between (full 14, pruned)
    candidates = [
        ("m5i_full14", auc_s_full, auc_g_full, meta_oof_s, meta_oof_g, meta_test_s, meta_test_g, names, coef_s_full),
        ("m5i_pruned", auc_s_p, auc_g_p, meta_oof_s_p, meta_oof_g_p, meta_test_s_p, meta_test_g_p, names_p, coef_s_p),
    ]
    best = max(candidates, key=lambda x: x[1])  # Strat OOF as decider
    label, auc_s_b, auc_g_b, oof_s_b, oof_g_b, test_s_b, test_g_b, names_b, coef_b = best
    print(f"\n=== M5i WINNER: {label} ===")
    print(f"  Strat={auc_s_b:.5f} (Δ M5h={(auc_s_b-M5H_S)*1e4:+.1f}bp)  "
          f"GroupKF={auc_g_b:.5f} (Δ M5h={(auc_g_b-M5H_G)*1e4:+.1f}bp)")

    np.save(ART / "oof_m5i_strat.npy", np.column_stack([1 - oof_s_b, oof_s_b]))
    np.save(ART / "test_m5i_strat.npy", np.column_stack([1 - test_s_b, test_s_b]))
    np.save(ART / "oof_m5i_groupkf.npy", np.column_stack([1 - oof_g_b, oof_g_b]))
    np.save(ART / "test_m5i_groupkf.npy", np.column_stack([1 - test_g_b, test_g_b]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s_b
    sub.to_csv("submissions/submission_m5i_lr_meta.csv", index=False)

    feat_names = ([f"raw_{n}" for n in names_b] + [f"rank_{n}" for n in names_b]
                  + [f"logit_{n}" for n in names_b])
    res = dict(
        winner=label,
        full14=dict(strat=auc_s_full, groupkf=auc_g_full,
                    delta_m5h_strat_bp=(auc_s_full - M5H_S) * 1e4,
                    delta_m5h_groupkf_bp=(auc_g_full - M5H_G) * 1e4,
                    l1_strat=l1_full_s, l1_groupkf=l1_full_g,
                    median_l1_strat=median_l1),
        pruned=dict(strat=auc_s_p, groupkf=auc_g_p,
                    delta_m5h_strat_bp=(auc_s_p - M5H_S) * 1e4,
                    delta_m5h_groupkf_bp=(auc_g_p - M5H_G) * 1e4,
                    survivors=[p[0] for p in surv],
                    dropped=dropped),
        winner_pool=names_b,
        winner_coefs={n: float(c) for n, c in zip(feat_names, coef_b)},
    )
    (ART / "m5i_lr_meta_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n→ scripts/artifacts/m5i_lr_meta_results.json")
    print(f"→ submissions/submission_m5i_lr_meta.csv")


if __name__ == "__main__":
    main()
