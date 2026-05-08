"""Day-10 — Strat-side match for d10b GKF stack rebuild.

Build the same K=13 / K=15 stacks under Strat to get an apples-to-
apples comparison of FM-class lift. d10b found +2.01bp under
Race-only GKF; this script tells us the corresponding Strat lift.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

POOL_GKF_AVAILABLE = [
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


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F, y, splits):
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F, y)
    return meta_oof, lr_full.coef_.ravel()


def stack(label, oofs, names, y, splits):
    K = len(names)
    F = expand(np.column_stack(oofs))
    meta_oof, coef = fit_lr_meta(F, y, splits)
    auc = float(roc_auc_score(y, meta_oof))
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"\n=== {label} (K={K}) ===")
    print(f"  Strat stack OOF: {auc:.5f}")
    print(f"  L1 top-15:")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:15]:
        marker = "  ← FM-class" if n.startswith("FM") else ""
        print(f"    {n:<24s} L1={v:.3f}{marker}")
    return auc, l1


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    print(f"Loading {len(POOL_GKF_AVAILABLE)} bases (Strat OOFs)")
    base_oofs, base_names = [], []
    for label, fname in POOL_GKF_AVAILABLE:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        base_oofs.append(oo); base_names.append(label)

    auc_K13, l1_K13 = stack("K=13 Strat baseline (no FM)", base_oofs,
                             base_names, y, splits)

    oof_FM_A = np.load(ART / "oof_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    oof_FM_B = np.load(ART / "oof_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)
    full_oofs = base_oofs + [oof_FM_A, oof_FM_B]
    full_names = base_names + ["FM_A_driver_dynamics", "FM_B_race_context"]
    auc_K15, l1_K15 = stack("K=15 Strat + FM_A + FM_B", full_oofs,
                             full_names, y, splits)

    fm_lift_strat_bp = (auc_K15 - auc_K13) * 1e4

    # Read d10b's GKF result
    gkf_data = json.loads(
        (ART / "d10b_groupkf_stack_rebuild.json").read_text())
    auc_K13_gkf = gkf_data["race_only_gkf"]["stack_K13_oof"]
    auc_K15_gkf = gkf_data["race_only_gkf"]["stack_K15_oof"]
    fm_lift_gkf_bp = gkf_data["race_only_gkf"]["fm_class_lift_bp"]

    print("\n" + "=" * 72)
    print("FM-class lift: Strat vs Race-only GKF (apples-to-apples)")
    print("-" * 72)
    print(f"  K=13 baseline (no FM)   Strat: {auc_K13:.5f}   GKF: {auc_K13_gkf:.5f}   Δ {(auc_K13_gkf-auc_K13)*1e4:+.2f}bp")
    print(f"  K=15 + FM_A + FM_B      Strat: {auc_K15:.5f}   GKF: {auc_K15_gkf:.5f}   Δ {(auc_K15_gkf-auc_K15)*1e4:+.2f}bp")
    print(f"  FM-class LIFT (K15-K13)  Strat: {fm_lift_strat_bp:+.2f}bp  GKF: {fm_lift_gkf_bp:+.2f}bp")
    print()
    if fm_lift_gkf_bp >= fm_lift_strat_bp - 0.5:
        print("  VERDICT: FM lift PRESERVED or AMPLIFIED under GKF.")
        print("  → Private-LB robust. PRIMARY (d9f K=21 swap) holds.")
    elif fm_lift_gkf_bp >= 0:
        print("  VERDICT: FM lift PARTIALLY transfers under GKF.")
        print(f"  → {fm_lift_gkf_bp/fm_lift_strat_bp*100:.0f}% of Strat lift survives.")
    else:
        print("  VERDICT: FM lift NEGATIVE under GKF — leakage artifact.")
    print("=" * 72)

    final = dict(
        strat=dict(K13_oof=auc_K13, K15_oof=auc_K15,
                   fm_lift_bp=float(fm_lift_strat_bp),
                   l1_K13=l1_K13, l1_K15=l1_K15),
        gkf_race_only=dict(K13_oof=auc_K13_gkf, K15_oof=auc_K15_gkf,
                            fm_lift_bp=float(fm_lift_gkf_bp)),
        diagnostic_question="Does FM-class lift transfer GKF→Strat?",
        verdict="preserved" if fm_lift_gkf_bp >= fm_lift_strat_bp - 0.5
                 else "partial" if fm_lift_gkf_bp >= 0 else "leakage_artifact",
        wall_s=time.time() - t0,
    )
    (ART / "d10c_strat_match.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d10c_strat_match.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
