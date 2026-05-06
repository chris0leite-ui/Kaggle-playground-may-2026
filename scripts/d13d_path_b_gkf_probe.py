"""Day-13 — GroupKF probe of the Path B Stint hierarchical meta.

The Stint τ=100000 hierarchical meta landed LB 0.95041 (+7bp over
d9h/d9i, 11.6× OOF→LB amplification). Question: does the +0.86bp
OOF lift survive under leak-blocking GroupKF (Race-only)? If yes,
the mechanism transfers and the public-LB lift is private-robust.
If no, the Strat OOF lift is leakage-piggybacked and private will
revert.

Pool: K=20 GKF (matches d12_groupkf_meta_no_realmlp baseline).
realmlp dropped (only Strat artifact exists). FM_A + FM_B trained
under Race-only GKF in d10b/d10d.

Compare:
  global LR meta (= PRIMARY equivalent, K=20 GKF):  baseline OOF
  Stint hier meta τ=100000 (K=20 GKF):              probe OOF
  Δ = probe − baseline

If Δ > 0, hier-meta lifts under leak-blocking → private-LB robust.
If Δ ≈ 0, the Strat lift is leakage-piggybacked → private will compress.
If Δ < 0, the lift was entirely leakage-amplified → expect regression.

~3-5 min wall.
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
from sklearn.model_selection import GroupKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
TAU = 100000

POOL_GKF = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"), ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"), ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"), ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"), ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    # realmlp dropped — no GKF artifact
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
]
FM_GKF = [
    ("FM_A", "d9f_FM_A_groupkf_race"),
    ("FM_B", "d9f_FM_B_groupkf_race"),
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_aug(F, y):
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict_aug(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    # Load K=20 GKF pool
    base_oofs = []
    names = []
    for label, fname in POOL_GKF:
        oo = np.load(ART / f"oof_{fname}_groupkf.npy")[:, 1].astype(np.float64)
        base_oofs.append(oo); names.append(label)
    for label, fname in FM_GKF:
        oo = np.load(ART / f"oof_{fname}.npy")[:, 1].astype(np.float64)
        base_oofs.append(oo); names.append(label)
    K = len(base_oofs)
    print(f"K={K} GKF bases loaded "
          f"({len(POOL_GKF)} structural + {len(FM_GKF)} FM)")
    F_oof = expand(np.column_stack(base_oofs))
    print(f"F shape: {F_oof.shape}")

    # GroupKFold by Race (matches existing pool partition)
    grp = train["Race"].astype(str).values
    print(f"Race-only GKF: {len(np.unique(grp))} groups, {N_FOLDS} folds")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.zeros(len(y)), y, groups=grp))

    # Stint segmentation (5 of 6 populated)
    seg = np.clip(train["Stint"].astype(int).values, 0, 5)
    n_seg = 6

    # ---- Baseline: global LR meta under GKF ----
    print("\n--- Baseline: K=20 global LR meta under GKF ---")
    meta_global = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global[va] = lr.predict_proba(F_oof[va])[:, 1]
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s")
    auc_global_gkf = float(roc_auc_score(y, meta_global))
    print(f"  → Global GKF stack OOF: {auc_global_gkf:.5f}")

    # ---- Probe: Stint hier-meta τ=100000 under GKF ----
    print(f"\n--- Probe: K=20 Stint hier-meta τ={TAU} under GKF ---")
    meta_hier = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = fit_lr_aug(F_oof[tr], y[tr])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg[tr] == s)[0]
            counts[s] = len(idx)
            if len(idx) < 200 or len(np.unique(y[tr][idx])) < 2:
                continue
            W_local[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            mask[s] = True
        n_local = counts.astype(np.float64)
        alpha = n_local / (n_local + TAU)
        W_shrunk = (alpha[:, None] * W_local +
                    (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg[va]):
            idx = np.where(seg[va] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            meta_hier[va[idx]] = predict_aug(F_oof[va[idx]], w)
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s")
    auc_hier_gkf = float(roc_auc_score(y, meta_hier))
    delta_gkf_bp = (auc_hier_gkf - auc_global_gkf) * 1e4
    print(f"  → Stint hier-meta GKF OOF: {auc_hier_gkf:.5f}")

    # ---- Compare to Strat-side reference ----
    # From d13b: global Strat = 0.95073, Stint hier τ=100000 Strat = 0.95082
    strat_global = 0.95073
    strat_hier_tau100k = 0.95082
    delta_strat_bp = (strat_hier_tau100k - strat_global) * 1e4

    print("\n" + "=" * 72)
    print("Path B Stint hier-meta lift: Strat vs GKF")
    print("-" * 72)
    print(f"  Global LR meta     Strat: {strat_global:.5f}   "
          f"GKF: {auc_global_gkf:.5f}   "
          f"Δ {(auc_global_gkf-strat_global)*1e4:+.2f}bp")
    print(f"  Stint hier τ=100k  Strat: {strat_hier_tau100k:.5f}   "
          f"GKF: {auc_hier_gkf:.5f}   "
          f"Δ {(auc_hier_gkf-strat_hier_tau100k)*1e4:+.2f}bp")
    print(f"  Hier LIFT          Strat: {delta_strat_bp:+.2f}bp   "
          f"GKF: {delta_gkf_bp:+.2f}bp")
    print()
    if delta_gkf_bp >= delta_strat_bp - 0.3:
        print("  VERDICT: Hier-meta lift PRESERVED under GKF.")
        print("  → Path B mechanism is leakage-robust; full +7bp public LB")
        print("    expected to transfer to private (within sample variance).")
    elif delta_gkf_bp >= 0:
        ratio = delta_gkf_bp / delta_strat_bp * 100 if delta_strat_bp > 0 else 0
        print(f"  VERDICT: Hier-meta lift PARTIALLY transfers ({ratio:.0f}%).")
        print(f"  → Expected private LB transfer: {7 * ratio/100:+.1f}bp range.")
        print(f"  → ~{7 * (1 - ratio/100):.1f}bp of public lift was leakage.")
    else:
        print(f"  VERDICT: Hier-meta lift NEGATIVE under GKF ({delta_gkf_bp:+.2f}bp).")
        print(f"  → Public-LB lift was leakage-amplified.")
        print(f"  → Private LB likely REGRESSES vs HEDGE (d9h/d9i).")
    print("=" * 72)

    final = dict(
        gkf=dict(global_oof=auc_global_gkf, hier_oof=auc_hier_gkf,
                  hier_lift_bp=float(delta_gkf_bp)),
        strat=dict(global_oof=strat_global, hier_oof=strat_hier_tau100k,
                    hier_lift_bp=float(delta_strat_bp)),
        ratio_gkf_over_strat=(float(delta_gkf_bp / delta_strat_bp)
                              if delta_strat_bp > 0 else None),
        wall_s=time.time() - t0,
    )
    (ART / "d13d_gkf_probe_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13d_gkf_probe_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
