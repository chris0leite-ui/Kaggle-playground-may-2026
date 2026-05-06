"""Day-13c — Move C pool refactor on Strat axis.

Day-12 found GBDTs leak 200-247bp under GroupKF; d13b confirmed
d9c_FM is redundant given d9f + d13a partition FMs (FULL_22 0.94607
vs SWAP_21 0.94606, Δ -0.01bp under GKF).

This script tests whether dropping leakage-eaters from the Strat pool
holds (or improves) OOF on the truth axis (public LB ≈ Strat per U3).

PRIMARY = d9i_S1_K21_swap_aug2way (Strat OOF 0.95071, LB 0.95034).

Variants (all built from existing _strat.npy artifacts via LR meta):
  T0_S3_K24       baseline reproduction (POOL_KEEP + TOP_3 + 5 FMs)
  T1_drop_d9c     K=23 — drop d9c_FM (Move C minimal: redundancy claim)
  T2_drop_d9c_e5  K=22 — also drop e5_optuna_lgbm (heaviest leakage eater)
  T3_drop_3leak   K=21 — also drop cb_slow-wide-bag (top-3 leakage eaters)

Submit gate (per HANDOVER critical-rules §7-style):
  OOF ≥ PRIMARY (0.95071) AND ρ < 0.9995  →  PRIMARY-submit candidate
  OOF ≥ PRIMARY AND ρ ≥ 0.9995            →  TIE_EXPECTED hold
  OOF < PRIMARY                            →  HEDGE only (R5 rule)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

PRIMARY_S = 0.95071
PRIMARY_LB = 0.95034
RHO_TIE = 0.9995

# Full S3 K=24 pool (matches d13a S3 build)
POOL_KEEP = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"), ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"), ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"), ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"), ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    ("realmlp", "realmlp"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]
TOP_3_D9 = [
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
]
FM_BASES = [
    ("d9c_FM", "oof_d9c_fm_strat.npy", "test_d9c_fm_strat.npy"),
    ("d9f_FM_A", "oof_d9f_FM_A_strat.npy", "test_d9f_FM_A_strat.npy"),
    ("d9f_FM_B", "oof_d9f_FM_B_strat.npy", "test_d9f_FM_B_strat.npy"),
    ("FM_A_53", "oof_d13a_FM_A_53_strat.npy", "test_d13a_FM_A_53_strat.npy"),
    ("FM_B_53", "oof_d13a_FM_B_53_strat.npy", "test_d13a_FM_B_53_strat.npy"),
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_test = np.load(ART / "test_d9i_S1_K21_swap_aug2way_strat.npy"
                           )[:, 1].astype(np.float64)

    # Load all base + FM artifacts once
    all_oof, all_test, all_names = {}, {}, []
    for label, fname in POOL_KEEP + TOP_3_D9:
        all_oof[label] = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        all_test[label] = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        all_names.append(label)
    for label, of, tf in FM_BASES:
        all_oof[label] = np.load(ART / of)[:, 1].astype(np.float64)
        all_test[label] = np.load(ART / tf)[:, 1].astype(np.float64)
        all_names.append(label)

    def stack(label, drop):
        keep = [n for n in all_names if n not in drop]
        Xs = [all_oof[n] for n in keep]
        Ts = [all_test[n] for n in keep]
        K = len(keep)
        F_oof = expand(np.column_stack(Xs))
        F_test = expand(np.column_stack(Ts))
        mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
        auc = float(roc_auc_score(y, mo))
        rho, _ = spearmanr(tp, primary_test)
        delta_bp = (auc - PRIMARY_S) * 1e4
        l1 = {keep[i]: float(abs(coef[i]) + abs(coef[K+i]) + abs(coef[2*K+i]))
              for i in range(K)}
        if auc >= PRIMARY_S and rho < RHO_TIE:
            verdict = "PRIMARY-CANDIDATE ✓"
        elif auc >= PRIMARY_S:
            verdict = "TIE_EXPECTED (hold)"
        else:
            verdict = "REGRESS / HEDGE-only"
        print(f"\n=== {label} (K={K}) ===")
        print(f"  Strat OOF: {auc:.5f}  Δ PRIMARY {delta_bp:+.2f}bp  "
              f"ρ vs PRIMARY {rho:.5f}  →  {verdict}")
        print(f"  L1 top-12:")
        for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:12]:
            mk = ""
            if n_ in ("FM_A_53", "FM_B_53"): mk = "  ← d13a"
            elif n_ in ("d9f_FM_A", "d9f_FM_B"): mk = "  ← d9f"
            elif n_ == "d9c_FM": mk = "  ← d9c"
            elif n_.startswith("rule_"): mk = "  ← rule"
            elif n_ in ("R6_next_compound", "R10_driver_eb", "R7_prev_compound"):
                mk = "  ← d9-rule"
            print(f"    {n_:<24s} L1={v:.3f}{mk}")
        if drop:
            print(f"  dropped: {sorted(drop)}")
        return dict(K=K, strat_oof=auc, delta_bp=delta_bp,
                    rho_vs_primary=float(rho), verdict=verdict, l1=l1,
                    test_pred=tp)

    print(f"PRIMARY = d9i_S1_K21_swap_aug2way (Strat {PRIMARY_S:.5f}, LB {PRIMARY_LB:.5f})\n")
    print(f"Submit gate: OOF ≥ {PRIMARY_S:.5f} AND ρ < {RHO_TIE}")

    r0 = stack("T0_S3_K24 (baseline reproduction)", set())
    r1 = stack("T1_drop_d9c (K=23 Move C minimal)", {"d9c_FM"})
    r2 = stack("T2_drop_d9c_e5 (K=22 + heaviest GBDT leak-eater)",
               {"d9c_FM", "e5_optuna_lgbm"})
    r3 = stack("T3_drop_3leak (K=21 + cb_slow-wide-bag)",
               {"d9c_FM", "e5_optuna_lgbm", "cb_slow-wide-bag"})
    r4 = stack("T4_drop_d9c_keep_e5 (K=23 swap d9c only — control for T2)",
               {"d9c_FM"})  # same as T1, kept as a name for clarity (skip)
    # T4 is duplicate of T1, drop it from list

    print(f"\n=== Move C refactor matrix summary ===")
    rows = [("T0_S3_K24", r0), ("T1_drop_d9c", r1),
            ("T2_drop_d9c_e5", r2), ("T3_drop_3leak", r3)]
    for name, r in rows:
        marker = " ★" if r["strat_oof"] >= PRIMARY_S and r["rho_vs_primary"] < RHO_TIE else ""
        print(f"  {name:<22s} K={r['K']}  Strat {r['strat_oof']:.5f}  "
              f"Δ {r['delta_bp']:+.2f}bp  ρ {r['rho_vs_primary']:.5f}  "
              f"{r['verdict']}{marker}")

    # Save the leading non-baseline candidate's submission for review
    candidates = [r1, r2, r3]
    leader = max(candidates, key=lambda r: r["strat_oof"])
    leader_name = {id(r1): "T1_drop_d9c", id(r2): "T2_drop_d9c_e5",
                   id(r3): "T3_drop_3leak"}[id(leader)]
    tp_leader = leader["test_pred"]
    np.save(ART / f"test_d13c_{leader_name}_strat.npy",
            np.column_stack([1 - tp_leader, tp_leader]))
    sub = sample_sub.copy(); sub[TARGET] = tp_leader
    sub_path = f"submissions/submission_d13c_{leader_name}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n→ leader: {leader_name} (Strat {leader['strat_oof']:.5f})")
    print(f"  wrote {sub_path}")

    # Persist results (drop test_pred numpy from JSON)
    serial = {n: {k: v for k, v in r.items() if k != "test_pred"}
              for n, r in rows}
    final = dict(
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB,
                     ref="test_d9i_S1_K21_swap_aug2way_strat.npy"),
        leader=leader_name,
        variants=serial,
        total_wall_s=time.time() - t0,
    )
    (ART / "d13c_strat_refactor_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13c_strat_refactor_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
