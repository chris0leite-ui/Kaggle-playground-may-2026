"""Round 3 P0 — Proxy-substitution audit + C-sweep + residual correlation.

Tests the Senior ML Engineer's hypothesis: the Round-2 11/11 null was
gated against K=4 (homogeneous tree pool); the actual PRIMARY is
K=11+K=9 with slim-kNN diversity. If candidates revive under a more
diverse anchor, the row-feature-ceiling claim is anchor-conditional.

Available diverse anchors in the snapshot:
  - K=21 (the historical pool, 21 bases, FM/MLP/RealMLP/CatBoost/HGBC mix)
  - K=27 + Path-B as a single super-base (the d18 PRIMARY OOF)

We use both for cross-validation of the hypothesis.

P0.1: Retest the 3 most-plausible Round-2 nulls at K=21+1 and K=27+1.
P0.2: LR-meta C-sweep at K=4+1 for the LGBM-rank candidate (Rule 21
      family-falsification).
P0.3: K=4 vs K=27 residual correlation (sanity check).

Origin: 2026-05-18 round-3 plan (/root/.claude/plans/read-the-...).
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

K4_BASES = [
    "d17_h1d_yekenot_full", "p1_single_cb_v4_gpu",
    "f1_hgbc_deep", "d16_orig_continuous_only",
]

K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
]

# K=4 + (K=27 + Path-B as super-base). 5-base "K=27 representation".
K4_PLUS_K27SUPER = K4_BASES + [
    "d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000"
]

CANDIDATES_TO_RETEST = [
    "K4_conformal_widths",   # P2.2 — per-bin uncertainty
    "K4_rrf_k60",            # P0.1 — RRF blend score
    "K4_meta_lgbm_rank",     # P1.1 — LGBM ranker output
]


def pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def meta_oof_lr(y, F, C=1.0):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def gate_one(y, pool_oofs, cand_oof, label):
    F_base = expand(np.column_stack(pool_oofs))
    F_with = expand(np.column_stack(pool_oofs + [cand_oof]))
    _, base_auc = meta_oof_lr(y, F_base)
    _, with_auc = meta_oof_lr(y, F_with)
    delta_bp = (with_auc - base_auc) * 1e4
    return base_auc, with_auc, delta_bp


def main():
    y = pd.read_csv("data/train.csv")[TARGET].astype(int).values
    print(f"y: n={len(y):,}, mean={y.mean():.4f}")

    # ===== P0.1 — Retest 3 candidates at K=4, K=21, K=27-augmented anchors =====
    print("\n" + "=" * 80)
    print("P0.1 — Proxy-substitution audit (multi-anchor retest)")
    print("=" * 80)

    cands = {c: pos(ART / f"oof_{c}_strat.npy") for c in CANDIDATES_TO_RETEST}

    pools = {}
    print("\nLoading pool OOFs...")
    pools["K=4"] = [pos(ART / f"oof_{b}_strat.npy") for b in K4_BASES]
    print(f"  K=4: {len(pools['K=4'])} bases")
    pools["K=21"] = [pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]
    print(f"  K=21: {len(pools['K=21'])} bases")
    pools["K=4+K27super"] = [pos(ART / f"oof_{b}_strat.npy")
                              for b in K4_PLUS_K27SUPER]
    print(f"  K=4+K27super: {len(pools['K=4+K27super'])} bases")

    primary_test = pos(ART / "test_d13e_compound_stint_tau20000_strat.npy")

    print(f"\n{'candidate':<25s} {'anchor':<15s} {'base_AUC':>10s} {'with_AUC':>10s} {'Δ bp':>8s}")
    print("-" * 75)
    p0_1_results = []
    for cname, cand_oof in cands.items():
        cand_test = pos(ART / f"test_{cname}_strat.npy")
        for anchor_name, pool_oofs in pools.items():
            t0 = time.time()
            base_auc, with_auc, delta_bp = gate_one(y, pool_oofs, cand_oof,
                                                     f"{anchor_name}+{cname}")
            wall = time.time() - t0
            # rho check (vs PRIMARY d13e)
            # Build full-train test prediction for ρ vs PRIMARY
            P_test_base = np.column_stack([pos(ART / f"test_{b}_strat.npy")
                                            for b in (K4_BASES if anchor_name == "K=4"
                                                       else K21_BASES if anchor_name == "K=21"
                                                       else K4_PLUS_K27SUPER)])
            P_test_with = np.column_stack([P_test_base, cand_test])
            F_test_with = expand(P_test_with)
            F_oof_with = expand(np.column_stack(pool_oofs + [cand_oof]))
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            lr.fit(F_oof_with, y)
            test_with = lr.predict_proba(F_test_with)[:, 1]
            rho, _ = spearmanr(test_with, primary_test)
            print(f"{cname:<25s} {anchor_name:<15s} {base_auc:.5f}   "
                  f"{with_auc:.5f}   {delta_bp:+7.3f}  ρ={rho:.4f}  "
                  f"({wall:.1f}s)")
            p0_1_results.append(dict(candidate=cname, anchor=anchor_name,
                                      base_auc=base_auc, with_auc=with_auc,
                                      delta_bp=delta_bp, rho=float(rho)))

    # ===== P0.2 — LR-meta C-sweep on K=4+1 (LGBM-rank candidate) =====
    print("\n" + "=" * 80)
    print("P0.2 — LR-meta C-sweep (Rule 21 family-falsification on K=4+1)")
    print("=" * 80)

    cand_oof = cands["K4_meta_lgbm_rank"]
    F_base = expand(np.column_stack(pools["K=4"]))
    F_with = expand(np.column_stack(pools["K=4"] + [cand_oof]))

    print(f"\n{'C':>8s} {'base_AUC':>10s} {'with_AUC':>10s} {'Δ bp':>8s}")
    print("-" * 50)
    p0_2_results = []
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        _, base_auc = meta_oof_lr(y, F_base, C=C)
        _, with_auc = meta_oof_lr(y, F_with, C=C)
        delta_bp = (with_auc - base_auc) * 1e4
        print(f"{C:>8.2f} {base_auc:.5f}   {with_auc:.5f}   {delta_bp:+7.3f}")
        p0_2_results.append(dict(C=C, base_auc=base_auc, with_auc=with_auc,
                                  delta_bp=delta_bp))

    # ===== P0.3 — K=4 vs K=27 residual correlation =====
    print("\n" + "=" * 80)
    print("P0.3 — Residual correlation (K=4 OOF vs K=27+Path-B OOF)")
    print("=" * 80)

    # K=4 LR-meta baseline OOF
    F_oof_k4 = expand(np.column_stack(pools["K=4"]))
    k4_oof, _ = meta_oof_lr(y, F_oof_k4)
    # K=27+Path-B is a single OOF file
    k27_oof = pos(ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")
    resid_k4 = y.astype(float) - k4_oof
    resid_k27 = y.astype(float) - k27_oof
    rho_resid = float(np.corrcoef(resid_k4, resid_k27)[0, 1])
    rho_pred = float(spearmanr(k4_oof, k27_oof)[0])
    print(f"\n  K=4 LR-meta OOF AUC      : {roc_auc_score(y, k4_oof):.5f}")
    print(f"  K=27+Path-B OOF AUC      : {roc_auc_score(y, k27_oof):.5f}")
    print(f"  Pearson(resid_K4, resid_K27): {rho_resid:.5f}")
    print(f"  Spearman(K4_pred, K27_pred): {rho_pred:.5f}")
    if rho_resid < 0.95:
        print(f"  ⚠ FLAG: residuals differ meaningfully — proxy may be unfit.")
    else:
        print(f"  ✓ Residuals are highly correlated; proxy substitution is "
              f"unlikely to be the flaw.")

    # ===== Save aggregate JSON =====
    out = dict(
        p0_1=p0_1_results,
        p0_2=p0_2_results,
        p0_3=dict(K4_OOF_auc=float(roc_auc_score(y, k4_oof)),
                   K27_OOF_auc=float(roc_auc_score(y, k27_oof)),
                   resid_pearson=rho_resid,
                   pred_spearman=rho_pred))
    (ART / "probe_round3_p0_proxy_audit_results.json").write_text(
        json.dumps(out, indent=2))
    print(f"\n→ scripts/artifacts/probe_round3_p0_proxy_audit_results.json")


if __name__ == "__main__":
    main()
