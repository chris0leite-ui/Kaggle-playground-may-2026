"""SVM gate report: standalone OOF + ρ vs PRIMARY + K=27+1 min-meta lift.

Reads:
  - candidate OOF/test from scripts/artifacts/oof_<NAME>_strat.npy
  - K=27 PRIMARY pool (matches scripts/d18_path_b.py k27_v4h1d_d16_d18_e2_f2)
  - PRIMARY: oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy

Reports:
  - candidate OOF AUC vs LR-bank ceiling (0.92776) and PRIMARY OOF
  - ρ_test (Spearman) vs PRIMARY → predicted-LB-delta band
  - G3 rare-class flip ratio at top-1% threshold
  - K=27 LR-meta OOF (no candidate) vs K=27+1 LR-meta OOF (with candidate)

Usage:
  python scripts/svm_gate.py --candidate svm_nystroem_linsvc_g0.02
"""
from __future__ import annotations

import argparse
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

PRIMARY_NAME = "d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000"
LR_BANK_CEILING = 0.92776  # mega-LR OOF (knowledge-base anchor)

# K=27 pool from scripts/d18_path_b.py k27_v4h1d_d16_d18_e2_f2 (current PRIMARY).
POOL_KEEP = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
]
TOP_3_D9 = ["d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound"]
FM_AB = ["d9f_FM_A", "d9f_FM_B"]
K27_EXTRAS = [
    "p1_single_cb_v4_gpu", "d17_h1d_yekenot_full",
    "d16_orig_continuous_only", "d18_chain_decomp",
    "d18_e2_preimage_knn", "d18_f2_constraint",
]
K27_BASES = POOL_KEEP + TOP_3_D9 + FM_AB + K27_EXTRAS

# K=10 forward-selected core from scripts/t2_k10_primary.py (E9 pick order).
# Audit: K=10 = K=24 in OOF AUC; effective-rank-aware sparse pool.
# Sparser pool means less rank-lock → orthogonal base has more room to add.
K10_BASES = [
    "d17_h1d_yekenot_full", "p1_single_cb_v3_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _meta_oof(y: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="artifact stem; reads oof_<stem>_strat.npy etc.")
    args = ap.parse_args()
    name = args.candidate

    print(f"=== SVM gate: {name} ===\n")
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    cand_oof = _pos(ART / f"oof_{name}_strat.npy")
    cand_test = _pos(ART / f"test_{name}_strat.npy")
    primary_oof = _pos(ART / f"oof_{PRIMARY_NAME}_strat.npy")
    primary_test = _pos(ART / f"test_{PRIMARY_NAME}_strat.npy")

    cand_auc = float(roc_auc_score(y, cand_oof))
    primary_auc = float(roc_auc_score(y, primary_oof))
    print(f"  candidate OOF AUC: {cand_auc:.5f}")
    print(f"    vs LR-bank ceiling 0.92776  Δ {(cand_auc - LR_BANK_CEILING)*1e4:+.2f}bp")
    print(f"    vs PRIMARY      0.{int(primary_auc*100000):05d}  "
          f"Δ {(cand_auc - primary_auc)*1e4:+.2f}bp")

    rho, _ = spearmanr(cand_test, primary_test)
    rho = float(rho)
    print(f"\n  ρ_test vs PRIMARY: {rho:.5f}")

    # G3 rare-class flip ratio at top-1%
    rare_thr = float(np.quantile(primary_test, 0.99))
    pp = primary_test >= rare_thr
    cp = cand_test >= rare_thr
    n_to_neg = int((pp & ~cp).sum())
    n_to_pos = int((~pp & cp).sum())
    flip_ratio = (min(n_to_pos, n_to_neg) / max(n_to_pos, n_to_neg, 1)
                  if max(n_to_pos, n_to_neg) > 0 else 1.0)
    print(f"  G3 flip ratio (top-1%): {flip_ratio:.3f}  "
          f"(+→−: {n_to_neg}, −→+: {n_to_pos})")

    # Min-meta gate against both pools (sparse K=10 + dense K=27)
    deltas = {}
    for pool_name, bases in [("K=10 forward-selected", K10_BASES),
                              ("K=27 PRIMARY", K27_BASES)]:
        print(f"\n  --- {pool_name} + 1 LR-meta gate ---")
        pool_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in bases]
        P_oof_base = np.column_stack(pool_oofs)
        P_oof_with = np.column_stack(pool_oofs + [cand_oof])
        F_base = _expand(P_oof_base)
        F_with = _expand(P_oof_with)

        print(f"  K={len(bases)} base pool  feat={F_base.shape[1]}")
        _, auc_base = _meta_oof(y, F_base)
        _, auc_with = _meta_oof(y, F_with)
        delta_bp = (auc_with - auc_base) * 1e4
        print(f"  K={len(bases)}   LR-meta OOF: {auc_base:.5f}")
        print(f"  K={len(bases)}+1 LR-meta OOF: {auc_with:.5f}   "
              f"Δ {delta_bp:+.2f}bp")
        deltas[pool_name] = (auc_base, auc_with, delta_bp)

    # Verdict
    print(f"\n  --- verdict ---")
    best_delta = max(d[2] for d in deltas.values())
    if best_delta >= 0.5 and rho < 0.99:
        v = "PASS"
    elif best_delta >= 0:
        v = "WEAK_PASS"
    else:
        v = "FAIL"
    print(f"  verdict: {v}  (best Δ {best_delta:+.2f}bp,  "
          f"ρ {rho:.4f},  flip {flip_ratio:.2f})")
    for k, (b, w, d) in deltas.items():
        print(f"    {k}: K+1 Δ {d:+.2f}bp")


if __name__ == "__main__":
    main()
