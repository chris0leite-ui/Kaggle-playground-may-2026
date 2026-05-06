"""scripts/probe_min_meta.py — min-meta gate for K=K+1 stack additions.

Standardized probe: compare K=21 PRIMARY-pool LR meta OOF vs
K=21 + N candidate bases LR meta OOF. Reports per-candidate delta-bp.

Usage:
  python scripts/probe_min_meta.py --candidates d6_rule_compound_stint d12_lr_meta

Loads y from data/train.csv, K=21 from PRIMARY-pool list (d13e
convention), candidate from `oof_<NAME>_strat.npy` / `test_<NAME>_strat.npy`.
"""
from __future__ import annotations

import argparse
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

K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
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


def _meta_oof(y, F):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def _meta_full_test(y, F_oof, F_test):
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr.fit(F_oof, y)
    return lr.predict_proba(F_test)[:, 1], lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="base name(s); each must have oof_<NAME>_strat.npy "
                         "and test_<NAME>_strat.npy under scripts/artifacts/")
    ap.add_argument("--mode", choices=["add", "swap"], default="add",
                    help="add = K22; swap = replace lowest-L1 K=21 base")
    ap.add_argument("--save-prefix", default=None,
                    help="save artifacts as oof_{prefix}_strat.npy etc")
    args = ap.parse_args()

    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    primary_test = _pos(ART / "test_d13e_compound_stint_tau20000_strat.npy")

    pool_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]
    pool_tests = [_pos(ART / f"test_{b}_strat.npy") for b in K21_BASES]
    P_oof_base = np.column_stack(pool_oofs)
    P_test_base = np.column_stack(pool_tests)
    F_oof_base = _expand(P_oof_base)
    F_test_base = _expand(P_test_base)

    t0 = time.time()
    oof_base, auc_base = _meta_oof(y, F_oof_base)
    print(f"\n=== K=21 LR-meta baseline ===")
    print(f"  OOF: {auc_base:.5f}  ({time.time()-t0:.1f}s)")

    cand_oofs = []
    cand_tests = []
    for c in args.candidates:
        cand_oofs.append(_pos(ART / f"oof_{c}_strat.npy"))
        cand_tests.append(_pos(ART / f"test_{c}_strat.npy"))
    cand_oof_arr = np.column_stack(cand_oofs)
    cand_test_arr = np.column_stack(cand_tests)

    P_oof_with = np.column_stack([P_oof_base, cand_oof_arr])
    P_test_with = np.column_stack([P_test_base, cand_test_arr])
    F_oof_with = _expand(P_oof_with)
    F_test_with = _expand(P_test_with)

    t1 = time.time()
    oof_with, auc_with = _meta_oof(y, F_oof_with)
    print(f"\n=== K=21 + {len(args.candidates)} ({'+'.join(args.candidates)}) ===")
    print(f"  OOF: {auc_with:.5f}  ({time.time()-t1:.1f}s)")
    delta_bp = (auc_with - auc_base) * 1e4
    print(f"  Δ vs K=21 baseline: {delta_bp:+.3f} bp")

    test_with, lr_full = _meta_full_test(y, F_oof_with, F_test_with)
    rho, _ = spearmanr(test_with, primary_test)
    print(f"  ρ vs PRIMARY (d13e Compound×Stint τ=20k): {rho:.6f}")

    # L1 of last cand cols (raw + rank + logit weights summed)
    K = P_oof_with.shape[1]
    n_cand = len(args.candidates)
    raw_w = lr_full.coef_.ravel()
    # column layout: [raw_K, rank_K, logit_K]
    print(f"\n  Per-candidate weight summary (sum-of-3-columns |w|):")
    for j, name in enumerate(args.candidates):
        col = K - n_cand + j
        w_raw = raw_w[col]
        w_rank = raw_w[K + col]
        w_logit = raw_w[2 * K + col]
        l1 = abs(w_raw) + abs(w_rank) + abs(w_logit)
        print(f"    {name:<40s}  |w| = {l1:.4f}  "
              f"(raw {w_raw:+.3f}, rank {w_rank:+.3f}, logit {w_logit:+.3f})")

    summary = dict(
        candidates=args.candidates, mode=args.mode,
        k_pool=len(K21_BASES), auc_base=auc_base, auc_with=auc_with,
        delta_bp=float(delta_bp), rho_vs_primary=float(rho),
    )
    if args.save_prefix:
        oof2 = np.column_stack([1 - oof_with, oof_with])
        test2 = np.column_stack([1 - test_with, test_with])
        np.save(ART / f"oof_{args.save_prefix}_strat.npy", oof2)
        np.save(ART / f"test_{args.save_prefix}_strat.npy", test2)
        print(f"\n  → saved oof/test_{args.save_prefix}_strat.npy")
    json_out = ART / "probe_min_meta_last.json"
    json_out.write_text(json.dumps(summary, indent=2))
    print(f"  → {json_out}")


if __name__ == "__main__":
    main()
