"""D17 H1 SWAP gate — replace `realmlp` slot in K=21 with d17_h1_yekenot_realmlp.

Companion to scripts/probe_min_meta.py (which is K=22 ADD).
This SWAPS our default-config realmlp base out and yekenot's recipe in,
preserving K=21 cardinality. Reports K=21-baseline-vs-K=21-swap delta.

Usage:
  python scripts/d17_h1_swap_gate.py
  python scripts/d17_h1_swap_gate.py --candidate d17_h1_yekenot_realmlp_strong
"""
from __future__ import annotations

import argparse
import json
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

# Same K=21 list as probe_min_meta.py
K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
]
SWAP_OUT = "realmlp"  # the slot to replace


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
    ap.add_argument("--candidate", default="d17_h1_yekenot_realmlp")
    args = ap.parse_args()

    cand = args.candidate
    swap_bases = [b if b != SWAP_OUT else cand for b in K21_BASES]
    print(f"=== K=21 SWAP gate: {SWAP_OUT} → {cand} ===")
    print(f"  pool size: {len(swap_bases)} (unchanged)")

    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    primary_test = _pos(ART / "test_d13e_compound_stint_tau20000_strat.npy")

    # Baseline K=21 (with default realmlp)
    base_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]
    base_tests = [_pos(ART / f"test_{b}_strat.npy") for b in K21_BASES]
    P_o = np.column_stack(base_oofs)
    P_t = np.column_stack(base_tests)
    F_o = _expand(P_o)
    F_t = _expand(P_t)
    _, auc_base = _meta_oof(y, F_o)
    print(f"  K=21 baseline OOF: {auc_base:.5f}")

    # Swap variant
    swap_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in swap_bases]
    swap_tests = [_pos(ART / f"test_{b}_strat.npy") for b in swap_bases]
    P_os = np.column_stack(swap_oofs)
    P_ts = np.column_stack(swap_tests)
    F_os = _expand(P_os)
    F_ts = _expand(P_ts)
    oof_swap, auc_swap = _meta_oof(y, F_os)
    print(f"  K=21 swap OOF:     {auc_swap:.5f}")
    delta_bp = (auc_swap - auc_base) * 1e4
    print(f"  Δ (swap − base):   {delta_bp:+.3f} bp")

    test_swap, lr_full = _meta_full_test(y, F_os, F_ts)
    rho, _ = spearmanr(test_swap, primary_test)
    print(f"  ρ vs PRIMARY (d13e Compound×Stint τ=20k): {rho:.6f}")

    # Per-base |w|
    K = len(swap_bases)
    raw_w = lr_full.coef_.ravel()
    print(f"\n  Per-base weight (sum of raw+rank+logit |w|):")
    rows = []
    for j, name in enumerate(swap_bases):
        w_raw = raw_w[j]
        w_rank = raw_w[K + j]
        w_logit = raw_w[2 * K + j]
        l1 = abs(w_raw) + abs(w_rank) + abs(w_logit)
        rows.append((name, l1, w_raw, w_rank, w_logit))
    # show all sorted descending by |w|
    for name, l1, wr, wk, wl in sorted(rows, key=lambda r: -r[1]):
        marker = " <-- swapped in" if name == cand else ""
        print(f"    {name:<40s}  |w|={l1:.4f} "
              f"(raw {wr:+.3f}, rank {wk:+.3f}, logit {wl:+.3f}){marker}")

    summary = dict(
        candidate=cand, swap_out=SWAP_OUT,
        auc_base=auc_base, auc_swap=auc_swap,
        delta_bp=float(delta_bp),
        rho_vs_primary=float(rho),
        per_base_weights={r[0]: dict(l1=r[1], raw=r[2], rank=r[3], logit=r[4])
                          for r in rows},
    )
    out = ART / f"d17_h1_swap_gate__{cand}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
