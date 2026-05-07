"""Day-16 multi-candidate K=22+N min-meta gate against PRIMARY's K=22 pool.

PRIMARY is hier-meta(K=22, Compound×Stint, τ=20k) over K=21 + d15b_dae_only.
For min-meta we use the LR-meta with [raw,rank,logit] over the same K=22
plus N candidates. Compare to LR-meta(K=22) baseline and to PRIMARY itself.

Reports per-candidate L1 weight + multi-add OOF Δ + ρ vs PRIMARY.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5

K22_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
    "d15b_lgbm_dae_only",
]

PRIMARY_OOF = ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"


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


def _meta_full(y, F_oof, F_test):
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr.fit(F_oof, y)
    return lr.predict_proba(F_test)[:, 1], lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    y = pd.read_csv("data/train.csv", usecols=["PitNextLap"])["PitNextLap"].astype(int).values

    print(f"[mm] loading K=22 pool ...", flush=True)
    pool_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K22_BASES]
    pool_tests = [_pos(ART / f"test_{b}_strat.npy") for b in K22_BASES]
    P_oof_base = np.column_stack(pool_oofs)
    P_test_base = np.column_stack(pool_tests)
    F_oof_base = _expand(P_oof_base)
    F_test_base = _expand(P_test_base)

    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))
    print(f"[mm] PRIMARY hier-meta OOF AUC = {auc_primary:.5f}", flush=True)

    t0 = time.time()
    oof_base, auc_base = _meta_oof(y, F_oof_base)
    print(f"[mm] LR-meta(K={len(K22_BASES)}) OOF AUC = {auc_base:.5f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    cand_oofs = []
    cand_tests = []
    valid_cands = []
    for c in args.candidates:
        p_oof = ART / f"oof_{c}_strat.npy"
        p_te = ART / f"test_{c}_strat.npy"
        if not p_oof.exists() or not p_te.exists():
            print(f"[mm] WARN: missing artifact for {c} (skipping)", flush=True)
            continue
        cand_oofs.append(_pos(p_oof))
        cand_tests.append(_pos(p_te))
        valid_cands.append(c)
    if not valid_cands:
        print("[mm] no candidates loaded; abort", flush=True)
        return
    cand_oof_arr = np.column_stack(cand_oofs)
    cand_test_arr = np.column_stack(cand_tests)

    P_oof_with = np.column_stack([P_oof_base, cand_oof_arr])
    P_test_with = np.column_stack([P_test_base, cand_test_arr])
    F_oof_with = _expand(P_oof_with)
    F_test_with = _expand(P_test_with)

    t1 = time.time()
    oof_with, auc_with = _meta_oof(y, F_oof_with)
    print(f"[mm] LR-meta(K={len(K22_BASES)}+{len(valid_cands)}) OOF AUC "
          f"= {auc_with:.5f}  ({time.time()-t1:.0f}s)", flush=True)
    delta_lr = (auc_with - auc_base) * 1e4
    delta_pr = (auc_with - auc_primary) * 1e4
    print(f"[mm] Δ vs LR-meta(K=22): {delta_lr:+.3f} bp", flush=True)
    print(f"[mm] Δ vs PRIMARY hier:  {delta_pr:+.3f} bp", flush=True)

    test_with, lr_full = _meta_full(y, F_oof_with, F_test_with)
    rho_pr = float(spearmanr(test_with, primary_test)[0])
    print(f"[mm] ρ vs PRIMARY: {rho_pr:.6f}", flush=True)

    K = P_oof_with.shape[1]
    n_cand = len(valid_cands)
    raw_w = lr_full.coef_.ravel()
    print("\n[mm] Per-candidate |w| (sum of raw + rank + logit):", flush=True)
    cand_weights = []
    for j, name in enumerate(valid_cands):
        col = K - n_cand + j
        w_raw = raw_w[col]
        w_rank = raw_w[K + col]
        w_logit = raw_w[2 * K + col]
        l1 = abs(w_raw) + abs(w_rank) + abs(w_logit)
        print(f"    {name:<35s} |w|={l1:.4f}  raw={w_raw:+.3f} rank={w_rank:+.3f} logit={w_logit:+.3f}",
              flush=True)
        cand_weights.append(dict(name=name, l1=float(l1),
                                 raw=float(w_raw), rank=float(w_rank),
                                 logit=float(w_logit)))

    res = dict(
        candidates=valid_cands,
        k_pool=len(K22_BASES),
        auc_lr_meta_K22=auc_base,
        auc_lr_meta_K22_plus_N=auc_with,
        delta_vs_lr_meta_bp=float(delta_lr),
        primary_hier_oof_auc=auc_primary,
        delta_vs_primary_bp=float(delta_pr),
        rho_vs_primary=rho_pr,
        cand_weights=cand_weights,
    )
    out_path = args.out or (ART / f"d16_min_meta__{'+'.join(valid_cands)[:60]}.json")
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(f"\n[mm] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
