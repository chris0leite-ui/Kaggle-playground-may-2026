"""Day-16 H2 — Twin parallel-pool 2-meta blend.

ε2 axis. Friction `lr-meta-rank-lock-strong-anchor` says LR-meta on a
strong-anchor pool is rank-saturated. The mechanism distinct from
'add another base' is to build TWO independent metas over DISJOINT
base subsets, then blend the two metas with a top-level LR.

Pool A = 6 strong GBDT/baseline bases (single-architecture-class).
Pool B = 5 FM-class + LR + rule + DAE (model-class diverse).
Each gets its own LR meta with [raw, rank, logit] expand. Top-level LR
combines metaA + metaB.

If ρ(metaA, metaB) is meaningfully below 0.999, the 2-meta-blend
captures structure that the SINGLE LR-meta over all 11 bases cannot
(the latter has access to the same info but the meta-routing is
constrained to a SINGLE convex combination of all 11 logits).

Output:
  oof_d16_h2_twin_meta_strat.npy   (n_train, 2)
  test_d16_h2_twin_meta_strat.npy  (n_test, 2)
  d16_h2_twin_meta_results.json
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
SEED, N_FOLDS = 42, 5

# Pool A — 6 strong GBDT / HGBC / single-architecture
POOL_A = [
    "e3_hgbc",
    "e5_optuna_lgbm",
    "cb_year-cat",
    "cb_lossguide",
    "cb_slow-wide-bag",
    "f1_hgbc_deep",
]

# Pool B — 5 model-class diverse: FM-class, sparse-LR, rule-residual, DAE
POOL_B = [
    "d9f_FM_A",
    "d9f_FM_B",
    "d6_rule_compound_stint",
    "d6_rule_year_race",
    "d15b_lgbm_dae_only",
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


def _meta_full(y, F_oof, F_test):
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr.fit(F_oof, y)
    return lr.predict_proba(F_test)[:, 1]


def main():
    t0 = time.time()
    y = pd.read_csv("data/train.csv", usecols=["PitNextLap"])["PitNextLap"].astype(int).values

    # Load pools
    print(f"[h2] loading Pool A ({len(POOL_A)}): {POOL_A}", flush=True)
    A_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in POOL_A]
    A_tests = [_pos(ART / f"test_{b}_strat.npy") for b in POOL_A]
    print(f"[h2] loading Pool B ({len(POOL_B)}): {POOL_B}", flush=True)
    B_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in POOL_B]
    B_tests = [_pos(ART / f"test_{b}_strat.npy") for b in POOL_B]

    # Pool A meta
    PA_oof = np.column_stack(A_oofs)
    PA_test = np.column_stack(A_tests)
    FA_oof = _expand(PA_oof)
    FA_test = _expand(PA_test)
    metaA_oof, aucA = _meta_oof(y, FA_oof)
    metaA_test = _meta_full(y, FA_oof, FA_test)
    print(f"[h2] meta A (K={len(POOL_A)}) OOF AUC = {aucA:.5f}", flush=True)

    # Pool B meta
    PB_oof = np.column_stack(B_oofs)
    PB_test = np.column_stack(B_tests)
    FB_oof = _expand(PB_oof)
    FB_test = _expand(PB_test)
    metaB_oof, aucB = _meta_oof(y, FB_oof)
    metaB_test = _meta_full(y, FB_oof, FB_test)
    print(f"[h2] meta B (K={len(POOL_B)}) OOF AUC = {aucB:.5f}", flush=True)

    # Top-level meta over [metaA, metaB]
    rho_AB = float(spearmanr(metaA_test, metaB_test)[0])
    print(f"[h2] ρ(meta A test, meta B test) = {rho_AB:.6f}", flush=True)

    P_top = np.column_stack([metaA_oof, metaB_oof])
    P_top_test = np.column_stack([metaA_test, metaB_test])
    F_top = _expand(P_top)
    F_top_test = _expand(P_top_test)
    top_oof, aucTop = _meta_oof(y, F_top)
    top_test = _meta_full(y, F_top, F_top_test)
    print(f"[h2] top-level meta [A, B] OOF AUC = {aucTop:.5f}", flush=True)

    # Compare to single LR meta over A+B (11 bases combined)
    P_all = np.column_stack(A_oofs + B_oofs)
    P_all_test = np.column_stack(A_tests + B_tests)
    F_all = _expand(P_all)
    F_all_test = _expand(P_all_test)
    all_oof, aucAll = _meta_oof(y, F_all)
    all_test = _meta_full(y, F_all, F_all_test)
    print(f"[h2] single LR-meta [A∪B, K={len(POOL_A)+len(POOL_B)}] OOF AUC = {aucAll:.5f}",
          flush=True)
    delta_vs_single = (aucTop - aucAll) * 1e4
    print(f"[h2] Δ twin vs single = {delta_vs_single:+.3f}bp", flush=True)

    # Save twin output as a candidate base for K=22+1 stack
    np.save(ART / "oof_d16_h2_twin_meta_strat.npy",
            np.column_stack([1.0 - top_oof, top_oof]))
    np.save(ART / "test_d16_h2_twin_meta_strat.npy",
            np.column_stack([1.0 - top_test, top_test]))

    # Compare to PRIMARY too
    PR_oof = _pos(ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy")
    PR_test = _pos(ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy")
    aucPR = float(roc_auc_score(y, PR_oof))
    rho_top_pr = float(spearmanr(top_test, PR_test)[0])
    print(f"[h2] PRIMARY OOF AUC: {aucPR:.5f}", flush=True)
    print(f"[h2] ρ(twin top, PRIMARY) = {rho_top_pr:.6f}", flush=True)

    res = dict(pool_A=POOL_A, pool_B=POOL_B,
               aucA=aucA, aucB=aucB, aucTop=aucTop, aucAllSingle=aucAll,
               delta_twin_vs_single_bp=float(delta_vs_single),
               rho_AB=float(rho_AB),
               rho_top_vs_primary=float(rho_top_pr),
               primary_oof_auc=aucPR,
               wall_s=time.time() - t0)
    (ART / "d16_h2_twin_meta_results.json").write_text(json.dumps(res, indent=2))
    print(f"[h2] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
