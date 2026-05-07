"""scripts/lr_bank_stacking_fast.py — fast stacking experiments.

Streamlined version of lr_bank_stacking.py — drops forward selection
(too expensive on 15 bases × multiple dims), keeps the high-info questions:

  1. LR-meta over LR-bank only (Chris's core architecture).
  2. K=24 GBDT + LR-bank → LR-meta gate (does adding LR move the dial?).
  3. K=24 + single-LR-add sweep (top 5 by ρ-orthogonality and AUC).

Output: scripts/artifacts/lr_bank_stacking_fast.json + console.
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

K24_GBDT_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
    "d16_orig_continuous_only", "p1_single_cb_v3_gpu",
    "d17_h1d_yekenot_full",
]

LR_BANK = [
    "lr_raw_std", "lr_raw_std_balanced",
    "lr_raw_freq", "lr_raw_te", "lr_raw_ohe",
    "lr_poly2_std",
    "lr_kbins5_ohe", "lr_kbins20_ohe", "lr_kbins50_uniform", "lr_kbins_yekenot",
    "lr_splines_5",
    "lr_C_low_kbins20", "lr_C_high_kbins20", "lr_balanced_kbins20",
    "lr_l1_lasso_kbins20",
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
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values

    lr_bases = [b for b in LR_BANK if (ART / f"oof_{b}_strat.npy").exists()]
    gbdt_bases = [b for b in K24_GBDT_BASES if (ART / f"oof_{b}_strat.npy").exists()]
    print(f"LR bank: {len(lr_bases)} ; GBDT pool: {len(gbdt_bases)}", flush=True)

    P_lr = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in lr_bases])
    P_gbdt = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in gbdt_bases])

    out = dict(lr_bases=lr_bases, gbdt_bases=gbdt_bases)

    # ----- (1) LR-meta-of-LRs only
    F_lr = _expand(P_lr)
    print(f"\n[1] LR-meta over LR-bank ({F_lr.shape[1]}-dim) ...", flush=True)
    t0 = time.time()
    oof_lr_meta, auc_lr_meta = _meta_oof(y, F_lr)
    print(f"    OOF {auc_lr_meta:.5f}  ({time.time()-t0:.1f}s)", flush=True)
    single_aucs = [(b, float(roc_auc_score(y, P_lr[:, i]))) for i, b in enumerate(lr_bases)]
    single_aucs.sort(key=lambda x: -x[1])
    print(f"    best single LR base: {single_aucs[0][0]} OOF {single_aucs[0][1]:.5f}", flush=True)
    print(f"    LR-meta lift over best-single: {(auc_lr_meta - single_aucs[0][1]) * 1e4:+.2f} bp", flush=True)
    out["exp1_lr_meta_only"] = dict(
        n_bases=len(lr_bases), auc=auc_lr_meta,
        best_single=single_aucs[0],
        lift_over_best_single_bp=(auc_lr_meta - single_aucs[0][1]) * 1e4,
    )

    # ----- (2) K=24 GBDT baseline + K=24 + LR-bank
    F_gbdt = _expand(P_gbdt)
    print(f"\n[2] K=24 GBDT baseline LR-meta ({F_gbdt.shape[1]}-dim) ...", flush=True)
    t0 = time.time()
    oof_gbdt_meta, auc_gbdt = _meta_oof(y, F_gbdt)
    print(f"    OOF {auc_gbdt:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    F_full = _expand(np.column_stack([P_gbdt, P_lr]))
    print(f"\n    K=24 + LR-bank ({F_full.shape[1]}-dim) ...", flush=True)
    t0 = time.time()
    oof_full, auc_full = _meta_oof(y, F_full)
    delta_bp = (auc_full - auc_gbdt) * 1e4
    print(f"    OOF {auc_full:.5f}  Δ {delta_bp:+.3f} bp  ({time.time()-t0:.1f}s)", flush=True)
    out["exp2_k24_full_lr_bank"] = dict(
        k24_auc=auc_gbdt, full_auc=auc_full, delta_bp=delta_bp,
        n_lr_added=len(lr_bases),
    )

    # ----- (3) K=24 + single-LR sweep (top 5 by ρ-orthogonality vs PRIMARY)
    prim_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    rho_prim = []
    for i, b in enumerate(lr_bases):
        r, _ = spearmanr(P_lr[:, i], prim_oof)
        rho_prim.append((b, i, float(r)))
    rho_prim.sort(key=lambda x: abs(x[2]))
    print(f"\n[3] K=24 + single-LR add sweep (top 5 by orthogonality):", flush=True)
    out["exp3_k24_plus_one_sweep"] = []
    for b, i, r in rho_prim[:5]:
        F1 = _expand(np.column_stack([P_gbdt, P_lr[:, i:i+1]]))
        t0 = time.time()
        _, auc1 = _meta_oof(y, F1)
        d_bp = (auc1 - auc_gbdt) * 1e4
        print(f"    add {b:<26s}  ρ_PRIM {r:+.4f}  K=25 OOF {auc1:.5f}  Δ {d_bp:+.3f} bp  ({time.time()-t0:.0f}s)", flush=True)
        out["exp3_k24_plus_one_sweep"].append(
            dict(name=b, rho_prim=r, auc=auc1, delta_bp=d_bp))

    # ----- Persist
    out_json = ART / "lr_bank_stacking_fast.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n→ saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
