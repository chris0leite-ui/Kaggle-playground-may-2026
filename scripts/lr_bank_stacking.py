"""scripts/lr_bank_stacking.py — stacking experiments on the LR bank.

Three Chris-Deotte-style experiments:

  1. **LR-meta over LR-bank only** (Chris's core architecture):
     fit LR-meta on the [P, rank, logit]-expansion of all LR bases.
     Compare OOF AUC to the best single LR base.

  2. **Forward selection within LR bank** (greedy CV):
     start with empty pool, add the LR base whose addition gives the
     largest OOF AUC bump. Stop when no positive-Δ addition exists.

  3. **K=24 ⊕ LR-bank min-meta gate**: does adding the LR bank to the
     existing K=24 GBDT pool lift the LR-meta OOF? Standard Δ vs PRIMARY.

Output: scripts/artifacts/lr_bank_stacking.json + console.
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

LR_BANK_DEFAULT = [
    "lr_raw_std", "lr_raw_std_balanced",
    "lr_raw_freq", "lr_raw_te", "lr_raw_ohe",
    "lr_poly2_std", "lr_poly2_ohe", "lr_poly3_std",
    "lr_kbins5_ohe", "lr_kbins20_ohe", "lr_kbins50_uniform", "lr_kbins_yekenot",
    "lr_splines_5", "lr_splines_10",
    "lr_hash_2way_2k", "lr_hash_3way_8k",
    "lr_l1_lasso_kbins20", "lr_C_low_kbins20", "lr_C_high_kbins20", "lr_balanced_kbins20",
    "lr_perseg_compound", "lr_perseg_year",
    "lr_on_top_models",
]


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P: np.ndarray) -> np.ndarray:
    """[P, rank/n, logit] expansion — same convention as probe_min_meta."""
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _meta_oof(y: np.ndarray, F: np.ndarray, lr_kwargs: dict | None = None) -> tuple[np.ndarray, float]:
    if lr_kwargs is None:
        lr_kwargs = dict(C=1.0, max_iter=2000, solver="lbfgs")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(**lr_kwargs)
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values

    lr_bases = [b for b in LR_BANK_DEFAULT if (ART / f"oof_{b}_strat.npy").exists()]
    gbdt_bases = [b for b in K24_GBDT_BASES if (ART / f"oof_{b}_strat.npy").exists()]
    print(f"LR bank: {len(lr_bases)} bases; GBDT pool: {len(gbdt_bases)} bases")

    P_lr = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in lr_bases])
    P_gbdt = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in gbdt_bases])

    out = dict(lr_bases=lr_bases, gbdt_bases=gbdt_bases)

    # ----- (1) LR-meta over LR-bank only
    F_lr = _expand(P_lr)
    t0 = time.time()
    oof_lr_meta, auc_lr_meta = _meta_oof(y, F_lr)
    print(f"\n[1] LR-meta over LR-bank only: OOF {auc_lr_meta:.5f}  ({time.time()-t0:.1f}s)")

    # Best single LR base (for context)
    single_aucs = [(b, float(roc_auc_score(y, P_lr[:, i]))) for i, b in enumerate(lr_bases)]
    single_aucs.sort(key=lambda x: -x[1])
    print(f"    best single LR base: {single_aucs[0][0]} OOF {single_aucs[0][1]:.5f}")
    print(f"    LR-meta lift over best-single: {(auc_lr_meta - single_aucs[0][1]) * 1e4:+.2f} bp")
    out["exp1_lr_meta_only"] = dict(
        auc=auc_lr_meta,
        best_single=single_aucs[0],
        lift_over_best_single_bp=(auc_lr_meta - single_aucs[0][1]) * 1e4,
    )

    # Save the LR-only meta as a candidate base for downstream stacking
    F_lr_test = None  # we don't need test in this script (no submission)

    # ----- (2) Forward selection within LR bank (greedy CV)
    print(f"\n[2] Forward selection within LR bank (greedy CV)")
    selected: list[int] = []
    remaining = set(range(len(lr_bases)))
    history = []
    cur_auc = 0.5  # placeholder; first iter picks best-single
    for step in range(len(lr_bases)):
        best_gain = -np.inf
        best_idx = None
        for j in remaining:
            cols = selected + [j]
            F = _expand(P_lr[:, cols])
            _, auc_j = _meta_oof(y, F)
            gain = auc_j - cur_auc
            if gain > best_gain:
                best_gain = gain
                best_idx = j
                best_auc = auc_j
        # stop when no positive gain (after at least 1 base)
        if step > 0 and best_gain <= 0:
            print(f"    step {step}: best-add yields {best_gain * 1e4:+.3f} bp — STOP")
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        cur_auc = best_auc
        history.append(dict(step=step + 1, added=lr_bases[best_idx],
                            n_selected=len(selected),
                            oof_auc=round(best_auc, 6),
                            gain_bp=round(best_gain * 1e4, 3)))
        print(f"    step {step + 1}: added {lr_bases[best_idx]:<26s}  "
              f"OOF {best_auc:.5f}  (+{best_gain * 1e4:.3f} bp)")
        if step >= 10:
            # cap depth at 11 to bound cost
            break
    out["exp2_forward_select_lr"] = dict(
        history=history,
        final_selected=[lr_bases[i] for i in selected],
        final_auc=cur_auc,
    )

    # ----- (3) K=24 + LR-bank min-meta gate
    print(f"\n[3] K=24 + LR-bank min-meta gate")
    F_gbdt = _expand(P_gbdt)
    t0 = time.time()
    oof_gbdt_meta, auc_gbdt = _meta_oof(y, F_gbdt)
    print(f"    K=24 LR-meta baseline:        OOF {auc_gbdt:.5f}  ({time.time()-t0:.1f}s)")

    # Add full LR bank
    F_full = _expand(np.column_stack([P_gbdt, P_lr]))
    t1 = time.time()
    oof_full, auc_full = _meta_oof(y, F_full)
    delta_bp = (auc_full - auc_gbdt) * 1e4
    print(f"    K=24 + LR-bank ({len(lr_bases)}) :  OOF {auc_full:.5f}  ({time.time()-t1:.1f}s)")
    print(f"    Δ vs K=24:                       {delta_bp:+.3f} bp")
    out["exp3a_k24_full_lr_bank"] = dict(
        k24_auc=auc_gbdt, full_auc=auc_full, delta_bp=delta_bp,
    )

    # Subset: only forward-selected LR bases
    if selected:
        sel_oofs = P_lr[:, selected]
        F_sub = _expand(np.column_stack([P_gbdt, sel_oofs]))
        oof_sub, auc_sub = _meta_oof(y, F_sub)
        delta_sub_bp = (auc_sub - auc_gbdt) * 1e4
        print(f"    K=24 + FS-selected ({len(selected)}) :  OOF {auc_sub:.5f}  Δ {delta_sub_bp:+.3f} bp")
        out["exp3b_k24_fs_lr_subset"] = dict(
            k24_auc=auc_gbdt, sub_auc=auc_sub, delta_bp=delta_sub_bp,
            selected=[lr_bases[i] for i in selected],
        )

    # Single-orthogonal-LR add (replicates the original min-meta gate behaviour)
    print(f"\n    K=24 + single-LR-base sweep (top 8 by lowest |ρ_PRIM|):")
    prim_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    rho_prim = []
    for i, b in enumerate(lr_bases):
        r, _ = spearmanr(P_lr[:, i], prim_oof)
        rho_prim.append((b, i, float(r)))
    rho_prim.sort(key=lambda x: abs(x[2]))
    out["exp3c_k24_single_add_sweep"] = []
    for b, i, r in rho_prim[:8]:
        F1 = _expand(np.column_stack([P_gbdt, P_lr[:, i:i+1]]))
        _, auc1 = _meta_oof(y, F1)
        d_bp = (auc1 - auc_gbdt) * 1e4
        print(f"      add {b:<26s}  ρ_PRIM {r:+.4f}  K=25 OOF {auc1:.5f}  Δ {d_bp:+.3f} bp")
        out["exp3c_k24_single_add_sweep"].append(
            dict(name=b, rho_prim=r, auc=auc1, delta_bp=d_bp))

    out_json = ART / "lr_bank_stacking.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n→ JSON saved: {out_json}")


if __name__ == "__main__":
    main()
