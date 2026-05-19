"""scripts/probe_r13_operator_sweep.py — Phase B operator-class diagnostic.

Per experiment-loop.md §95-114: cb_horizon proved itself under Path-B
DCS τ=100k. Confirm whether the +0.03 bp LB lift came from the base
diversity (cb_horizon's count-regression signal) or from the operator
(Path-B's segmented LR-meta shrinkage). Sweep LR-meta with C ∈ {1, 10}
on the K=13 pool with and without cb_horizon — if cb_horizon adds at
the LR-meta level, base diversity is doing the work; if it doesn't,
Path-B segmentation is the entire mechanism.

Not a gate; informational only. Per Rule 19, log the result to
`audit/decisions.jsonl` regardless of direction.

Outputs:
- `audit/2026-05-19-r13-operator-sweep.json` — per-C OOF AUC for
  K=13 baseline + K=14 (K=13 + cb_horizon) + reads.

Usage:
  python scripts/probe_r13_operator_sweep.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from build_K13_pathb_multiseg import K13_FILES
from build_K11_full_pathb import _pos, expand

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_ITER = 2000


def lr_meta_oof(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """5-fold StratifiedKFold OOF predictions from LR-meta on X."""
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for ti, vi in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(X[ti], y[ti])
        oof[vi] = lr.predict_proba(X[vi])[:, 1]
    return oof


def main() -> None:
    t0 = time.time()
    print("== R13 Phase B: operator-class diagnostic on cb_horizon ==",
          flush=True)
    y = pd.read_csv("data/train.csv")[TARGET].astype(int).values
    print(f"  y: {len(y)} rows, prior {y.mean():.4f}", flush=True)

    # K=13 OOF columns
    k13_oof_cols = []
    k13_names = []
    for name, oof_file, _ in K13_FILES:
        p = ART / oof_file
        if not p.exists():
            print(f"  MISSING: {p}", flush=True)
            return
        col = _pos(p)
        if len(col) != len(y):
            print(f"  WARN shape {name}: {len(col)} != {len(y)}", flush=True)
            return
        k13_oof_cols.append(col)
        k13_names.append(name)
    K13 = np.column_stack(k13_oof_cols)
    print(f"  K=13 OOF matrix: {K13.shape}", flush=True)

    # cb_horizon OOF column
    cbh_oof = np.load(ART / "oof_R12_cb_horizon_strat.npy").astype(np.float64)
    print(f"  cb_horizon OOF: {cbh_oof.shape} mean {cbh_oof.mean():.4f}",
          flush=True)
    K14 = np.column_stack([K13, cbh_oof])

    # Expand both: [raw, rank, logit] columns (same as Path-B)
    K13_exp = expand(K13)
    K14_exp = expand(K14)
    print(f"  K=13 expanded: {K13_exp.shape}; K=14 expanded: {K14_exp.shape}",
          flush=True)

    # R12-2 PRIMARY OOF (Path-B operator) for reference
    r12_oof = np.load(ART / "oof_K13_pathb_driverclass_stint_tau100000.npy")
    # Note: this file was OVERWRITTEN by the K=14 cb_horizon Path-B build
    # earlier today. Recompute against the canonical R12-2 by using the
    # builder's output for the K=14+cb_horizon variant if it exists, or
    # just use cb_horizon K=14 oof.
    # build_K13_pathb_multiseg writes oof_K14_pathb_<seg>_tau<X>.npy
    r12_oof_path = ART / "oof_K14_pathb_driverclass_stint_tau100000.npy"
    if r12_oof_path.exists():
        r12_oof = np.load(r12_oof_path)
    auc_r12 = float(roc_auc_score(y, r12_oof))
    print(f"  R12-2 PRIMARY OOF (Path-B reference): {auc_r12:.6f}", flush=True)

    results = {}
    for C in (0.1, 1.0, 10.0):
        for X_exp, label in ((K13_exp, "K13"), (K14_exp, "K14")):
            oof_lr = lr_meta_oof(X_exp, y, C)
            auc = float(roc_auc_score(y, oof_lr))
            key = f"LR-meta C={C} on {label}"
            results[key] = dict(auc=auc, n_features=X_exp.shape[1])
            print(f"  {key}:  OOF AUC = {auc:.6f}", flush=True)

    # Read: does cb_horizon add at LR-meta (any C)?
    cb_adds_at_lr = False
    cb_lift_bp = {}
    for C in (0.1, 1.0, 10.0):
        k13_auc = results[f"LR-meta C={C} on K13"]["auc"]
        k14_auc = results[f"LR-meta C={C} on K14"]["auc"]
        lift_bp = (k14_auc - k13_auc) * 1e4
        cb_lift_bp[f"C={C}"] = lift_bp
        if lift_bp > 0.05:
            cb_adds_at_lr = True
    print(f"\n  cb_horizon lift at LR-meta per C: {cb_lift_bp}", flush=True)

    # Compare LR-meta baseline to Path-B PRIMARY (R12-2 OOF)
    # LR-meta-baseline = K=13 LR-meta best C; Path-B PRIMARY = R12-2 OOF
    best_k13_lr = max(results[f"LR-meta C={C} on K13"]["auc"]
                      for C in (0.1, 1.0, 10.0))
    best_k14_lr = max(results[f"LR-meta C={C} on K14"]["auc"]
                      for C in (0.1, 1.0, 10.0))
    print(f"\n  Path-B R12-2 PRIMARY:  {auc_r12:.6f}", flush=True)
    print(f"  Best K=13 LR-meta:    {best_k13_lr:.6f}", flush=True)
    print(f"  Best K=14 LR-meta:    {best_k14_lr:.6f}", flush=True)
    print(f"  Path-B vs best K=14 LR: Δ={ (auc_r12 - best_k14_lr) * 1e4:+.2f} bp",
          flush=True)

    # Verdict
    if cb_adds_at_lr:
        verdict = ("cb_horizon ADDS at LR-meta — base diversity is doing "
                   "the work. Scale base diversity (more orthogonal-target "
                   "CBs).")
    else:
        verdict = ("cb_horizon does NOT add at LR-meta — Path-B segmentation "
                   "is the mechanism. Scale Path-B (alt-seg / tau sweep).")
    print(f"\n  VERDICT: {verdict}", flush=True)

    summary = dict(
        round="R13_B_operator_sweep",
        results=results,
        cb_lift_bp_per_C=cb_lift_bp,
        path_b_primary_oof=auc_r12,
        best_k13_lr=best_k13_lr,
        best_k14_lr=best_k14_lr,
        path_b_vs_best_k14_lr_bp=(auc_r12 - best_k14_lr) * 1e4,
        cb_adds_at_lr=cb_adds_at_lr,
        verdict=verdict,
        wall_s=time.time() - t0,
    )
    out_json = Path("audit/2026-05-19-r13-operator-sweep.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
