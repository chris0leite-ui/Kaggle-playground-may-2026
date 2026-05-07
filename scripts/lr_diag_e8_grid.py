"""scripts/lr_diag_e8_grid.py — E8: class_weight × C × penalty grid.

Diagnostic. Settles empirically (for our binary AUC) the 12th-place
"three axes" claim. Grid over LR-meta(K=24):
  - class_weight ∈ {None, 'balanced'}
  - C ∈ {0.001, 0.01, 0.1, 1, 10, 100}
  - penalty ∈ {l2, l1}  (l1 only for compatible solvers)

Records OOF AUC + ρ vs anchor (class_weight=None, C=1.0, l2) so we
see RANK-CORRELATION between configs (not just AUC). For binary AUC
class_weight is theoretically rank-no-op; this experiment verifies.

Output: scripts/artifacts/lr_diag_e8_grid.json + console.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
]
EXTRAS = ["d16_orig_continuous_only", "p1_single_cb_v3_gpu",
          "d17_h1d_yekenot_full"]
ALL_BASES = K21_BASES + EXTRAS


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P):
    n = len(P)
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    rk = np.column_stack([
        np.argsort(np.argsort(c)) / n for c in P.T
    ])
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _oof(F, y, cw, C, penalty):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    solver = "saga" if penalty == "l1" else "lbfgs"
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(
            C=C, class_weight=cw, penalty=penalty,
            solver=solver, max_iter=4000
        )
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values
    P = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in ALL_BASES])
    F = _expand(P)

    # Anchor
    print("Fitting anchor (cw=None, C=1.0, l2)...")
    anchor_oof, anchor_auc = _oof(F, y, None, 1.0, "l2")

    grid = []
    cw_vals = [None, "balanced"]
    C_vals = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    pen_vals = ["l2", "l1"]
    total = len(cw_vals) * len(C_vals) * len(pen_vals)
    i = 0
    for cw in cw_vals:
        for pen in pen_vals:
            for C in C_vals:
                i += 1
                print(f"[{i}/{total}] cw={cw} C={C:>7g} penalty={pen} ...",
                      end="", flush=True)
                try:
                    oof, auc = _oof(F, y, cw, C, pen)
                    rho, _ = spearmanr(oof, anchor_oof)
                    bp = (auc - anchor_auc) * 1e4
                    print(f"  AUC={auc:.5f}  Δ={bp:+.2f}bp  ρ={rho:.5f}")
                    grid.append({
                        "class_weight": str(cw),
                        "C": C,
                        "penalty": pen,
                        "oof_auc": round(auc, 6),
                        "delta_bp_vs_anchor": round(float(bp), 3),
                        "spearman_rho_vs_anchor": round(float(rho), 6),
                    })
                except Exception as e:
                    print(f"  FAILED: {e}")
                    grid.append({
                        "class_weight": str(cw), "C": C, "penalty": pen,
                        "error": str(e)[:80],
                    })

    out = {
        "anchor_auc": round(anchor_auc, 6),
        "grid": grid,
    }
    json_path = ART / "lr_diag_e8_grid.json"
    json_path.write_text(json.dumps(out, indent=2))

    print("\n=== E8 grid summary ===")
    print(f"anchor (cw=None, C=1.0, l2): AUC={anchor_auc:.5f}")
    rows_ok = [r for r in grid if "oof_auc" in r]
    if rows_ok:
        best = max(rows_ok, key=lambda r: r["oof_auc"])
        worst = min(rows_ok, key=lambda r: r["oof_auc"])
        print(f"best:  {best}")
        print(f"worst: {worst}")
        # rank-correlation perspective
        rhos = [r["spearman_rho_vs_anchor"] for r in rows_ok]
        print(f"Spearman ρ to anchor: min={min(rhos):.5f}, "
              f"median={float(np.median(rhos)):.5f}, max={max(rhos):.5f}")
        bps = [r["delta_bp_vs_anchor"] for r in rows_ok]
        print(f"Δ bp range: [{min(bps):+.2f}, {max(bps):+.2f}]")
    print(f"\n→ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
