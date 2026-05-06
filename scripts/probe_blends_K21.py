"""scripts/probe_blends_K21.py — alternative-blender probe.

Cheap probe: replace the K=21 LR meta with 4 alternative aggregators.
- mean        arithmetic mean of probabilities
- gmean       geometric mean (log-mean → exp)
- rank_mean   mean of per-base ranks (then rescaled to [0,1])
- trimmed     mean after dropping per-row top-3 and bot-3 base values

Each blend writes:
  scripts/artifacts/oof_blend_{kind}_K21_strat.npy
  scripts/artifacts/test_blend_{kind}_K21_strat.npy

Then runs the harness gate (ρ-only) against current PRIMARY. Prints the
table; saves a JSON summary at scripts/artifacts/probe_blends_K21.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ART = Path("scripts/artifacts")

# K=21 PRIMARY pool from d13e_path_b_compound_stint.py
K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
]


def _load_pos_col(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1].astype(np.float64)
    return arr.astype(np.float64).ravel()


def main():
    oofs = np.column_stack([
        _load_pos_col(ART / f"oof_{b}_strat.npy") for b in K21_BASES])
    tests = np.column_stack([
        _load_pos_col(ART / f"test_{b}_strat.npy") for b in K21_BASES])
    print(f"Loaded K=21 stack — OOF {oofs.shape}, test {tests.shape}")

    blends = {}
    # arithmetic mean
    blends["mean"] = (oofs.mean(axis=1), tests.mean(axis=1))
    # geometric mean (clip to avoid log(0))
    eps = 1e-9
    blends["gmean"] = (
        np.exp(np.log(np.clip(oofs, eps, 1 - eps)).mean(axis=1)),
        np.exp(np.log(np.clip(tests, eps, 1 - eps)).mean(axis=1)),
    )
    # rank-mean
    def rank_mean(M):
        n = len(M)
        ranks = np.column_stack([rankdata(c) / n for c in M.T])
        return ranks.mean(axis=1)
    blends["rank_mean"] = (rank_mean(oofs), rank_mean(tests))
    # trimmed mean (drop top-3 + bot-3 per row, mean of middle 15)
    def trimmed(M, k=3):
        sorted_M = np.sort(M, axis=1)
        return sorted_M[:, k:M.shape[1]-k].mean(axis=1)
    blends["trimmed"] = (trimmed(oofs, 3), trimmed(tests, 3))

    # Save
    for kind, (oof_blend, test_blend) in blends.items():
        np.save(ART / f"oof_blend_{kind}_K21_strat.npy",
                np.column_stack([1 - oof_blend, oof_blend]))
        np.save(ART / f"test_blend_{kind}_K21_strat.npy",
                np.column_stack([1 - test_blend, test_blend]))

    # Gate via the harness (ρ-only mode runs without data)
    from probe import gate, PRIMARY_OOF, PRIMARY_TEST

    summary = {}
    for kind in ["mean", "gmean", "rank_mean", "trimmed"]:
        res = gate(
            f"blend_{kind}_K21",
            ART / f"oof_blend_{kind}_K21_strat.npy",
            ART / f"test_blend_{kind}_K21_strat.npy",
            primary_oof_path=PRIMARY_OOF,
            primary_test_path=PRIMARY_TEST,
        )
        summary[kind] = {
            "rho_vs_primary": res["rho_vs_primary"],
            "g3_flip_ratio": res["g3_flip_ratio"],
            "verdict": res["verdict"],
            "delta_oof_bp": res["delta_oof_bp"],
            "y_available": res["y_available"],
        }

    (ART / "probe_blends_K21.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ {ART / 'probe_blends_K21.json'}")

    # Compact comparison table
    print("\n=== Blend ρ vs PRIMARY summary ===")
    print(f"{'kind':<12s} {'ρ vs PRIMARY':>14s} {'flips':>10s} {'verdict':>14s}")
    for kind, r in summary.items():
        print(f"{kind:<12s} {r['rho_vs_primary']:>14.6f} "
              f"{r['g3_flip_ratio']:>10.3f} {r['verdict']:>14s}")


if __name__ == "__main__":
    main()
