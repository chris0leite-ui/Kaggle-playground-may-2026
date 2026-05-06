"""scripts/probe_rho_inventory.py — ρ inventory of held candidates.

Sweeps ρ vs current PRIMARY across every held submission/OOF artifact
on disk. Cheap (~5s); zero data dependency. Outputs a sorted ledger
and saves JSON. Useful for HEDGE candidate selection (rank-shift
maximizers from PRIMARY) and for spotting near-duplicate held subs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ART = Path("scripts/artifacts")
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"

# Candidate tests — held submissions worth ρ-mapping.
CANDIDATES = sorted(set([
    # d13e tau variants
    "test_d13e_compound_stint_tau5000_strat.npy",
    "test_d13e_compound_stint_tau100000_strat.npy",
    "test_d13e_compound_stint_tau500000_strat.npy",
    # d13 Stint Path B (prior PRIMARY)
    "test_d13_path_b_stint_tau100000_strat.npy",
    "test_d13_path_b_stint_tau20000_strat.npy",
    # d13c compound only
    "test_d13c_path_b_compound_tau100000_strat.npy",
    # d12 GroupKF meta (R5 hedge candidate)
    "test_d12_groupkf_meta_strat.npy",
    "test_d12_lr_meta_strat.npy",
    # d14 Path B alt-cohort variants
    "test_d14_path_b_year_tau5000_strat.npy",
    "test_d14_path_b_year_tau20000_strat.npy",
    "test_d14_path_b_year_tau100000_strat.npy",
    "test_d14_path_b_year_stint_tau5000_strat.npy",
    "test_d14_path_b_year_stint_tau20000_strat.npy",
    "test_d14_path_b_year_stint_tau100000_strat.npy",
    "test_d14_path_b_race_tau5000_strat.npy",
    "test_d14_path_b_race_tau20000_strat.npy",
    "test_d14_path_b_race_tau100000_strat.npy",
    # d14 H1 (failed FM aug15)
    "test_d14_h1_fm_aug13_3way_strat.npy",
    # d10d leak-corrected meta
    "test_d10d_leak_corrected_meta_strat.npy",
    # K=21 LR-meta blends just produced
    "test_blend_mean_K21_strat.npy",
    "test_blend_gmean_K21_strat.npy",
    "test_blend_rank_mean_K21_strat.npy",
    "test_blend_trimmed_K21_strat.npy",
]))


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def main():
    primary = _pos(PRIMARY_TEST)
    rare_thr = float(np.quantile(primary, 0.99))
    primary_pos = primary >= rare_thr

    rows = []
    for fname in CANDIDATES:
        p = ART / fname
        if not p.exists():
            continue
        cand = _pos(p)
        if len(cand) != len(primary):
            continue
        rho, _ = spearmanr(cand, primary)
        cand_pos = cand >= rare_thr
        f_to_neg = int(np.sum(primary_pos & ~cand_pos))
        f_to_pos = int(np.sum(~primary_pos & cand_pos))
        flip_ratio = (min(f_to_pos, f_to_neg) / max(f_to_pos, f_to_neg)
                      if max(f_to_pos, f_to_neg) > 0 else 1.0)
        rows.append(dict(
            name=fname.replace("test_", "").replace("_strat.npy", ""),
            rho=float(rho),
            flips_to_neg=f_to_neg, flips_to_pos=f_to_pos,
            flip_ratio=float(flip_ratio),
        ))

    rows.sort(key=lambda r: r["rho"])  # most-diverse first
    print(f"\n=== ρ inventory vs PRIMARY (d13e Compound×Stint τ=20k) ===")
    print(f"{'name':<48s} {'ρ':>10s} {'+→−':>7s} {'−→+':>7s} {'ratio':>7s}")
    for r in rows:
        print(f"{r['name']:<48s} {r['rho']:>10.6f} "
              f"{r['flips_to_neg']:>7d} {r['flips_to_pos']:>7d} "
              f"{r['flip_ratio']:>7.3f}")

    # Buckets
    tie = [r for r in rows if r["rho"] >= 0.999]
    near_tie = [r for r in rows if 0.995 <= r["rho"] < 0.999]
    diverse = [r for r in rows if r["rho"] < 0.995]
    print(f"\nBuckets: TIE_EXPECTED (ρ≥0.999): {len(tie)} | "
          f"near-tie (0.995–0.999): {len(near_tie)} | "
          f"diverse (<0.995): {len(diverse)}")

    out = ART / "probe_rho_inventory.json"
    out.write_text(json.dumps({"rows": rows,
                               "primary_test": str(PRIMARY_TEST)}, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
