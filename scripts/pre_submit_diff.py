"""Pre-submit diff helper — call BEFORE any kaggle submit.

Compares a candidate submission CSV against the most recently submitted
one (or a specified reference). Reports:
  - Spearman rank correlation (the load-bearing AUC predictor)
  - max abs diff
  - rank-shift distribution
  - fraction of rows differing > thresholds

Decision rule (s6e5 specific): if Spearman > 0.999 vs prior submission,
the LB will tie within Kaggle's 5-decimal quantization. Slot is
wasted as a calibration probe; abort.

Usage:
  python3 scripts/pre_submit_diff.py <candidate.csv> [reference.csv]

If reference omitted, defaults to comparing against M5h L1-pruned
(the current production submission).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

DEFAULT_REF = "submissions/submission_m5h_lr_meta_l1pruned.csv"
TARGET = "PitNextLap"
RHO_THRESHOLD = 0.999   # above this, LB will tie at 5 decimals


def diff_report(cand_path: str, ref_path: str = DEFAULT_REF) -> dict:
    cand = pd.read_csv(cand_path)
    ref = pd.read_csv(ref_path)
    assert len(cand) == len(ref), f"length mismatch: {len(cand)} vs {len(ref)}"
    if (cand["id"].values == ref["id"].values).all():
        a, b = cand[TARGET].values, ref[TARGET].values
    else:
        merged = cand.merge(ref, on="id", suffixes=("_cand", "_ref"))
        a, b = merged[f"{TARGET}_cand"].values, merged[f"{TARGET}_ref"].values
    abs_d = np.abs(a - b)
    rho, _ = spearmanr(a, b)
    ra, rb = rankdata(a), rankdata(b)
    rank_shift = np.abs(ra - rb)
    res = dict(
        candidate=cand_path, reference=ref_path,
        n=len(a),
        max_abs_diff=float(abs_d.max()),
        mean_abs_diff=float(abs_d.mean()),
        median_abs_diff=float(np.median(abs_d)),
        spearman=float(rho),
        rank_shift_max=float(rank_shift.max()),
        rank_shift_mean=float(rank_shift.mean()),
        rank_shift_median=float(np.median(rank_shift)),
        rows_diff_gt_1e6=int((abs_d > 1e-6).sum()),
        rows_diff_gt_1e4=int((abs_d > 1e-4).sum()),
        rows_diff_gt_1e3=int((abs_d > 1e-3).sum()),
    )
    pct_e6 = 100 * res["rows_diff_gt_1e6"] / res["n"]
    pct_e4 = 100 * res["rows_diff_gt_1e4"] / res["n"]
    pct_e3 = 100 * res["rows_diff_gt_1e3"] / res["n"]

    print(f"\n=== Pre-submit diff ===")
    print(f"candidate: {cand_path}")
    print(f"reference: {ref_path}")
    print(f"n_test:    {res['n']}")
    print(f"abs diff:  max={res['max_abs_diff']:.6e}  "
          f"mean={res['mean_abs_diff']:.6e}  median={res['median_abs_diff']:.6e}")
    print(f"rows diff > 1e-6: {res['rows_diff_gt_1e6']:>7d} ({pct_e6:.2f}%)")
    print(f"rows diff > 1e-4: {res['rows_diff_gt_1e4']:>7d} ({pct_e4:.2f}%)")
    print(f"rows diff > 1e-3: {res['rows_diff_gt_1e3']:>7d} ({pct_e3:.2f}%)")
    print(f"Spearman ρ: {res['spearman']:.6f}")
    print(f"rank shift: max={res['rank_shift_max']:.0f}  "
          f"mean={res['rank_shift_mean']:.2f}  median={res['rank_shift_median']:.0f}")

    if res["spearman"] > RHO_THRESHOLD:
        print(f"\n⚠️  Spearman > {RHO_THRESHOLD} — LB will likely TIE the reference.")
        print("    Slot will be wasted as a calibration probe. Abort unless you")
        print("    explicitly want to confirm the tie.")
        verdict = "TIE_EXPECTED"
    else:
        print(f"\n✓ Spearman ≤ {RHO_THRESHOLD} — predictions structurally different.")
        print("  LB delta possible.")
        verdict = "DIFFERS"
    res["verdict"] = verdict
    return res


def main(argv: list[str]):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    cand = argv[1]
    ref = argv[2] if len(argv) >= 3 else DEFAULT_REF
    diff_report(cand, ref)


if __name__ == "__main__":
    main(sys.argv)
