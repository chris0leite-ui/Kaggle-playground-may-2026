"""scripts/probe_r10_operator_vs_pool.py — Round 10 operator-vs-pool check

Resolves the R9 EOD open question
(`knowledge-base/questions/2026-05-18-pathb-specific-vs-pool-structural-lock.md`):

  Is the K=14 base-add absorption at Path-B DriverClass×Stint
  τ=100k OPERATOR-specific or POOL-structural?

Test: plain LR-meta (no segmentation / no shrinkage) on K=13 vs
K=14+NB4 vs K=14+C1. If LR-meta finds Δ ≥ +0.05 bp for either,
the lock is operator-specific (Path-B absorbs what plain LR
extracts).

Cost: ~3 min CPU (3 LR-meta runs × ~1 min each).

Decision rule:
  - ANY K=14 LR-meta Δ ≥ +0.05 bp → operator-specific lock; reopen
    NB4/C1 + parallel-track mechanism expansion in R10.
  - ALL K=14 LR-meta Δ < +0.05 bp → pool-structural lock; commit
    to mechanism expansion (seq2seq / graph / survival) sole-track.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from build_K11_full_pathb import _pos, expand, lr_meta_oof, ART, DATA, TARGET  # noqa
from build_K13_pathb_multiseg import K13_FILES  # noqa


def lr_meta_run(label: str, oofs: list[np.ndarray], y: np.ndarray) -> dict:
    t0 = time.time()
    F = expand(np.column_stack(oofs))
    oof = lr_meta_oof(F, y)
    auc = float(roc_auc_score(y, oof))
    wall = time.time() - t0
    return {"label": label, "K": len(oofs), "F_dim": F.shape[1],
            "auc": auc, "wall": wall}


def main() -> None:
    print("=== R10 operator-vs-pool probe — plain LR-meta sweep ===")
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].astype(int).values

    # K=13 pool
    k13_oofs = [_pos(ART / o) for _, o, _ in K13_FILES]
    nb4_oof = _pos(ART / "oof_NB4_compound_stint_te_strat.npy")
    c1_oof  = _pos(ART / "oof_C1_race_external_strat.npy")

    pools = [
        ("K=13 (baseline)",     k13_oofs),
        ("K=14 + NB4 (TE-base)", k13_oofs + [nb4_oof]),
        ("K=14 + C1 (ext)",      k13_oofs + [c1_oof]),
        ("K=15 + NB4 + C1",      k13_oofs + [nb4_oof, c1_oof]),
    ]

    results = [lr_meta_run(label, oofs, y) for label, oofs in pools]

    print(f"\n{'Pool':<25}{'K':>5}{'F':>6}{'OOF AUC':>10}{'wall':>8}")
    print("-" * 60)
    base_auc = results[0]["auc"]
    for r in results:
        delta_bp = (r["auc"] - base_auc) * 1e4
        marker = " ★" if delta_bp >= 0.05 else ("  " if delta_bp >= 0 else " ↓")
        print(f"{r['label']:<25}{r['K']:>5}{r['F_dim']:>6}"
              f"{r['auc']:>10.5f}{r['wall']:>7.1f}s   "
              f"Δ vs K=13 LR-meta: {delta_bp:+.3f} bp{marker}")

    # Reference: R7.1 PRIMARY Path-B OOF
    print()
    primary = _pos(ART / "oof_K13_pathb_driverclass_stint_tau100000.npy")
    primary_auc = float(roc_auc_score(y, primary))
    print(f"  Reference: R7.1 PRIMARY K=13+Path-B DC×Stint τ=100k OOF: {primary_auc:.5f}")
    pathb_lift = (primary_auc - base_auc) * 1e4
    print(f"  Path-B-vs-LR-meta lift on K=13: +{pathb_lift:.3f} bp")
    print()
    print("  Decision rule (resolve operator-vs-pool open question):")
    for r in results[1:]:
        delta_bp = (r["auc"] - base_auc) * 1e4
        verdict = "OPERATOR-SPECIFIC" if delta_bp >= 0.05 else "POOL-STRUCTURAL"
        print(f"    {r['label']:<25}: Δ {delta_bp:+.3f} bp → {verdict}")


if __name__ == "__main__":
    main()
