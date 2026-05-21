"""scripts/lr_v2_phaseA_K19_eval.py — Δ vs K=18 PRIMARY after K=19 gate.

The `build_K13_pathb_multiseg.py --extra-bases ...` harness reports Δ
vs R5.2 K=13 baseline (its built-in reference). For Phase A we need
Δ vs K=18 PRIMARY. This script computes:

  K=19 OOF AUC
  K=18 OOF AUC (PRIMARY)
  Δ (K=19 - K=18) in bp on OOF
  ρ_OOF(K=19, K=18)
  ρ_test(K=19, K=18)

And echoes the Phase A decision per the plan:
  Δ ≥ +0.30 bp ∧ ρ_test ≤ 0.97  →  FUND PHASE B
  Δ ∈ [0, +0.30)                →  NARROW PHASE B (drop per-seg arm)
  Δ < 0                          →  KILL PIVOT (R22/R5d redirect)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
DATA = Path("data")
AUDIT = Path("audit")


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k19-oof", default=str(
        ART / "oof_K19_pathb_driverclass_stint_tau100000.npy"))
    ap.add_argument("--k19-test", default=str(
        ART / "test_K19_pathb_driverclass_stint_tau100000.npy"))
    ap.add_argument("--k18-oof", default=str(
        ART / "oof_K18_pathb_driverclass_stint_tau100000.npy"))
    ap.add_argument("--k18-test", default=str(
        ART / "test_K18_pathb_driverclass_stint_tau100000.npy"))
    ap.add_argument("--probe-oof", default=str(
        ART / "oof_lr_phaseA_leafprobe_strat.npy"),
        help="Phase A leaf+LR base OOF (for ρ vs K=18 diagnostic).")
    ap.add_argument("--probe-test", default=str(
        ART / "test_lr_phaseA_leafprobe_strat.npy"))
    ap.add_argument("--out", default=str(
        AUDIT / "2026-05-22-phaseA-K19-decision.json"))
    args = ap.parse_args()

    y = pd.read_csv(DATA / "train.csv")["PitNextLap"].astype(int).values

    print(f"=== Phase A K=19 stack-add evaluation ===")
    missing = [p for p in [args.k19_oof, args.k19_test, args.k18_oof,
                            args.k18_test] if not Path(p).exists()]
    if missing:
        print("MISSING files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(2)

    k19_oof = _pos(Path(args.k19_oof))
    k19_te = _pos(Path(args.k19_test))
    k18_oof = _pos(Path(args.k18_oof))
    k18_te = _pos(Path(args.k18_test))

    assert len(k19_oof) == len(k18_oof) == len(y), \
        f"OOF length mismatch: K19={len(k19_oof)} K18={len(k18_oof)} y={len(y)}"
    assert len(k19_te) == len(k18_te) == 188165, \
        f"test length mismatch: K19={len(k19_te)} K18={len(k18_te)}"

    k19_auc = float(roc_auc_score(y, k19_oof))
    k18_auc = float(roc_auc_score(y, k18_oof))
    delta_bp = (k19_auc - k18_auc) * 1e4
    rho_oof = float(spearmanr(k19_oof, k18_oof)[0])
    rho_test = float(spearmanr(k19_te, k18_te)[0])

    print(f"  K=18 PRIMARY OOF AUC : {k18_auc:.6f}")
    print(f"  K=19  (+ LR_probe)   : {k19_auc:.6f}")
    print(f"  Δ                    : {delta_bp:+.3f} bp")
    print(f"  ρ_OOF  K19↔K18       : {rho_oof:.6f}")
    print(f"  ρ_test K19↔K18       : {rho_test:.6f}")

    probe_diag = {}
    if Path(args.probe_oof).exists() and Path(args.probe_test).exists():
        probe_oof = _pos(Path(args.probe_oof))
        probe_te = _pos(Path(args.probe_test))
        probe_auc = float(roc_auc_score(y, probe_oof))
        rho_probe_k18_oof = float(spearmanr(probe_oof, k18_oof)[0])
        rho_probe_k18_test = float(spearmanr(probe_te, k18_te)[0])
        print(f"  --- probe (leaf+LR) standalone diagnostic ---")
        print(f"  probe standalone OOF : {probe_auc:.6f}  "
              f"(gap to K18: {(probe_auc - k18_auc)*1e4:+.2f} bp)")
        print(f"  ρ_OOF  probe↔K18     : {rho_probe_k18_oof:.6f}")
        print(f"  ρ_test probe↔K18     : {rho_probe_k18_test:.6f}")
        probe_diag = dict(
            standalone_oof_auc=probe_auc,
            standalone_gap_vs_k18_bp=(probe_auc - k18_auc) * 1e4,
            rho_oof_vs_k18=rho_probe_k18_oof,
            rho_test_vs_k18=rho_probe_k18_test,
        )

    # Decision per plan
    if delta_bp >= 0.30 and rho_test <= 0.97:
        decision = "FUND_PHASE_B"
        why = (f"Δ {delta_bp:+.3f} bp ≥ +0.30 AND ρ_test {rho_test:.4f} "
               f"≤ 0.97 — GPU sweep justified")
    elif delta_bp >= 0.0:
        decision = "NARROW_PHASE_B"
        why = (f"Δ {delta_bp:+.3f} bp ∈ [0, +0.30) — fund Phase B "
               f"without per-segment specialist arm")
    else:
        decision = "KILL_PIVOT"
        why = (f"Δ {delta_bp:+.3f} bp < 0 — redirect to R22 + R5d/R7d")

    print(f"\n  DECISION: {decision}")
    print(f"  WHY:      {why}")

    AUDIT.mkdir(exist_ok=True)
    out = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        k18_oof_auc=k18_auc, k19_oof_auc=k19_auc,
        delta_bp=delta_bp,
        rho_oof_k19_k18=rho_oof,
        rho_test_k19_k18=rho_test,
        decision=decision, why=why,
        probe=probe_diag,
    )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
