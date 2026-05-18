"""Phase 0 P0.3 — Hand-coded stint-cap multiplier sweep.

Post-process the K=4 + Path-B PRIMARY (oof_K4_fwd_pathb / test_K4_fwd_pathb)
by multiplying P by a fixed scalar on rows where Compound==C and
TyreLife >= threshold[C]. Sweeps 3 threshold-configs × 5 multipliers
= 15 cells. Pure heuristic; cannot regress at OOF beyond a single
multiplier (clip to [0, 1]).

Origin: 2026-05-18 round-2 plan (P0.3); skill Rule 23 free-form FE
slot.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

THRESHOLDS = [
    {"SOFT": 22, "MEDIUM": 32, "HARD": 42},  # baseline svanikkolli
    {"SOFT": 18, "MEDIUM": 28, "HARD": 38},  # tighter
    {"SOFT": 24, "MEDIUM": 34, "HARD": 44},  # looser
]
MULTIPLIERS = [1.05, 1.10, 1.20, 1.30, 1.50]


def apply_cap(p: np.ndarray, compound: np.ndarray,
              tyrelife: np.ndarray, thresholds: dict, mult: float) -> np.ndarray:
    out = p.copy()
    for c, t in thresholds.items():
        mask = (compound == c) & (tyrelife >= t)
        out[mask] = np.clip(out[mask] * mult, 0.001, 0.999)
    return out


def asymmetric_flip(p_a: np.ndarray, p_b: np.ndarray,
                    base_rate: float = 0.199):
    from scipy.stats import rankdata
    n = len(p_a)
    k = max(1, int(round(base_rate * n)))
    rank_a = rankdata(-p_a, method="ordinal")
    rank_b = rankdata(-p_b, method="ordinal")
    top_a = rank_a <= k
    top_b = rank_b <= k
    return int((top_b & ~top_a).sum()), int((top_a & ~top_b).sum())


def main():
    print("Loading K=4 + Path-B PRIMARY OOF + test predictions...")
    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")[:, 1]
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")[:, 1]
    print(f"  OOF shape {primary_oof.shape}, test shape {primary_test.shape}")

    train = pd.read_csv("data/train.csv", usecols=[TARGET, "Compound",
                                                    "TyreLife"])
    test = pd.read_csv("data/test.csv", usecols=["Compound", "TyreLife"])
    y = train[TARGET].astype(int).values
    train_compound = train["Compound"].values
    train_tyrelife = train["TyreLife"].astype(float).values
    test_compound = test["Compound"].values
    test_tyrelife = test["TyreLife"].astype(float).values

    base_auc = roc_auc_score(y, primary_oof)
    print(f"\nBaseline (K=4 + Path-B PRIMARY) OOF AUC: {base_auc:.5f}")

    print("\n=== Sweep (threshold-config × multiplier) ===")
    print(f"{'threshold-set':<35s} {'mult':>5s}  {'OOF AUC':>9s}  "
          f"{'Δ bp':>8s}  {'flip+':>6s}  {'flip-':>6s}")
    results = []
    best = (-np.inf, None, None, None, None)
    for i, thresholds in enumerate(THRESHOLDS):
        ts_label = f"S{thresholds['SOFT']}/M{thresholds['MEDIUM']}/H{thresholds['HARD']}"
        for mult in MULTIPLIERS:
            oof_capped = apply_cap(primary_oof, train_compound,
                                    train_tyrelife, thresholds, mult)
            auc = roc_auc_score(y, oof_capped)
            delta = (auc - base_auc) * 1e4
            n_changed = ((train_compound == "SOFT") & (train_tyrelife >= thresholds["SOFT"])).sum() \
                      + ((train_compound == "MEDIUM") & (train_tyrelife >= thresholds["MEDIUM"])).sum() \
                      + ((train_compound == "HARD") & (train_tyrelife >= thresholds["HARD"])).sum()
            fp, fn = asymmetric_flip(primary_oof, oof_capped)
            print(f"{ts_label:<35s} {mult:>5.2f}  {auc:.5f}  "
                  f"{delta:+8.3f}  {fp:>6d}  {fn:>6d}  (changed {n_changed:,})")
            results.append({"thresholds": thresholds, "mult": mult,
                            "oof_auc": auc, "delta_bp": delta,
                            "n_changed": int(n_changed),
                            "flip_pos": fp, "flip_neg": fn})
            if auc > best[0]:
                best = (auc, thresholds, mult, oof_capped, None)

    print(f"\nBest cell: thresholds={best[1]}, mult={best[2]}, "
          f"OOF={best[0]:.5f}, Δ={(best[0]-base_auc)*1e4:+.3f} bp")

    # Save best-cell test predictions for downstream gating
    if best[0] > base_auc:
        test_capped = apply_cap(primary_test, test_compound,
                                 test_tyrelife, best[1], best[2])
        np.save(ART / "oof_K4_stint_cap_best_strat.npy",
                np.column_stack([1 - best[3], best[3]]).astype(np.float64))
        np.save(ART / "test_K4_stint_cap_best_strat.npy",
                np.column_stack([1 - test_capped, test_capped]).astype(np.float64))
        print(f"  → saved oof_/test_K4_stint_cap_best_strat.npy")

    (ART / "probe_a2_stint_cap_results.json").write_text(
        json.dumps({"baseline_oof": base_auc, "results": results,
                    "best_thresholds": best[1], "best_mult": best[2]},
                   indent=2))


if __name__ == "__main__":
    main()
