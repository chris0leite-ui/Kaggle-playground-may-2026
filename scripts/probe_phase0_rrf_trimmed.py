"""Phase 0 P0.1 + P0.2 — Reciprocal-rank-fusion + trimmed-rank blends.

Pure post-process on K=4 base predictions. Two operators:

  RRF (Anserini-style): score_i = sum_b  1 / (k + rank_b(i))
  Trimmed-rank: drop highest and lowest rank per row; mean middle 2

For each, computes:
  (a) standalone OOF AUC vs K=4 LR-meta baseline 0.95399
  (b) saves oof_/test_ artifacts for K=4+1 gating downstream

Origin: 2026-05-18 round-2 plan (audit/2026-05-18-plateau-brainstorm.md
+ /root/.claude/plans/read-the-handover-look-toasty-candle.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

K4 = ["d17_h1d_yekenot_full", "p1_single_cb_v4_gpu",
      "f1_hgbc_deep", "d16_orig_continuous_only"]


def pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def rrf_score(preds: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion. Returns a score per row in [0, 1]-ish."""
    n = preds[0].shape[0]
    # rankdata of -p gives rank 1 for highest probability (best).
    score = np.zeros(n, dtype=np.float64)
    for p in preds:
        r = rankdata(-p, method="average")
        score += 1.0 / (k + r)
    # Normalize to [0, 1] range; not strictly necessary for AUC.
    score = (score - score.min()) / (score.max() - score.min() + 1e-12)
    return score


def trimmed_rank_score(preds: list[np.ndarray],
                       trim_low: int = 1,
                       trim_high: int = 1) -> np.ndarray:
    """Drop trim_low lowest and trim_high highest ranks per row;
    mean of middle ranks. Normalized rank per base."""
    n = preds[0].shape[0]
    nb = len(preds)
    # Per-base normalized rank (1/n .. 1)
    R = np.column_stack([rankdata(p, method="average") / n for p in preds])
    # Sort ranks per row
    R_sorted = np.sort(R, axis=1)
    middle = R_sorted[:, trim_low: nb - trim_high]
    return middle.mean(axis=1)


def main():
    train_y = pd.read_csv("data/train.csv")[TARGET].astype(int).values
    print(f"y shape {train_y.shape}, sum={train_y.sum():,}, "
          f"mean={train_y.mean():.4f}")

    # Load K=4 base OOFs + tests
    oofs = [pos(ART / f"oof_{b}_strat.npy") for b in K4]
    tests = [pos(ART / f"test_{b}_strat.npy") for b in K4]
    print(f"\nLoaded K=4: {len(oofs)} OOFs, lengths "
          f"{[len(o) for o in oofs]}")

    # K=4 baseline reference AUCs (per-base, for sanity)
    print("\n--- Per-base OOF AUC ---")
    for b, o in zip(K4, oofs):
        print(f"  {b:<35s}  {roc_auc_score(train_y, o):.5f}")

    results = {}

    # ============ P0.1 RRF sweep ============
    print("\n=== P0.1 — Reciprocal-rank-fusion sweep ===")
    for k in [30, 60, 100]:
        s_oof = rrf_score(oofs, k=k)
        s_test = rrf_score(tests, k=k)
        auc = roc_auc_score(train_y, s_oof)
        name = f"K4_rrf_k{k}"
        # Save as 2-col layout (mirrors existing artifact convention)
        np.save(ART / f"oof_{name}_strat.npy",
                np.column_stack([1 - s_oof, s_oof]).astype(np.float64))
        np.save(ART / f"test_{name}_strat.npy",
                np.column_stack([1 - s_test, s_test]).astype(np.float64))
        results[name] = {"oof_auc": auc, "type": "rrf", "k": k}
        print(f"  k={k:<3d}  OOF AUC={auc:.5f}  "
              f"→ oof_{name}_strat.npy")

    # ============ P0.2 Trimmed-rank ============
    print("\n=== P0.2 — Trimmed-rank blend ===")
    s_oof = trimmed_rank_score(oofs, trim_low=1, trim_high=1)
    s_test = trimmed_rank_score(tests, trim_low=1, trim_high=1)
    auc = roc_auc_score(train_y, s_oof)
    name = "K4_trimmed_rank_t1_1"
    np.save(ART / f"oof_{name}_strat.npy",
            np.column_stack([1 - s_oof, s_oof]).astype(np.float64))
    np.save(ART / f"test_{name}_strat.npy",
            np.column_stack([1 - s_test, s_test]).astype(np.float64))
    results[name] = {"oof_auc": auc, "type": "trimmed_rank",
                     "trim": [1, 1]}
    print(f"  trim=(1,1)  OOF AUC={auc:.5f}  "
          f"→ oof_{name}_strat.npy")

    # ============ Headlines ============
    print("\n=== Phase 0 headline ===")
    print(f"K=4 LR-meta baseline OOF (from prior runs): 0.95399")
    print(f"\n{'name':<25s}  {'AUC':>8s}  {'Δ bp vs K=4 LR-meta':>22s}")
    for name, r in results.items():
        delta = (r["oof_auc"] - 0.95399) * 1e4
        marker = " *" if delta > 0 else ""
        print(f"  {name:<25s}  {r['oof_auc']:.5f}  {delta:+.3f}{marker}")

    # Save aggregate JSON
    (ART / "probe_phase0_rrf_trimmed_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nSaved → {ART}/probe_phase0_rrf_trimmed_results.json")


if __name__ == "__main__":
    main()
