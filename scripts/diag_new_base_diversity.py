"""New-base diversity scorecard vs M5h consensus.

For each new base candidate (EBM, LR-FE, H1, RealMLP if available),
compute:
  1. Standalone Strat OOF AUC.
  2. Spearman ρ vs M5h pool consensus (median of 13 bases).
  3. Spearman ρ vs M5h stacker test predictions.
  4. Mean |new − M5h-consensus| on the top-decile-disagreement test rows.
  5. Mean |new − M5h-consensus| on Stint=2 test rows (the blind-spot
     segment).

Lower ρ + higher |new − consensus| on hard subsets = highest LB-lift
potential when added to a stack.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

# Candidate new bases (suffix _strat); skip if file missing
CANDIDATES = [
    ("d3e_ebm", "EBM (GA²M)"),
    ("d3f_pseudo_lgbm", "H1 pseudo-label LGBM"),
    ("d3g_lr_fe", "LR with FE"),
    ("realmlp", "RealMLP-TD (Kaggle GPU)"),
]


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # Pool consensus + per-row std
    consensus = np.load(ART / "test_pool_consensus.npy")
    row_std = np.load(ART / "test_pool_row_std.npy")
    hi_thresh = np.percentile(row_std, 90)
    hi_mask = row_std >= hi_thresh
    s2_mask = (test["Stint"].values == 2)

    # M5h test prediction for ρ comparison
    m5h_test = np.load(ART / "test_m5h_strat.npy")[:, 1]

    print(f"=== New-base diversity scorecard ===")
    print(f"M5h Strat OOF: 0.95043; LB: 0.94991\n")
    print(f"{'Base':<24} {'Strat OOF':>10} {'Δ M5h':>8} "
          f"{'ρ M5h cons':>11} {'ρ M5h test':>11} "
          f"{'|Δ|@hi-dis':>11} {'|Δ|@Stint2':>11}")

    rows = []
    for name, label in CANDIDATES:
        oof_p = ART / f"oof_{name}_strat.npy"
        test_p = ART / f"test_{name}_strat.npy"
        if not oof_p.exists() or not test_p.exists():
            print(f"  {label:<24}  (artifacts not yet available)")
            continue
        oof = np.load(oof_p)[:, 1]
        tp = np.load(test_p)[:, 1]
        auc = float(roc_auc_score(y, oof))
        delta_m5h = (auc - 0.95043) * 1e4

        rho_cons, _ = spearmanr(tp, consensus)
        rho_m5h, _ = spearmanr(tp, m5h_test)
        diff_hi = float(np.abs(tp[hi_mask] - consensus[hi_mask]).mean())
        diff_s2 = float(np.abs(tp[s2_mask] - consensus[s2_mask]).mean())

        print(f"  {label:<24} {auc:>10.5f} {delta_m5h:>+8.1f} "
              f"{rho_cons:>11.5f} {rho_m5h:>11.5f} "
              f"{diff_hi:>11.4f} {diff_s2:>11.4f}")
        rows.append(dict(name=name, label=label, oof=auc,
                         delta_m5h_bp=delta_m5h,
                         rho_consensus=rho_cons, rho_m5h_test=rho_m5h,
                         diff_hi_disagreement=diff_hi,
                         diff_stint2=diff_s2))

    # Ranking by diversity score: lower ρ + higher diff on hard subsets
    if rows:
        print("\n=== Diversity ranking ===")
        for r in sorted(rows, key=lambda x: x["rho_m5h_test"]):
            score = (1 - r["rho_m5h_test"]) + r["diff_stint2"]
            print(f"  {r['label']:<24}  diversity_score={score:.4f}  "
                  f"(1-ρ_m5h={1-r['rho_m5h_test']:.4f} + |Δ|@S2={r['diff_stint2']:.4f})")

    import json
    (ART / "new_base_diversity_scorecard.json").write_text(json.dumps(rows, indent=2))
    print(f"\n→ scripts/artifacts/new_base_diversity_scorecard.json")


if __name__ == "__main__":
    main()
