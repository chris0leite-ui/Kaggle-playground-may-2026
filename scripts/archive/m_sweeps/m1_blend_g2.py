"""M1 — free blend-G2 of baseline + d2a_te on existing OOF .npy.

Tests D2-A audit hypothesis cause #3: TE may only help in a stack,
not standalone. No new compute beyond reading 4 .npy files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def best_blend(oof_a: np.ndarray, oof_b: np.ndarray, y: np.ndarray):
    """Search 0..1 in 0.05 steps for best convex blend; return (w, auc)."""
    best_w, best_auc = 0.0, 0.0
    for w in np.linspace(0, 1, 21):
        b = w * oof_a + (1 - w) * oof_b
        auc = roc_auc_score(y, b)
        if auc > best_auc:
            best_w, best_auc = float(w), float(auc)
    return best_w, best_auc


def main():
    train = pd.read_csv("data/train.csv")
    y = train["PitNextLap"].astype(int).values

    # Strat anchor
    base_strat = np.load("scripts/artifacts/oof_baseline_two_anchor_strat.npy")[:, 1]
    te_strat = np.load("scripts/artifacts/oof_d2a_te_strat.npy")[:, 1]
    auc_base_s = roc_auc_score(y, base_strat)
    auc_te_s = roc_auc_score(y, te_strat)
    auc_blend50_s = roc_auc_score(y, 0.5 * base_strat + 0.5 * te_strat)
    w_s, auc_best_s = best_blend(base_strat, te_strat, y)

    # GroupKF anchor
    base_grp = np.load("scripts/artifacts/oof_baseline_two_anchor_groupkf.npy")[:, 1]
    te_grp = np.load("scripts/artifacts/oof_d2a_te_groupkf.npy")[:, 1]
    auc_base_g = roc_auc_score(y, base_grp)
    auc_te_g = roc_auc_score(y, te_grp)
    auc_blend50_g = roc_auc_score(y, 0.5 * base_grp + 0.5 * te_grp)
    w_g, auc_best_g = best_blend(base_grp, te_grp, y)

    delta_s_50 = (auc_blend50_s - auc_base_s) * 1e4
    delta_s_best = (auc_best_s - auc_base_s) * 1e4
    delta_g_50 = (auc_blend50_g - auc_base_g) * 1e4
    delta_g_best = (auc_best_g - auc_base_g) * 1e4

    print("=== M1 blend-G2 (baseline + d2a_te) ===")
    print(f"Strat:    base={auc_base_s:.5f}  te={auc_te_s:.5f}")
    print(f"          blend50={auc_blend50_s:.5f}  Δ={delta_s_50:+.2f}bp")
    print(f"          best   ={auc_best_s:.5f}  Δ={delta_s_best:+.2f}bp  (w_base={w_s:.2f})")
    print(f"GroupKF:  base={auc_base_g:.5f}  te={auc_te_g:.5f}")
    print(f"          blend50={auc_blend50_g:.5f}  Δ={delta_g_50:+.2f}bp")
    print(f"          best   ={auc_best_g:.5f}  Δ={delta_g_best:+.2f}bp  (w_base={w_g:.2f})")

    verdict_s = "PASS" if delta_s_best > 1.0 else "NULL"
    verdict_g = "PASS" if delta_g_best > 1.0 else "NULL"
    print(f"G2 verdict: Strat {verdict_s}, GroupKF {verdict_g}")

    import datetime as dt
    out = f"audit/{dt.date.today().isoformat()}-m1-blend-g2.md"
    with open(out, "w") as f:
        f.write(
            f"# M1 — blend-G2 (baseline + d2a_te) ({dt.date.today()})\n\n"
            f"Tests D2-A audit cause #3: TE only helps in stack.\n\n"
            f"## Results\n\n"
            f"| anchor | base | te | 50/50 | best | best_w(base) | Δ best vs base |\n"
            f"|---|---:|---:|---:|---:|---:|---:|\n"
            f"| Strat | {auc_base_s:.5f} | {auc_te_s:.5f} | {auc_blend50_s:.5f} | "
            f"{auc_best_s:.5f} | {w_s:.2f} | {delta_s_best:+.2f}bp |\n"
            f"| GroupKF | {auc_base_g:.5f} | {auc_te_g:.5f} | {auc_blend50_g:.5f} | "
            f"{auc_best_g:.5f} | {w_g:.2f} | {delta_g_best:+.2f}bp |\n\n"
            f"## Verdict\n\n"
            f"G2 Strat: **{verdict_s}** (Δ{delta_s_best:+.2f}bp, threshold +1bp).\n"
            f"G2 GroupKF: **{verdict_g}** (Δ{delta_g_best:+.2f}bp).\n\n"
            f"Closes D2-A postmortem cause #3. Free probe — no compute.\n"
        )
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
