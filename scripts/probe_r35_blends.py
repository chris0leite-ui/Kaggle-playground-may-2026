"""Probe R35 FT-Transformer as blend ingredient.

Run after R35 kernel completes and outputs are copied to scripts/artifacts/.

Usage: python scripts/probe_r35_blends.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold

ART = "scripts/artifacts"
y = pd.read_csv("data/train.csv")["PitNextLap"].values
test_ids = pd.read_csv("data/test.csv")["id"].values

def load(p):
    a = np.load(f"{ART}/{p}")
    if a.ndim == 2:
        a = a[:, -1] if a.shape[1] >= 2 else a.ravel()
    return a.astype(np.float64)

def rk(a): return rankdata(a, method="average") / len(a)


def main():
    K18 = load("oof_K18_pathb_driverclass_stint_tau100000.npy")
    R21 = load("oof_R21_yetirank_yrl_strat.npy")
    R30 = load("oof_R30_r21_bag_yrl_strat.npy")
    R72 = load("oof_K13_dcs_pathb_foldbag_strat.npy")
    R35 = load("oof_R35_ft_transformer_strat.npy")

    tK18 = load("test_K18_pathb_driverclass_stint_tau100000.npy")
    tR21 = load("test_R21_yetirank_yrl_strat.npy")
    tR30 = load("test_R30_r21_bag_yrl_strat.npy")
    tR72 = load("test_K13_dcs_pathb_foldbag_strat.npy")
    tR35 = load("test_R35_ft_transformer_strat.npy")

    print(f"R35 standalone OOF: {roc_auc_score(y, R35):.6f}")
    print(f"K18 standalone OOF: {roc_auc_score(y, K18):.6f}")
    print(f"R30 standalone OOF: {roc_auc_score(y, R30):.6f}")
    print(f"\nrho(R35, K18) = {spearmanr(R35, K18).statistic:.4f}")
    print(f"rho(R35, R21) = {spearmanr(R35, R21).statistic:.4f}")
    print(f"rho(R35, R30) = {spearmanr(R35, R30).statistic:.4f}")
    print(f"rho(R35, R72) = {spearmanr(R35, R72).statistic:.4f}")

    r_K18, r_R21, r_R30, r_R72, r_R35 = rk(K18), rk(R21), rk(R30), rk(R72), rk(R35)
    tr_K18, tr_R21, tr_R30, tr_R72, tr_R35 = rk(tK18), rk(tR21), rk(tR30), rk(tR72), rk(tR35)

    R25_oof = 0.89 * r_K18 + 0.11 * r_R21
    R25_test = 0.89 * tr_K18 + 0.11 * tr_R21
    print(f"\nR25 OOF: {roc_auc_score(y, R25_oof):.6f}")

    # 2-way K18 + R35
    print("\n=== K18 + R35 ===")
    best = (0, None)
    for w in np.arange(0.70, 0.99, 0.01):
        b = w * r_K18 + (1 - w) * r_R35
        auc = roc_auc_score(y, b)
        if auc > best[0]:
            best = (auc, w)
    print(f"  best: AUC={best[0]:.6f} w={best[1]:.2f}")

    # 3-way K18 + R30 + R35
    print("\n=== K18 + R30 + R35 ===")
    best = (0, None)
    for a in np.arange(0.50, 0.95, 0.05):
        for b in np.arange(0.05, 0.30, 0.05):
            c = round(1.0 - a - b, 3)
            if c < 0.02 or c > 0.30: continue
            blend = a * r_K18 + b * r_R30 + c * r_R35
            auc = roc_auc_score(y, blend)
            if auc > best[0]:
                best = (auc, (a, round(b, 2), c))
    print(f"  best: AUC={best[0]:.6f} weights={best[1]}")

    # 3-way K18 + R21 + R35
    print("\n=== K18 + R21 + R35 ===")
    best = (0, None)
    for a in np.arange(0.50, 0.95, 0.05):
        for b in np.arange(0.05, 0.30, 0.05):
            c = round(1.0 - a - b, 3)
            if c < 0.02 or c > 0.30: continue
            blend = a * r_K18 + b * r_R21 + c * r_R35
            auc = roc_auc_score(y, blend)
            if auc > best[0]:
                best = (auc, (a, round(b, 2), c))
    print(f"  best: AUC={best[0]:.6f} weights={best[1]}")

    # 4-way K18 + R30 + R72 + R35
    print("\n=== K18 + R30 + R72 + R35 ===")
    best = (0, None)
    for a in np.arange(0.40, 0.85, 0.05):
        for b in np.arange(0.05, 0.25, 0.05):
            for c in np.arange(0.05, 0.30, 0.05):
                d = round(1.0 - a - b - c, 3)
                if d < 0.02 or d > 0.25: continue
                blend = a * r_K18 + b * r_R30 + c * r_R72 + d * r_R35
                auc = roc_auc_score(y, blend)
                if auc > best[0]:
                    best = (auc, (a, round(b, 2), round(c, 2), d))
    print(f"  best: AUC={best[0]:.6f} weights={best[1]}")

    # Nested CV for the best 3-way K18+R30+R35
    print("\n=== Nested CV: K18 + R30 + R35 ===")
    grid = [(a, b, round(1-a-b, 3))
            for a in [round(0.05*i, 2) for i in range(10, 19)]
            for b in [round(0.05*i, 2) for i in range(1, 7)]
            if 0.05 <= round(1-a-b, 3) <= 0.40]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros_like(y), y))
    test_aucs, chosen = [], []
    for k, (tr, te) in enumerate(folds):
        best = (-1, None)
        for ag, bg, cg in grid:
            blend = ag * r_K18[tr] + bg * r_R30[tr] + cg * r_R35[tr]
            auc = roc_auc_score(y[tr], blend)
            if auc > best[0]:
                best = (auc, (ag, bg, cg))
        w = best[1]
        chosen.append(w)
        b_te = w[0] * r_K18[te] + w[1] * r_R30[te] + w[2] * r_R35[te]
        test_aucs.append(roc_auc_score(y[te], b_te))
    print(f"  nested mean: {np.mean(test_aucs):.6f}")
    print(f"  per-fold weights: {chosen}")

    # Build submission candidates if any blend OOF >= R25 OOF
    R25_oof_auc = roc_auc_score(y, R25_oof)
    candidates_to_build = []
    for name, w in [
        ("R38_K18_R35_2way", (0.92, 0.08, None, None)),
        ("R39_K18_R30_R35_3way", (0.80, 0.15, 0.05, None)),
        ("R40_K18_R30_R72_R35_4way", (0.55, 0.15, 0.20, 0.10)),
    ]:
        candidates_to_build.append((name, w))

    # TODO: build CSVs only if blend lifts vs R25

if __name__ == "__main__":
    main()
