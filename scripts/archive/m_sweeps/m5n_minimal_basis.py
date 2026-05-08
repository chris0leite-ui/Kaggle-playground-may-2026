"""M5n — Minimal-orthogonal-basis stack (PI Day-3 hypothesis).

Per the disagreement diagnostic:
  - a_horizon         ρ vs others = 0.729 (most diverse)
  - b_lapsuntilpit    ρ vs others = 0.734
  - cb_slow-wide-bag  ρ vs others = 0.840
  - cb_lossguide      ρ vs others = 0.884 (4th, gap)
  - (all others)      ρ ≥ 0.89 (GBDT consensus clones)

Hypothesis: the 10 redundant bases inflate OOF but don't change rank.
Stacking only the orthogonal basis should give a SIMILAR ranking
(within fold-noise of M5h) but with LOWER OOF — and crucially,
DIFFERENT test predictions (Spearman significantly < 0.999 vs M5h).
That makes it a slot-9 candidate to break the 0.94991 LB tie.

Variants tested (all use raw+rank+logit expansion, LR meta):
  M5n_3   = [a_horizon, b_lapsuntilpit, cb_slow-wide-bag]
  M5n_3b  = [a_horizon, b_lapsuntilpit, cb_slow-wide-bag, baseline]
  M5n_4   = above + cb_lossguide
  M5n_5   = above + d2a_te (4th-most diverse at ρ=0.897)
  M5n_6   = above + cb_year-cat (5th)

Compare each to M5h (Strat OOF + Spearman ρ vs M5h test).

R1: Strat-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5H_S = 0.95043
SEED, N_FOLDS = 42, 5

# Diversity-ranked (Spearman ρ vs mean of other 12; lower = more diverse)
DIVERSITY = [
    ("a_horizon", "a_horizon"),                      # 0.729
    ("b_lapsuntilpit", "b_lapsuntilpit"),            # 0.734
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),        # 0.840
    ("cb_lossguide", "cb_lossguide"),                # 0.884
    ("d2a_te", "d2a_te"),                            # 0.897
    ("baseline", "baseline_two_anchor"),             # 0.909
    ("cb_year-cat", "cb_year-cat"),                  # 0.912
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),          # 0.914
    ("e3_hgbc", "e3_hgbc"),                          # 0.914
    ("f1_hgbc_deep", "f1_hgbc_deep"),                # 0.915
    ("m2_xgb", "m2_xgb"),                            # 0.916
    ("e1_cb_sub", "e1_catboost_sub"),                # 0.920
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),            # 0.922
]


def load(name, suffix="strat"):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rank, logit])


def fit_meta(F_oof, F_test, y, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc


def assemble(pool, suffix="strat"):
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool:
        oo, te = load(name, suffix)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    return expand(np.column_stack(Xs_oof)), expand(np.column_stack(Xs_test)), names


def main():
    train = pd.read_csv("data/train.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    m5h_test = np.load(ART / "test_m5h_strat.npy")[:, 1].astype(np.float64)

    variants = [
        ("M5n_3",   DIVERSITY[:3]),
        ("M5n_3b",  DIVERSITY[:3] + [DIVERSITY[5]]),  # +baseline (most "anchor"-like)
        ("M5n_4",   DIVERSITY[:4]),
        ("M5n_5",   DIVERSITY[:5]),
        ("M5n_6",   DIVERSITY[:6]),
        ("M5n_7",   DIVERSITY[:7]),
    ]

    print(f"=== Minimal-orthogonal-basis sweep (M5h Strat = {M5H_S:.5f}) ===\n")
    print(f"{'variant':<10} {'K':>3} {'Strat OOF':>11} {'Δ M5h (bp)':>12} "
          f"{'ρ vs M5h test':>14} {'rank-shift mean':>16}")

    results = {}
    for label, pool in variants:
        F_oof, F_test, names = assemble(pool, "strat")
        oof, test_p, auc = fit_meta(F_oof, F_test, y)
        rho, _ = spearmanr(test_p, m5h_test)
        ra, rb = rankdata(test_p), rankdata(m5h_test)
        rank_shift_mean = float(np.abs(ra - rb).mean())
        delta_m5h = (auc - M5H_S) * 1e4
        print(f"{label:<10} {len(names):>3} {auc:>11.5f} {delta_m5h:>+12.1f} "
              f"{rho:>14.5f} {rank_shift_mean:>16.0f}")
        results[label] = dict(K=len(names), pool=names, strat=auc,
                              delta_m5h_bp=delta_m5h, spearman_vs_m5h=rho,
                              rank_shift_mean=rank_shift_mean,
                              oof=oof, test=test_p)

    # Decision: candidate that maximizes (low Spearman vs M5h) AND keeps
    # OOF within 30bp of M5h. The combination = "differs structurally
    # without sacrificing measured ranking quality".
    print("\n=== Slot-candidate scoring ===")
    print("(target: ρ < 0.999 AND OOF ≥ M5h - 30bp)")
    candidates = []
    for label, r in results.items():
        if r["spearman_vs_m5h"] < 0.999 and r["delta_m5h_bp"] >= -30:
            print(f"  {label}: ρ={r['spearman_vs_m5h']:.5f}  Δ={r['delta_m5h_bp']:+.1f}bp  "
                  f"K={r['K']}  ✓ candidate")
            candidates.append(label)
        else:
            reason = []
            if r["spearman_vs_m5h"] >= 0.999: reason.append(f"ρ≥0.999")
            if r["delta_m5h_bp"] < -30: reason.append(f"OOF too low")
            print(f"  {label}: ρ={r['spearman_vs_m5h']:.5f}  Δ={r['delta_m5h_bp']:+.1f}bp  "
                  f"K={r['K']}  ✗ ({', '.join(reason)})")

    if candidates:
        # Best candidate = lowest Spearman (most rank-different from M5h)
        best = min(candidates, key=lambda l: results[l]["spearman_vs_m5h"])
        print(f"\n=== M5n WINNER: {best} ===")
        print(f"  K={results[best]['K']}  Strat={results[best]['strat']:.5f}  "
              f"ρ vs M5h={results[best]['spearman_vs_m5h']:.5f}")
        winner = results[best]
        np.save(ART / f"oof_{best.lower()}_strat.npy",
                np.column_stack([1 - winner["oof"], winner["oof"]]))
        np.save(ART / f"test_{best.lower()}_strat.npy",
                np.column_stack([1 - winner["test"], winner["test"]]))
        sub = sample_sub.copy()
        sub[TARGET] = winner["test"]
        sub.to_csv(f"submissions/submission_{best.lower()}.csv", index=False)
        print(f"→ submissions/submission_{best.lower()}.csv (held)")
    else:
        print("\nNo candidate clears both ρ<0.999 and OOF≥M5h-30bp gates.")
        print("All minimal-basis variants are either too redundant or too weak.")

    summary = {
        label: {k: v for k, v in r.items() if k not in ("oof", "test")}
        for label, r in results.items()
    }
    summary["candidates"] = candidates
    (ART / "m5n_minimal_basis_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/m5n_minimal_basis_results.json")


if __name__ == "__main__":
    main()
