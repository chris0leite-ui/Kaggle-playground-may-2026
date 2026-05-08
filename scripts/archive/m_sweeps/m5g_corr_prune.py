"""H3 — Pairwise correlation prune on the 15-base M5f pool.

Per main's Day-2-strategy-critique: M5d gap widened to -6.0bp adding
correlated HGBC β variants. M5f adds 3 more bases on top, raising
redundancy risk. This script:

1. Computes pairwise Pearson ρ on the 15 base Strat OOFs.
2. For any pair with ρ ≥ 0.97, drops the lower-standalone-OOF member.
3. Refits LR-meta on the pruned pool → M5g.
4. Compares M5g vs M5f on both anchors.

Output files:
  scripts/artifacts/m5g_*  (refit OOFs + test + results)
  submissions/submission_m5g_lr_meta_pruned.csv
  audit/2026-05-04-m5g-corr-prune.md

Pruning rule: greedy on descending standalone Strat OOF. For a pair
(A, B) with ρ ≥ 0.97 where AUC(A) > AUC(B), drop B. Iterate until no
pair exceeds the threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
BASE_S, BASE_G = 0.94075, 0.92059
M5C_S, M5C_G = 0.95000, 0.92963
M5D_S, M5D_G = 0.95023, 0.92994
M5E_S, M5E_G = 0.95027, 0.93084
M5F_S, M5F_G = 0.95042, 0.93105
SEED, N_FOLDS = 42, 5
CORR_THRESH = 0.97

POOL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("m3_catboost", "m3_catboost"),
    ("m4_relstate", "m4_relstate"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"),
    ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
]


def load(name, suffix):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1].astype(np.float64)
    return oof, test


def expand(P):
    n = len(P)
    rank = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) / (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rank, logit])


def stack_refit(suffix, base_auc, pool_subset):
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    Xs_oof, Xs_test, names = [], [], []
    for label, name in pool_subset:
        oo, te = load(name, suffix)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    meta_test = lr_full.predict_proba(F_test)[:, 1]
    return meta_oof, meta_test, auc, names


def main():
    print("=== H3 corr-prune on 15-base M5f pool ===\n")

    # Load all base Strat OOFs and standalone AUCs
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    base_oofs = {}
    base_aucs = {}
    for label, name in POOL:
        oo, _ = load(name, "strat")
        base_oofs[label] = oo
        base_aucs[label] = float(roc_auc_score(y, oo))
    print("Per-base Strat OOF AUC (sorted desc):")
    for label, auc in sorted(base_aucs.items(), key=lambda kv: -kv[1]):
        print(f"  {label:24s} {auc:.5f}")

    # Pairwise correlation matrix (Pearson on raw probs)
    labels = [p[0] for p in POOL]
    M = np.column_stack([base_oofs[lbl] for lbl in labels])
    C = np.corrcoef(M.T)
    print(f"\nPairs with |ρ| ≥ {CORR_THRESH}:")
    pairs_above = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if abs(C[i, j]) >= CORR_THRESH:
                pairs_above.append((labels[i], labels[j], float(C[i, j])))
                print(f"  {labels[i]:24s} ↔ {labels[j]:24s} ρ={C[i,j]:+.4f}")

    # Greedy prune: for each high-corr pair, drop lower-AUC member
    surviving = list(labels)
    dropped = []
    iteration = 0
    while True:
        iteration += 1
        S_idx = {lbl: labels.index(lbl) for lbl in surviving}
        worst = None
        for li, lbl_i in enumerate(surviving):
            for lj, lbl_j in enumerate(surviving):
                if lj <= li:
                    continue
                rho = C[S_idx[lbl_i], S_idx[lbl_j]]
                if abs(rho) >= CORR_THRESH:
                    drop = lbl_i if base_aucs[lbl_i] < base_aucs[lbl_j] else lbl_j
                    keep = lbl_j if drop == lbl_i else lbl_i
                    if worst is None or abs(rho) > worst[0]:
                        worst = (abs(rho), drop, keep, rho)
        if worst is None:
            break
        rho_abs, drop, keep, rho_signed = worst
        print(f"  iter{iteration}: dropping {drop} (AUC {base_aucs[drop]:.5f}) "
              f"-- ρ={rho_signed:+.4f} with {keep} (AUC {base_aucs[keep]:.5f})")
        surviving.remove(drop)
        dropped.append((drop, keep, rho_signed))
    print(f"\nSurviving pool ({len(surviving)}/{len(labels)}): {surviving}")
    print(f"Dropped ({len(dropped)}): {[d[0] for d in dropped]}")

    pruned_pool = [p for p in POOL if p[0] in surviving]

    print("\n=== M5g — Strat (pruned pool) ===")
    oof_s, test_s, auc_s, names_s = stack_refit("strat", BASE_S, pruned_pool)
    print(f"  M5g Strat: {auc_s:.5f}  Δbase={(auc_s-BASE_S)*1e4:+.1f}bp  "
          f"K={len(names_s)}")

    print("=== M5g — GroupKF (pruned pool) ===")
    oof_g, test_g, auc_g, _ = stack_refit("groupkf", BASE_G, pruned_pool)
    print(f"  M5g GroupKF: {auc_g:.5f}  Δbase={(auc_g-BASE_G)*1e4:+.1f}bp")

    np.save(ART / "oof_m5g_strat.npy", np.column_stack([1 - oof_s, oof_s]))
    np.save(ART / "test_m5g_strat.npy", np.column_stack([1 - test_s, test_s]))
    np.save(ART / "oof_m5g_groupkf.npy", np.column_stack([1 - oof_g, oof_g]))
    np.save(ART / "test_m5g_groupkf.npy", np.column_stack([1 - test_g, test_g]))

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv("submissions/submission_m5g_lr_meta_pruned.csv", index=False)

    res = dict(
        threshold=CORR_THRESH,
        pool_size_before=len(POOL),
        pool_size_after=len(surviving),
        surviving=surviving,
        dropped=[(d, k, r) for d, k, r in dropped],
        strat=dict(auc=auc_s, delta_base_bp=(auc_s - BASE_S) * 1e4,
                   delta_m5f_bp=(auc_s - M5F_S) * 1e4,
                   delta_m5d_bp=(auc_s - M5D_S) * 1e4),
        groupkf=dict(auc=auc_g, delta_base_bp=(auc_g - BASE_G) * 1e4,
                     delta_m5f_bp=(auc_g - M5F_G) * 1e4),
        per_base_auc=base_aucs,
    )
    (ART / "m5g_lr_meta_results.json").write_text(json.dumps(res, indent=2))

    body = (
        f"# M5g — corr-pruned stack ρ≥{CORR_THRESH} (2026-05-04)\n\n"
        f"Pool ({len(surviving)}): {surviving}\n\n"
        f"Dropped ({len(dropped)}):\n"
        + "\n".join([f"  - {d} (kept {k}, ρ={r:+.4f})" for d, k, r in dropped])
        + "\n\n## Results\n\n"
        f"| anchor | M5g | M5f | Δ vs M5f | Δ vs M5d (LB 0.94963) |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| Strat | **{auc_s:.5f}** | {M5F_S:.5f} | "
        f"{(auc_s - M5F_S) * 1e4:+.1f}bp | {(auc_s - M5D_S) * 1e4:+.1f}bp |\n"
        f"| GroupKF | **{auc_g:.5f}** | {M5F_G:.5f} | "
        f"{(auc_g - M5F_G) * 1e4:+.1f}bp | {(auc_g - M5D_G) * 1e4:+.1f}bp |\n\n"
        f"Submission: submissions/submission_m5g_lr_meta_pruned.csv (held).\n"
    )
    Path("audit/2026-05-04-m5g-corr-prune.md").write_text(body)
    print(f"\nM5g vs M5f: Δstrat={(auc_s-M5F_S)*1e4:+.1f}bp  "
          f"Δgroupkf={(auc_g-M5F_G)*1e4:+.1f}bp")


if __name__ == "__main__":
    main()
