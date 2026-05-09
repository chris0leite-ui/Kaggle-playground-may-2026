"""Phase 10 — Path-B C × stint_start_imputed_bin cohort on K=4.

Per d18 K1/K2/K3 friction "cohort-axis variation isn't the amp axis":
mode-id cohorts gave ~1.0× amp at K=28. NEVER tested with the
DGP-recovered orig-stint cohort `stint_start_imputed_bin`.

Mechanism: synth's Stint label is fabricated (P1). The TRUE orig-stint
identifier is `stint_start_imputed = LapNumber − TyreLife + 1`. Path-B
partial-pooling on the recovered cohort routes shrinkage along the
DGP's actual structure rather than synth's noise.

Cohort axes tested:
  - stint_start_imputed_bin (8 bins) — NEW (DGP-recovered)
  - Compound × stint_start_imputed_bin (5 × 8 = 40 cells) — NEW
  - Compound × Stint (current PRIMARY: 5 × 8 = 40 cells) — REFERENCE

τ ∈ {5000, 20000, 100000}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
SEED, N_FOLDS = 42, 5

K4 = [
    "d17_h1d_yekenot_full",
    "p1_single_cb_v4_gpu",
    "f1_hgbc_deep",
    "d16_orig_continuous_only",
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    P_clip = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(P_clip / (1 - P_clip))
    return np.hstack([P, rk, logit])


def fit_lr(F, y, max_iter=300):
    lr = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def predict(F, w):
    F_aug = np.column_stack([np.ones(len(F)), F])
    return 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))


def main():
    ts = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["PitNextLap"].astype(int).values
    primary_test = np.load(ART / "test_K4_fwd_pathb.npy")[:, 1]

    def load_p(path):
        a = np.load(path).astype(np.float64)
        if a.ndim == 2 and a.shape[1] == 2:
            return a[:, 1]
        return a.ravel()

    base_oofs, base_tests = [], []
    for fname in K4:
        oo = load_p(ART / f"oof_{fname}_strat.npy")
        te = load_p(ART / f"test_{fname}_strat.npy")
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"K={len(base_oofs)} bases; F shape {F_oof.shape}", flush=True)

    # Cohort axes
    train["ss"] = train["LapNumber"] - train["TyreLife"] + 1
    test["ss"] = test["LapNumber"] - test["TyreLife"] + 1
    bins = [0, 1, 5, 10, 15, 20, 25, 30, 80]
    train["ss_bin"] = pd.cut(train["ss"], bins=bins, labels=False).fillna(0).astype(int)
    test["ss_bin"] = pd.cut(test["ss"], bins=bins, labels=False).fillna(0).astype(int)
    cmp_levels = sorted(set(train["Compound"]) | set(test["Compound"]))
    cmp = {c: i for i, c in enumerate(cmp_levels)}
    c_tr = train["Compound"].map(cmp).values
    c_te = test["Compound"].map(cmp).values
    ss_tr = train["ss_bin"].values
    ss_te = test["ss_bin"].values

    # Cohorts to test
    cohorts = {
        "ss_only_8": (ss_tr, ss_te),
        "C_x_ss_40": (c_tr * 8 + ss_tr, c_te * 8 + ss_te),
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Global LR-meta on K=4 (reference)
    print("\n--- Reference: K=4 plain LR-meta ---", flush=True)
    oof_global = np.zeros(len(y))
    for tr, va in splits:
        w = fit_lr(F_oof[tr], y[tr])
        oof_global[va] = predict(F_oof[va], w)
    auc_global = roc_auc_score(y, oof_global)
    print(f"  global LR-meta OOF: {auc_global:.5f}", flush=True)

    taus = [5000, 20000, 100000]

    for cohort_name, (cohort_tr, cohort_te) in cohorts.items():
        print(f"\n--- Path-B cohort: {cohort_name} "
              f"({len(np.unique(cohort_tr))} cells) ---", flush=True)
        oofs = {tau: np.zeros(len(y)) for tau in taus}
        # Per-fold LR-aug
        for tr, va in splits:
            for tau in taus:
                # global w
                w_global = fit_lr(F_oof[tr], y[tr])
                # per-cohort weighted shrinkage
                cohort_va = cohort_tr[va]
                for cell in np.unique(cohort_tr[tr]):
                    cell_tr = cohort_tr[tr] == cell
                    n_local = cell_tr.sum()
                    alpha = n_local / (n_local + tau)
                    if n_local < 200:
                        # too small for separate fit -- use global
                        continue
                    w_local = fit_lr(F_oof[tr][cell_tr], y[tr][cell_tr],
                                      max_iter=300)
                    # shrunk weights
                    w_shrunk = alpha * w_local + (1 - alpha) * w_global
                    cell_va_idx = va[cohort_va == cell]
                    if len(cell_va_idx) == 0:
                        continue
                    oofs[tau][cell_va_idx] = predict(F_oof[cell_va_idx], w_shrunk)
                # rows in val with cells unseen in train use global
                for cell in np.unique(cohort_tr[va]):
                    if cell not in np.unique(cohort_tr[tr]):
                        cell_va_idx = va[cohort_va == cell]
                        oofs[tau][cell_va_idx] = predict(F_oof[cell_va_idx],
                                                         w_global)
                # also rows where cell was too small
                for cell in np.unique(cohort_tr[tr]):
                    if (cohort_tr[tr] == cell).sum() < 200:
                        cell_va_idx = va[cohort_va == cell]
                        if len(cell_va_idx) > 0:
                            oofs[tau][cell_va_idx] = predict(F_oof[cell_va_idx],
                                                             w_global)

        for tau in taus:
            auc = roc_auc_score(y, oofs[tau])
            delta_vs_global = (auc - auc_global) * 1e4
            print(f"  τ={tau:>7d}: OOF {auc:.5f}  Δ vs global LR {delta_vs_global:+.2f}bp",
                  flush=True)
            np.save(ART / f"oof_p10_pathb_{cohort_name}_tau{tau}_strat.npy",
                    np.column_stack([1 - oofs[tau], oofs[tau]]))

    # Compare to current PRIMARY (Compound × Stint τ=100k)
    primary_oof = np.load(ART / "oof_K4_fwd_pathb.npy")[:, 1]
    primary_auc = roc_auc_score(y, primary_oof)
    print(f"\n--- Reference: K=4 + Path-B Compound × Stint τ=100k OOF: "
          f"{primary_auc:.5f} ---", flush=True)
    print(f"Total time: {time.time()-ts:.0f}s", flush=True)


if __name__ == "__main__":
    main()
