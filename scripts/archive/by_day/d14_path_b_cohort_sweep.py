"""d14 Path B cohort sweep: Year, Year×Stint, Race.

Extends d13_path_b_hier_meta with three untested cohorts:
  Year (4 segs)  -- leverages 2023 anomaly directly; per-segment LR
                    can downweight 2023's flat-rate dilution
  Year×Stint (4×6 = 24 segs)  -- interacts the FM-hub field (Phase E
                    found Year×Stint dominant) with Stint (current best)
  Race (26 segs)  -- highest cohort count; small-n risk per ~17k/seg

Run as a sibling to d13_path_b_hier_meta with the same K=21 PRIMARY
pool, same fold structure, same tau sweep.  Submits NOTHING (Rule 1).

Output:
  scripts/artifacts/oof_d14_path_b_{kind}_tau{tau}_strat.npy
  scripts/artifacts/test_d14_path_b_{kind}_tau{tau}_strat.npy
  scripts/artifacts/d14_path_b_cohort_sweep_results.json
  submissions/submission_d14_path_b_{kind}_tau{tau}.csv  (held)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "scripts")
from d13_path_b_hier_meta import (  # noqa: E402
    POOL_KEEP, TOP_3_D9, expand, hier_oof, predicted_lb_delta,
)

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def make_segments_extra(train, test, kind):
    """Year, Year×Stint, Race cohorts not in d13 script."""
    def stint_clip(s):
        return np.clip(s.astype(int).values, 0, 5)
    if kind == "year":
        years = sorted(set(train["Year"].astype(int).unique()) |
                       set(test["Year"].astype(int).unique()))
        mp = {y: i for i, y in enumerate(years)}
        st = train["Year"].astype(int).map(mp).astype(int).values
        ste = test["Year"].astype(int).map(mp).astype(int).values
        return st, ste, len(years), f"Year ({len(years)})"
    elif kind == "year_stint":
        years = sorted(set(train["Year"].astype(int).unique()) |
                       set(test["Year"].astype(int).unique()))
        ymp = {y: i for i, y in enumerate(years)}
        y_tr = train["Year"].astype(int).map(ymp).astype(int).values
        y_te = test["Year"].astype(int).map(ymp).astype(int).values
        s_tr = stint_clip(train["Stint"])
        s_te = stint_clip(test["Stint"])
        st = y_tr * 6 + s_tr
        ste = y_te * 6 + s_te
        return st, ste, len(years) * 6, f"Year×Stint ({len(years) * 6})"
    elif kind == "race":
        races = sorted(set(train["Race"].astype(str).unique()) |
                       set(test["Race"].astype(str).unique()))
        mp = {r: i for i, r in enumerate(races)}
        st = train["Race"].astype(str).map(mp).astype(int).values
        ste = test["Race"].astype(str).map(mp).astype(int).values
        return st, ste, len(races), f"Race ({len(races)})"
    raise ValueError(kind)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy"
                           )[:, 1].astype(np.float64)
    # current PRIMARY-of-record on LB
    current_primary_oof = np.load(ART / "oof_d13e_compound_stint_tau20000_strat.npy"
                                   )[:, 1].astype(np.float64)
    current_primary_test = np.load(ART / "test_d13e_compound_stint_tau20000_strat.npy"
                                    )[:, 1].astype(np.float64)
    current_primary_auc = roc_auc_score(y, current_primary_oof)

    print(f"Loading K=21 PRIMARY pool…")
    base_oofs, base_tests, names = [], [], []
    for label, fname in POOL_KEEP + TOP_3_D9:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        base_oofs.append(oo); base_tests.append(te); names.append(label)
    # FM_A and FM_B (per d13 script)
    for label, fname in [("FM_A", "d9f_FM_A"), ("FM_B", "d9f_FM_B")]:
        oo = np.load(ART / f"oof_{fname}_strat.npy").astype(np.float64)
        if oo.ndim == 2: oo = oo[:, 1]
        te = np.load(ART / f"test_{fname}_strat.npy").astype(np.float64)
        if te.ndim == 2: te = te[:, 1]
        base_oofs.append(oo); base_tests.append(te); names.append(label)

    P_oof = np.column_stack(base_oofs)
    P_test = np.column_stack(base_tests)
    F_oof = expand(P_oof); F_test = expand(P_test)
    print(f"K = {P_oof.shape[1]} bases, F_oof shape {F_oof.shape}")

    # global LR meta as baseline (matches d13 logic)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(F_oof, y))
    oof_global = np.zeros(len(y), dtype=np.float64)
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        oof_global[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc_global = float(roc_auc_score(y, oof_global))
    print(f"Global LR meta OOF: {auc_global:.5f} (matches d9f K=21 swap PRIMARY)")
    print(f"d13e Compound×Stint τ=20k OOF: {current_primary_auc:.5f}")

    taus = [5000, 20000, 100000]
    kinds = ["year", "year_stint", "race"]
    results = {}

    for kind in kinds:
        seg_tr, seg_te, n_seg, name = make_segments_extra(train, test, kind)
        sizes = np.bincount(seg_tr, minlength=n_seg)
        sized = sizes[sizes > 0]
        print(f"\n--- {name}: populated={len(sized)}/{n_seg}, "
              f"min/median/max rows = "
              f"{sized.min()}/{int(np.median(sized))}/{sized.max()} ---")
        t_kind = time.time()
        oofs_tau, tests_tau = hier_oof(F_oof, F_test, y, seg_tr, seg_te,
                                        n_seg, taus, splits)
        print(f"  hier_oof time: {time.time() - t_kind:.0f}s")
        for tau in taus:
            auc = float(roc_auc_score(y, oofs_tau[tau]))
            rho_d9f, _ = spearmanr(tests_tau[tau], primary_test)
            rho_curr, _ = spearmanr(tests_tau[tau], current_primary_test)
            d_global = (auc - auc_global) * 1e4
            d_curr = (auc - current_primary_auc) * 1e4
            d_lb_pred = predicted_lb_delta(d_global, rho_d9f)
            # flip count vs current PRIMARY
            new = tests_tau[tau]
            cur = current_primary_test
            top10 = np.percentile(cur, 90)
            bot10 = np.percentile(cur, 10)
            flips_neg = int(((cur > top10) & (new < np.percentile(new, 90))).sum())
            flips_pos = int(((cur < bot10) & (new > np.percentile(new, 10))).sum())

            print(f"  τ={tau:>6}: OOF {auc:.5f} "
                  f"Δd9f {d_global:+.2f}bp  Δcurr {d_curr:+.2f}bp  "
                  f"ρ_d9f {rho_d9f:.5f}  ρ_curr {rho_curr:.5f}  "
                  f"flips {flips_neg}/{flips_pos}")

            np.save(ART / f"oof_d14_path_b_{kind}_tau{tau}_strat.npy",
                    oofs_tau[tau].astype(np.float32))
            np.save(ART / f"test_d14_path_b_{kind}_tau{tau}_strat.npy",
                    tests_tau[tau].astype(np.float32))
            sub = sample_sub.copy()
            sub[TARGET] = tests_tau[tau]
            sub.to_csv(f"submissions/submission_d14_path_b_{kind}_tau{tau}.csv",
                       index=False)

            results[(kind, tau)] = dict(
                segment=name, tau=tau, oof=auc,
                delta_d9f_bp=float(d_global),
                delta_current_primary_bp=float(d_curr),
                rho_vs_d9f=float(rho_d9f),
                rho_vs_current_primary=float(rho_curr),
                flips_neg=flips_neg, flips_pos=flips_pos,
                pred_lb_delta_bp=float(d_lb_pred),
            )

    print(f"\n=== Sweep complete ({time.time() - t0:.0f}s wall) ===\n")
    print("rank\tcohort\tτ\tOOF\tΔd9f\tΔcurr\tρ_d9f\tρ_curr\tflips_n/p")
    for (k, t), r in sorted(results.items(),
                              key=lambda kv: -kv[1]["oof"]):
        print(f"\t{k:14s}\t{t:>6}\t{r['oof']:.5f}\t"
              f"{r['delta_d9f_bp']:+.2f}\t{r['delta_current_primary_bp']:+.2f}\t"
              f"{r['rho_vs_d9f']:.5f}\t{r['rho_vs_current_primary']:.5f}\t"
              f"{r['flips_neg']}/{r['flips_pos']}")

    out = {f"{k}__tau{t}": v for (k, t), v in results.items()}
    (ART / "d14_path_b_cohort_sweep_results.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nresults saved to scripts/artifacts/d14_path_b_cohort_sweep_results.json")


if __name__ == "__main__":
    main()
