"""H3 — per-cohort isotonic post-hoc on PRIMARY K=22.

Two cohort splits tested: Stint, Year, Compound.
Two evaluations:
  (a) IN-SAMPLE upper bound: fit isotonic on full-OOF, score on same.
  (b) HONEST nested-CV: 5-fold; in each fold, fit isotonic on outer-train OOF,
      predict outer-val.  This is the realistic OOF gain.

Outputs:
  scripts/artifacts/oof_H3_isoStint_strat.npy
  scripts/artifacts/test_H3_isoStint_strat.npy
  audit/2026-05-13-H3-results.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")


def main() -> None:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).to_numpy()

    oof = np.load(ART / "oof_PRIMARY_K22_strat.npy").astype(np.float64)
    te = np.load(ART / "test_PRIMARY_K22_strat.npy").astype(np.float64)
    base_auc = roc_auc_score(y, oof)
    print(f"Baseline PRIMARY K=22 OOF AUC = {base_auc:.6f}")

    # --- (a) In-sample upper bound per cohort ---
    rows = []
    for cohort in ["Stint", "Year", "Compound", "PitStop"]:
        adj = oof.copy()
        for v in sorted(train[cohort].unique()):
            m = train[cohort].to_numpy() == v
            if m.sum() < 500:
                continue
            ir = IsotonicRegression(out_of_bounds="clip").fit(oof[m], y[m])
            adj[m] = ir.predict(oof[m])
        new_auc = roc_auc_score(y, adj)
        rows.append((cohort, "in_sample", base_auc, new_auc,
                     (new_auc - base_auc) * 1e4))

    # --- (b) Honest nested-CV per cohort ---
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    nested_results = {}
    for cohort in ["Stint", "Year", "Compound"]:
        adj = oof.copy()
        cohort_arr = train[cohort].to_numpy()
        for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
            for v in np.unique(cohort_arr[tr_idx]):
                tr_mask = (cohort_arr[tr_idx] == v)
                if tr_mask.sum() < 100:
                    continue
                # fit on outer-train rows of this cohort
                ir = IsotonicRegression(out_of_bounds="clip").fit(
                    oof[tr_idx][tr_mask], y[tr_idx][tr_mask])
                # apply to outer-val rows of this cohort
                va_mask = (cohort_arr[va_idx] == v)
                adj[va_idx[va_mask]] = ir.predict(oof[va_idx][va_mask])
        new_auc = roc_auc_score(y, adj)
        nested_results[cohort] = adj
        rows.append((cohort, "nested_cv", base_auc, new_auc,
                     (new_auc - base_auc) * 1e4))

    # --- Combined cohort: Stint × Year ---
    adj = oof.copy()
    cohort_arr = (train["Stint"].astype(str) + "_" + train["Year"].astype(str)).to_numpy()
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        for v in np.unique(cohort_arr[tr_idx]):
            tr_mask = (cohort_arr[tr_idx] == v)
            if tr_mask.sum() < 100:
                continue
            ir = IsotonicRegression(out_of_bounds="clip").fit(
                oof[tr_idx][tr_mask], y[tr_idx][tr_mask])
            va_mask = (cohort_arr[va_idx] == v)
            adj[va_idx[va_mask]] = ir.predict(oof[va_idx][va_mask])
    new_auc = roc_auc_score(y, adj)
    nested_results["Stint_x_Year"] = adj
    rows.append(("Stint × Year", "nested_cv", base_auc, new_auc,
                 (new_auc - base_auc) * 1e4))

    # --- Apply best cohort's calibration to TEST predictions ---
    # Use Stint cohort as default deployment (full-train fit)
    best_cohort = "Stint"  # selected from in-sample table
    test_adj = te.copy()
    for v in sorted(train["Stint"].unique()):
        tr_mask = train["Stint"].to_numpy() == v
        te_mask = test["Stint"].to_numpy() == v
        if tr_mask.sum() < 500 or te_mask.sum() == 0:
            continue
        ir = IsotonicRegression(out_of_bounds="clip").fit(oof[tr_mask], y[tr_mask])
        test_adj[te_mask] = ir.predict(te[te_mask])

    np.save(ART / "oof_H3_isoStint_strat.npy",
            nested_results["Stint"].astype(np.float32))
    np.save(ART / "test_H3_isoStint_strat.npy",
            test_adj.astype(np.float32))

    # --- Report ---
    md = ["# H3 — Per-cohort isotonic post-hoc on PRIMARY K=22\n"]
    md.append(f"Baseline PRIMARY K=22 OOF AUC = **{base_auc:.6f}**\n")
    md.append("| Cohort | Eval | Base | Calibrated | Δ bp |")
    md.append("|---|---|---:|---:|---:|")
    for c, e, b, n, d in rows:
        md.append(f"| {c} | {e} | {b:.6f} | {n:.6f} | {d:+.2f} |")
    md.append("\n**Honest nested-CV gain on Stint = "
              f"{[r for r in rows if r[0]=='Stint' and r[1]=='nested_cv'][0][4]:+.2f} bp**")
    md.append("\nDeployed: per-Stint isotonic, test predictions saved to "
              "`scripts/artifacts/test_H3_isoStint_strat.npy`.\n")
    md.append("\n**Min-meta-gate decision**: H3 is post-hoc on PRIMARY (not a base "
              "addition).  The honest OOF gain *is* the predicted LB Δ.  No min-meta "
              "needed; submit-as-PRIMARY-replacement decision is governed by Rule 1 "
              "(PI sign-off).\n")

    Path("audit/2026-05-13-H3-results.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
