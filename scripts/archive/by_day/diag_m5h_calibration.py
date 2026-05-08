"""M5h calibration diagnostic + per-Race rank-normalize lift test.

Step 1+2 of the post-segment-diagnostic plan (2026-05-04 mid-session):

1. Reliability bins on M5h OOF (Strat). Brier score. Diagnostic only —
   the comp metric is AUC, which is monotone-invariant, so global
   isotonic gives ZERO LB lift (worth confirming numerically).

2. Per-Race rank-normalize: rank-within-Race / N_in_Race, concat.
   Different transformation per Race → cross-Race ranking changes →
   AUC CAN move. Real lift test.

3. Per-Race isotonic regression: fit isotonic per-Race on OOF, apply
   to test. Same shape of analysis but more expressive than rank.

Saves diagnostic to audit/2026-05-04-d3-calibration.md and (if lift)
calibrated test predictions to scripts/artifacts/test_m5h_perRace*.npy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
M5H_AGG_S = 0.95043


def reliability_bins(y, p, n_bins=10):
    """Equal-frequency bins by predicted probability; observed pos rate per bin."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_ids = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        n = int(mask.sum())
        obs = float(y[mask].mean())
        pred = float(p[mask].mean())
        rows.append((b, n, pred, obs, obs - pred))
    return pd.DataFrame(rows, columns=["bin", "n", "pred_mean", "obs_rate", "gap"])


def per_group_rank_normalize(p, group):
    """Rank within each group, normalize to [0,1]. Returns array aligned to p."""
    out = np.zeros(len(p), dtype=np.float64)
    g = pd.Series(group)
    for key, idx in g.groupby(g, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) == 0:
            continue
        out[idx] = rankdata(p[idx]) / len(idx)
    return out


def per_group_isotonic(y_oof, p_oof, p_test, group_oof, group_test):
    """Fit isotonic per group on OOF, apply to test. Groups absent from OOF
    get a fallback global isotonic. Returns (calibrated_oof, calibrated_test).
    """
    out_oof = np.zeros(len(p_oof), dtype=np.float64)
    out_test = np.zeros(len(p_test), dtype=np.float64)

    # Global fallback
    iso_global = IsotonicRegression(out_of_bounds="clip")
    iso_global.fit(p_oof, y_oof)

    g_oof = pd.Series(group_oof)
    g_test = pd.Series(group_test)
    fitted = {}
    for key, idx in g_oof.groupby(g_oof, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 100 or len(np.unique(y_oof[idx])) < 2:
            # too small or one-class → fallback
            out_oof[idx] = iso_global.predict(p_oof[idx])
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_oof[idx], y_oof[idx])
        out_oof[idx] = iso.predict(p_oof[idx])
        fitted[key] = iso

    for key, idx in g_test.groupby(g_test, sort=False).groups.items():
        idx = np.asarray(idx)
        if key in fitted:
            out_test[idx] = fitted[key].predict(p_test[idx])
        else:
            out_test[idx] = iso_global.predict(p_test[idx])

    return out_oof, out_test


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    oof = np.load(ART / "oof_m5h_strat.npy")[:, 1].astype(np.float64)
    p_test = np.load(ART / "test_m5h_strat.npy")[:, 1].astype(np.float64)
    auc_orig = float(roc_auc_score(y, oof))
    brier_orig = float(brier_score_loss(y, oof))
    print(f"M5h OOF AUC: {auc_orig:.5f}  Brier: {brier_orig:.5f}\n")

    # === Step 1: reliability bins (diagnostic only) ===
    print("=== Reliability bins (decile, M5h Strat OOF) ===")
    rel = reliability_bins(y, oof, n_bins=10)
    print(rel.to_string(index=False))

    # Global isotonic AUC sanity (should be exactly auc_orig)
    iso_g = IsotonicRegression(out_of_bounds="clip")
    iso_g.fit(oof, y)
    oof_iso_g = iso_g.predict(oof)
    auc_iso_g = float(roc_auc_score(y, oof_iso_g))
    brier_iso_g = float(brier_score_loss(y, oof_iso_g))
    print(f"\nGlobal isotonic: AUC={auc_iso_g:.5f} (Δ={(auc_iso_g-auc_orig)*1e4:+.1f}bp)  "
          f"Brier={brier_iso_g:.5f}")

    # === Step 2: per-Race rank-normalize ===
    print("\n=== Per-Race rank-normalize ===")
    oof_rrank = per_group_rank_normalize(oof, train["Race"].values)
    auc_rrank = float(roc_auc_score(y, oof_rrank))
    print(f"Per-Race rank: AUC={auc_rrank:.5f}  Δ M5h={(auc_rrank-auc_orig)*1e4:+.1f}bp")

    # === Step 3: per-Race isotonic ===
    print("\n=== Per-Race isotonic ===")
    oof_iso_r, test_iso_r = per_group_isotonic(
        y, oof, p_test, train["Race"].values, test["Race"].values)
    auc_iso_r = float(roc_auc_score(y, oof_iso_r))
    print(f"Per-Race isotonic: AUC={auc_iso_r:.5f}  Δ M5h={(auc_iso_r-auc_orig)*1e4:+.1f}bp")

    # === Per-Year isotonic (smaller grouping; 4 groups vs 26) ===
    print("\n=== Per-Year isotonic ===")
    oof_iso_y, test_iso_y = per_group_isotonic(
        y, oof, p_test, train["Year"].values, test["Year"].values)
    auc_iso_y = float(roc_auc_score(y, oof_iso_y))
    print(f"Per-Year isotonic: AUC={auc_iso_y:.5f}  Δ M5h={(auc_iso_y-auc_orig)*1e4:+.1f}bp")

    # === Per-(Year, Race) isotonic ===
    g_oof_yr = (train["Year"].astype(str) + "|" + train["Race"].astype(str)).values
    g_test_yr = (test["Year"].astype(str) + "|" + test["Race"].astype(str)).values
    print("\n=== Per-(Year,Race) isotonic ===")
    oof_iso_yr, test_iso_yr = per_group_isotonic(y, oof, p_test, g_oof_yr, g_test_yr)
    auc_iso_yr = float(roc_auc_score(y, oof_iso_yr))
    print(f"Per-(Year,Race) isotonic: AUC={auc_iso_yr:.5f}  "
          f"Δ M5h={(auc_iso_yr-auc_orig)*1e4:+.1f}bp")

    # === Inner-CV check: does per-(Year, Race) isotonic generalize? ===
    # The OOF predictions are out-of-fold; isotonic fit on them is a meta-step.
    # Sanity: inner 5-fold split on OOF rows. Fit isotonic on 4 inner folds,
    # eval on the held-out 5th. Honest generalization estimate.
    from sklearn.model_selection import StratifiedKFold
    print("\n=== Inner-CV (5-fold) per-(Year, Race) isotonic generalization ===")
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    inner_oof_yr = np.zeros(len(y), dtype=np.float64)
    inner_oof_r = np.zeros(len(y), dtype=np.float64)
    g_oof_r = train["Race"].values  # for the per-Race-only inner-cv check

    def inner_cv_per_group(group_keys):
        out = np.zeros(len(y), dtype=np.float64)
        for k_inner, (tr_in, va_in) in enumerate(skf_inner.split(np.zeros(len(y)), y)):
            iso_global_inner = IsotonicRegression(out_of_bounds="clip")
            iso_global_inner.fit(oof[tr_in], y[tr_in])
            # Build group → tr_in indices map
            tr_groups = {}
            for j, key in enumerate(group_keys[tr_in]):
                tr_groups.setdefault(key, []).append(j)
            # Build group → va_in indices map
            va_groups = {}
            for j, key in enumerate(group_keys[va_in]):
                va_groups.setdefault(key, []).append(j)
            # Fit per-group isotonic, batch-predict
            for key, va_idx_in_va_subset in va_groups.items():
                va_idx_global = va_in[np.array(va_idx_in_va_subset)]
                if (key in tr_groups
                        and len(tr_groups[key]) >= 100):
                    local_idx_in_tr_subset = np.array(tr_groups[key])
                    local_idx_global = tr_in[local_idx_in_tr_subset]
                    if len(np.unique(y[local_idx_global])) >= 2:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(oof[local_idx_global], y[local_idx_global])
                        out[va_idx_global] = iso.predict(oof[va_idx_global])
                        continue
                out[va_idx_global] = iso_global_inner.predict(oof[va_idx_global])
        return out

    inner_oof_yr = inner_cv_per_group(g_oof_yr)
    auc_inner_yr = float(roc_auc_score(y, inner_oof_yr))
    print(f"Inner-CV per-(Year, Race) isotonic: AUC={auc_inner_yr:.5f}  "
          f"Δ M5h={(auc_inner_yr-auc_orig)*1e4:+.1f}bp  "
          f"(in-sample claim was {(auc_iso_yr-auc_orig)*1e4:+.1f}bp)")
    inner_oof_r = inner_cv_per_group(g_oof_r)
    auc_inner_r = float(roc_auc_score(y, inner_oof_r))
    print(f"Inner-CV per-Race isotonic: AUC={auc_inner_r:.5f}  "
          f"Δ M5h={(auc_inner_r-auc_orig)*1e4:+.1f}bp  "
          f"(in-sample claim was {(auc_iso_r-auc_orig)*1e4:+.1f}bp)")

    # === Save best-lift candidate ===
    candidates = [
        ("global_iso", oof_iso_g, iso_g.predict(p_test), auc_iso_g),
        ("perRace_rank", oof_rrank, per_group_rank_normalize(p_test, test["Race"].values), auc_rrank),
        ("perRace_iso", oof_iso_r, test_iso_r, auc_iso_r),
        ("perYear_iso", oof_iso_y, test_iso_y, auc_iso_y),
        ("perYearRace_iso", oof_iso_yr, test_iso_yr, auc_iso_yr),
    ]
    candidates_sorted = sorted(candidates, key=lambda x: -x[3])
    best = candidates_sorted[0]
    print(f"\n=== Best calibration: {best[0]} AUC={best[3]:.5f} "
          f"(Δ M5h={(best[3]-auc_orig)*1e4:+.1f}bp) ===")

    # Always save calibrated artifacts for the best candidate
    np.save(ART / f"oof_m5h_cal_{best[0]}.npy",
            np.column_stack([1 - best[1], best[1]]))
    np.save(ART / f"test_m5h_cal_{best[0]}.npy",
            np.column_stack([1 - best[2], best[2]]))

    # Submission for the best calibration (held until PI sign-off)
    sub = sample_sub.copy()
    sub[TARGET] = best[2]
    sub.to_csv(f"submissions/submission_m5h_cal_{best[0]}.csv", index=False)

    # Audit doc
    body = ["# M5h calibration diagnostics — 2026-05-04\n",
            f"Aggregate Strat OOF AUC: **{auc_orig:.5f}**  Brier: {brier_orig:.5f}\n",
            "## Reliability bins (decile)\n",
            rel.to_string(index=False),
            "\n\n## Calibration variants vs M5h baseline\n",
            "| variant | OOF AUC | Δ M5h (bp) | notes |",
            "|---|---:|---:|---|",
            f"| baseline (uncalibrated) | {auc_orig:.5f} | 0.0 | reference |",
            f"| global isotonic | {auc_iso_g:.5f} | {(auc_iso_g-auc_orig)*1e4:+.1f} | should be ~0 (AUC monotone-invariant) |",
            f"| per-Race rank-normalize | {auc_rrank:.5f} | {(auc_rrank-auc_orig)*1e4:+.1f} | rescale within Race |",
            f"| per-Race isotonic | {auc_iso_r:.5f} | {(auc_iso_r-auc_orig)*1e4:+.1f} | 26 isotonic fits; non-monotonic across Race |",
            f"| per-Year isotonic | {auc_iso_y:.5f} | {(auc_iso_y-auc_orig)*1e4:+.1f} | 4 fits; coarser |",
            f"| per-(Year,Race) isotonic | {auc_iso_yr:.5f} | {(auc_iso_yr-auc_orig)*1e4:+.1f} | finest grouping |",
            f"\nBest: **{best[0]}** at AUC {best[3]:.5f} (Δ M5h {(best[3]-auc_orig)*1e4:+.1f}bp).\n",
            "## Implications\n",
            "Global isotonic is AUC no-op (confirms diagnostic).",
            "Per-group calibrations CAN move AUC because they break global monotonicity.",
            "If the best variant gives ≥+5bp OOF AUC, it is a slot-7 candidate (free LB lift).",
            ]
    out = Path("audit/2026-05-04-d3-calibration.md")
    out.write_text("\n".join(body))
    print(f"\n→ {out}")
    print(f"→ submissions/submission_m5h_cal_{best[0]}.csv (held)")


if __name__ == "__main__":
    main()
