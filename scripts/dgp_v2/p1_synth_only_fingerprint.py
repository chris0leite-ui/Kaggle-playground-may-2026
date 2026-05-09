"""Phase 1 — pure-synth DGP fingerprint (no public CSV).

PI directive: find DGP without leaning on the aadigupta1601 public CSV.

Probes synth (train + test) directly to characterize:

  Q1. Per-column quantization grid: do values lie on a fixed step?
      A discrete grid is a strong CTGAN/CopulaGAN fingerprint
      (mode-specific normalization re-emits empirical values).

  Q2. Mode count via DBSCAN-on-values for each numeric column. Compare
      to the BGMM(10) used in d18_g_mode_id (CTGAN modes).

  Q3. id-ordering structure: are id-adjacent rows more similar than
      id-distant rows? Tests the hypothesis that CTGAN sampled in
      batches with shared latent state.

  Q4. Ghost-driver clustering: 887 drivers, 31 real + 856 ghost
      (D001-D856 + 3-letter historical). Each ghost driver has a
      cluster of synth rows. What's their centroid distribution?

  Q5. Within-stint coherence on synth: what fraction of (Race, Driver,
      Year, Stint) groups have monotonic LapNumber, monotonic TyreLife,
      consistent Compound?

Output: audit/2026-05-08/2026-05-08-p1-synth-fingerprint.md and
scripts/artifacts/p1_synth_fingerprint.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
ART = ROOT / "scripts/artifacts"
AUDIT = ROOT / "audit/2026-05-08"

ART.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:5.1f}s] {label}", flush=True)


def main() -> dict:
    ts = time.time()
    print("Loading synth train + test...", flush=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    t(f"train {train.shape} | test {test.shape}", ts)

    full = pd.concat([train.drop(columns=["PitNextLap"]), test],
                     ignore_index=True)
    n_full = len(full)

    numeric_cols = [
        "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change", "Year",
    ]
    cat_cols = ["Driver", "Compound", "Race"]

    # ===== Q1. Quantization grid per numeric column =====
    print("\nQ1. Quantization grid analysis...", flush=True)
    grid_summary = {}
    for c in numeric_cols:
        v = full[c].dropna().to_numpy()
        n_unique = len(np.unique(v))
        # Smallest non-zero ULP between consecutive sorted unique values
        u = np.sort(np.unique(v))
        diffs = np.diff(u)
        diffs = diffs[diffs > 0]
        min_step = float(np.min(diffs)) if len(diffs) else 0.0
        # Are most diffs an integer multiple of min_step? (grid hypothesis)
        if min_step > 0 and len(diffs):
            ratios = diffs / min_step
            integer_frac = float(np.mean(np.abs(ratios - np.round(ratios)) < 1e-6))
        else:
            integer_frac = 1.0
        grid_summary[c] = {
            "n_unique": int(n_unique),
            "min_step": min_step,
            "integer_grid_frac": integer_frac,
            "min": float(v.min()),
            "max": float(v.max()),
        }
        print(f"    {c:25s} unique={n_unique:7d} step={min_step:.6g} "
              f"int_grid_frac={integer_frac:.3f}", flush=True)
    t("Q1 done", ts)

    # ===== Q2. id-ordering structure =====
    print("\nQ2. id-ordering structure...", flush=True)
    # Are id-adjacent rows more similar than id-distant rows?
    # Take L2 distance on standardized 8 KS-low numerics.
    from sklearn.preprocessing import StandardScaler
    KS_LOW = ["TyreLife", "Position", "LapTime (s)",
              "Cumulative_Degradation", "RaceProgress",
              "LapTime_Delta", "LapNumber", "Stint"]
    sc = StandardScaler()
    Xs = sc.fit_transform(full[KS_LOW].fillna(0).to_numpy())
    # adjacent
    adj_dist = np.linalg.norm(Xs[1:] - Xs[:-1], axis=1)
    # random pairs
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n_full, size=len(adj_dist))
    idx_b = rng.integers(0, n_full, size=len(adj_dist))
    rnd_dist = np.linalg.norm(Xs[idx_a] - Xs[idx_b], axis=1)
    id_order_summary = {
        "adjacent_dist_mean": float(adj_dist.mean()),
        "adjacent_dist_p50": float(np.median(adj_dist)),
        "random_dist_mean": float(rnd_dist.mean()),
        "random_dist_p50": float(np.median(rnd_dist)),
        "ratio_adj_to_rnd_mean": float(adj_dist.mean() / max(rnd_dist.mean(), 1e-9)),
    }
    for k, v in id_order_summary.items():
        print(f"    {k}: {v:.4f}", flush=True)
    t("Q2 done", ts)

    # ===== Q3. Ghost-driver clustering =====
    print("\nQ3. Ghost-driver structure...", flush=True)
    drv_counts = full["Driver"].value_counts()
    ghost_mask_d = full["Driver"].str.startswith("D")
    # 3-letter historicals (e.g. MAS, RAI) are uppercase 3-char alphabet
    abbrev_mask = full["Driver"].str.len() == 3
    ghost_drv = full.loc[ghost_mask_d & full["Driver"].str.match(r"^D\d{3}$")]
    abbrev_drv = full.loc[abbrev_mask]
    real_or_ghost_other = ~(ghost_mask_d | abbrev_mask)
    real_drv = full.loc[real_or_ghost_other]
    print(f"    drivers total: {drv_counts.shape[0]}", flush=True)
    print(f"    D-prefix ghosts (D###): rows={len(ghost_drv)} "
          f"distinct={ghost_drv['Driver'].nunique()}", flush=True)
    print(f"    3-letter abbrev (e.g. MAS): rows={len(abbrev_drv)} "
          f"distinct={abbrev_drv['Driver'].nunique()}", flush=True)
    print(f"    other (assumed real-name): rows={len(real_drv)} "
          f"distinct={real_drv['Driver'].nunique()}", flush=True)

    # Per-ghost-driver row count distribution
    g_counts = full[ghost_mask_d & full["Driver"].str.match(r"^D\d{3}$")
                    ]["Driver"].value_counts()
    print(f"    D### rows/driver: median={g_counts.median():.0f}, "
          f"p10={g_counts.quantile(0.1):.0f}, p90={g_counts.quantile(0.9):.0f}", flush=True)

    # Race × Driver overlap: how many races does each ghost appear in?
    races_per_ghost = (full[ghost_mask_d & full["Driver"].str.match(r"^D\d{3}$")]
                       .groupby("Driver")["Race"].nunique())
    print(f"    D### races/driver: median={races_per_ghost.median():.1f}, "
          f"min={races_per_ghost.min()}, max={races_per_ghost.max()}", flush=True)

    ghost_summary = {
        "n_drivers_total": int(drv_counts.shape[0]),
        "n_d_prefix": int(ghost_drv["Driver"].nunique()),
        "n_abbrev": int(abbrev_drv["Driver"].nunique()),
        "n_other": int(real_drv["Driver"].nunique()),
        "d_prefix_rows_per_driver": {
            "median": float(g_counts.median()),
            "p10": float(g_counts.quantile(0.1)),
            "p90": float(g_counts.quantile(0.9)),
        },
        "d_prefix_races_per_driver": {
            "median": float(races_per_ghost.median()),
            "min": float(races_per_ghost.min()),
            "max": float(races_per_ghost.max()),
        },
    }
    t("Q3 done", ts)

    # ===== Q4. Within-stint coherence on synth =====
    print("\nQ4. Within-stint coherence...", flush=True)
    # Group by (Race, Driver, Year, Stint). For each group, check:
    # - LapNumber strictly increasing?
    # - TyreLife strictly increasing?
    # - Compound constant?
    g = full.groupby(["Race", "Driver", "Year", "Stint"])
    n_groups = g.ngroups
    print(f"    n_groups: {n_groups}", flush=True)
    # Sample 50k groups for speed if large
    sample_keys = None
    if n_groups > 50000:
        rng = np.random.default_rng(0)
        keys = list(g.groups.keys())
        sample_keys = [keys[i] for i in rng.choice(len(keys), 50000, replace=False)]
        print(f"    sampling {len(sample_keys)} groups", flush=True)

    n_check = 0
    n_lap_mono = 0
    n_tyre_mono = 0
    n_compound_const = 0
    n_lap_strict_consec = 0  # adjacent diff = 1?
    iterator = sample_keys if sample_keys is not None else g.groups.keys()
    for k in iterator:
        sub = g.get_group(k) if sample_keys is None else full.loc[g.groups[k]]
        if len(sub) < 2:
            continue
        n_check += 1
        ln = sub["LapNumber"].to_numpy()
        tl = sub["TyreLife"].to_numpy()
        cp = sub["Compound"].to_numpy()
        if np.all(np.diff(ln) > 0):
            n_lap_mono += 1
        if np.all(np.diff(tl) > 0):
            n_tyre_mono += 1
        if (cp == cp[0]).all():
            n_compound_const += 1
        if np.all(np.diff(ln) == 1):
            n_lap_strict_consec += 1
    coherence = {
        "n_groups_total": int(n_groups),
        "n_groups_checked_with_2plus_rows": n_check,
        "lap_monotonic_frac": n_lap_mono / max(n_check, 1),
        "tyre_monotonic_frac": n_tyre_mono / max(n_check, 1),
        "compound_constant_frac": n_compound_const / max(n_check, 1),
        "lap_strict_consecutive_frac": n_lap_strict_consec / max(n_check, 1),
    }
    for k, v in coherence.items():
        print(f"    {k}: {v}", flush=True)
    t("Q4 done", ts)

    # ===== Q5. Class-conditional mode rates per BGMM-like binning =====
    print("\nQ5. Class-conditional mode rates...", flush=True)
    # Use 10-quantile binning per numeric column on TRAIN; then class rate.
    cls = {}
    if "PitNextLap" in train.columns:
        for c in ["TyreLife", "LapNumber", "Stint", "Position",
                  "RaceProgress", "Cumulative_Degradation",
                  "LapTime (s)", "LapTime_Delta"]:
            v = train[c].fillna(train[c].median())
            qs = np.quantile(v, np.linspace(0, 1, 11))
            qs = np.unique(qs)  # de-dup
            bins = np.digitize(v, qs) - 1
            bins = np.clip(bins, 0, len(qs) - 2)
            y = train["PitNextLap"].to_numpy()
            rates = []
            for b in range(len(qs) - 1):
                m = bins == b
                if m.sum() > 0:
                    rates.append((float(np.mean(y[m])), int(m.sum())))
            spread = max(r[0] for r in rates) - min(r[0] for r in rates)
            cls[c] = {
                "n_bins": len(rates),
                "rate_min": min(r[0] for r in rates),
                "rate_max": max(r[0] for r in rates),
                "spread_max_minus_min": float(spread),
            }
            print(f"    {c:25s} rate_spread={spread:.4f} "
                  f"({cls[c]['rate_min']:.3f} → {cls[c]['rate_max']:.3f})",
                  flush=True)
    t("Q5 done", ts)

    # ===== Q6. Joint mutual-information tables =====
    print("\nQ6. Pairwise feature MI on a subsample...", flush=True)
    from sklearn.feature_selection import mutual_info_regression
    sub_n = 100_000
    rng = np.random.default_rng(123)
    idx = rng.choice(n_full, sub_n, replace=False)
    Xfeat = full.iloc[idx][KS_LOW].fillna(0).to_numpy()
    mi_matrix = np.zeros((Xfeat.shape[1], Xfeat.shape[1]))
    for i in range(Xfeat.shape[1]):
        scores = mutual_info_regression(Xfeat, Xfeat[:, i],
                                        n_neighbors=3, random_state=42)
        mi_matrix[i, :] = scores
    mi_summary = {
        f"{KS_LOW[i]}__vs__{KS_LOW[j]}": float(mi_matrix[i, j])
        for i in range(len(KS_LOW)) for j in range(len(KS_LOW)) if i < j
    }
    # top-5 highest off-diagonal
    pairs = [(KS_LOW[i], KS_LOW[j], mi_matrix[i, j])
             for i in range(len(KS_LOW)) for j in range(i+1, len(KS_LOW))]
    pairs.sort(key=lambda x: -x[2])
    print("    top-5 MI pairs:", flush=True)
    for a, b, m in pairs[:5]:
        print(f"      {a} ↔ {b}: {m:.4f}", flush=True)
    t("Q6 done", ts)

    summary = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_full": int(n_full),
        "Q1_quantization": grid_summary,
        "Q2_id_ordering": id_order_summary,
        "Q3_ghost_drivers": ghost_summary,
        "Q4_within_stint_coherence": coherence,
        "Q5_class_conditional_rates": cls,
        "Q6_top5_mi_pairs": [{"a": a, "b": b, "mi": float(m)}
                             for a, b, m in pairs[:5]],
    }

    out = ART / "p1_synth_fingerprint.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}", flush=True)
    return summary


if __name__ == "__main__":
    main()
