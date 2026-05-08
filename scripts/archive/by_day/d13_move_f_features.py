"""Day-13 Move F — new FM-input features (leak-free build + sanity).

HANDOVER thesis (Day-13): same-12-field FM partition variants are
dead (V1 K=22 5/3 SUBMITTED LB 0.95032 TIE; V2/V3 falsified by
shared thesis). The leakage-robust vein within d9h_aug12's 12-field
set is fully mined by d9c FM + d9f FM_A/B + d9h aug12. Future
FM-class lift requires NEW INPUT FIELDS.

This script builds 4 candidate fields none of which are present in
the existing FM input set (D, C, R, Y, S, T_q5, Rp_q5, P_q5, Nx,
Pv, Cd_q5, Ld_q5):

    F1 PitWindow      laps_since_last_pit (d3b precedent: +18bp
                      std-alone, FAIL gate by 35bp -- but never
                      tested as FM input, where pairwise low-rank
                      crosses with Driver/Compound/Stint may be the
                      live lever).
    F2 HazardDecay    exp(-alpha * TyreLife / mu_compound_stint).
                      Smooth nonlinear transform of TyreLife
                      conditioned on Compound; bins capture
                      "survival pressure" zones distinct from T_q5.
    F3 CompoundPress  TyreLife - mu_compound_stint (signed dev
                      from typical stint length per compound).
                      mu computed on TRAIN ONLY using Stint-end
                      lap counts to avoid target leakage.
    F4 RaceStage      Non-uniform RaceProgress bins (opening /
                      early / mid / late / closing / final) -
                      distinct from Rp_q5 (uniform quintiles).

Output: ``scripts/artifacts/d13_move_f_features.parquet`` with
columns ``[F1_PitWindow_q5, F2_HazardDecay_q5, F3_CompoundPress_q5,
F4_RaceStage]`` for the concatenated train+test rows in original
order, plus ``__src in {tr,te}`` for downstream split.

Leakage contract:
- F1 uses observed PitStop only (lag, not lead).
- F2/F3 mu_compound_stint computed on TRAIN ONLY, applied to both.
- F4 is a deterministic transform of RaceProgress (no fit).

All quantile binners fit on train; applied to both. RaceStage uses
fixed cuts (no fitting).

To run:
    python scripts/d13_move_f_features.py

Next steps (Move F-2, separate script): add F1-F4 to d9h_aug12's
12-field FM (-> 16-field FM); also smoke a F-only 4-field FM to
isolate the new-feature signal vs the existing pool.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
DATA = Path("data")
TARGET = "PitNextLap"

# Feature parameters.
HAZARD_ALPHA = 1.0      # exp(-alpha * TyreLife / mu_C)
N_BINS_QUANT = 5        # F1, F2, F3 quintile-binned to match T_q5 etc.

# RaceStage fixed cuts on RaceProgress in [0,1].
# Chosen by inspection of typical F1 race phasing:
# - opening:   0.00 - 0.10  (lap 1 chaos, undercut window)
# - early:     0.10 - 0.30  (first stint settle)
# - mid_a:     0.30 - 0.55  (first pit window)
# - mid_b:     0.55 - 0.75  (second pit window)
# - late:      0.75 - 0.92  (overcut / last-pit decisions)
# - final:     0.92 - 1.00  (closing laps, near-zero hazard)
RACE_STAGE_CUTS = np.array([0.0, 0.10, 0.30, 0.55, 0.75, 0.92, 1.0001])
RACE_STAGE_LABELS = ["opening", "early", "mid_a", "mid_b", "late", "final"]


# ---------------------------------------------------------------------------
# F1 -- laps_since_last_pit
# ---------------------------------------------------------------------------

def build_pit_window(full: pd.DataFrame) -> np.ndarray:
    """laps_since_last_pit per (Year, Race, Driver), leak-free.

    Pattern from scripts/d3b_seqfe.py (verified leak-free): for each
    row, compute LapNumber - (LapNumber of most recent prior PitStop
    in this group); if no prior pit, fall back to LapNumber (laps
    from race start).
    """
    sort_idx = full.sort_values(["Year", "Race", "Driver", "LapNumber"]).index
    s = full.loc[sort_idx].copy()
    grp = s.groupby(["Year", "Race", "Driver"], sort=False)
    s["_marker"] = s["LapNumber"].where(s["PitStop"] == 1)
    s["_last_pit_lap"] = grp["_marker"].ffill()
    s["pit_window"] = (s["LapNumber"] - s["_last_pit_lap"]).fillna(s["LapNumber"])
    s = s.sort_index()
    return s["pit_window"].astype(float).values


# ---------------------------------------------------------------------------
# F2 -- HazardDecay  &  F3 -- CompoundPressure
# ---------------------------------------------------------------------------

def fit_compound_mu(train: pd.DataFrame) -> dict[str, float]:
    """Per-compound mean stint length (in laps), TRAIN ONLY.

    Approximated by max TyreLife within each (Year, Race, Driver,
    Stint) cell averaged per Compound. Stint identity is itself a
    direct feature so this leverages TyreLife for finer structure.
    Target is NEVER inspected here.
    """
    g = (train.groupby(["Year", "Race", "Driver", "Stint", "Compound"],
                       sort=False)["TyreLife"].max()
         .reset_index().rename(columns={"TyreLife": "stint_len"}))
    mu = g.groupby("Compound")["stint_len"].mean().to_dict()
    return {str(k): float(v) for k, v in mu.items()}


def build_hazard_and_pressure(full: pd.DataFrame, mu: dict[str, float]
                              ) -> tuple[np.ndarray, np.ndarray]:
    comp = full["Compound"].astype(str).values
    tyre = full["TyreLife"].astype(float).values
    fallback_mu = float(np.mean(list(mu.values())))
    mu_arr = np.array([mu.get(c, fallback_mu) for c in comp], dtype=float)
    hazard = np.exp(-HAZARD_ALPHA * tyre / np.maximum(mu_arr, 1e-6))
    pressure = tyre - mu_arr
    return hazard, pressure


# ---------------------------------------------------------------------------
# F4 -- RaceStage (non-uniform RaceProgress bins)
# ---------------------------------------------------------------------------

def build_race_stage(full: pd.DataFrame) -> np.ndarray:
    rp = np.clip(full["RaceProgress"].astype(float).values, 0.0, 1.0)
    idx = np.searchsorted(RACE_STAGE_CUTS, rp, side="right") - 1
    idx = np.clip(idx, 0, len(RACE_STAGE_LABELS) - 1)
    return np.array([RACE_STAGE_LABELS[i] for i in idx])


# ---------------------------------------------------------------------------
# Quantile binner (train-fit)
# ---------------------------------------------------------------------------

def quantile_bin(arr_train: np.ndarray, arr_query: np.ndarray, n_bins: int
                 ) -> np.ndarray:
    edges = np.quantile(arr_train, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                   0, n_bins - 1).astype(int)


# ---------------------------------------------------------------------------
# Sanity printout: per-bin positive rate (uses target -- TRAIN ONLY,
# for diagnostic, NOT a feature).
# ---------------------------------------------------------------------------

def per_bin_rate(bins: np.ndarray, y: np.ndarray, label: str) -> None:
    print(f"  {label} positive rate by bin (train):")
    for b in sorted(set(bins.tolist())):
        m = bins == b
        n = int(m.sum())
        rate = float(y[m].mean()) if n else float("nan")
        print(f"    bin={b!s:<10s} n={n:>7d}  rate={rate:.4f}")


# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    train_path, test_path = DATA / "train.csv", DATA / "test.csv"
    if not train_path.exists():
        raise SystemExit(f"missing {train_path}; run kaggle-comp setup first")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y = train[TARGET].astype(int).values
    print(f"train rows {len(train):,}  test rows {len(test):,}")

    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    n_tr = len(train)

    # F1 -- pit_window (raw, before binning)
    print("\nF1 PitWindow (laps_since_last_pit) ...")
    pit_window = build_pit_window(full)
    f1_train = pit_window[:n_tr]; f1_test = pit_window[n_tr:]
    print(f"  raw describe (train): min={f1_train.min():.1f} "
          f"median={np.median(f1_train):.1f} mean={f1_train.mean():.2f} "
          f"max={f1_train.max():.1f}")
    f1_q5_full = quantile_bin(f1_train, pit_window, N_BINS_QUANT)

    # F2 / F3 -- need per-compound mu
    print("\nF2 HazardDecay / F3 CompoundPressure ...")
    mu = fit_compound_mu(train)
    print(f"  mu_compound_stint (train-fit): {mu}")
    hazard, pressure = build_hazard_and_pressure(full, mu)
    f2_q5_full = quantile_bin(hazard[:n_tr], hazard, N_BINS_QUANT)
    f3_q5_full = quantile_bin(pressure[:n_tr], pressure, N_BINS_QUANT)

    # F4 -- race_stage
    print("\nF4 RaceStage (non-uniform RaceProgress) ...")
    f4_full = build_race_stage(full)
    print(f"  stage value counts (full):")
    vc = pd.Series(f4_full).value_counts().reindex(RACE_STAGE_LABELS,
                                                    fill_value=0)
    for k, v in vc.items():
        print(f"    {k:<10s} {int(v):>8d}")

    # Diagnostic: per-bin positive rate on TRAIN slice only
    print("\n--- diagnostic: per-bin train positive-rate ---")
    per_bin_rate(f1_q5_full[:n_tr], y, "F1_PitWindow_q5")
    per_bin_rate(f2_q5_full[:n_tr], y, "F2_HazardDecay_q5")
    per_bin_rate(f3_q5_full[:n_tr], y, "F3_CompoundPress_q5")
    per_bin_rate(f4_full[:n_tr], y, "F4_RaceStage")

    # Save -- parquet for compactness; also raw npy for scriptability
    out = pd.DataFrame({
        "F1_PitWindow_q5":     f1_q5_full,
        "F2_HazardDecay_q5":   f2_q5_full,
        "F3_CompoundPress_q5": f3_q5_full,
        "F4_RaceStage":        f4_full,
        "__src":               full["__src"].values,
    })
    ART.mkdir(parents=True, exist_ok=True)
    pq_path = ART / "d13_move_f_features.parquet"
    try:
        out.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"  parquet write failed ({e}); falling back to csv.gz")
        pq_path = ART / "d13_move_f_features.csv.gz"
        out.to_csv(pq_path, index=False, compression="gzip")
    print(f"\n-> wrote {pq_path} ({len(out):,} rows)")

    meta = dict(
        features=["F1_PitWindow_q5", "F2_HazardDecay_q5",
                  "F3_CompoundPress_q5", "F4_RaceStage"],
        params=dict(hazard_alpha=HAZARD_ALPHA, n_bins=N_BINS_QUANT,
                    race_stage_cuts=RACE_STAGE_CUTS.tolist(),
                    race_stage_labels=RACE_STAGE_LABELS),
        mu_compound_stint=mu,
        n_train=n_tr, n_test=int(len(test)),
        wall_s=time.time() - t0,
    )
    (ART / "d13_move_f_features_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"-> wrote {ART / 'd13_move_f_features_meta.json'}  "
          f"(wall {time.time()-t0:.1f}s)")
    print("\nNext: scripts/d13_move_f_fm_aug16.py -- add F1-F4 to d9h_aug12 "
          "12-field FM; std/ρ-vs-PRIMARY/min-meta gate.")


if __name__ == "__main__":
    main()
