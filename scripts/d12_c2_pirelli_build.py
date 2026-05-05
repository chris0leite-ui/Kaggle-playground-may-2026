"""C2 — Build 6 Pirelli pit-window rule_residual bases [DEPRECATED 2026-05-12].

DEPRECATED before execution alongside `d12_c2_pirelli_scrape.py`. See
`audit/2026-05-12-d12-c2-pirelli-prep.md` for the synthetic-DGP-
incompatibility rationale. Skeleton preserved for reference.

Original docstring follows.
---

Consumes `data/external/pirelli_windows.csv` (built by
`d12_c2_pirelli_scrape.py`) and produces 6 rule_residual L1 bases
following the F1.2 multi-rule template (`scripts/d6_multi_rule.py`).

Each base is a Bayesian-smoothed (alpha=50) lookup over a key
involving Pirelli window features, with an HGBC residual on raw
features. Apply Q6 filter (ρ vs new PRIMARY ≤ 0.997 standalone) per
base; demote any base failing the gate.

Bases (per Day-12 prep doc §4):
  1. in_window         — (Race, Year, Compound, lap_in_window_flag)
  2. dist_to_center    — (Race, Year, Compound, signed_dist_decile)
  3. dist_to_edge      — (Race, Year, Compound, abs_dist_outside_decile)
  4. stops_to_go       — (Race, Year, Driver, n_stops_remaining)
  5. window_progress   — (Race, Year, Compound, lap/center_decile)
  6. multi_strategy    — (Race, Year, n_window_options)

After per-base build:
  - Standalone OOF + ρ vs PRIMARY (d9f K=21, OOF 0.95073) for each
  - Q6 gate: keep bases with ρ ≤ 0.997
  - Build K=N+m stack (m = survivor count) with LR-meta on
    [raw, rank, logit] expansion (d6 template)
  - Min-meta sanity check (PRIMARY + this base) per base before pool add
  - K=N stack must beat PRIMARY OOF by ≥ +0.5bp AND ρ vs PRIMARY ≤ 0.999

This is a SKELETON; the data-loading + window-feature-derivation
helpers are stubs. Real execution requires the scrape output to
exist at `data/external/pirelli_windows.csv`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
PIRELLI_CSV = Path("data/external/pirelli_windows.csv")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

# New PRIMARY anchors (post-d9f K=21 swap, Day-10)
PRIMARY_S = 0.95073   # d9f K=21 swap OOF
PRIMARY_LB = 0.95031
RHO_TIE = 0.999       # K=N stack gate
RHO_Q6 = 0.997        # standalone-vs-PRIMARY filter (Day-9 Q6)
ALPHA = 50.0


# ---------------------------------------------------------------------------
# Pirelli-window feature derivation
# ---------------------------------------------------------------------------

def load_pirelli_windows() -> pd.DataFrame:
    """Load the scrape output. Schema per `d12_c2_pirelli_scrape.py`."""
    if not PIRELLI_CSV.exists():
        raise FileNotFoundError(
            f"{PIRELLI_CSV} missing. Run d12_c2_pirelli_scrape.py first."
        )
    return pd.read_csv(PIRELLI_CSV)


def derive_window_features(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Add per-row Pirelli-window features.

    For each train/test row (LapNumber, Race, Year, Compound), look up
    the matching set of windows and compute:
      - lap_in_window_flag  ∈ {0, 1}
      - signed_dist_to_center  (lap − nearest_window_center; sign matters)
      - abs_dist_outside  (0 if inside any window, else min distance OUT)
      - lap_over_center_ratio  (lap / nearest_center; quintile-bucketed)
      - n_window_options  (count of recommended strategies for race-year)

    Missing data: rows where (Race, Year) lacks any window get
    `lap_in_window_flag=0` and median-imputed numerics with a
    `pirelli_data_missing=1` flag column for downstream gating.
    """
    raise NotImplementedError("derive_window_features: stub")


# ---------------------------------------------------------------------------
# Base specs (key tuples + per-row arrays for the Bayesian lookup)
# ---------------------------------------------------------------------------

def base_specs() -> list[tuple]:
    """Return list of (name, key_columns, transforms) per base."""
    return [
        ("in_window",          ("Race", "Year", "Compound", "lap_in_window_flag"),  None),
        ("dist_to_center",     ("Race", "Year", "Compound", "signed_dist_decile"),   None),
        ("dist_to_edge",       ("Race", "Year", "Compound", "abs_dist_outside_decile"), None),
        ("stops_to_go",        ("Race", "Year", "Driver",   "n_stops_remaining"),    None),
        ("window_progress",    ("Race", "Year", "Compound", "lap_over_center_decile"), None),
        ("multi_strategy",     ("Race", "Year", "n_window_options"),                 None),
    ]


# ---------------------------------------------------------------------------
# Reused F1.2 builder primitives (copy-paste from d6_multi_rule.py)
# ---------------------------------------------------------------------------

def fit_lookup(keys_train, y_train, alpha=ALPHA):
    df = pd.DataFrame({"k": list(keys_train), "y": y_train})
    g = df.groupby("k", observed=True)["y"]
    counts = g.count(); means = g.mean()
    glob = float(np.mean(y_train))
    smoothed = (means * counts + glob * alpha) / (counts + alpha)
    return smoothed.to_dict(), glob


def apply_lookup(keys, lookup, glob):
    out = np.full(len(keys), glob, dtype=np.float64)
    for i, k in enumerate(keys):
        v = lookup.get(k)
        if v is not None:
            out[i] = v
    return out


def make_hgbc_regressor():
    return HistGradientBoostingRegressor(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=SEED,
    )


# ---------------------------------------------------------------------------
# Q6 filter — ρ vs new PRIMARY
# ---------------------------------------------------------------------------

def q6_pass(oof_base: np.ndarray, oof_primary: np.ndarray) -> tuple[bool, float]:
    """Q6 filter: standalone ρ vs PRIMARY ≤ RHO_Q6 (= 0.997)."""
    rho = float(spearmanr(oof_base, oof_primary).correlation)
    return rho <= RHO_Q6, rho


# ---------------------------------------------------------------------------
# Main pipeline (skeleton)
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=== C2 Pirelli pit-window base build (SKELETON) ===")
    print("Loading Pirelli scrape output...")
    windows = load_pirelli_windows()  # raises if scrape not run
    print(f"  {len(windows)} window records covering "
          f"{windows.race.nunique()} races × {windows.year.nunique()} years")

    print("Loading train/test + deriving window features...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train_w = derive_window_features(train, windows)  # NotImplementedError
    test_w = derive_window_features(test, windows)

    print("Loading PRIMARY OOF (d9f K=21) for Q6 filter...")
    oof_primary = np.load(ART / "oof_d9f_K21_swap_partA_partB_strat.npy")  # main-branch artifact

    y = train[TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Per-base build + Q6 filter
    survivors = []
    for name, key_cols, transforms in base_specs():
        print(f"\n--- Building '{name}' (key={key_cols}) ---")
        # build_rule_residual_base(...) per d6 template
        # oof_base, test_base, std_oof = build_rule_residual_base(...)
        # passed, rho = q6_pass(oof_base, oof_primary)
        # if passed:
        #     survivors.append((name, oof_base, test_base, std_oof, rho))
        #     np.save(ART / f"oof_d12_c2_{name}_strat.npy", oof_base)
        #     np.save(ART / f"test_d12_c2_{name}_strat.npy", test_base)
        # else:
        #     print(f"  Q6 FAIL: ρ={rho:.4f} > {RHO_Q6}")
        raise NotImplementedError(f"build_rule_residual_base for '{name}' not wired up; needs derive_window_features() implemented first")

    if not survivors:
        print("\nNo bases survived Q6. C2 confirmed dead-list category.")
        return

    print(f"\n=== Q6 survivors: {len(survivors)}/6 ===")
    # K=N stack: load existing K=21 pool + add survivors
    # K=21 pool members per origin/main d9f: see scripts/d9f_multi_fm.py POOL
    # F_oof = stack([POOL_OOFS, *[s[1] for s in survivors]], axis=1)
    # F_test = stack([POOL_TESTS, *[s[2] for s in survivors]], axis=1)
    # meta_oof, meta_test = fit_lr_meta(expand(F_oof), expand(F_test), y)
    # final_auc = roc_auc_score(y, meta_oof)
    # rho_final = float(spearmanr(meta_oof, oof_primary).correlation)
    # print(f"K={21+len(survivors)} stack OOF: {final_auc:.5f}  ρ vs PRIMARY: {rho_final:.5f}")

    print(f"\nTotal wall: {time.time()-t0:.1f}s")
    print("DONE (skeleton). Wire derive_window_features() + uncomment build/stack to execute.")


if __name__ == "__main__":
    print("WARNING: this is a SKELETON.")
    print("Pre-reqs: data/external/pirelli_windows.csv exists (run d12_c2_pirelli_scrape.py first)")
    print("DO NOT EXECUTE without PI sign-off after scrape completes.")
    main()
