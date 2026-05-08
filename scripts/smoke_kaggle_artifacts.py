"""Smoke test: prove the Kaggle-Dataset migration didn't break anything.

Run after a fresh clone / clean cache. Confirms:
  1. ARTIFACT_DIR resolver picks the right location.
  2. K=21 base OOFs load via numpy.load with the expected shape/dtype.
  3. probe_min_meta.py-style K=21 LR-meta OOF AUC is reproducible
     (target: ~0.951+ on this comp; precise value is the calibration anchor).
  4. A representative single-base OOF (CB v4 = K=22 add) loads.

Usage:
    # local: artifacts must be in scripts/artifacts/ (download via:
    #   kaggle datasets download chrisleitescha/s6e5-artifacts -p scripts/artifacts/ --unzip)
    python scripts/smoke_kaggle_artifacts.py

    # Kaggle notebook: attach Add Data -> chrisleitescha/s6e5-artifacts; auto-detected.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import ART, SEED, N_FOLDS  # noqa: E402

# K=21 PRIMARY pool (matches probe_min_meta.K21_BASES exactly)
K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
]

# Day-19 expected K=21 LR-meta OOF AUC (calibration anchor; matches d18 trail).
EXPECTED_K21_AUC_LOWER = 0.94800
EXPECTED_K21_AUC_UPPER = 0.95300


def load_y() -> np.ndarray:
    p = Path("data/train.csv")
    if not p.exists():
        print(f"[FAIL] data/train.csv not found at {p.absolute()}; run bootstrap.sh")
        sys.exit(1)
    return pd.read_csv(p, usecols=["PitNextLap"])["PitNextLap"].astype(int).values


def load_pos(name: str) -> np.ndarray:
    """Load oof_<NAME>_strat.npy and return positive-class column."""
    p = ART / f"oof_{name}_strat.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (ART={ART})")
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def step(label: str) -> None:
    print(f"\n--- {label} ---")


def main() -> int:
    failures: list[str] = []
    t0 = time.time()

    step("step 1: ARTIFACT_DIR resolver")
    print(f"  ART = {ART.absolute()}")
    if not ART.exists():
        failures.append("ART dir does not exist")
    n_files = len(list(ART.glob("*.npy"))) if ART.exists() else 0
    print(f"  {n_files} .npy files visible")
    if n_files < 40:
        failures.append(f"only {n_files} .npy files; expect at least the 21 K=21 OOFs + 21 tests")

    step("step 2: load y from comp data")
    y = load_y()
    print(f"  y.shape={y.shape}, mean={y.mean():.4f} (target ~0.199)")
    if abs(y.mean() - 0.199) > 0.02:
        failures.append(f"y class prior {y.mean():.4f} drifted from 0.199 anchor")

    step("step 3: load all 21 K=21 base OOFs")
    oofs = []
    missing = []
    for name in K21_BASES:
        try:
            p = load_pos(name)
            oofs.append(p)
            assert p.shape == y.shape, f"{name} shape {p.shape} != y {y.shape}"
        except FileNotFoundError as e:
            missing.append(name)
            print(f"  MISSING: {name}")
    if missing:
        failures.append(f"{len(missing)} K=21 bases missing: {missing}")
    else:
        print(f"  loaded {len(oofs)} bases, all shape {oofs[0].shape}")

    if len(oofs) == 21:
        step("step 4: K=21 LR-meta OOF AUC reproducibility")
        F = np.column_stack(oofs)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_meta = np.zeros(len(y))
        for tr, va in skf.split(np.zeros(len(y)), y):
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            lr.fit(F[tr], y[tr])
            oof_meta[va] = lr.predict_proba(F[va])[:, 1]
        auc = roc_auc_score(y, oof_meta)
        print(f"  K=21 LR-meta OOF AUC = {auc:.5f}")
        print(f"  expected band:        [{EXPECTED_K21_AUC_LOWER:.5f}, {EXPECTED_K21_AUC_UPPER:.5f}]")
        if not (EXPECTED_K21_AUC_LOWER <= auc <= EXPECTED_K21_AUC_UPPER):
            failures.append(f"K=21 OOF AUC {auc:.5f} outside expected band")

    step("step 5: representative high-value base load (d18_chain_decomp)")
    try:
        d18 = load_pos("d18_chain_decomp")
        print(f"  loaded d18_chain_decomp: shape={d18.shape}, range=[{d18.min():.4f}, {d18.max():.4f}]")
        single_auc = roc_auc_score(y, d18)
        print(f"  single-feat AUC = {single_auc:.4f}")
    except FileNotFoundError as e:
        failures.append(f"d18_chain_decomp missing: {e}")

    step("step 6: PRIMARY-pool sanity (d18 K=27 PRIMARY OOF)")
    try:
        primary = load_pos("d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000")
        primary_auc = roc_auc_score(y, primary)
        print(f"  d18 K=27 PRIMARY OOF AUC = {primary_auc:.5f}")
        print(f"  (calibration ladder anchor: 0.95432; LB landed 0.95368)")
        if primary_auc < 0.954 or primary_auc > 0.956:
            failures.append(f"PRIMARY OOF AUC {primary_auc:.5f} drifted from 0.95432 anchor")
    except FileNotFoundError:
        print("  (PRIMARY OOF not found by exact filename; not fatal — naming may differ)")

    elapsed = time.time() - t0
    print(f"\n=== smoke test {'PASSED' if not failures else 'FAILED'} in {elapsed:.1f}s ===")
    if failures:
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
