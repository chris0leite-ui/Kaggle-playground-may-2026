"""Day-16 H4 — Year=2023 ∩ rare-Driver hard-mask post-process.

d13 G4 (queued d13 evening, never run). Probe Q3 finding:
  Year=2023 has 887 unique drivers (vs ~547 in 2022/2024/2025) at pos
  rate 0.96% (vs ~28%). 24 of 26 (Year=2023, Race) cohorts <5% pit rate.
  Year=2023 is a synthetic flat-rate generator with a long-tail driver
  cohort. Rare-driver rows in 2023 are near-certain negatives.

This is post-processing on PRIMARY OOF/test predictions:
  mask_rows(r) = (Year=2023) & (Driver-count-across-all-years < K)
  pred_new[r] = 0.0   (for r in mask)

Sweep K in {5, 10, 20, 50, 100, 200}; pick the one that maximizes
OOF AUC on full train (only OOF rows are in-distribution). Apply to
test predictions for the matched mask criterion.

Output (all under scripts/artifacts/):
  oof_d16_h4_year_mask_strat.npy      (n_train, 2)
  test_d16_h4_year_mask_strat.npy     (n_test, 2)
  d16_h4_year_mask_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")

PRIMARY_OOF = ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"


def main():
    t0 = time.time()
    print("[h4] loading data + PRIMARY artifacts ...", flush=True)
    train = pd.read_csv("data/train.csv", usecols=["PitNextLap", "Driver", "Year"])
    test = pd.read_csv("data/test.csv", usecols=["Driver", "Year"])
    y = train["PitNextLap"].astype(int).values
    n_train, n_test = len(train), len(test)
    print(f"[h4] train {n_train}  test {n_test}", flush=True)

    primary_oof = np.load(PRIMARY_OOF)[:, 1].astype(np.float64)
    primary_test = np.load(PRIMARY_TEST)[:, 1].astype(np.float64)
    auc_base = float(roc_auc_score(y, primary_oof))
    print(f"[h4] PRIMARY OOF AUC = {auc_base:.6f}", flush=True)

    # Driver count across all years (use train+test combined for stability;
    # we want to identify "rare" drivers regardless of where they show up).
    full_drivers = pd.concat([train["Driver"], test["Driver"]],
                             axis=0, ignore_index=True)
    driver_counts = full_drivers.value_counts()

    train_mask_2023 = (train["Year"].values == 2023)
    test_mask_2023 = (test["Year"].values == 2023)

    train_pos_rate_2023 = float(y[train_mask_2023].mean()) if train_mask_2023.sum() else float("nan")
    print(f"[h4] Year=2023 train rows: {int(train_mask_2023.sum())} "
          f"({100*train_mask_2023.mean():.1f}%) pos rate {train_pos_rate_2023:.4f}",
          flush=True)
    print(f"[h4] Year=2023 test rows : {int(test_mask_2023.sum())} "
          f"({100*test_mask_2023.mean():.1f}%)", flush=True)

    # Sweep K.
    Ks = [3, 5, 10, 20, 50, 100, 200, 500, 99999]   # 99999 = "all 2023 rows"
    sweep = []
    best = None
    for K in Ks:
        rare_drivers = set(driver_counts[driver_counts < K].index)
        mask_train = train_mask_2023 & train["Driver"].isin(rare_drivers).values
        mask_test = test_mask_2023 & test["Driver"].isin(rare_drivers).values
        # Pos rate in train among masked rows
        if mask_train.sum() > 0:
            pos_train_masked = float(y[mask_train].mean())
        else:
            pos_train_masked = float("nan")
        # Apply mask: set predictions to 0.0
        oof_new = primary_oof.copy()
        oof_new[mask_train] = 0.0
        auc_new = float(roc_auc_score(y, oof_new))
        delta_bp = (auc_new - auc_base) * 1e4
        rec = dict(K=K,
                   n_train_masked=int(mask_train.sum()),
                   n_test_masked=int(mask_test.sum()),
                   pos_rate_train_masked=pos_train_masked,
                   oof_auc=auc_new,
                   delta_bp=delta_bp)
        sweep.append(rec)
        print(f"  K={K:>5}  n_train_mask={mask_train.sum():>7}  "
              f"pos_rate={pos_train_masked:.4f}  "
              f"OOF AUC={auc_new:.6f}  Δ {delta_bp:+.3f}bp",
              flush=True)
        if best is None or auc_new > best["oof_auc"]:
            best = rec

    # Also try: instead of zeroing, REPLACE with the rare-driver pos rate
    # (more conservative; keeps relative ordering above the floor).
    print("\n[h4] -- replace-with-rate variant (instead of zero) --", flush=True)
    sweep_replace = []
    for K in Ks:
        rare_drivers = set(driver_counts[driver_counts < K].index)
        mask_train = train_mask_2023 & train["Driver"].isin(rare_drivers).values
        mask_test = test_mask_2023 & test["Driver"].isin(rare_drivers).values
        if mask_train.sum() > 0:
            replace_val = float(y[mask_train].mean())
        else:
            replace_val = 0.0
        oof_new = primary_oof.copy()
        oof_new[mask_train] = replace_val
        auc_new = float(roc_auc_score(y, oof_new))
        delta_bp = (auc_new - auc_base) * 1e4
        rec = dict(K=K, replace_val=replace_val,
                   n_train_masked=int(mask_train.sum()),
                   oof_auc=auc_new,
                   delta_bp=delta_bp)
        sweep_replace.append(rec)
        print(f"  K={K:>5}  n={mask_train.sum():>7}  "
              f"replace_val={replace_val:.4f}  "
              f"OOF AUC={auc_new:.6f}  Δ {delta_bp:+.3f}bp",
              flush=True)

    # Pick best zero-mask
    best_K = best["K"]
    rare_drivers = set(driver_counts[driver_counts < best_K].index)
    mask_train_best = train_mask_2023 & train["Driver"].isin(rare_drivers).values
    mask_test_best = test_mask_2023 & test["Driver"].isin(rare_drivers).values
    oof_best = primary_oof.copy()
    oof_best[mask_train_best] = 0.0
    test_best = primary_test.copy()
    test_best[mask_test_best] = 0.0

    # Save best zero-mask artifacts
    oof2 = np.column_stack([1.0 - oof_best, oof_best])
    test2 = np.column_stack([1.0 - test_best, test_best])
    np.save(ART / "oof_d16_h4_year_mask_strat.npy", oof2)
    np.save(ART / "test_d16_h4_year_mask_strat.npy", test2)

    res = dict(
        primary_oof=auc_base,
        train_pos_rate_2023=train_pos_rate_2023,
        n_train_2023=int(train_mask_2023.sum()),
        n_test_2023=int(test_mask_2023.sum()),
        sweep_zero=sweep,
        sweep_replace=sweep_replace,
        best=dict(best),
        wall_s=time.time() - t0,
    )
    (ART / "d16_h4_year_mask_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n[h4] BEST K={best['K']}  Δ OOF {best['delta_bp']:+.3f}bp  "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
