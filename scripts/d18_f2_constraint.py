"""d18 F2 — Constraint violation audit.

10 physical/logical constraints from the F1 domain. For each synth row,
compute violation indicators. Use as features in 5-fold LGBM.

Constraints (per (Driver, Race, Year) group, sorted by LapNumber):
  C1  TyreLife monotone within stint (resets on new Stint)
  C2  LapNumber monotone within race
  C3  CumDeg ≈ cumsum(LapTime_Delta) within stint (cumdeg_drift)
  C4  Stint counter monotone non-decreasing within race
  C5  Position ∈ [1, 22]  (some F1 grids extend to 22)
  C6  Position_Change ≈ Position[t] - Position[t-1] within race
  C7  RaceProgress monotone non-decreasing within race
  C8  TyreLife consistent with within-stint LapNumber count
  C9  PitStop=1 should coincide with Stint increment next row
  C10 LapTime within reasonable bounds (μ ± 4σ per Race)

Per-row features:
  viol_<i>  binary indicator that the row violates constraint i
  viol_count  sum of violations
  group_size  number of rows in (Driver, Race, Year) group

Then 5-fold LGBM on (raw + 11 violation features) → PitNextLap.

Also outputs per-row violation rate computed on orig (control) for
comparison.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"
LAPTIME = "LapTime (s)"
GROUP_KEYS = ["Driver", "Race", "Year"]

CAT_OK = ["Compound", "Race"]
NUM_FEATS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             LAPTIME, "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change"]


def compute_violations(df: pd.DataFrame, sort_keys=GROUP_KEYS,
                       laptime_stats=None) -> pd.DataFrame:
    """Per-row violation indicators (0/1). Returns DataFrame aligned to df.index."""
    out = pd.DataFrame(index=df.index)
    n = len(df)

    # Sort within (Driver, Race, Year) by LapNumber for sequence checks
    df_s = df.copy()
    df_s["_idx"] = np.arange(n)
    df_s = df_s.sort_values(sort_keys + ["LapNumber"], kind="mergesort")
    g = df_s.groupby(sort_keys, sort=False)

    # C1 TyreLife monotone within stint
    tl_diff = df_s.groupby(sort_keys + ["Stint"], sort=False)["TyreLife"].diff()
    c1 = (tl_diff < 0).astype(np.int8).fillna(0).values

    # C2 LapNumber monotone within race
    ln_diff = df_s.groupby(sort_keys, sort=False)["LapNumber"].diff()
    c2 = (ln_diff < 0).astype(np.int8).fillna(0).values

    # C3 CumDeg drift = |CumDeg_actual − cumsum(LapTime_Delta) within stint|
    sg = df_s.groupby(sort_keys + ["Stint"], sort=False)
    cumdelta = sg["LapTime_Delta"].cumsum()
    cumdeg_actual = df_s["Cumulative_Degradation"]
    drift = np.abs(cumdeg_actual.values - cumdelta.values)
    c3 = (drift > 5.0).astype(np.int8)

    # C4 Stint monotone within race
    stint_diff = df_s.groupby(sort_keys, sort=False)["Stint"].diff()
    c4 = (stint_diff < 0).astype(np.int8).fillna(0).values

    # C5 Position in [1, 22]
    pos = pd.to_numeric(df_s["Position"], errors="coerce").fillna(-1).values
    c5 = ((pos < 1) | (pos > 22)).astype(np.int8)

    # C6 Position_Change matches diff(Position)
    pos_diff_actual = df_s.groupby(sort_keys, sort=False)["Position"].diff()
    pos_change_col = pd.to_numeric(df_s["Position_Change"], errors="coerce")
    c6 = (np.abs(pos_diff_actual.values - pos_change_col.values) > 0.5).astype(np.int8)
    # Mask first-row-of-group (NaN diff) → mark 0
    first_row_mask = pos_diff_actual.isna().values
    c6 = np.where(first_row_mask, 0, c6)

    # C7 RaceProgress monotone within race
    rp_diff = df_s.groupby(sort_keys, sort=False)["RaceProgress"].diff()
    c7 = (rp_diff < -1e-6).astype(np.int8).fillna(0).values

    # C8 TyreLife consistent: within stint, TyreLife should equal stint_lap_idx
    # (lap counter inside stint, 0/1-indexed). Test: TyreLife = within-stint
    # rank(LapNumber). Allow ±1 slack.
    sg2 = df_s.groupby(sort_keys + ["Stint"], sort=False)
    within_idx = sg2.cumcount()
    c8 = (np.abs(df_s["TyreLife"].values - within_idx.values - 1) > 2).astype(np.int8)

    # C9 PitStop=1 should coincide with Stint increment in NEXT row
    # (current stint ends, next stint begins).
    next_stint = df_s.groupby(sort_keys, sort=False)["Stint"].shift(-1)
    pit = pd.to_numeric(df_s["PitStop"], errors="coerce").fillna(0).astype(int).values
    next_stint_diff = next_stint.values - df_s["Stint"].values
    # Pit=1 expects next_stint_diff = 1; Pit=0 expects 0; last row → ignore.
    expected = np.where(pit == 1, 1.0, 0.0)
    actual = next_stint_diff
    c9 = np.where(np.isnan(actual), 0,
                  (np.abs(actual - expected) > 0.5).astype(np.int8))

    # C10 LapTime within μ ± 4σ per Race
    if laptime_stats is None:
        # Compute on this df; pass in for synth-on-orig-stats consistency
        lt_grp = df_s.groupby("Race")[LAPTIME].agg(["mean", "std"])
    else:
        lt_grp = laptime_stats
    lt_mean = df_s["Race"].astype(str).map(lt_grp["mean"]).values
    lt_std = df_s["Race"].astype(str).map(lt_grp["std"]).values
    z = np.abs((df_s[LAPTIME].values - lt_mean) /
               np.where(lt_std > 0, lt_std, 1.0))
    c10 = (z > 4.0).astype(np.int8)

    # Group-size feature (small groups indicate fragmented synth)
    group_size = g[LAPTIME].transform("count").values

    # Restore original order
    cols = pd.DataFrame(dict(
        viol_C1_tl_mono_stint=c1, viol_C2_ln_mono_race=c2,
        viol_C3_cumdeg_drift=c3, viol_C4_stint_mono=c4,
        viol_C5_pos_bounds=c5, viol_C6_poschange=c6,
        viol_C7_rp_mono=c7, viol_C8_tl_within_stint=c8,
        viol_C9_pit_stint_next=c9, viol_C10_laptime_4sigma=c10,
        viol_count=c1+c2+c3+c4+c5+c6+c7+c8+c9+c10,
        group_size=group_size,
    ), index=df_s.index)
    cols = cols.loc[df.sort_values(sort_keys + ["LapNumber"],
                                   kind="mergesort").index]
    # Re-sort to match df order via _idx
    cols["_idx"] = df_s["_idx"].values
    cols = cols.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)
    cols.index = df.index
    return cols, lt_grp


def main():
    t0 = time.time()
    print("[F2 constraint violations]  loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    print(f"  train {tr.shape}  test {te.shape}  orig {orig.shape}")

    # Compute orig violation rates as control (reference)
    print("\n[orig violation rates]")
    orig_v, _ = compute_violations(orig)
    orig_rates = {c: float(orig_v[c].mean()) for c in orig_v.columns
                  if c.startswith("viol_")}
    for c, r in sorted(orig_rates.items()):
        print(f"  {c:32s}  rate={r:.4%}")

    # Use orig laptime stats so synth violations are referenced to orig
    _, lt_stats = compute_violations(orig)

    print("\n[train violation rates]")
    tr_v, _ = compute_violations(tr, laptime_stats=lt_stats)
    tr_rates = {c: float(tr_v[c].mean()) for c in tr_v.columns
                if c.startswith("viol_")}
    for c, r in sorted(tr_rates.items()):
        delta = r - orig_rates[c]
        print(f"  {c:32s}  rate={r:.4%}  Δ vs orig {delta:+.4%}")
    te_v, _ = compute_violations(te, laptime_stats=lt_stats)

    # Class-conditional violation rates in train
    y = tr[TARGET].astype(int).values
    print("\n[train violation rates by y]")
    for c in [c for c in tr_v.columns if c.startswith("viol_")]:
        v = tr_v[c].values
        r0 = float(v[y == 0].mean()); r1 = float(v[y == 1].mean())
        print(f"  {c:32s}  y=0 {r0:.4%}  y=1 {r1:.4%}  Δ {r1-r0:+.4%}")

    # ---- Downstream LGBM ----
    print("\n[downstream LGBM raw + 12 violation features]")
    # Encode cats
    cmps = sorted(set(tr["Compound"].astype(str)) | set(te["Compound"].astype(str)))
    cm = {c: i for i, c in enumerate(cmps)}
    races = sorted(set(tr["Race"].astype(str)) | set(te["Race"].astype(str)))
    rm = {r: i for i, r in enumerate(races)}

    raw_cols = ["Compound", "Race"] + NUM_FEATS
    tr_X = tr[raw_cols].copy()
    te_X = te[raw_cols].copy()
    tr_X["Compound"] = tr["Compound"].astype(str).map(cm).astype(int)
    te_X["Compound"] = te["Compound"].astype(str).map(cm).astype(int)
    tr_X["Race"] = tr["Race"].astype(str).map(rm).astype(int)
    te_X["Race"] = te["Race"].astype(str).map(rm).astype(int)

    tr_X = pd.concat([tr_X.reset_index(drop=True),
                      tr_v.reset_index(drop=True)], axis=1)
    te_X = pd.concat([te_X.reset_index(drop=True),
                      te_v.reset_index(drop=True)], axis=1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test_avg = np.zeros(len(te), dtype=np.float64)
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=127, min_data_in_leaf=80, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbosity=-1, seed=SEED)
    cat_idx = [tr_X.columns.get_loc(c) for c in ["Compound", "Race"]]
    for fi, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        ds_tr = lgb.Dataset(tr_X.iloc[tr_i], label=y[tr_i],
                            categorical_feature=cat_idx, free_raw_data=False)
        ds_va = lgb.Dataset(tr_X.iloc[va_i], label=y[va_i],
                            categorical_feature=cat_idx, reference=ds_tr,
                            free_raw_data=False)
        m = lgb.train(p, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_i] = m.predict(tr_X.iloc[va_i], num_iteration=m.best_iteration)
        test_avg += m.predict(te_X, num_iteration=m.best_iteration) / N_FOLDS
        print(f"  fold {fi}: AUC={roc_auc_score(y[va_i], oof[va_i]):.5f}")

    auc = float(roc_auc_score(y, oof))
    print(f"  OOF AUC = {auc:.5f}")

    np.save(ART / "oof_d18_f2_constraint_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d18_f2_constraint_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))

    summary = dict(
        oof_auc=auc, orig_violation_rates=orig_rates,
        train_violation_rates=tr_rates,
        train_violation_rates_by_y={
            c: dict(y0=float(tr_v[c].values[y==0].mean()),
                    y1=float(tr_v[c].values[y==1].mean()))
            for c in tr_v.columns if c.startswith("viol_")
        },
        wall_s=time.time() - t0,
    )
    (ART / "d18_f2_constraint_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done F2]  wall {time.time()-t0:.0f}s  OOF {auc:.5f}")


if __name__ == "__main__":
    main()
