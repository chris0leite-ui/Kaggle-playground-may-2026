"""scripts/probe_r9_race_external_scalars.py — Round 9 Phase B (C1)

Per-Race scalar aggregates from Aadigupta external dataset
(`data/original/f1_strategy_dataset_v4.csv`, 101 371 rows) joined to
s6e5 train/test by Race name. The aggregates are FEATURE-derived (not
target-derived) — orthogonal to yekenot's existing TE_CONFIGS (which
include te_race_yr / te_race_comp covering target-mean per Race).

The hypothesis: external feature-aggregate scalars inject row-rank
information not derivable from the 14-column s6e5 schema. If TRUE,
this is the first new-information mechanism to break the row-feature
ceiling; if FALSE, the ceiling is confirmed structural and R9
strategic posture finalises to hedge-prep.

Gates:
  G1 standalone OOF ≥ 0.948  (yekenot-level proxy)
  G2 K=14 + Path-B Δ ≥ +0.05 bp vs R7.1 PRIMARY (0.95447)
  G3 ρ_test vs PRIMARY ∈ [0.999, 0.9999]
  STRETCH: Δ ≥ +0.20 bp → first ceiling-break
  KILL: standalone < 0.945 OR K=14 Δ < +0.01 bp

Usage:
  python scripts/probe_r9_race_external_scalars.py [--smoke]
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)
EXT_CSV = Path("data/original/f1_strategy_dataset_v4.csv")
# Forbidden col per comp-context.md (host-removed from synth)
FORBIDDEN = {"Normalized_TyreLife"}

LGB_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    lambda_l1=0.0, lambda_l2=1.0, max_depth=-1, n_jobs=-1,
    verbose=-1, random_state=SEED,
)


def per_race_scalars(ext_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5 FEATURE-derived per-Race scalars (NOT target-derived).

    Orthogonal to yekenot's TE_CONFIGS which capture target-mean per
    (Race, Year) / (Race, Compound). Output: one row per Race.
    """
    agg = ext_df.groupby("Race").agg(
        lap_time_median_race=("LapTime (s)", "median"),
        lap_time_std_race=("LapTime (s)", "std"),
        cum_deg_max_race=("Cumulative_Degradation", "max"),
        pos_chg_std_race=("Position_Change", "std"),
        race_len_max_lap=("LapNumber", "max"),
    ).reset_index()
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold on 50k rows to verify wall-time + signal")
    ap.add_argument("--max_rounds", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    print(f"== R9 Phase B: C1 external per-Race scalars (Aadigupta) ==")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}  prior {y_all.mean():.4f}")

    print(f"  Loading external: {EXT_CSV}")
    ext = pd.read_csv(EXT_CSV)
    # Drop forbidden col if present
    ext = ext.drop(columns=[c for c in FORBIDDEN if c in ext.columns])
    print(f"    ext rows: {len(ext)}, Race levels (ext): {ext['Race'].nunique()}")

    s6_races = set(train["Race"].astype(str).unique()) | set(test["Race"].astype(str).unique())
    ext_races = set(ext["Race"].astype(str).unique())
    overlap = s6_races & ext_races
    print(f"    s6e5 Race levels: {len(s6_races)}, ext Race levels: {len(ext_races)}, "
          f"overlap: {len(overlap)}")
    if len(overlap) < 20:
        print(f"  WARN: only {len(overlap)} of {len(s6_races)} Races overlap — fallback heavy")

    # Compute per-Race scalars
    agg = per_race_scalars(ext)
    print(f"  Aggregates: {list(agg.columns)}")
    print(f"    {len(agg)} Race rows, sample:")
    print(agg.head(3).to_string(index=False))

    scalar_cols = [c for c in agg.columns if c != "Race"]
    # Per-column global means for fallback (unseen races)
    fallback = {c: float(agg[c].mean()) for c in scalar_cols}

    # Join to s6e5 train + test by Race
    train_j = train.merge(agg, on="Race", how="left")
    test_j  = test.merge(agg, on="Race", how="left")
    for c in scalar_cols:
        if train_j[c].isna().any() or test_j[c].isna().any():
            train_j[c] = train_j[c].fillna(fallback[c])
            test_j[c]  = test_j[c].fillna(fallback[c])
    n_na_tr = train_j[scalar_cols].isna().any(axis=1).sum()
    n_na_te = test_j[scalar_cols].isna().any(axis=1).sum()
    print(f"    Joined: train NA-rows={n_na_tr}, test NA-rows={n_na_te}")

    # Categorical encoding
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        u = pd.concat([train_j[c], test_j[c]], ignore_index=True).unique()
        m = {v: i for i, v in enumerate(u)}
        train_j[c] = train_j[c].map(m).astype(np.int32)
        test_j[c]  = test_j[c].map(m).astype(np.int32)

    feat_cols = [c for c in train_j.columns if c not in {"id", TARGET}]
    print(f"  features: {len(feat_cols)} (raw 14 + {len(scalar_cols)} ext scalars)")
    print(f"    {feat_cols}")

    if args.smoke:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(train_j), size=50_000, replace=False))
        train_j = train_j.iloc[idx].reset_index(drop=True)
        y_all = y_all[idx]
        args.max_rounds = min(args.max_rounds, 500)
        print(f"  SMOKE -> train {train_j.shape}, 1 fold")

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_all)), y_all))
    if args.smoke:
        fold_list = fold_list[:1]

    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(test_j), dtype=np.float64)
    fold_aucs = []

    for fold, (ti, vi) in enumerate(fold_list, 1):
        t_fold = time.time()
        X_tr = train_j.iloc[ti][feat_cols].fillna(0).values
        X_va = train_j.iloc[vi][feat_cols].fillna(0).values
        X_te = test_j[feat_cols].fillna(0).values
        y_tr = y_all[ti]
        y_va = y_all[vi]

        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=args.max_rounds)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

        oof_va = m.predict_proba(X_va)[:, 1]
        oof[vi] = oof_va
        if not args.smoke:
            test_pred += m.predict_proba(X_te)[:, 1] / N_FOLDS

        auc_va = roc_auc_score(y_va, oof_va)
        fold_aucs.append(float(auc_va))
        print(f"  Fold {fold}: AUC={auc_va:.5f} iters={m.best_iteration_} "
              f"wall={time.time()-t_fold:.1f}s")

    if args.smoke:
        auc_full = fold_aucs[0]
    else:
        auc_full = float(roc_auc_score(y_all, oof))
    fold_std = float(np.std(fold_aucs))
    print(f"\n  Standalone OOF AUC: {auc_full:.5f}  "
          f"fold-std={fold_std:.5f}  total wall={time.time()-t0:.1f}s")

    if auc_full >= 0.948:
        print(f"  G1 PASS (standalone ≥ 0.948)")
    elif auc_full >= 0.945:
        print(f"  G1 WARN (standalone {auc_full:.5f} below yekenot-level 0.948)")
    else:
        print(f"  G1 FAIL (standalone < 0.945)")
        if not args.smoke:
            print(f"  Aborting save to avoid polluting K=14 pool")
            return

    if args.smoke:
        print("  SMOKE complete.")
        return

    np.save(ART / "oof_C1_race_external_strat.npy", oof.astype(np.float64))
    np.save(ART / "test_C1_race_external_strat.npy", test_pred.astype(np.float64))
    print(f"  Saved: oof_C1_race_external_strat.npy "
          f"test_C1_race_external_strat.npy")

    # Per-Race AUC on weak races (Strategy-critic Section 1 surfaced
    # Spanish / Bahrain / Emilia GP as worst)
    train_raw = pd.read_csv("data/train.csv")
    for r in ["Spanish Grand Prix", "Bahrain Grand Prix",
              "Emilia Romagna Grand Prix", "Saudi Arabian Grand Prix"]:
        mask = (train_raw["Race"] == r).values
        if mask.sum() > 100 and y_all[mask].sum() > 5 and (1 - y_all[mask]).sum() > 5:
            auc_r = roc_auc_score(y_all[mask], oof[mask])
            print(f"  Diagnostic: C1 OOF AUC on {r:30s} ({mask.sum()} rows): {auc_r:.4f}")


if __name__ == "__main__":
    main()
