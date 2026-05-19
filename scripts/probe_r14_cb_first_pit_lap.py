"""scripts/probe_r14_cb_first_pit_lap.py — Phase 5 (promoted): CB on first-pit lap.

Replaces dropped Phase 3 (cb_next_compound). Target: absolute lap of
the driver's FIRST pit in this race, per (Driver, Race) summary.

Different from cb_horizon (laps until NEXT pit, per-row) and
cb_stint_completion (stint-fraction-elapsed, per-row). This is a
PER-(Driver, Race) GROUP-UNIFORM target — every row in the same
(Driver, Race) gets the same target value. The model has to predict
this group-summary value from row features (TyreLife, Compound,
LapNumber, Year, etc.).

Strict per-fold target derivation: find first pit lap using ONLY
rows in the fold's train set. If a (Driver, Race) group has NO ti
row with PitStop=1, target = right-censor (use race-end lap).

`log(first_pit_lap + 1)` for numerical stability, capped at LAPS_CAP=80.

Output rank signal: 1 / (1 + predicted_first_pit_lap) — higher when
early-pit prediction. Path-B rank-mapped to (0, 1).

Usage:
  python scripts/probe_r14_cb_first_pit_lap.py [--smoke] [--max-rounds 3000]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, str(Path(__file__).parent))
from p1_features import (
    TE_CONFIGS, apply_fs_a, feature_columns_for_lgbm,
    fit_fs_a, make_features_static,
)
from p1_single_cb import fold_safe_te_for_fold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
LAPS_CAP = 80


def cb_first_pit_params(max_iters: int, seed: int, depth: int = 8) -> dict:
    return dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=max_iters,
        learning_rate=0.05,
        depth=depth,
        l2_leaf_reg=8.0,
        one_hot_max_size=10,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        min_data_in_leaf=20,
        od_type="Iter",
        od_wait=300,
        random_seed=seed,
        verbose=500,
        allow_writing_files=False,
        task_type="CPU",
        thread_count=-1,
        rsm=0.8,
    )


def build_first_pit_lap_target(df: pd.DataFrame) -> np.ndarray:
    """Per row, log1p of the lap of FIRST pit in this (Driver, Race) group
    using only the rows in `df`. If no pit observed in the group's rows,
    right-censor to log1p(max(LapNumber) in group). Returns np.float32.
    """
    work = df.copy()
    # per (Driver, Race), find min LapNumber where PitStop == 1
    pit_rows = work[work["PitStop"] == 1]
    first_pit = (pit_rows.groupby(["Driver", "Race"])["LapNumber"]
                 .min().rename("_first_pit_lap"))
    # Right-censor: race-end lap per (Driver, Race) where no pit observed
    race_end = (work.groupby(["Driver", "Race"])["LapNumber"]
                .max().rename("_race_end_lap"))
    summary = pd.concat([first_pit, race_end], axis=1)
    summary["_target_lap"] = summary["_first_pit_lap"].fillna(
        summary["_race_end_lap"])
    # Map back to per-row
    keys = list(zip(work["Driver"].values, work["Race"].values))
    tgt = np.array([summary["_target_lap"].get((d, r), np.nan)
                    for d, r in keys])
    tgt = np.clip(np.nan_to_num(tgt, nan=LAPS_CAP), 0, LAPS_CAP)
    return np.log1p(tgt).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("== R14 Phase 5: cb_first_pit_lap (CB on log1p(first-pit lap)) ==",
          flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    # Diagnostic: full-train target stats
    full_target = build_first_pit_lap_target(train)
    print(f"  full-train target: mean {full_target.mean():.3f} "
          f"std {full_target.std():.3f} min {full_target.min():.3f} "
          f"max {full_target.max():.3f}", flush=True)

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(train), 50_000,
                                                  replace=False)
        train = train.iloc[idx].reset_index(drop=True)
        print(f"  SMOKE: subset to {train.shape}", flush=True)

    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y_pn = train_S[TARGET].astype(int).reset_index(drop=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_pn)), y_pn))
    sample_ti = fold_list[0][0]
    sample_fs_a = fit_fs_a(train_S.iloc[sample_ti])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in feats and c not in cat_cols:
            cat_cols.append(c)
    feats = feats + [n for _, _, n in TE_CONFIGS]
    print(f"  feats: {len(feats)}  cat: {len(cat_cols)}", flush=True)

    sorted_ids = train_S[ID_COL].values
    orig_train_ids = train[ID_COL].values
    id_to_sorted_pos = {tid: i for i, tid in enumerate(sorted_ids)}
    test_sorted_ids = test_S[ID_COL].values
    test_orig_ids = test[ID_COL].values
    test_id_to_sorted_pos = {tid: i for i, tid in enumerate(test_sorted_ids)}

    oof_pred = np.zeros(len(y_pn), dtype=np.float64)
    test_pred = np.zeros(len(test_S), dtype=np.float64)
    fold_metrics = []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
        t_f = time.time()
        print(f"\n  --- Fold {fold}/{n_eff_folds} | ti={len(ti)} va={len(vi)} ---",
              flush=True)
        fs_a = fit_fs_a(train_S.iloc[ti])
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)

        y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
        fold_safe_te_for_fold(train_ti, train_va, test_fold,
                              y_ti, fold, N_FOLDS)

        X_tr = train_ti.reindex(columns=feats, fill_value=0).copy()
        X_va = train_va.reindex(columns=feats, fill_value=0).copy()
        X_te = test_fold.reindex(columns=feats, fill_value=0).copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("int32")
            X_va[c] = X_va[c].astype("int32")
            X_te[c] = X_te[c].astype("int32")
        num_cols = [c for c in feats if c not in cat_cols]
        for X in (X_tr, X_va, X_te):
            X[num_cols] = X[num_cols].fillna(0).astype(np.float32)
        cat_idx = [feats.index(c) for c in cat_cols]

        # STRICT per-fold target derivation
        t_ti = build_first_pit_lap_target(
            train_S.iloc[ti].reset_index(drop=True))
        t_va = build_first_pit_lap_target(
            train_S.iloc[vi].reset_index(drop=True))
        print(f"    target_ti: mean {t_ti.mean():.3f}  std {t_ti.std():.3f}  "
              f"target_va: mean {t_va.mean():.3f}", flush=True)

        params = cb_first_pit_params(args.max_rounds, SEED, depth=args.depth)
        m = cb.CatBoostRegressor(**params)
        m.fit(X_tr, t_ti, eval_set=(X_va, t_va),
              cat_features=cat_idx, use_best_model=True)

        pred_va_log = m.predict(X_va)
        pred_te_log = m.predict(X_te)
        # Convert log1p prediction → "earlier-pit score" = 1/(1+laps)
        pred_va_lap = np.maximum(np.expm1(pred_va_log), 0)
        pred_te_lap = np.maximum(np.expm1(pred_te_log), 0)
        score_va = 1.0 / (1.0 + pred_va_lap)
        score_te = 1.0 / (1.0 + pred_te_lap)
        oof_pred[vi] = score_va
        test_pred += score_te / n_eff_folds

        y_pn_va = train_va[TARGET].astype(int).values
        try:
            auc_va = float(roc_auc_score(y_pn_va, score_va))
        except ValueError:
            auc_va = float("nan")
        wall = time.time() - t_f
        fold_metrics.append(dict(
            fold=fold, iters=int(m.tree_count_),
            wall_s=float(wall), auc_va=auc_va,
        ))
        print(f"    iters={m.tree_count_} wall {wall:.0f}s  "
              f"AUC(score vs PitNextLap)={auc_va:.5f}", flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold projection ≈ "
              f"{(time.time()-t0) * N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y_pn.values, oof_pred))
    print(f"\n  Standalone OOF AUC: {auc_full:.5f}", flush=True)

    R13_OOF = ART / "oof_K15_pathb_driverclass_stint_tau100000.npy"
    R12_CBH_OOF = ART / "oof_R12_cb_horizon_strat.npy"
    R13_CBSC_OOF = ART / "oof_R13_cb_stint_completion_strat.npy"
    r13_oof = np.load(R13_OOF)
    auc_r13 = float(roc_auc_score(y_pn.values, r13_oof))
    rho_vs_r13, _ = spearmanr(oof_pred, r13_oof)
    cbh_oof = np.load(R12_CBH_OOF)
    rho_vs_cbh, _ = spearmanr(oof_pred, cbh_oof)
    cbsc_oof = np.load(R13_CBSC_OOF)
    rho_vs_cbsc, _ = spearmanr(oof_pred, cbsc_oof)
    print(f"  R13 PRIMARY OOF: {auc_r13:.6f}", flush=True)
    print(f"  ρ_OOF vs R13 PRIMARY: {rho_vs_r13:.6f}", flush=True)
    print(f"  ρ_OOF vs cb_horizon: {rho_vs_cbh:.6f}", flush=True)
    print(f"  ρ_OOF vs cb_stint_completion: {rho_vs_cbsc:.6f}", flush=True)

    # Rank-normalize and save
    combined = np.concatenate([oof_pred, test_pred])
    ranks = rankdata(combined)
    eps = 1.0 / (2 * len(ranks))
    uniform = np.clip((ranks - 0.5) / len(ranks), eps, 1 - eps)
    oof_uniform = uniform[:len(oof_pred)]
    test_uniform = uniform[len(oof_pred):]
    order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
    order_back_test = np.array([test_id_to_sorted_pos[t]
                                 for t in test_orig_ids])
    np.save(ART / "oof_R14_cb_first_pit_lap_strat.npy",
            oof_uniform[order_back_train].astype(np.float32))
    np.save(ART / "test_R14_cb_first_pit_lap_strat.npy",
            test_uniform[order_back_test].astype(np.float32))
    print(f"  Saved oof_R14_cb_first_pit_lap_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R14_Phase5_cb_first_pit_lap",
        oof_auc=auc_full,
        r13_primary_oof=auc_r13,
        rho_vs_r13=float(rho_vs_r13),
        rho_vs_cb_horizon=float(rho_vs_cbh),
        rho_vs_cb_stint_completion=float(rho_vs_cbsc),
        fold_metrics=fold_metrics,
        wall_total_s=time.time() - t0,
        depth=args.depth,
        max_rounds=args.max_rounds,
    )
    out_json = Path("audit/2026-05-19-r14-cb_first_pit_lap.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
