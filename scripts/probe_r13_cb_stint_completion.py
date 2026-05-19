"""scripts/probe_r13_cb_stint_completion.py — Phase D: CatBoost on stint-fraction.

A second alternative-target CatBoost on top of R12-2 PRIMARY. Target:
`1 − TyreLife / stint_duration`, where stint_duration is per-
(Driver, Race, Stint) max(TyreLife) within the fold's train rows
(strict per-fold, same pattern as cb_horizon to avoid Day-15
target-construction leakage 88-100 % collapse).

Stint_duration uses **TyreLife only, not PitNextLap** — so strictly
speaking it isn't label-derived. But the per-fold strict version
keeps the discipline consistent with cb_horizon and trivially safe.

The model is asked: given non-TyreLife row features (Compound,
Position, LapTime, etc.), predict the FRACTION of this stint that
has elapsed. This requires the model to learn an expected stint-
length from non-tyre-life features (Compound, Driver, Race, Stint,
LapNumber) — a different signal from cb_horizon's laps-until-pit
target.

Predicted to be orthogonal-but-smaller-than-cb_horizon (plan-agent
verdict: "the only target-variant not predicted to absorb"; my
analysis: target is partly inferrable from existing features so
the orthogonal lift is bounded).

Per Phase B (operator-class diagnostic), base diversity is the
right axis: cb_horizon added +0.02 to +0.16 bp at LR-meta and
+0.046 bp at Path-B. cb_stint_completion is a second base-class
probe on the winning axis.

Output: oof_R13_cb_stint_completion_strat.npy + test variant.
K=15 add via build_K13_pathb_multiseg --extra-bases R12_cb_horizon
R13_cb_stint_completion.

Usage:
  python scripts/probe_r13_cb_stint_completion.py [--smoke] [--depth 8]
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from p1_features import (
    TE_CONFIGS, apply_fs_a, feature_columns_for_lgbm,
    fit_fs_a, make_features_static,
)
from p1_single_cb import fold_safe_te_for_fold

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

R12_OOF = ART / "oof_K14_pathb_driverclass_stint_tau100000.npy"   # PRIMARY
CB_V4_OOF = ART / "oof_p1_single_cb_v4_gpu_strat.npy"


def cb_stint_params(max_iters: int, seed: int, depth: int = 8) -> dict:
    """CB regression on stint-completion fraction in [0, 1]. RMSE."""
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


def build_stint_completion(df: pd.DataFrame) -> np.ndarray:
    """Per-row stint-completion fraction = TyreLife / max(TyreLife) within
    that row's (Driver, Race, Stint) group, computed on the rows of `df`.

    Strict semantic: if `df` is a fold's train subset, the returned
    fraction uses ONLY train rows for the max. Train and val rows of the
    same stint can have different max(TyreLife) and thus different
    fractions.

    Returns: TyreLife / max(TyreLife) ∈ [0, 1] (closer to 1 = END of
    stint = high pit-probability; 0 = start of stint). Cap denom at 1
    to avoid div-by-zero for single-row stints.
    """
    grp = df.groupby(["Driver", "Race", "Stint"])
    stint_max = grp["TyreLife"].transform("max").astype(np.float32)
    stint_max = np.maximum(stint_max, 1.0)
    frac = df["TyreLife"].astype(np.float32).values / stint_max
    return np.clip(frac, 0.0, 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("== R13 Phase D: cb_stint_completion (1 - TyreLife/stint_max) ==",
          flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    # Diagnostic: full-train target distribution
    full_target = build_stint_completion(train)
    print(f"  full-train target: mean {full_target.mean():.4f} "
          f"std {full_target.std():.4f} pct-zero {(full_target == 0).mean():.4f}",
          flush=True)

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(train), 50_000,
                                                  replace=False)
        train = train.iloc[idx].reset_index(drop=True)
        print(f"  SMOKE: subset to {train.shape}", flush=True)

    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
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

    oof_pred = np.zeros(len(y), dtype=np.float64)
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

        # STRICT per-fold target: stint_completion computed on ti rows only.
        t_ti = build_stint_completion(
            train_S.iloc[ti].reset_index(drop=True))
        t_va = build_stint_completion(
            train_S.iloc[vi].reset_index(drop=True))
        print(f"    target_ti: mean {t_ti.mean():.4f}  std {t_ti.std():.4f}  "
              f"target_va: mean {t_va.mean():.4f}", flush=True)

        params = cb_stint_params(args.max_rounds, SEED, depth=args.depth)
        m = cb.CatBoostRegressor(**params)
        m.fit(X_tr, t_ti, eval_set=(X_va, t_va),
              cat_features=cat_idx, use_best_model=True)

        pred_va = m.predict(X_va)
        pred_te = m.predict(X_te)
        oof_pred[vi] = pred_va
        test_pred += pred_te / n_eff_folds

        y_va = train_va[TARGET].astype(int).values
        try:
            auc_va = float(roc_auc_score(y_va, pred_va))
        except ValueError:
            auc_va = float("nan")
        wall = time.time() - t_f
        fold_metrics.append(dict(
            fold=fold, n_ti=int(len(ti)), n_va=int(len(vi)),
            iters=int(m.tree_count_), wall_s=float(wall),
            auc_va=auc_va, pred_range=[float(pred_va.min()), float(pred_va.max())],
        ))
        print(f"    iters={m.tree_count_} wall {wall:.0f}s  AUC(pred)={auc_va:.5f}",
              flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold projection "
              f"~ {(time.time()-t0) * N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y.values, oof_pred))
    print(f"\n  Standalone OOF AUC (proxy): {auc_full:.5f}", flush=True)

    # ρ vs R12-2 PRIMARY OOF (Path-B output)
    r12_oof = np.load(R12_OOF)
    auc_r12 = float(roc_auc_score(y.values, r12_oof))
    rho_vs_r12, _ = spearmanr(oof_pred, r12_oof)
    print(f"  R12-2 PRIMARY OOF: {auc_r12:.6f}", flush=True)
    print(f"  ρ_OOF (stint base) vs R12-2: {rho_vs_r12:.6f}", flush=True)

    # ρ vs cb_horizon base
    cbh_oof = np.load(ART / "oof_R12_cb_horizon_strat.npy")
    rho_vs_cbh, _ = spearmanr(oof_pred, cbh_oof)
    print(f"  ρ_OOF (stint base) vs cb_horizon base: {rho_vs_cbh:.6f}",
          flush=True)

    # Rank-normalize and save (Path-B add requires (0, 1))
    from scipy.stats import rankdata
    combined = np.concatenate([oof_pred, test_pred])
    ranks = rankdata(combined)
    eps = 1.0 / (2 * len(ranks))
    uniform = np.clip((ranks - 0.5) / len(ranks), eps, 1 - eps)
    oof_uniform = uniform[:len(oof_pred)]
    test_uniform = uniform[len(oof_pred):]

    # Map sorted → original order
    order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
    order_back_test = np.array([test_id_to_sorted_pos[t]
                                 for t in test_orig_ids])
    np.save(ART / "oof_R13_cb_stint_completion_strat.npy",
            oof_uniform[order_back_train].astype(np.float32))
    np.save(ART / "test_R13_cb_stint_completion_strat.npy",
            test_uniform[order_back_test].astype(np.float32))
    print(f"  Saved oof_R13_cb_stint_completion_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R13_D_cb_stint_completion",
        oof_auc=auc_full,
        r12_primary_oof=auc_r12,
        rho_vs_r12=float(rho_vs_r12),
        rho_vs_cb_horizon=float(rho_vs_cbh),
        fold_metrics=fold_metrics,
        wall_total_s=time.time() - t0,
        depth=args.depth, max_rounds=args.max_rounds,
    )
    out_json = Path("audit/2026-05-19-r13-cb_stint_completion.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
