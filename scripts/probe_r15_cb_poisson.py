"""scripts/probe_r15_cb_poisson.py — R15 Phase 5: CB Poisson on LapsUntilPit.

BASE absorption check. CatBoost native `Poisson` loss on the integer
count target (raw LapsUntilPit, NOT log). Same target as cb_horizon
but Poisson gradient instead of log-RMSE.

PREDICTION: should ABSORB at K=N+1 add (monotone-transform of the same
target → ρ_OOF predicted ≥ 0.95 vs cb_horizon). Cheap to verify the
prediction; if it does NOT absorb, that itself is informative.

Target: raw LapsUntilPit count (integer). Strict per-fold derivation.

Output: predicted rate → rank-normalized to (eps, 1-eps) for Path-B.

Usage:
  python scripts/probe_r15_cb_poisson.py [--smoke] [--max-rounds 3000]
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
from b_laps_until_pit import build_laps_until_pit

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5


def cb_poisson_params(max_iters: int, seed: int, depth: int = 8) -> dict:
    return dict(
        loss_function="Poisson",
        eval_metric="Poisson",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("== R15 Phase 5: cb_poisson (CB Poisson on raw LapsUntilPit count) ==",
          flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"  train {train.shape}  test {test.shape}", flush=True)

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

        # STRICT per-fold target: raw LapsUntilPit count (no log)
        t_ti_log = build_laps_until_pit(
            train_S.iloc[ti].reset_index(drop=True))
        t_va_log = build_laps_until_pit(
            train_S.iloc[vi].reset_index(drop=True))
        # Invert log1p to raw count (Poisson expects non-negative integer)
        t_ti = np.round(np.expm1(t_ti_log)).astype(np.float32)
        t_va = np.round(np.expm1(t_va_log)).astype(np.float32)
        print(f"    target_ti: mean {t_ti.mean():.3f}  std {t_ti.std():.3f}  "
              f"target_va: mean {t_va.mean():.3f}", flush=True)

        params = cb_poisson_params(args.max_rounds, SEED, depth=args.depth)
        m = cb.CatBoostRegressor(**params)
        m.fit(X_tr, t_ti, eval_set=(X_va, t_va),
              cat_features=cat_idx, use_best_model=True)

        pred_va = m.predict(X_va)
        pred_te = m.predict(X_te)
        # Lower predicted count → closer pit → higher pit-likelihood
        score_va = -pred_va
        score_te = -pred_te
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
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold proj "
              f"~{(time.time()-t0)*N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y_pn.values, oof_pred))
    print(f"\n  Standalone OOF AUC: {auc_full:.5f}", flush=True)

    # ρ vs R15 PRIMARY (K=17)
    r15_oof = np.load(ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy")
    auc_r15 = float(roc_auc_score(y_pn.values, r15_oof))
    rho_vs_r15, _ = spearmanr(oof_pred, r15_oof)
    print(f"  R15 PRIMARY OOF: {auc_r15:.6f}", flush=True)
    print(f"  ρ_OOF vs R15 PRIMARY: {rho_vs_r15:.6f}", flush=True)

    # ρ vs cb_horizon (same target, RMSE on log) — absorption check
    cbh = np.load(ART / "oof_R12_cb_horizon_strat.npy")
    rho_vs_cbh, _ = spearmanr(oof_pred, cbh)
    print(f"  ρ_OOF vs cb_horizon (same target, RMSE-log loss): "
          f"{rho_vs_cbh:.6f}", flush=True)
    absorbed = rho_vs_cbh >= 0.95
    print(f"  ABSORPTION (ρ≥0.95): {absorbed}", flush=True)

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
    np.save(ART / "oof_R15_cb_poisson_strat.npy",
            oof_uniform[order_back_train].astype(np.float32))
    np.save(ART / "test_R15_cb_poisson_strat.npy",
            test_uniform[order_back_test].astype(np.float32))
    print(f"  Saved oof_R15_cb_poisson_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R15_Phase5_cb_poisson",
        oof_auc=auc_full,
        r15_primary_oof=auc_r15,
        rho_vs_r15=float(rho_vs_r15),
        rho_vs_cb_horizon=float(rho_vs_cbh),
        absorbed_by_cb_horizon=bool(absorbed),
        fold_metrics=fold_metrics,
        wall_total_s=time.time() - t0,
        depth=args.depth, max_rounds=args.max_rounds,
    )
    out_json = Path("audit/2026-05-19-r15-cb_poisson.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
