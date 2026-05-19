"""scripts/probe_r15_cb_pairlogit.py — R15 Phase 2: CB PairLogit on PitNextLap.

BASE pairwise-ranking-loss alternative to cb_v4's pointwise Logloss.
Same target (PitNextLap binary) but trained via CatBoost's native
`PairLogit` objective — within each group (Year, Race), the model
learns to rank PitNextLap=1 rows above PitNextLap=0 rows. Directly
optimizes pairwise AUC; no logloss bias.

Reuses cb_v4 FE pipeline (yekenot recipe + per-fold FS_A + per-fold
CV TE).

Output: predicted ranking score per row (CB PairLogit returns
real-valued scores). Rank-normalized to (eps, 1-eps) for Path-B
K=17 stack-add.

Usage:
  python scripts/probe_r15_cb_pairlogit.py [--smoke] [--max-rounds 3000]
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


def cb_pairlogit_params(max_iters: int, seed: int, depth: int = 8) -> dict:
    return dict(
        loss_function="PairLogit",
        eval_metric="PairLogit",
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
    )


def make_group_ids(df_subset: pd.DataFrame) -> np.ndarray:
    """Return per-row integer group_id from (Year, Race)."""
    keys = (df_subset["Year"].astype(str) + "|" +
            df_subset["Race"].astype(str)).values
    # Sort-stable factorize via pandas
    cat = pd.Categorical(keys)
    return cat.codes.astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("== R15 Phase 2: cb_pairlogit (CB pairwise ranking on PitNextLap) ==",
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

        # Group IDs per (Year, Race) — CatBoost PairLogit requirement.
        # Rows MUST be sorted by group_id (CatBoost throws
        # "queryIds should be grouped" if not). Sort each subset
        # stably by group_id.
        gid_tr = make_group_ids(train_ti)
        gid_va = make_group_ids(train_va)
        gid_te = make_group_ids(test_fold)
        sort_tr = np.argsort(gid_tr, kind="stable")
        sort_va = np.argsort(gid_va, kind="stable")
        sort_te = np.argsort(gid_te, kind="stable")
        X_tr_s = X_tr.iloc[sort_tr].reset_index(drop=True)
        X_va_s = X_va.iloc[sort_va].reset_index(drop=True)
        X_te_s = X_te.iloc[sort_te].reset_index(drop=True)
        y_tr_s = y_ti.values[sort_tr]
        y_va_s = train_va[TARGET].astype(int).values[sort_va]
        gid_tr_s = gid_tr[sort_tr]
        gid_va_s = gid_va[sort_va]
        gid_te_s = gid_te[sort_te]
        # Pools with sorted group_ids
        pool_tr = cb.Pool(X_tr_s, label=y_tr_s, cat_features=cat_idx,
                          group_id=gid_tr_s)
        pool_va = cb.Pool(X_va_s, label=y_va_s, cat_features=cat_idx,
                          group_id=gid_va_s)
        pool_te = cb.Pool(X_te_s, cat_features=cat_idx, group_id=gid_te_s)

        params = cb_pairlogit_params(args.max_rounds, SEED, depth=args.depth)
        m = cb.CatBoostRanker(**params)
        m.fit(pool_tr, eval_set=pool_va, use_best_model=True)

        # Predict on sorted pools, then UNSORT to original index order
        pred_va_sorted = m.predict(pool_va)
        pred_te_sorted = m.predict(pool_te)
        # Unsort: argsort(sort_va) gives the inverse permutation
        inv_va = np.argsort(sort_va)
        inv_te = np.argsort(sort_te)
        pred_va = pred_va_sorted[inv_va]
        pred_te = pred_te_sorted[inv_te]
        oof_pred[vi] = pred_va
        test_pred += pred_te / n_eff_folds

        y_va = train_va[TARGET].astype(int).values
        try:
            auc_va = float(roc_auc_score(y_va, pred_va))
        except ValueError:
            auc_va = float("nan")
        wall = time.time() - t_f
        fold_metrics.append(dict(
            fold=fold, iters=int(m.tree_count_),
            wall_s=float(wall), auc_va=auc_va,
        ))
        print(f"    iters={m.tree_count_} wall {wall:.0f}s  AUC={auc_va:.5f}",
              flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold proj "
              f"~{(time.time()-t0)*N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y.values, oof_pred))
    print(f"\n  Standalone OOF AUC: {auc_full:.5f}", flush=True)

    # ρ vs R14 PRIMARY
    r14_oof = np.load(ART / "oof_K16_tabm_pathb_dcs_tau100000.npy")
    auc_r14 = float(roc_auc_score(y.values, r14_oof))
    rho_vs_r14, _ = spearmanr(oof_pred, r14_oof)
    print(f"  R14 PRIMARY OOF: {auc_r14:.6f}", flush=True)
    print(f"  ρ_OOF vs R14 PRIMARY: {rho_vs_r14:.6f}", flush=True)

    # ρ vs cb_v4 (the pointwise Logloss reference)
    cb_v4 = np.load(ART / "oof_p1_single_cb_v4_gpu_strat.npy")
    cb_v4_p = cb_v4[:, 1] if cb_v4.ndim == 2 else cb_v4
    # cb_v4 is in original order; oof_pred is in sorted order — align via id
    cb_v4_sorted = np.array([cb_v4_p[orig_train_ids.tolist().index(t)]
                              if t in orig_train_ids else np.nan
                              for t in sorted_ids[:100]])  # quick sample
    # Use index map for full ρ
    id_to_orig = {tid: i for i, tid in enumerate(orig_train_ids)}
    cb_v4_sorted_full = np.array([cb_v4_p[id_to_orig[t]] for t in sorted_ids])
    rho_vs_cbv4, _ = spearmanr(oof_pred, cb_v4_sorted_full)
    print(f"  ρ_OOF vs cb_v4 base: {rho_vs_cbv4:.6f}", flush=True)

    # Rank-normalize for Path-B add
    combined = np.concatenate([oof_pred, test_pred])
    ranks = rankdata(combined)
    eps = 1.0 / (2 * len(ranks))
    uniform = np.clip((ranks - 0.5) / len(ranks), eps, 1 - eps)
    oof_uniform = uniform[:len(oof_pred)]
    test_uniform = uniform[len(oof_pred):]

    order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
    order_back_test = np.array([test_id_to_sorted_pos[t]
                                 for t in test_orig_ids])
    np.save(ART / "oof_R15_cb_pairlogit_strat.npy",
            oof_uniform[order_back_train].astype(np.float32))
    np.save(ART / "test_R15_cb_pairlogit_strat.npy",
            test_uniform[order_back_test].astype(np.float32))
    print(f"  Saved oof_R15_cb_pairlogit_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R15_Phase2_cb_pairlogit",
        oof_auc=auc_full,
        r14_primary_oof=auc_r14,
        rho_vs_r14=float(rho_vs_r14),
        rho_vs_cb_v4=float(rho_vs_cbv4),
        fold_metrics=fold_metrics,
        wall_total_s=time.time() - t0,
        depth=args.depth,
        max_rounds=args.max_rounds,
    )
    out_json = Path("audit/2026-05-19-r15-cb_pairlogit.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
