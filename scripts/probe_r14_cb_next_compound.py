"""scripts/probe_r14_cb_next_compound.py — Phase 3: CB MultiClass on next-stint Compound.

Qualitatively different orthogonal-target CB on TOP of R13 PRIMARY.
Target: the Compound type (5-class) of the NEXT stint (Stint+1)
within the same (Driver, Race) group. Categorical, not real-valued —
distinct from cb_horizon (laps-until-pit count) and cb_stint_completion
(stint-fraction float).

Output base column: P(next_compound != current_compound) — probability
the driver switches to a different tyre at the next stop.

Fold-safety: target derived from ti rows of the same (Driver, Race)
group ONLY. Rows whose next stint has no ti row → target unknown,
excluded from training (val rows + test rows still get predictions).
This is strict per-fold derivation, same discipline as cb_horizon.

Per Phase B's read (base diversity is doing the work) + the R13 win
on cb_stint_completion (ρ ≈ 0 vs PRIMARY), categorical orthogonality
may produce another stackable base.

Usage:
  python scripts/probe_r14_cb_next_compound.py [--smoke] [--max-rounds 3000]
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

R13_OOF = ART / "oof_K15_pathb_driverclass_stint_tau100000.npy"
R12_CBH_OOF = ART / "oof_R12_cb_horizon_strat.npy"
R13_CBSC_OOF = ART / "oof_R13_cb_stint_completion_strat.npy"

COMPOUND_VOCAB = {"HARD": 0, "MEDIUM": 1, "SOFT": 2,
                  "INTERMEDIATE": 3, "WET": 4}


def cb_mc_params(max_iters: int, seed: int, depth: int = 8) -> dict:
    """CatBoost MultiClass for 5-class compound prediction."""
    return dict(
        loss_function="MultiClass",
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
        classes_count=5,
    )


def build_next_compound_target(df_subset: pd.DataFrame) -> np.ndarray:
    """For each row in df_subset, find the NEXT stint's compound (5-class
    int label, 0-4 per COMPOUND_VOCAB) using ONLY rows in df_subset.

    Returns:
        target: int array of length len(df_subset). Value in {0..4} if a
        next-stint row exists in df_subset for same (Driver, Race); -1
        otherwise (excluded from training).
    """
    df = df_subset.reset_index().rename(columns={"index": "_orig_idx"})
    df = df.sort_values(["Driver", "Race", "Stint", "LapNumber"],
                        kind="stable").reset_index(drop=True)
    # Map each (Driver, Race, Stint) → Compound (constant within stint)
    stint_compound = (df.groupby(["Driver", "Race", "Stint"])["Compound"]
                      .first()
                      .map(COMPOUND_VOCAB).fillna(-1).astype(int))
    # For each row, look up next stint's compound
    next_keys = list(zip(df["Driver"].values, df["Race"].values,
                          (df["Stint"].astype(int) + 1).values))
    target = np.array([stint_compound.get((d, r, s), -1)
                       for d, r, s in next_keys], dtype=int)
    df["_target"] = target
    df = df.sort_values("_orig_idx", kind="stable")
    return df["_target"].values.astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("== R14 Phase 3: cb_next_compound (5-class on next-stint Compound) ==",
          flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    # Diagnostic: full-train target stats
    full_target = build_next_compound_target(train)
    print(f"  full-train target dist (compound idx):", flush=True)
    vals, cnts = np.unique(full_target, return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"    {v} ({'NONE' if v == -1 else list(COMPOUND_VOCAB.keys())[v]}): "
              f"{c} ({c/len(full_target)*100:.1f}%)", flush=True)

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

    # The "base column" output: P(next_compound != current_compound).
    # Per row, accumulate across folds.
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

        # STRICT per-fold target: next-compound computed on ti rows only.
        t_ti_full = build_next_compound_target(
            train_S.iloc[ti].reset_index(drop=True))
        t_va_full = build_next_compound_target(
            train_S.iloc[vi].reset_index(drop=True))

        # Exclude rows where target is unknown (-1)
        keep_ti = t_ti_full >= 0
        keep_va = t_va_full >= 0
        X_tr_k = X_tr.loc[keep_ti].reset_index(drop=True)
        y_tr_k = t_ti_full[keep_ti]
        X_va_k = X_va.loc[keep_va].reset_index(drop=True)
        y_va_k = t_va_full[keep_va]
        print(f"    target_ti: keep {keep_ti.sum()}/{len(t_ti_full)}; "
              f"target_va: keep {keep_va.sum()}/{len(t_va_full)}", flush=True)

        params = cb_mc_params(args.max_rounds, SEED, depth=args.depth)
        m = cb.CatBoostClassifier(**params)
        m.fit(X_tr_k, y_tr_k, eval_set=(X_va_k, y_va_k),
              cat_features=cat_idx, use_best_model=True)

        # Predict on val + test (ALL rows, not just kept rows)
        # predict_proba returns (n, 5) — probability per compound class
        proba_va = m.predict_proba(X_va)  # (n_va, 5)
        proba_te = m.predict_proba(X_te)

        # Map current compound (per row) to its class index
        cur_va = train_va["Compound"].map(COMPOUND_VOCAB).fillna(-1).astype(int).values
        cur_te = test_fold["Compound"].map(COMPOUND_VOCAB).fillna(-1).astype(int).values

        # P(next != current) = 1 - P(next = current)
        p_same_va = np.array([proba_va[i, c] if 0 <= c < 5 else 0.0
                              for i, c in enumerate(cur_va)])
        p_same_te = np.array([proba_te[i, c] if 0 <= c < 5 else 0.0
                              for i, c in enumerate(cur_te)])
        p_change_va = 1.0 - p_same_va
        p_change_te = 1.0 - p_same_te
        oof_pred[vi] = p_change_va
        test_pred += p_change_te / n_eff_folds

        y_pn_va = train_va[TARGET].astype(int).values
        try:
            auc_va = float(roc_auc_score(y_pn_va, p_change_va))
        except ValueError:
            auc_va = float("nan")
        wall = time.time() - t_f
        fold_metrics.append(dict(
            fold=fold, n_ti_kept=int(keep_ti.sum()),
            n_va_kept=int(keep_va.sum()),
            iters=int(m.tree_count_), wall_s=float(wall),
            auc_va=auc_va,
        ))
        print(f"    iters={m.tree_count_} wall {wall:.0f}s  "
              f"AUC(p_change vs PitNextLap)={auc_va:.5f}", flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold projection ≈ "
              f"{(time.time()-t0) * N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y_pn.values, oof_pred))
    print(f"\n  Standalone OOF AUC (p_change vs PitNextLap): {auc_full:.5f}",
          flush=True)

    # ρ vs R13 PRIMARY OOF (Path-B output on K=15)
    r13_oof = np.load(R13_OOF)
    auc_r13 = float(roc_auc_score(y_pn.values, r13_oof))
    rho_vs_r13, _ = spearmanr(oof_pred, r13_oof)
    print(f"  R13 PRIMARY OOF: {auc_r13:.6f}", flush=True)
    print(f"  ρ_OOF (p_change) vs R13 PRIMARY: {rho_vs_r13:.6f}", flush=True)

    cbh_oof = np.load(R12_CBH_OOF)
    cbsc_oof = np.load(R13_CBSC_OOF)
    rho_vs_cbh, _ = spearmanr(oof_pred, cbh_oof)
    rho_vs_cbsc, _ = spearmanr(oof_pred, cbsc_oof)
    print(f"  ρ_OOF (p_change) vs cb_horizon base: {rho_vs_cbh:.6f}",
          flush=True)
    print(f"  ρ_OOF (p_change) vs cb_stint_completion base: {rho_vs_cbsc:.6f}",
          flush=True)

    # Rank-normalize to (eps, 1-eps) for Path-B add
    combined = np.concatenate([oof_pred, test_pred])
    ranks = rankdata(combined)
    eps = 1.0 / (2 * len(ranks))
    uniform = np.clip((ranks - 0.5) / len(ranks), eps, 1 - eps)
    oof_uniform = uniform[:len(oof_pred)]
    test_uniform = uniform[len(oof_pred):]

    order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
    order_back_test = np.array([test_id_to_sorted_pos[t]
                                 for t in test_orig_ids])
    np.save(ART / "oof_R14_cb_next_compound_strat.npy",
            oof_uniform[order_back_train].astype(np.float32))
    np.save(ART / "test_R14_cb_next_compound_strat.npy",
            test_uniform[order_back_test].astype(np.float32))
    print(f"  Saved oof_R14_cb_next_compound_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R14_Phase3_cb_next_compound",
        oof_auc_pchange=auc_full,
        r13_primary_oof=auc_r13,
        rho_pchange_vs_r13=float(rho_vs_r13),
        rho_pchange_vs_cb_horizon=float(rho_vs_cbh),
        rho_pchange_vs_cb_stint_completion=float(rho_vs_cbsc),
        fold_metrics=fold_metrics,
        wall_total_s=time.time() - t0,
        depth=args.depth,
        max_rounds=args.max_rounds,
    )
    out_json = Path("audit/2026-05-19-r14-cb_next_compound.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
