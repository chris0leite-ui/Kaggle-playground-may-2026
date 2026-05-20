"""scripts/probe_r16_real_f1_specialist.py — Round 16 Phase 3a:
real-F1-subset CatBoost specialist sub-model.

Strategy-critic 2026-05-19 identified the 120 bp real-F1 vs synthetic-
D### AUC gap (real-F1 AUC 0.94672 vs synthetic 0.95874) as the only
remaining failure axis. Cheap row-level absorption (is_real flag +
Driver TE) was NULL — K=17 PRIMARY pool already absorbs both. This
probe tests the alternative: train a separate CB classifier on the
is_real==1 row subset (168,324 rows, 38% of train) and blend back at
the slice via per-fold LR-meta.

Mechanism: synthetic generator's 887 pseudo-drivers dissolve the
real-F1 (~22-driver) cohort signal at the row-aggregate level. A
sub-model trained ONLY on real-F1 rows can extract driver-specific
patterns that the full-population model averages away.

Per the strategy-critic EV: +0.05 to +0.15 bp midpoint on overall
OOF AUC if sub-model lifts the real-F1 slice AUC by 2+ bp.

Inherits cb_horizon's fold-safety: PitNextLap target only (no
reformulation — Bayes-ceiling forbids residual-style targets;
cf. R12-1 cb_resid AUC 0.478 below random).

Usage:
  python scripts/probe_r16_real_f1_specialist.py [--smoke] [--max-rounds 3000]
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import logit, expit
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

REF_R15_OOF = ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy"
REF_R15_TEST = ART / "test_K17_xendcg_pathb_dcs_tau100000.npy"


def cb_params(max_iters: int, seed: int, depth: int = 8) -> dict:
    return dict(
        loss_function="Logloss",
        eval_metric="AUC",
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
    print("== R16 Phase 3a: real-F1 specialist sub-model ==", flush=True)

    train_full = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    # Full-train ID → R15 OOF position mapping (preserved across smoking)
    full_orig_train_ids = train_full[ID_COL].values
    id_to_r15_pos = {tid: i for i, tid in enumerate(full_orig_train_ids)}

    is_real_full = (~train_full["Driver"].str.startswith("D")).values
    is_real_test_orig = (~test["Driver"].str.startswith("D")).values
    print(f"  train {train_full.shape}  is_real=1: {is_real_full.sum()} "
          f"({100*is_real_full.mean():.1f}%)", flush=True)
    print(f"  test  {test.shape}  is_real=1: {is_real_test_orig.sum()} "
          f"({100*is_real_test_orig.mean():.1f}%)", flush=True)

    train = train_full

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(train), 50_000,
                                                  replace=False)
        train = train.iloc[idx].reset_index(drop=True)
        print(f"  SMOKE: subset to {train.shape}  is_real=1: "
              f"{(~train['Driver'].str.startswith('D')).sum()}", flush=True)
    is_real_train_orig = (~train["Driver"].str.startswith("D")).values
    y_all = train[TARGET].astype(int).values

    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    # Reconstruct is_real for the sorted features frame.
    sorted_ids = train_S[ID_COL].values
    orig_train_ids = train[ID_COL].values
    id_to_orig_pos = {tid: i for i, tid in enumerate(orig_train_ids)}
    id_to_sorted_pos = {tid: i for i, tid in enumerate(sorted_ids)}
    is_real_S = np.array([is_real_train_orig[id_to_orig_pos[t]]
                          for t in sorted_ids], dtype=bool)
    print(f"  is_real_S=1 in sorted: {is_real_S.sum()} "
          f"({100*is_real_S.mean():.1f}%)", flush=True)

    test_sorted_ids = test_S[ID_COL].values
    test_orig_ids = test[ID_COL].values
    test_id_to_orig_pos = {tid: i for i, tid in enumerate(test_orig_ids)}
    test_id_to_sorted_pos = {tid: i for i, tid in enumerate(test_sorted_ids)}
    is_real_test_S = np.array([is_real_test_orig[test_id_to_orig_pos[t]]
                                for t in test_sorted_ids], dtype=bool)
    print(f"  is_real_test_S=1 in sorted: {is_real_test_S.sum()}", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    sample_ti = fold_list[0][0]
    # Restrict sample_ti to is_real=1 for feature schema fitting
    sample_ti_real = sample_ti[is_real_S[sample_ti]]
    sample_fs_a = fit_fs_a(train_S.iloc[sample_ti_real])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in feats and c not in cat_cols:
            cat_cols.append(c)
    feats = feats + [n for _, _, n in TE_CONFIGS]
    print(f"  feats: {len(feats)}  cat: {len(cat_cols)}", flush=True)

    # Outputs: sub-model OOF (only is_real=1 slots filled), sub-model test
    # (only is_real=1 slots filled).
    submodel_oof_real_only = np.zeros(len(y), dtype=np.float64)
    submodel_test_real_only = np.zeros(len(test_S), dtype=np.float64)
    fold_metrics = []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    # Load R15 reference. R15 OOF .npy is in full-train.csv row order
    # (439,140 entries). Use full-train id_to_r15_pos to map sorted IDs
    # → R15 positions; works correctly under smoking (which leaves
    # train_full untouched).
    r15_oof_orig = np.load(REF_R15_OOF)
    r15_oof_sorted = np.array([r15_oof_orig[id_to_r15_pos[t]]
                                for t in sorted_ids])
    r15_test_orig = np.load(REF_R15_TEST)
    r15_test_sorted = np.array([r15_test_orig[test_id_to_orig_pos[t]]
                                 for t in test_sorted_ids])

    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds], 1):
        t_f = time.time()
        ti_real = ti[is_real_S[ti]]
        vi_real = vi[is_real_S[vi]]
        print(f"\n  --- Fold {fold}/{n_eff_folds} | ti={len(ti)} "
              f"ti_real={len(ti_real)} va={len(vi)} va_real={len(vi_real)} ---",
              flush=True)

        fs_a = fit_fs_a(train_S.iloc[ti_real])
        train_ti = apply_fs_a(train_S.iloc[ti_real].reset_index(drop=True),
                              fs_a)
        train_va = apply_fs_a(train_S.iloc[vi_real].reset_index(drop=True),
                              fs_a)
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

        params = cb_params(args.max_rounds, SEED, depth=args.depth)
        m = cb.CatBoostClassifier(**params)
        m.fit(X_tr, y_ti, eval_set=(X_va, train_va[TARGET].astype(int)),
              cat_features=cat_idx, use_best_model=True)

        p_va_real = m.predict_proba(X_va)[:, 1]
        p_te_real = m.predict_proba(X_te)[:, 1]
        # Fill sub-model OOF/test ONLY for is_real=1 sorted positions
        submodel_oof_real_only[vi_real] = p_va_real
        submodel_test_real_only[is_real_test_S] += p_te_real[is_real_test_S] / n_eff_folds

        y_va_real = train_va[TARGET].astype(int).values
        auc_va_real = float(roc_auc_score(y_va_real, p_va_real))
        # R15 baseline on the same is_real=1 val slice
        auc_r15_va_real = float(roc_auc_score(y_va_real, r15_oof_sorted[vi_real]))

        wall = time.time() - t_f
        fold_metrics.append(dict(
            fold=fold, n_ti_real=int(len(ti_real)),
            n_va_real=int(len(vi_real)),
            iters=int(m.tree_count_), wall_s=float(wall),
            auc_va_real_submodel=auc_va_real,
            auc_va_real_R15=auc_r15_va_real,
            slice_delta_bp=1e4 * (auc_va_real - auc_r15_va_real),
        ))
        print(f"    iters={m.tree_count_}  wall {wall:.0f}s", flush=True)
        print(f"    real-F1 slice AUC: sub={auc_va_real:.5f}  R15={auc_r15_va_real:.5f}  "
              f"Δ={1e4*(auc_va_real-auc_r15_va_real):+.3f} bp", flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold projection "
              f"≈ {(time.time()-t0) * N_FOLDS:.0f}s", flush=True)
        return

    # === Slice analysis: replace-at-slice blend vs LR-meta blend ===
    print("\n  Slice analysis:", flush=True)
    y_arr = y.values
    real_mask_S = is_real_S
    synth_mask_S = ~is_real_S

    # 1) Replace-at-slice: combined = sub_model where is_real, R15 elsewhere
    combined_replace = r15_oof_sorted.copy()
    combined_replace[real_mask_S] = submodel_oof_real_only[real_mask_S]
    auc_combined_replace = roc_auc_score(y_arr, combined_replace)
    auc_r15_baseline = roc_auc_score(y_arr, r15_oof_sorted)
    print(f"  Overall OOF R15 baseline:       {auc_r15_baseline:.6f}",
          flush=True)
    print(f"  Overall OOF replace-at-slice:   {auc_combined_replace:.6f}  "
          f"Δ {1e4*(auc_combined_replace-auc_r15_baseline):+.4f} bp",
          flush=True)

    # Per-slice AUC for diagnostic
    auc_real_r15 = roc_auc_score(y_arr[real_mask_S], r15_oof_sorted[real_mask_S])
    auc_real_sub = roc_auc_score(y_arr[real_mask_S], submodel_oof_real_only[real_mask_S])
    auc_synth_r15 = roc_auc_score(y_arr[synth_mask_S], r15_oof_sorted[synth_mask_S])
    print(f"  real-F1 slice AUC: R15={auc_real_r15:.5f}  sub={auc_real_sub:.5f}  "
          f"Δ {1e4*(auc_real_sub-auc_real_r15):+.3f} bp", flush=True)
    print(f"  synthetic slice AUC: R15={auc_synth_r15:.5f}", flush=True)

    # 2) Per-fold LR-meta blend on is_real=1 slice
    # Inputs: [logit(R15), logit(sub_model)] → PitNextLap
    blend_oof = r15_oof_sorted.copy()
    eps = 1e-6
    coef_log = []
    for fold, (ti, vi) in enumerate(fold_list, 1):
        ti_real = ti[real_mask_S[ti]]
        vi_real = vi[real_mask_S[vi]]
        Xm_ti = np.column_stack([
            logit(np.clip(r15_oof_sorted[ti_real], eps, 1 - eps)),
            logit(np.clip(submodel_oof_real_only[ti_real], eps, 1 - eps)),
        ])
        Xm_va = np.column_stack([
            logit(np.clip(r15_oof_sorted[vi_real], eps, 1 - eps)),
            logit(np.clip(submodel_oof_real_only[vi_real], eps, 1 - eps)),
        ])
        y_meta = y_arr[ti_real]
        lr = LogisticRegression(C=1.0, max_iter=500)
        lr.fit(Xm_ti, y_meta)
        blend_oof[vi_real] = lr.predict_proba(Xm_va)[:, 1]
        coef_log.append({"fold": fold, "intercept": float(lr.intercept_[0]),
                         "coef_r15": float(lr.coef_[0, 0]),
                         "coef_sub": float(lr.coef_[0, 1])})
    auc_blend = roc_auc_score(y_arr, blend_oof)
    print(f"  Overall OOF LR-meta blend:      {auc_blend:.6f}  "
          f"Δ {1e4*(auc_blend-auc_r15_baseline):+.4f} bp", flush=True)
    print(f"  LR-meta coefs (per fold): {coef_log}", flush=True)

    # === Test predictions: same blend strategy, full-train LR-meta on real ===
    # Fit one LR-meta on full-train is_real=1 OOF predictions
    real_mask_train_S = real_mask_S
    Xm_full = np.column_stack([
        logit(np.clip(r15_oof_sorted[real_mask_train_S], eps, 1 - eps)),
        logit(np.clip(submodel_oof_real_only[real_mask_train_S], eps, 1 - eps)),
    ])
    y_full_real = y_arr[real_mask_train_S]
    lr_full = LogisticRegression(C=1.0, max_iter=500)
    lr_full.fit(Xm_full, y_full_real)

    # Apply to test
    test_blend = r15_test_sorted.copy()
    Xm_te = np.column_stack([
        logit(np.clip(r15_test_sorted[is_real_test_S], eps, 1 - eps)),
        logit(np.clip(submodel_test_real_only[is_real_test_S], eps, 1 - eps)),
    ])
    test_blend[is_real_test_S] = lr_full.predict_proba(Xm_te)[:, 1]

    # Pre-submit ρ vs R15 .npy
    rho_test_vs_r15 = float(spearmanr(test_blend, r15_test_sorted).statistic)
    print(f"  Pre-submit ρ_test vs R15 PRIMARY (.npy): {rho_test_vs_r15:.6f}",
          flush=True)
    if rho_test_vs_r15 >= 0.9999:
        verdict = "TIE_ZONE"
    elif rho_test_vs_r15 < 0.999:
        verdict = "REGRESSION_RISK"
    else:
        verdict = "OK"
    print(f"  Band verdict: {verdict}", flush=True)

    # Save artifacts in original train.csv order
    order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
    order_back_test = np.array([test_id_to_sorted_pos[t]
                                 for t in test_orig_ids])
    np.save(ART / "oof_R16_real_f1_specialist_strat.npy",
            blend_oof[order_back_train].astype(np.float32))
    np.save(ART / "test_R16_real_f1_specialist_strat.npy",
            test_blend[order_back_test].astype(np.float32))
    np.save(ART / "oof_R16_submodel_real_only.npy",
            submodel_oof_real_only[order_back_train].astype(np.float32))
    np.save(ART / "test_R16_submodel_real_only.npy",
            submodel_test_real_only[order_back_test].astype(np.float32))

    # Build submission CSV
    SUBS = Path("submissions")
    SUBS.mkdir(exist_ok=True)
    sub_df = pd.DataFrame({
        "id": test_orig_ids,
        "PitNextLap": test_blend[order_back_test],
    })
    sub_csv = SUBS / "R16_real_f1_specialist_lrmeta.csv"
    sub_df.to_csv(sub_csv, index=False)
    print(f"  Wrote {sub_csv}", flush=True)

    summary = dict(
        round="R16_Phase3a_real_f1_specialist",
        n_train_real=int(real_mask_S.sum()),
        n_test_real=int(is_real_test_S.sum()),
        auc_r15_baseline=float(auc_r15_baseline),
        auc_combined_replace=float(auc_combined_replace),
        auc_combined_lrmeta=float(auc_blend),
        delta_replace_bp=float(1e4 * (auc_combined_replace - auc_r15_baseline)),
        delta_lrmeta_bp=float(1e4 * (auc_blend - auc_r15_baseline)),
        auc_real_r15=float(auc_real_r15),
        auc_real_submodel=float(auc_real_sub),
        slice_delta_bp=float(1e4 * (auc_real_sub - auc_real_r15)),
        auc_synth_r15=float(auc_synth_r15),
        rho_test_vs_r15=rho_test_vs_r15,
        verdict_band=verdict,
        fold_metrics=fold_metrics,
        lr_meta_coefs=coef_log,
        wall_total_s=time.time() - t0,
    )
    out_json = Path("audit/2026-05-20-round-16-real-f1-specialist.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
