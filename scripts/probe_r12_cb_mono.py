"""scripts/probe_r12_cb_mono.py — Round 12 Phase 3: CatBoost with monotone constraints.

cb_v4 architecture/FE/hyperparams + `monotone_constraints` forcing the
decision surface to be monotone-increasing in the canonical "tyre wears
out → driver pits" indicators. Forces a structurally different model
even on the same feature set: the unconstrained CB can split arbitrarily
along these features; the constrained CB cannot.

Per the plan-agent BOTE: P(G2-clear) ≈ 0.15, E[Δ vs R7.1] ≈ +0.05 bp,
wall ~30 min CPU.

Monotone direction +1 (feature increases ⇒ pit-prob increases) applied
to the canonical tyre-end-of-life indicators that have unambiguous
domain semantics:
  - TyreLife              (older tyre)
  - tyre_overdue_norm     (TyreLife > 85 % of compound_max_life)
  - overdue_pit           (TyreLife > compound_avg_life from FS_A)
  - pit_imminent          (laps_until_stop ≤ 2)
  - pit_in_5              (laps_until_stop ≤ 5)
  - drv_hist_overdue      (TyreLife > driver's historical avg pit lap)

Reuses:
- cb_v4 FE pipeline (`p1_features.py`).
- cb_v4 hyperparams via `p1_single_cb.cb_params`.

Output: standard CatBoostClassifier predict_proba in (0, 1), saved
in original CSV order.

Usage:
  python scripts/probe_r12_cb_mono.py [--smoke] [--max-rounds 3000]
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
from p1_single_cb import fold_safe_te_for_fold, cb_params

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

REF_OOF = ART / "oof_K13_pathb_driverclass_stint_tau100000.npy"
CB_V4_OOF = ART / "oof_p1_single_cb_v4_gpu_strat.npy"

# Domain-knowledge monotone direction: feature value ↑ ⇒ P(pit) ↑
MONO_PLUS = [
    "TyreLife", "tyre_overdue_norm", "overdue_pit",
    "pit_imminent", "pit_in_5", "drv_hist_overdue",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("== R12-3: cb_mono (CatBoost binary + monotone constraints) ==",
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

    # Build monotone_constraints array (one entry per feature; 0 = unconstrained)
    mono_arr = [0] * len(feats)
    applied = []
    for i, fname in enumerate(feats):
        if fname in MONO_PLUS:
            mono_arr[i] = 1
            applied.append(fname)
    print(f"  monotone +1 applied to: {applied}", flush=True)
    if not applied:
        print(f"  WARN: no MONO_PLUS features found in feats list -- "
              f"falling back to plain cb_v4 (no constraint)", flush=True)

    sorted_ids = train_S[ID_COL].values
    orig_train_ids = train[ID_COL].values
    id_to_orig_pos = {tid: i for i, tid in enumerate(orig_train_ids)}
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

        params = cb_params(use_gpu=False, max_iters=args.max_rounds,
                           seed=SEED, depth=args.depth)
        # Override iter cap if requested; add mono
        params["iterations"] = args.max_rounds
        if any(m != 0 for m in mono_arr):
            params["monotone_constraints"] = mono_arr

        m = cb.CatBoostClassifier(**params)
        m.fit(X_tr, y_ti.values, eval_set=(X_va, train_va[TARGET].astype(int).values),
              cat_features=cat_idx, use_best_model=True)

        proba_va = m.predict_proba(X_va)[:, 1]
        proba_te = m.predict_proba(X_te)[:, 1]
        oof_pred[vi] = proba_va
        test_pred += proba_te / n_eff_folds

        y_va = train_va[TARGET].astype(int).values
        auc_va = float(roc_auc_score(y_va, proba_va))

        wall = time.time() - t_f
        fold_metrics.append(dict(
            fold=fold, n_ti=int(len(ti)), n_va=int(len(vi)),
            iters=int(m.tree_count_), wall_s=float(wall),
            auc_va=auc_va,
        ))
        print(f"    iters={m.tree_count_}  wall {wall:.0f}s  "
              f"AUC={auc_va:.5f}", flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold projection "
              f"≈ {(time.time()-t0) * N_FOLDS:.0f}s", flush=True)
        return

    auc_full = float(roc_auc_score(y.values, oof_pred))

    oof_R71 = np.load(REF_OOF)
    oof_R71_sorted = np.array([oof_R71[id_to_orig_pos[t]] for t in sorted_ids])
    rho_vs_R71, _ = spearmanr(oof_pred, oof_R71_sorted)

    rho_vs_cbv4 = None
    if CB_V4_OOF.exists():
        cb_v4 = np.load(CB_V4_OOF)
        cb_v4_p = cb_v4[:, 1] if cb_v4.ndim == 2 else cb_v4
        cb_v4_sorted = np.array([cb_v4_p[id_to_orig_pos[t]] for t in sorted_ids])
        rho_vs_cbv4, _ = spearmanr(oof_pred, cb_v4_sorted)

    print(f"\n  Standalone OOF AUC: {auc_full:.5f}", flush=True)
    print(f"  ρ_OOF vs R7.1 PRIMARY: {rho_vs_R71:.6f}", flush=True)
    if rho_vs_cbv4 is not None:
        print(f"  ρ_OOF vs cb_v4: {rho_vs_cbv4:.6f}", flush=True)

    order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
    order_back_test = np.array([test_id_to_sorted_pos[t]
                                 for t in test_orig_ids])
    np.save(ART / "oof_R12_cb_mono_strat.npy",
            oof_pred[order_back_train].astype(np.float32))
    np.save(ART / "test_R12_cb_mono_strat.npy",
            test_pred[order_back_test].astype(np.float32))
    print(f"  Saved oof_R12_cb_mono_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R12_Phase3_cb_mono",
        oof_auc=auc_full,
        rho_vs_R71=float(rho_vs_R71),
        rho_vs_cbv4=(float(rho_vs_cbv4)
                      if rho_vs_cbv4 is not None else None),
        fold_metrics=fold_metrics,
        wall_total_s=time.time() - t0,
        depth=args.depth,
        max_rounds=args.max_rounds,
        mono_features=applied,
    )
    out_json = Path("audit/2026-05-19-round-12-cb_mono.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
