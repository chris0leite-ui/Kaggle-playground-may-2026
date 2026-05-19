"""scripts/probe_r12_cb_horizon.py — Round 12 Phase 2: CatBoost on LapsUntilPit.

Train CatBoost to predict `log(LapsUntilPit + 1)` (capped at LAPS_CAP=30
per `scripts/b_laps_until_pit.py`). Converts to pit-likelihood proxy
`1 / (1 + max(expm1(pred), 0))` — closer pit ⇒ higher proxy ∈ (0, 1].

Mechanism: different TARGET from the binary K=13 bases. The gradient
signal trains the model to localize WHERE in time the next pit is,
not WHETHER this lap precedes a pit. Different inductive bias ⇒
plausibly different ranking direction ⇒ G3 ρ-orthogonality.

Per the plan-agent BOTE: P(G2-clear) ≈ 0.25, E[Δ vs R7.1] ≈ +0.10 bp,
wall ~1 h CPU.

Reuses:
- `scripts/b_laps_until_pit.py::build_laps_until_pit` for target.
- cb_v4 FE pipeline (`p1_features.py`) — same as cb_resid.

Fold-safety: **PER-FOLD STRICT target derivation.** For each fold k,
the LapsUntilPit target is computed using ONLY rows in fold k's train
set (i.e., not in val). Rule 24 violation in the original
`b_laps_until_pit.py` (which derives target on FULL train) is
documented in mechanism-ledger lines 88-100 % collapse under strict
audit (e.g., inv-laps-until-pit OOF +1.899 bp → strict +0.234 bp;
pit-horizon +3.191 bp → +0.302 bp; reverse-cumulative-pits +4.867 bp
→ -0.005 bp). Strict construction guarantees train targets depend
only on train labels.

Output: pit-likelihood proxy in (eps, 1-eps) saved for Path-B K=14 add.

Usage:
  python scripts/probe_r12_cb_horizon.py [--smoke] [--max-rounds 3000]
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
from b_laps_until_pit import build_laps_until_pit

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

REF_OOF = ART / "oof_K13_pathb_driverclass_stint_tau100000.npy"
CB_V4_OOF = ART / "oof_p1_single_cb_v4_gpu_strat.npy"


def cb_horizon_params(max_iters: int, seed: int, depth: int = 8) -> dict:
    """CatBoost regression on log(LapsUntilPit+1). RMSE loss."""
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


def proxy_from_log_laps(pred_log_laps: np.ndarray) -> np.ndarray:
    """Map predicted log(laps+1) to pit-likelihood proxy in (0, 1]."""
    laps = np.maximum(np.expm1(pred_log_laps), 0.0)
    return 1.0 / (1.0 + laps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("== R12-2: cb_horizon (CatBoost regression on log(LapsUntilPit+1)) ==",
          flush=True)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    y_all = train[TARGET].astype(int).values

    # Diagnostic: full-train target stats (for reference only; NOT used
    # for training -- per-fold strict targets computed inside the loop).
    y_log_lup_full = build_laps_until_pit(train).astype(np.float32)
    print(f"  log(LapsUntilPit+1) full-train (ref): mean "
          f"{y_log_lup_full.mean():.3f}  std {y_log_lup_full.std():.3f}",
          flush=True)

    if args.smoke:
        idx = np.random.default_rng(SEED).choice(len(train), 50_000,
                                                  replace=False)
        train = train.iloc[idx].reset_index(drop=True)
        y_all = train[TARGET].astype(int).values
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
    id_to_orig_pos = {tid: i for i, tid in enumerate(orig_train_ids)}
    id_to_sorted_pos = {tid: i for i, tid in enumerate(sorted_ids)}
    test_sorted_ids = test_S[ID_COL].values
    test_orig_ids = test[ID_COL].values
    test_id_to_sorted_pos = {tid: i for i, tid in enumerate(test_sorted_ids)}

    oof_log_pred = np.zeros(len(y), dtype=np.float64)
    test_log_pred = np.zeros(len(test_S), dtype=np.float64)
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

        # STRICT per-fold target derivation. Re-compute log(LapsUntilPit+1)
        # from train_S.iloc[ti] only. This is the fold-safe path Rule 24
        # requires; the full-train version (build_laps_until_pit(train))
        # leaks val-row PitNextLap into training-row targets and is the
        # documented 88-100 % collapse pattern from Day-15.
        ti_df = train_S.iloc[ti].reset_index(drop=True)
        vi_df = train_S.iloc[vi].reset_index(drop=True)
        t_ti = build_laps_until_pit(ti_df)
        t_va = build_laps_until_pit(vi_df)  # for ES only; val_labels-derived
        print(f"    target_ti: mean {t_ti.mean():.3f}  std {t_ti.std():.3f}  "
              f"target_va: mean {t_va.mean():.3f}", flush=True)

        params = cb_horizon_params(args.max_rounds, SEED, depth=args.depth)
        m = cb.CatBoostRegressor(**params)
        m.fit(X_tr, t_ti, eval_set=(X_va, t_va),
              cat_features=cat_idx, use_best_model=True)

        log_pred_va = m.predict(X_va)
        log_pred_te = m.predict(X_te)
        oof_log_pred[vi] = log_pred_va
        test_log_pred += log_pred_te / n_eff_folds

        y_va = train_va[TARGET].astype(int).values
        proxy_va = proxy_from_log_laps(log_pred_va)
        try:
            auc_proxy = float(roc_auc_score(y_va, proxy_va))
        except ValueError:
            auc_proxy = float("nan")

        wall = time.time() - t_f
        fold_metrics.append(dict(
            fold=fold, n_ti=int(len(ti)), n_va=int(len(vi)),
            iters=int(m.tree_count_), wall_s=float(wall),
            auc_proxy=auc_proxy,
            log_pred_range=[float(log_pred_va.min()),
                             float(log_pred_va.max())],
        ))
        print(f"    iters={m.tree_count_}  wall {wall:.0f}s  "
              f"AUC(proxy)={auc_proxy:.5f}", flush=True)

    if args.smoke:
        print(f"\n  SMOKE wall: {time.time()-t0:.0f}s; 5-fold projection "
              f"≈ {(time.time()-t0) * N_FOLDS:.0f}s", flush=True)
        return

    proxy_oof = proxy_from_log_laps(oof_log_pred)
    proxy_test = proxy_from_log_laps(test_log_pred)
    auc_full = float(roc_auc_score(y.values, proxy_oof))

    # ρ_OOF vs R7.1
    oof_R71 = np.load(REF_OOF)
    oof_R71_sorted = np.array([oof_R71[id_to_orig_pos[t]] for t in sorted_ids])
    rho_vs_R71, _ = spearmanr(proxy_oof, oof_R71_sorted)
    # ρ_OOF vs cb_v4
    rho_vs_cbv4 = None
    if CB_V4_OOF.exists():
        cb_v4 = np.load(CB_V4_OOF)
        cb_v4_p = cb_v4[:, 1] if cb_v4.ndim == 2 else cb_v4
        cb_v4_sorted = np.array([cb_v4_p[id_to_orig_pos[t]] for t in sorted_ids])
        rho_vs_cbv4, _ = spearmanr(proxy_oof, cb_v4_sorted)

    print(f"\n  Standalone OOF AUC (proxy = 1/(1+laps_pred)): {auc_full:.5f}",
          flush=True)
    print(f"  ρ_OOF proxy vs R7.1: {rho_vs_R71:.6f}", flush=True)
    if rho_vs_cbv4 is not None:
        print(f"  ρ_OOF proxy vs cb_v4: {rho_vs_cbv4:.6f}", flush=True)

    # Map back to original CSV order and save in (eps, 1-eps)
    eps = 1e-6
    proxy_oof = np.clip(proxy_oof, eps, 1 - eps)
    proxy_test = np.clip(proxy_test, eps, 1 - eps)
    order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
    order_back_test = np.array([test_id_to_sorted_pos[t]
                                 for t in test_orig_ids])
    np.save(ART / "oof_R12_cb_horizon_strat.npy",
            proxy_oof[order_back_train].astype(np.float32))
    np.save(ART / "test_R12_cb_horizon_strat.npy",
            proxy_test[order_back_test].astype(np.float32))
    # Raw log predictions for diagnostics
    np.save(ART / "oof_R12_cb_horizon_loglaps.npy",
            oof_log_pred[order_back_train].astype(np.float32))
    np.save(ART / "test_R12_cb_horizon_loglaps.npy",
            test_log_pred[order_back_test].astype(np.float32))
    print(f"  Saved oof_R12_cb_horizon_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R12_Phase2_cb_horizon",
        oof_auc_proxy=auc_full,
        rho_proxy_vs_R71=float(rho_vs_R71),
        rho_proxy_vs_cbv4=(float(rho_vs_cbv4)
                            if rho_vs_cbv4 is not None else None),
        fold_metrics=fold_metrics,
        wall_total_s=time.time() - t0,
        depth=args.depth,
        max_rounds=args.max_rounds,
    )
    out_json = Path("audit/2026-05-19-round-12-cb_horizon.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_json}", flush=True)
    print(f"  Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
