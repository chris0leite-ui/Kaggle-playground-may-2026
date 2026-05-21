"""scripts/lr_v2_phaseA_leaf_probe.py — Phase A of "many-LRs-on-GPU" pivot.

Kill-switch probe for the GPU sweep planned in
/root/.claude/plans/be-really-ambitious-and-hashed-noodle.md.

Builds ONE L2 logistic regression over:
    [cb_v4 per-fold leaf one-hot] + [11 raw numerics, std-scaled] + [6 TEs]

The base is gated at K=19 via the existing Path-B harness
(`scripts/build_K13_pathb_multiseg.py --extra-bases ... lr_phaseA_leafprobe`).

Decision rule (post-gate):
- K=19 OOF Δ vs K=18 PRIMARY ≥ +0.30 bp → fund Phase B GPU sweep.
- Δ ∈ [0, +0.30)            → narrow Phase B (drop per-segment arm).
- Δ < 0                     → kill the pivot, redirect to R22/R5d.

BOTE (Rule 16):
  Q1 prediction:  Δ K=19 vs K=18 OOF in [-0.05, +1.5] bp; median +0.4 bp.
  Q2 cost:        ~75 min CPU local (CB fit ~5 min × 5 folds + LR ~10 min × 5).
  Q3 metric:      binary CE on PitNextLap → AUC eval. Matches K=18 LR-meta. PASS.
  Q4 decision:    Δ ≥ +0.30 bp → fund Phase B; else narrow/kill.
  Q5 failure:     ρ(probe, K18) > 0.97 means orthogonality lever absent.
  Q6 (forced):    training objective == row-AUC metric class? YES (log-loss → AUC).

Fold safety (Rule 24):
- CB fit per-fold on ti rows only (uses same `fold_safe_te_for_fold` as p1_single_cb).
- Leaf dict enumerated on ti only; va/test map through with OOV bucket.
- LR fit on sparse CSR built per-fold; standardizer fit on ti only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
from scipy.sparse import csr_matrix, hstack as sp_hstack
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "scripts")
from p1_features import (
    TE_CONFIGS, apply_fs_a, cv_target_encode, feature_columns_for_lgbm,
    fit_fs_a, make_features_static,
)
from p1_single_cb import cb_params, fold_safe_te_for_fold

ART = Path("scripts/artifacts")
DATA = Path("data")
AUDIT = Path("audit")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

# Probe-quality CB (production cb_v4 uses iters=8000, depth=10).
# Sized for 15GB RAM budget: 800 trees × depth=8 (256 leaves cap/tree).
# After min_freq=200 pruning, expect ~60-80 kept leaves/tree → ~50-65k cols.
PROBE_ITERS = 800
PROBE_DEPTH = 8

# Raw numerics that the LR gets as smooth-direction input (standardized).
# Mirrors NUM_COLS in scripts/archive/lr_research/lr_torch_gpu.py:42.
LR_NUM_COLS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop", "Year",
]

# Frequency-pruning floor for leaf dict: drop leaves whose ti-freq < this.
# 200 ≈ 0.057% of 351k ti rows; bounds matrix width below ~65k.
LEAF_MIN_FREQ = 200


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def build_leaf_lookup(leaves_ti: np.ndarray, max_leaf_id: int,
                       min_freq: int = LEAF_MIN_FREQ):
    """Per-tree leaf dict + a single vectorized lookup table.

    Returns:
      tree_lookup: (n_trees, max_leaf_id+1) int32 array.
                   tree_lookup[t, leaf_id] = local col idx in tree t
                   (OOV bucket = n_kept[t] for unknown / pruned leaves).
      offsets:     (n_trees+1,) int32 array. Tree t's cols occupy
                   [offsets[t], offsets[t+1]); offsets[-1] = total cols.
      dicts:       list[dict] (for cache compatibility with Phase B).
    """
    n_rows, n_trees = leaves_ti.shape
    tree_lookup = np.zeros((n_trees, max_leaf_id + 1), dtype=np.int32)
    widths = np.empty(n_trees, dtype=np.int64)
    dicts = []
    for t in range(n_trees):
        col = leaves_ti[:, t]
        uniq, cnt = np.unique(col, return_counts=True)
        kept = uniq[cnt >= min_freq]
        oov = len(kept)  # OOV bucket idx in local coords
        tree_lookup[t, :] = oov  # default everywhere → OOV
        for i, lv in enumerate(kept):
            tree_lookup[t, int(lv)] = i
        dicts.append({int(lv): i for i, lv in enumerate(kept)})
        widths[t] = oov + 1  # +1 for OOV bucket
    offsets = np.concatenate(
        [[0], np.cumsum(widths)]).astype(np.int32)
    return tree_lookup, offsets, dicts


def leaves_to_csr(leaves: np.ndarray, tree_lookup: np.ndarray,
                  offsets: np.ndarray) -> csr_matrix:
    """Vectorized (n_rows, n_trees) leaf-id → (n_rows, total_cols) CSR.

    Each row has exactly n_trees nonzeros (one per tree). No Python
    per-row loops — only an O(n_trees) outer loop with O(n_rows) numpy
    fancy-indexing inside. Uses int32 indices and indptr to halve
    memory vs scipy's default int64.
    """
    n_rows, n_trees = leaves.shape
    total_cols = int(offsets[-1])
    # indices: row-major laid out, n_trees nonzeros per row
    indices = np.empty(n_rows * n_trees, dtype=np.int32)
    for t in range(n_trees):
        leaf_col = leaves[:, t]  # uint32 view, no copy
        local = tree_lookup[t][leaf_col]  # n_rows int32, vectorized lookup
        indices[t::n_trees] = offsets[t] + local
    data = np.ones(n_rows * n_trees, dtype=np.float32)
    indptr = np.arange(0, (n_rows + 1) * n_trees,
                       n_trees, dtype=np.int32)
    return csr_matrix((data, indices, indptr),
                      shape=(n_rows, total_cols))


def assemble_X(leaves_csr, num_dense, te_dense):
    """Sparse hstack: [leaf_one_hot | scaled_numerics | TEs]."""
    blocks = [leaves_csr]
    if num_dense is not None:
        blocks.append(csr_matrix(num_dense))
    if te_dense is not None:
        blocks.append(csr_matrix(te_dense))
    return sp_hstack(blocks, format="csr")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Fold-0 only for sanity (~15 min wall).")
    ap.add_argument("--name", default="lr_phaseA_leafprobe")
    ap.add_argument("--no-leaf-cache", action="store_true",
                    help="Don't dump per-fold leaf maps (skip Phase B prep).")
    args = ap.parse_args()
    t0_total = time.time()

    print(f"=== Phase A leaf+LR probe | name={args.name} | "
          f"iters={PROBE_ITERS} depth={PROBE_DEPTH} | smoke={args.smoke} ===",
          flush=True)

    # 1) Load + featurize (LABEL-INDEPENDENT first, mirrors p1_single_cb.py)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    # Canonical feature list via sample fold
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    sample_fs_a = fit_fs_a(train_S.iloc[fold_list[0][0]])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    cb_feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in cb_feats and c not in cat_cols:
            cat_cols.append(c)
    cb_feats = cb_feats + [n for _, _, n in TE_CONFIGS]
    print(f"  cb feats: {len(cb_feats)}  cat: {len(cat_cols)}", flush=True)
    print(f"  LR num cols: {len(LR_NUM_COLS)}  TE cols: {len(TE_CONFIGS)}",
          flush=True)

    n_train = len(y)
    n_test = len(test_S)
    assert n_train == 439140, f"unexpected train size {n_train}"
    assert n_test == 188165, f"unexpected test size {n_test}"

    oof = np.zeros(n_train, dtype=np.float64)
    test_pred = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []
    fold_walls = []
    fold_diag = []

    n_eff_folds = 1 if args.smoke else N_FOLDS

    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds]):
        t0 = time.time()
        print(f"\n  --- Fold {fold+1}/{n_eff_folds} | ti={len(ti)} va={len(vi)} ---",
              flush=True)

        # 2a) Per-fold FS_A + TE
        fs_a = fit_fs_a(train_S.iloc[ti])
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)
        y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
        fold_safe_te_for_fold(train_ti, train_va, test_fold,
                              y_ti, fold, N_FOLDS)

        # 2b) Assemble CB feature matrices (same as p1_single_cb.py)
        X_tr = train_ti.reindex(columns=cb_feats, fill_value=0).copy()
        X_va = train_va.reindex(columns=cb_feats, fill_value=0).copy()
        X_te = test_fold.reindex(columns=cb_feats, fill_value=0).copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("int32")
            X_va[c] = X_va[c].astype("int32")
            X_te[c] = X_te[c].astype("int32")
        num_cb = [c for c in cb_feats if c not in cat_cols]
        for X in [X_tr, X_va, X_te]:
            X[num_cb] = X[num_cb].fillna(0).astype(np.float32)
        cat_idx = [cb_feats.index(c) for c in cat_cols]

        # 2c) Fit probe-quality CB on ti
        t_cb = time.time()
        params = cb_params(use_gpu=False, max_iters=PROBE_ITERS,
                           seed=SEED, depth=PROBE_DEPTH)
        m = cb.CatBoostClassifier(**params)
        m.fit(X_tr, train_ti[TARGET].astype(int).values,
              eval_set=(X_va, train_va[TARGET].astype(int).values),
              cat_features=cat_idx, use_best_model=True)
        n_trees_used = int(m.tree_count_)
        cb_va_proba = m.predict_proba(X_va)[:, 1]
        cb_va_auc = float(roc_auc_score(
            train_va[TARGET].astype(int).values, cb_va_proba))
        print(f"    CB fit: {n_trees_used} trees | "
              f"AUC_va={cb_va_auc:.5f} | wall={time.time()-t_cb:.1f}s",
              flush=True)

        # 2d) Extract leaves (ti, va, test). Keep uint32 — no int64 cast.
        t_lf = time.time()
        leaves_ti = m.calc_leaf_indexes(X_tr)
        leaves_va = m.calc_leaf_indexes(X_va)
        leaves_te = m.calc_leaf_indexes(X_te)
        max_leaf_id = int(max(leaves_ti.max(),
                               leaves_va.max(),
                               leaves_te.max()))
        print(f"    leaves: ti{leaves_ti.shape} va{leaves_va.shape} "
              f"te{leaves_te.shape} max_id={max_leaf_id} "
              f"dtype={leaves_ti.dtype} | wall={time.time()-t_lf:.1f}s",
              flush=True)

        # Free CB model + X dataframes immediately (each ~1-2 GB)
        del m, X_tr, X_va, X_te
        import gc; gc.collect()

        # 2e) Build vectorized per-tree leaf lookup on ti only
        tree_lookup, offsets, dicts = build_leaf_lookup(
            leaves_ti, max_leaf_id=max_leaf_id, min_freq=LEAF_MIN_FREQ)
        total_leaf_cols = int(offsets[-1])
        avg_widths = float(np.diff(offsets).mean())
        print(f"    leaf dict: {total_leaf_cols} cols  "
              f"(avg {avg_widths:.1f} kept-leaves/tree)", flush=True)

        # 2f) Encode leaves → CSR (vectorized, int32 indices)
        t_enc = time.time()
        Xtr_leaf = leaves_to_csr(leaves_ti, tree_lookup, offsets)
        del leaves_ti; gc.collect()
        Xva_leaf = leaves_to_csr(leaves_va, tree_lookup, offsets)
        del leaves_va; gc.collect()
        Xte_leaf = leaves_to_csr(leaves_te, tree_lookup, offsets)
        del leaves_te; gc.collect()
        print(f"    leaf CSR encode: Xtr.nnz={Xtr_leaf.nnz} "
              f"({Xtr_leaf.data.nbytes / 1e9:.2f} GB data) "
              f"wall={time.time()-t_enc:.1f}s", flush=True)

        # 2g) Dense block: standardized numerics (fit on ti only) + TEs
        num_ti = train_ti[LR_NUM_COLS].fillna(0).astype(np.float32).values
        num_va = train_va[LR_NUM_COLS].fillna(0).astype(np.float32).values
        num_te = test_fold[LR_NUM_COLS].fillna(0).astype(np.float32).values
        scl = StandardScaler().fit(num_ti)
        num_ti = scl.transform(num_ti).astype(np.float32, copy=False)
        num_va = scl.transform(num_va).astype(np.float32, copy=False)
        num_te = scl.transform(num_te).astype(np.float32, copy=False)
        te_cols = [n for _, _, n in TE_CONFIGS]
        te_ti = train_ti[te_cols].fillna(0).astype(np.float32).values
        te_va = train_va[te_cols].fillna(0).astype(np.float32).values
        te_te = test_fold[te_cols].fillna(0).astype(np.float32).values

        # Free train_ti/va/test_fold dataframes — we have what we need
        y_tr_arr = train_ti[TARGET].astype(int).values
        y_va_arr = train_va[TARGET].astype(int).values
        del train_ti, train_va, test_fold; gc.collect()

        Xtr = assemble_X(Xtr_leaf, num_ti, te_ti)
        del Xtr_leaf, num_ti, te_ti; gc.collect()
        Xva = assemble_X(Xva_leaf, num_va, te_va)
        del Xva_leaf, num_va, te_va; gc.collect()
        Xte = assemble_X(Xte_leaf, num_te, te_te)
        del Xte_leaf, num_te, te_te; gc.collect()
        print(f"    full sparse X: {Xtr.shape}  "
              f"nnz={Xtr.nnz} ({Xtr.data.nbytes/1e9:.2f} GB)", flush=True)

        # 2h) Fit L2 LR with lbfgs. Smoke v3 max_iter=200 hit convergence
        # warning at AUC 0.899; bump to 500 for convergence headroom.
        # Standalone AUC is informational — the K=19 gate is the real
        # test (R17 has standalone 0.943 but ρ_K18=0.745, and pulls
        # weight in the pool — orthogonality > standalone strength).
        t_lr = time.time()
        lr = LogisticRegression(C=10.0, max_iter=500, solver="lbfgs",
                                penalty="l2", random_state=SEED)
        lr.fit(Xtr, y_tr_arr)
        lr_va_proba = lr.predict_proba(Xva)[:, 1]
        lr_te_proba = lr.predict_proba(Xte)[:, 1]
        lr_va_auc = float(roc_auc_score(y_va_arr, lr_va_proba))
        print(f"    LR fit (lbfgs C=10 max_iter=500): "
              f"AUC_va={lr_va_auc:.5f} n_iter={lr.n_iter_} | "
              f"wall={time.time()-t_lr:.1f}s", flush=True)

        # 2i) Stash OOF + accumulate test
        # train_S is sorted; the StratifiedKFold was built on its post-sort
        # indices, so ti/vi already index into train_S rows. We'll do
        # sort_back at the end (same convention as p1_single_cb.py:466).
        oof[vi] = lr_va_proba
        test_pred += lr_te_proba / n_eff_folds

        # 2j) Cache per-fold leaf dict for Phase B re-use
        if not args.no_leaf_cache:
            cache = dict(
                dicts=dicts, offsets=offsets, tree_lookup=tree_lookup,
                max_leaf_id=max_leaf_id,
                cb_iters=n_trees_used, cb_depth=PROBE_DEPTH,
                cb_seed=SEED,
                ti_idx_in_train_S=ti.astype(np.int64),
                va_idx_in_train_S=vi.astype(np.int64),
                lr_num_cols=LR_NUM_COLS,
                te_cols=te_cols,
                scaler_mean=scl.mean_.tolist(),
                scaler_scale=scl.scale_.tolist(),
            )
            cache_path = ART / f"lr_phaseA_leafmap_fold{fold}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            print(f"    cached leaf-map → {cache_path}", flush=True)

        fold_aucs.append(lr_va_auc)
        fold_walls.append(time.time() - t0)
        fold_diag.append(dict(
            fold=fold, cb_va_auc=cb_va_auc, lr_va_auc=lr_va_auc,
            n_trees=n_trees_used, total_leaf_cols=total_leaf_cols,
            X_shape=list(Xtr.shape), X_nnz=int(Xtr.nnz),
            wall_s=time.time() - t0,
        ))
        print(f"    Fold {fold+1}: lr_va_auc={lr_va_auc:.5f}  "
              f"wall={time.time()-t0:.1f}s", flush=True)

        # Free big objects before next fold
        del Xtr, Xva, Xte, lr, tree_lookup, dicts
        gc.collect()

    # 3) Smoke mode: stop here
    if args.smoke:
        print(f"\n  SMOKE fold-0 LR AUC: {fold_aucs[0]:.5f}  "
              f"total wall={time.time()-t0_total:.1f}s", flush=True)
        return

    # 4) Map back to original train.csv id order (same as p1_single_cb.py:465-472)
    order = train_S["id"].values
    sort_back = np.argsort(order)
    oof_aligned = oof[sort_back]
    order_te = test_S["id"].values
    id_to_pos = {tid: i for i, tid in enumerate(order_te)}
    orig_te = pd.read_csv(DATA / "test.csv", usecols=[ID_COL])[ID_COL].values
    test_aligned = np.array([test_pred[id_to_pos[t]] for t in orig_te])

    # Hard sanity check
    assert len(oof_aligned) == 439140, f"OOF length wrong: {len(oof_aligned)}"
    assert len(test_aligned) == 188165, f"test length wrong: {len(test_aligned)}"

    # 5) Save in project convention: oof_{name}_strat.npy as [1-p, p] float64
    oof_path = ART / f"oof_{args.name}_strat.npy"
    test_path = ART / f"test_{args.name}_strat.npy"
    np.save(oof_path, np.column_stack(
        [1 - oof_aligned, oof_aligned]).astype(np.float64))
    np.save(test_path, np.column_stack(
        [1 - test_aligned, test_aligned]).astype(np.float64))
    print(f"\n  → {oof_path}", flush=True)
    print(f"  → {test_path}", flush=True)

    # 6) Standalone AUC + ρ vs K=18 PRIMARY
    y_orig = pd.read_csv(DATA / "train.csv")[TARGET].values.astype(int)
    standalone_auc = float(roc_auc_score(y_orig, oof_aligned))

    k18_oof = np.load(ART / "oof_K18_pathb_driverclass_stint_tau100000.npy")
    k18_test = np.load(ART / "test_K18_pathb_driverclass_stint_tau100000.npy")
    k18_oof_p = k18_oof[:, 1] if k18_oof.ndim == 2 else k18_oof.ravel()
    k18_test_p = k18_test[:, 1] if k18_test.ndim == 2 else k18_test.ravel()
    rho_oof = float(spearmanr(oof_aligned, k18_oof_p)[0])
    rho_test = float(spearmanr(test_aligned, k18_test_p)[0])
    k18_auc = float(roc_auc_score(y_orig, k18_oof_p))

    print(f"\n  === Phase A standalone results ===", flush=True)
    print(f"  standalone OOF AUC : {standalone_auc:.5f}", flush=True)
    print(f"  K=18 PRIMARY AUC   : {k18_auc:.5f}", flush=True)
    print(f"  Δ vs K=18 standalone: {(standalone_auc - k18_auc)*1e4:+.3f} bp",
          flush=True)
    print(f"  ρ_OOF  vs K=18     : {rho_oof:.6f}", flush=True)
    print(f"  ρ_test vs K=18     : {rho_test:.6f}", flush=True)
    print(f"  per-fold AUCs      : {[f'{a:.5f}' for a in fold_aucs]}",
          flush=True)
    print(f"  fold-std           : {np.std(fold_aucs):.5f}", flush=True)
    print(f"  total wall         : {(time.time()-t0_total)/60:.1f} min",
          flush=True)

    # 7) Results JSON + audit decision log entry (R19)
    AUDIT.mkdir(exist_ok=True)
    results = dict(
        name=args.name,
        standalone_oof_auc=standalone_auc,
        k18_oof_auc=k18_auc,
        delta_standalone_bp=(standalone_auc - k18_auc) * 1e4,
        rho_oof_vs_k18=rho_oof,
        rho_test_vs_k18=rho_test,
        fold_aucs=fold_aucs,
        fold_walls=fold_walls,
        fold_std=float(np.std(fold_aucs)),
        total_wall_s=time.time() - t0_total,
        probe_iters=PROBE_ITERS,
        probe_depth=PROBE_DEPTH,
        leaf_min_freq=LEAF_MIN_FREQ,
        lr_num_cols=LR_NUM_COLS,
        te_cols=[n for _, _, n in TE_CONFIGS],
        fold_diag=fold_diag,
    )
    (ART / f"{args.name}_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"  → {ART}/{args.name}_results.json", flush=True)

    # Decisions ledger (R19)
    decisions_path = AUDIT / "decisions.jsonl"
    decision_entry = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        script="lr_v2_phaseA_leaf_probe",
        bote_predicted_bp_range=[-0.05, 1.5],
        bote_median_bp=0.4,
        cost_min=time.time() / 60,  # placeholder; updated below
        standalone_auc=standalone_auc,
        rho_test_vs_k18=rho_test,
        q6_metric_align=True,
        next_step="run scripts/build_K13_pathb_multiseg.py "
                  f"--extra-bases R12_cb_horizon R13_cb_stint_completion "
                  f"R14_tabm R15_xendcg_per_seg R17_listwise_race_lap {args.name} "
                  "--segs driverclass_stint --tau 100000",
    )
    decision_entry["cost_min"] = (time.time() - t0_total) / 60
    with open(decisions_path, "a") as f:
        f.write(json.dumps(decision_entry) + "\n")
    print(f"  → decision logged to {decisions_path}", flush=True)


if __name__ == "__main__":
    main()
