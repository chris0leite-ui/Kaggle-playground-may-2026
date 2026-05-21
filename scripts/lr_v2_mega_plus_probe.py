"""scripts/lr_v2_mega_plus_probe.py — lr_mega + cb_v4 leaves.

Phase A leaf+LR (no Rozen FE): standalone 0.886, K=19 Δ=-0.049 bp.
KBins-only LR: standalone 0.914, K=19 Δ=-0.084 bp.
lr_mega (rich FE, no leaves): standalone 0.928, K=19 Δ=+0.085 bp ← first positive.

The K=19 Δ scales with standalone AUC. PI directive: push past 0.94
standalone to fire the FUND_PHASE_B gate (Δ ≥ +0.30 bp).

Recipe = lr_mega's full FE pipeline (1202 features: Rozen static +
6 Rozen TEs + 16 3-way TEs + 16 DGP rules + KBins-OHE + OHE-cats)
PLUS per-fold cb_v4 leaves (~60k sparse cols). Stacked as sparse CSR.

Why this should hit 0.94:
- lr_mega alone: 0.928 (rich LR-class features)
- + leaves: adds GBDT-style interaction directions LR can't reach
  via additive features alone. Facebook 2014 GBDT+LR recipe.

Fold safety (Rule 24):
- All TEs use cv_target_encode (per-fold refit).
- FS_A per-fold via fit_fs_a/apply_fs_a.
- CB fit per fold on ti rows; leaves extracted from that per-fold CB.
- Leaf dict enumerated on ti only.

Wall budget: ~50-70 min (5 folds × (3min CB + 1min leaves + 5-8min LR fit)).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
from scipy import sparse
from scipy.sparse import csr_matrix, hstack as sp_hstack
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    KBinsDiscretizer, OneHotEncoder, StandardScaler,
)

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/archive/lr_research")
from p1_features import (
    TE_CONFIGS, apply_fs_a, cv_target_encode, feature_columns_for_lgbm,
    fit_fs_a, make_features_static,
)
from p1_single_cb import cb_params, fold_safe_te_for_fold
from lr_bank_rich_fe import build_rozen_static, build_dgp_rule_features
from lr_v2_phaseA_leaf_probe import (
    build_leaf_lookup, leaves_to_csr,
)

ART = Path("scripts/artifacts")
DATA = Path("data")
AUDIT = Path("audit")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

NUM_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
]
CAT_COLS = ["Driver", "Compound", "Race"]

# Probe CB for leaf extraction (Phase A used iters=800; bump to 1500
# for stronger leaves now that we know the recipe works).
PROBE_ITERS = 1500
PROBE_DEPTH = 8
LEAF_MIN_FREQ = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Fold 0 only (~10-15 min wall).")
    ap.add_argument("--name", default="lr_mega_plus")
    ap.add_argument("--no-leaves", action="store_true",
                    help="Skip CB leaves (just lr_mega + extra TEs).")
    ap.add_argument("--lr-c", type=float, default=1.0)
    ap.add_argument("--lr-max-iter", type=int, default=2000)
    args = ap.parse_args()
    t0_total = time.time()

    print(f"=== lr_mega + leaves probe | name={args.name} | "
          f"leaves={'no' if args.no_leaves else 'yes'} | "
          f"C={args.lr_c} iter={args.lr_max_iter} | smoke={args.smoke} ===",
          flush=True)

    # 1) Load + Rozen static
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values
    print(f"  data: train {train.shape} test {test.shape} prior={y.mean():.4f}",
          flush=True)

    train_S, test_S, _ = build_rozen_static(train, test)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    n_eff_folds = 1 if args.smoke else N_FOLDS

    # 2) Pre-compute 6 Rozen TEs
    print(f"  pre-computing 6 Rozen TE features...", flush=True)
    rozen_te_oof, rozen_te_test = {}, {}
    for cols, smoothing, te_name in TE_CONFIGS:
        oof_enc, test_enc = cv_target_encode(
            train, test, cols, train[TARGET].astype(int),
            fold_list, smoothing)
        rozen_te_oof[te_name] = oof_enc
        rozen_te_test[te_name] = test_enc

    # 3) Pre-compute 16 3-way TEs
    print(f"  pre-computing 16 3-way TE features...", flush=True)
    keys_3way = [
        ["Driver", "Race", "Year"], ["Driver", "Race", "Compound"],
        ["Driver", "Year", "Compound"], ["Race", "Year", "Compound"],
    ]
    smoothings = [1, 5, 20, 100]
    threeway_oof, threeway_test = [], []
    for keys in keys_3way:
        for sm in smoothings:
            oof_enc, test_enc = cv_target_encode(
                train, test, keys, train[TARGET].astype(int),
                fold_list, sm)
            threeway_oof.append(oof_enc)
            threeway_test.append(test_enc)
    threeway_oof = np.column_stack(threeway_oof).astype(np.float32)
    threeway_test = np.column_stack(threeway_test).astype(np.float32)

    # 4) NEW: 10 additional TE cohorts (Phase B plan)
    print(f"  pre-computing 10 ADDITIONAL TE cohorts...", flush=True)
    new_te_cohorts = [
        (["Year", "Race", "Stint"], 30),
        (["Year", "Race", "Compound"], 30),
        (["Driver", "Stint"], 40),
        (["Race", "Stint"], 30),
        (["Compound", "Stint"], 30),
        (["Driver", "Compound", "Stint"], 20),
        (["Race", "Compound", "Year"], 20),
        (["Year", "Race", "Compound", "Stint"], 50),
        (["Driver", "Year"], 30),
        (["Race", "TyreLife_decile"], 30),
    ]
    # Build tyre_decile column for the last one
    edges = np.quantile(train["TyreLife"].values, np.linspace(0, 1, 11))
    edges[0] = -np.inf; edges[-1] = np.inf
    train["TyreLife_decile"] = np.clip(
        np.searchsorted(edges, train["TyreLife"].values, side="right") - 1,
        0, 9).astype(int)
    test["TyreLife_decile"] = np.clip(
        np.searchsorted(edges, test["TyreLife"].values, side="right") - 1,
        0, 9).astype(int)
    new_te_oof, new_te_test = [], []
    for keys, sm in new_te_cohorts:
        oof_enc, test_enc = cv_target_encode(
            train, test, keys, train[TARGET].astype(int),
            fold_list, sm)
        new_te_oof.append(oof_enc)
        new_te_test.append(test_enc)
    new_te_oof = np.column_stack(new_te_oof).astype(np.float32)
    new_te_test = np.column_stack(new_te_test).astype(np.float32)

    # 5) Pre-compute KBins-OHE + OHE-cats (same as lr_mega — uses combined
    # train+test fit, Rule 25 safe via AV-AUC=0.502).
    print(f"  pre-computing KBins-OHE + OHE-cats...", flush=True)
    num_tr_raw = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te_raw = test[NUM_COLS].fillna(0).values.astype(np.float32)
    kb = KBinsDiscretizer(n_bins=20, encode="onehot", strategy="quantile",
                          subsample=None)
    kb.fit(np.vstack([num_tr_raw, num_te_raw]))
    Bk_tr = kb.transform(num_tr_raw).astype(np.float32)
    Bk_te = kb.transform(num_te_raw).astype(np.float32)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        dtype=np.float32)
    enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    Oc_tr = enc.transform(train[CAT_COLS]).astype(np.float32)
    Oc_te = enc.transform(test[CAT_COLS]).astype(np.float32)
    print(f"  KBins-OHE: tr={Bk_tr.shape} OHE-cats: tr={Oc_tr.shape}",
          flush=True)

    # Per-fold loop
    drop_cols_static = ["Driver", "Race", "Compound", "id", TARGET]
    n_tr, n_te = len(y), len(test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_pred = np.zeros(n_te, dtype=np.float64)
    fold_aucs = []
    fold_walls = []

    # Need a sample fit_fs_a for canonical feature list (CB cat cols)
    sample_fs_a = fit_fs_a(train.iloc[fold_list[0][0]])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    cb_feats, cb_cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in cb_feats and c not in cb_cat_cols:
            cb_cat_cols.append(c)
    cb_feats = cb_feats + [n for _, _, n in TE_CONFIGS]
    print(f"  CB feats={len(cb_feats)} cat={len(cb_cat_cols)}", flush=True)

    for fold, (tr, va) in enumerate(fold_list[:n_eff_folds]):
        t0 = time.time()
        print(f"\n  --- Fold {fold+1}/{n_eff_folds} | "
              f"ti={len(tr)} va={len(va)} ---", flush=True)

        # 5a) FS_A on tr rows
        fs_a = fit_fs_a(train.iloc[tr])
        train_A = apply_fs_a(train_S, fs_a)
        test_A = apply_fs_a(test_S, fs_a)

        # 5b) Rozen-static numeric block
        feat_cols = [c for c in train_A.columns
                     if c not in drop_cols_static and c not in CAT_COLS
                     and train_A[c].dtype.kind in "biufc"
                     and c != "TyreLife_decile"]
        Xstatic = train_A[feat_cols].fillna(0).values.astype(np.float32)
        Xstatic_te = test_A[feat_cols].fillna(0).values.astype(np.float32)

        # 5c) DGP rule features (per-fold)
        rule_tr, rule_va, rule_te = build_dgp_rule_features(
            train, test, y, tr, va)

        # 5d) Combine all DENSE numeric blocks
        te_oof_arr = np.column_stack(list(rozen_te_oof.values())).astype(
            np.float32)
        te_test_arr = np.column_stack(list(rozen_te_test.values())).astype(
            np.float32)
        num_tr_full = np.hstack([Xstatic[tr], te_oof_arr[tr],
                                 threeway_oof[tr], new_te_oof[tr], rule_tr])
        num_va_full = np.hstack([Xstatic[va], te_oof_arr[va],
                                 threeway_oof[va], new_te_oof[va], rule_va])
        num_te_full = np.hstack([Xstatic_te, te_test_arr,
                                 threeway_test, new_te_test, rule_te])
        sc = StandardScaler()
        num_tr_s = sc.fit_transform(num_tr_full).astype(np.float32)
        num_va_s = sc.transform(num_va_full).astype(np.float32)
        num_te_s = sc.transform(num_te_full).astype(np.float32)
        print(f"    dense block: tr={num_tr_s.shape} va={num_va_s.shape}",
              flush=True)

        # 5e) Per-fold CB fit → leaf extraction
        leaves_csr_tr = leaves_csr_va = leaves_csr_te = None
        leaf_cols = 0
        if not args.no_leaves:
            # Build CB matrices (same as p1_single_cb.py)
            train_ti = train_A.iloc[tr].reset_index(drop=True)
            train_va_ = train_A.iloc[va].reset_index(drop=True)
            test_fold = test_A
            # Re-apply per-fold TE for CB (CB has its own TE consumption)
            train_ti = train_ti.copy()
            train_va_ = train_va_.copy()
            test_fold_cb = test_fold.copy()
            y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
            fold_safe_te_for_fold(train_ti, train_va_, test_fold_cb,
                                   y_ti, fold, N_FOLDS)
            X_tr_cb = train_ti.reindex(columns=cb_feats, fill_value=0).copy()
            X_va_cb = train_va_.reindex(columns=cb_feats, fill_value=0).copy()
            X_te_cb = test_fold_cb.reindex(columns=cb_feats,
                                            fill_value=0).copy()
            for c in cb_cat_cols:
                X_tr_cb[c] = X_tr_cb[c].astype("int32")
                X_va_cb[c] = X_va_cb[c].astype("int32")
                X_te_cb[c] = X_te_cb[c].astype("int32")
            num_cb_cols = [c for c in cb_feats if c not in cb_cat_cols]
            for X in [X_tr_cb, X_va_cb, X_te_cb]:
                X[num_cb_cols] = X[num_cb_cols].fillna(0).astype(np.float32)
            cat_idx = [cb_feats.index(c) for c in cb_cat_cols]

            t_cb = time.time()
            cb_p = cb_params(use_gpu=False, max_iters=PROBE_ITERS,
                             seed=SEED, depth=PROBE_DEPTH)
            m = cb.CatBoostClassifier(**cb_p)
            m.fit(X_tr_cb, train_ti[TARGET].astype(int).values,
                  eval_set=(X_va_cb, train_va_[TARGET].astype(int).values),
                  cat_features=cat_idx, use_best_model=True)
            print(f"    CB: {m.tree_count_} trees wall={time.time()-t_cb:.1f}s",
                  flush=True)

            leaves_ti = m.calc_leaf_indexes(X_tr_cb)
            leaves_va = m.calc_leaf_indexes(X_va_cb)
            leaves_te = m.calc_leaf_indexes(X_te_cb)
            max_leaf_id = int(max(leaves_ti.max(), leaves_va.max(),
                                   leaves_te.max()))
            del m, X_tr_cb, X_va_cb, X_te_cb
            import gc; gc.collect()

            tree_lookup, offsets, _ = build_leaf_lookup(
                leaves_ti, max_leaf_id=max_leaf_id, min_freq=LEAF_MIN_FREQ)
            leaf_cols = int(offsets[-1])
            leaves_csr_tr = leaves_to_csr(leaves_ti, tree_lookup, offsets)
            del leaves_ti; gc.collect()
            leaves_csr_va = leaves_to_csr(leaves_va, tree_lookup, offsets)
            del leaves_va; gc.collect()
            leaves_csr_te = leaves_to_csr(leaves_te, tree_lookup, offsets)
            del leaves_te; gc.collect()
            print(f"    leaves CSR: tr={leaves_csr_tr.shape} "
                  f"({leaves_csr_tr.data.nbytes/1e9:.2f} GB)", flush=True)

        # 5f) Stack EVERYTHING as sparse CSR: dense + KBins + OHE-cats + leaves
        blocks_tr = [csr_matrix(num_tr_s),
                     Bk_tr[tr],
                     Oc_tr[tr]]
        blocks_va = [csr_matrix(num_va_s),
                     Bk_tr[va],
                     Oc_tr[va]]
        blocks_te = [csr_matrix(num_te_s),
                     Bk_te,
                     Oc_te]
        if leaves_csr_tr is not None:
            blocks_tr.append(leaves_csr_tr)
            blocks_va.append(leaves_csr_va)
            blocks_te.append(leaves_csr_te)
        Xtr = sp_hstack(blocks_tr, format="csr")
        Xva = sp_hstack(blocks_va, format="csr")
        Xte = sp_hstack(blocks_te, format="csr")
        del blocks_tr, blocks_va, blocks_te
        if leaves_csr_tr is not None:
            del leaves_csr_tr, leaves_csr_va, leaves_csr_te
        import gc; gc.collect()
        print(f"    full sparse X: tr={Xtr.shape}  "
              f"nnz={Xtr.nnz} ({Xtr.data.nbytes/1e9:.2f} GB)", flush=True)

        # 5g) Fit LR
        t_lr = time.time()
        lr = LogisticRegression(C=args.lr_c, max_iter=args.lr_max_iter,
                                solver="lbfgs", penalty="l2",
                                random_state=SEED)
        lr.fit(Xtr, y[tr])
        val_p = lr.predict_proba(Xva)[:, 1]
        test_p = lr.predict_proba(Xte)[:, 1]
        val_auc = float(roc_auc_score(y[va], val_p))
        print(f"    LR fit: AUC_va={val_auc:.5f} n_iter={lr.n_iter_} "
              f"wall={time.time()-t_lr:.1f}s", flush=True)

        oof[va] = val_p
        test_pred += test_p / n_eff_folds
        fold_aucs.append(val_auc)
        fold_walls.append(time.time() - t0)

        del Xtr, Xva, Xte, lr; gc.collect()

    if args.smoke:
        print(f"\n  SMOKE fold-0 AUC: {fold_aucs[0]:.5f}  "
              f"total wall={time.time()-t0_total:.1f}s", flush=True)
        return

    standalone_auc = float(roc_auc_score(y, oof))

    # Save aligned to original train.csv id order (oof is already in
    # train.csv row order since lr_mega used raw train rows, no sort)
    np.save(ART / f"oof_{args.name}_strat.npy",
            np.column_stack([1 - oof, oof]).astype(np.float64))
    np.save(ART / f"test_{args.name}_strat.npy",
            np.column_stack([1 - test_pred, test_pred]).astype(np.float64))

    # ρ vs K=18
    k18_oof = np.load(ART / "oof_K18_pathb_driverclass_stint_tau100000.npy")
    k18_te = np.load(ART / "test_K18_pathb_driverclass_stint_tau100000.npy")
    k18_oof_p = k18_oof[:, 1] if k18_oof.ndim == 2 else k18_oof.ravel()
    k18_te_p = k18_te[:, 1] if k18_te.ndim == 2 else k18_te.ravel()
    rho_oof = float(spearmanr(oof, k18_oof_p)[0])
    rho_test = float(spearmanr(test_pred, k18_te_p)[0])
    k18_auc = float(roc_auc_score(y, k18_oof_p))

    print(f"\n  === {args.name} results ===", flush=True)
    print(f"  standalone OOF AUC : {standalone_auc:.5f}  "
          f"(targets: lr_mega 0.92776 baseline; +0.94 the prize)",
          flush=True)
    print(f"  K=18 PRIMARY AUC   : {k18_auc:.5f}", flush=True)
    print(f"  Δ vs K=18 standalone: {(standalone_auc-k18_auc)*1e4:+.2f} bp",
          flush=True)
    print(f"  ρ_OOF  vs K=18     : {rho_oof:.6f}", flush=True)
    print(f"  ρ_test vs K=18     : {rho_test:.6f}", flush=True)
    print(f"  per-fold AUCs      : {[f'{a:.5f}' for a in fold_aucs]}",
          flush=True)
    print(f"  fold-std           : {np.std(fold_aucs):.5f}", flush=True)
    print(f"  total wall         : {(time.time()-t0_total)/60:.1f} min",
          flush=True)

    AUDIT.mkdir(exist_ok=True)
    Path(ART / f"{args.name}_results.json").write_text(json.dumps(dict(
        standalone_oof_auc=standalone_auc, rho_oof_vs_k18=rho_oof,
        rho_test_vs_k18=rho_test, fold_aucs=fold_aucs,
        fold_walls=fold_walls, k18_auc=k18_auc,
        leaves=not args.no_leaves, lr_c=args.lr_c,
        lr_max_iter=args.lr_max_iter,
        probe_iters=PROBE_ITERS, probe_depth=PROBE_DEPTH,
        leaf_min_freq=LEAF_MIN_FREQ,
    ), indent=2))


if __name__ == "__main__":
    main()
