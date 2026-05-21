"""scripts/lr_v2_kbins_probe.py — LR-only Phase A pivot probe.

Phase A leaf+LR (lr_v2_phaseA_leaf_probe.py) got standalone OOF 0.886,
K=19 Δ vs K=18 PRIMARY = -0.049 bp. lr_v2_phaseA_diag.py V1 confirmed
the L2-LR's converged optimum on the mixed feature matrix is ~0.877
— the std-scaled numerics + sparse leaf-OHE block has bad Hessian
conditioning that even saga / lbfgs full-convergence can't overcome.

This script tests the cleaner LR recipe (matching lr_kbins20_ohe which
hit standalone OOF 0.920 per lr_bank_diagnostics.json):

  Features: KBins(n=20, quantile)-OHE on 11 numerics + 6 fold-safe TEs
            + frequency-encoded categoricals (Driver, Compound, Race)

That's all sparse + binary scale → no conditioning issues, lbfgs/saga
converges cleanly. No CB fit, no leaves — pure LR-class probe.

DECISION RULE (same as plan Phase A):
  K=19 Δ vs K=18 PRIMARY ≥ +0.30 bp ∧ ρ_test ≤ 0.97 → FUND_PHASE_B
                                                       (with leaves
                                                        added next)
  Δ ∈ [0, +0.30)  → NARROW_PHASE_B
  Δ < 0           → KILL_PIVOT (LR-class fully absorbed by K=18)

Wall: ~5-10 min (no CB). Fold safety per Rule 24: TE refit per fold.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

import sys
sys.path.insert(0, "scripts")
from p1_features import (
    TE_CONFIGS, apply_fs_a, fit_fs_a, make_features_static,
)
from p1_single_cb import fold_safe_te_for_fold

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
KBINS_N = 20


def freq_encode(s_train: pd.Series, s_va: pd.Series, s_test: pd.Series):
    counts = s_train.value_counts()
    return (s_train.map(counts).fillna(0).astype(np.float32).values.reshape(-1, 1),
            s_va.map(counts).fillna(0).astype(np.float32).values.reshape(-1, 1),
            s_test.map(counts).fillna(0).astype(np.float32).values.reshape(-1, 1))


def build_X(train_ti, train_va, test_fold, te_cols):
    """Return sparse CSR (Xtr, Xva, Xte) with KBins(20)-OHE numerics +
    TE cols + frequency encodings."""
    # KBins-OHE on numerics (fit on ti only — Rule 24 safe)
    num_ti = train_ti[NUM_COLS].fillna(0).astype(np.float32).values
    num_va = train_va[NUM_COLS].fillna(0).astype(np.float32).values
    num_te = test_fold[NUM_COLS].fillna(0).astype(np.float32).values
    kb = KBinsDiscretizer(n_bins=KBINS_N, strategy="quantile",
                          encode="onehot", subsample=None)
    Xtr_n = kb.fit_transform(num_ti)
    Xva_n = kb.transform(num_va)
    Xte_n = kb.transform(num_te)
    # TE cols → dense float, hstack
    Xtr_te = csr_matrix(train_ti[te_cols].fillna(0).astype(np.float32).values)
    Xva_te = csr_matrix(train_va[te_cols].fillna(0).astype(np.float32).values)
    Xte_te = csr_matrix(test_fold[te_cols].fillna(0).astype(np.float32).values)
    # Freq-encode Driver, Compound, Race (raw cats)
    fe_blocks_tr, fe_blocks_va, fe_blocks_te = [], [], []
    for c in ("Driver", "Compound", "Race"):
        a, b, t = freq_encode(train_ti[c], train_va[c], test_fold[c])
        fe_blocks_tr.append(csr_matrix(a))
        fe_blocks_va.append(csr_matrix(b))
        fe_blocks_te.append(csr_matrix(t))
    Xtr = sp_hstack([Xtr_n, Xtr_te] + fe_blocks_tr, format="csr")
    Xva = sp_hstack([Xva_n, Xva_te] + fe_blocks_va, format="csr")
    Xte = sp_hstack([Xte_n, Xte_te] + fe_blocks_te, format="csr")
    return Xtr, Xva, Xte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Fold 0 only (~2 min wall).")
    ap.add_argument("--name", default="lr_v2_kbins_probe")
    ap.add_argument("--c", type=float, default=1.0,
                    help="LR C (inverse reg).")
    args = ap.parse_args()
    t0_total = time.time()

    print(f"=== LR-only Phase A probe | name={args.name} | "
          f"KBins={KBINS_N} | C={args.c} | smoke={args.smoke} ===",
          flush=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    te_cols = [n for _, _, n in TE_CONFIGS]

    n_train = len(y)
    n_test = len(test_S)
    oof = np.zeros(n_train, dtype=np.float64)
    test_pred = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []
    fold_walls = []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds]):
        t0 = time.time()
        print(f"\n  --- Fold {fold+1}/{n_eff_folds} | ti={len(ti)} va={len(vi)} ---",
              flush=True)

        # Per-fold FS_A + TE (same as cb pipeline)
        fs_a = fit_fs_a(train_S.iloc[ti])
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)
        y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
        fold_safe_te_for_fold(train_ti, train_va, test_fold,
                              y_ti, fold, N_FOLDS)

        Xtr, Xva, Xte = build_X(train_ti, train_va, test_fold, te_cols)
        y_tr_arr = train_ti[TARGET].astype(int).values
        y_va_arr = train_va[TARGET].astype(int).values
        print(f"    X shape: {Xtr.shape}  nnz={Xtr.nnz}", flush=True)

        # Fit LR — sparse binary block (KBins-OHE) + small TE + freq.
        # max_iter=1000 to ensure convergence (smoke fold-0 hit 300 cap
        # at AUC 0.914; reference is 0.920).
        t_lr = time.time()
        lr = LogisticRegression(C=args.c, max_iter=1000, solver="lbfgs",
                                penalty="l2", random_state=SEED)
        lr.fit(Xtr, y_tr_arr)
        lr_va_proba = lr.predict_proba(Xva)[:, 1]
        lr_te_proba = lr.predict_proba(Xte)[:, 1]
        lr_va_auc = float(roc_auc_score(y_va_arr, lr_va_proba))
        print(f"    LR: AUC_va={lr_va_auc:.5f} n_iter={lr.n_iter_} "
              f"wall={time.time()-t_lr:.1f}s", flush=True)

        oof[vi] = lr_va_proba
        test_pred += lr_te_proba / n_eff_folds
        fold_aucs.append(lr_va_auc)
        fold_walls.append(time.time() - t0)

    if args.smoke:
        print(f"\n  SMOKE fold-0 LR AUC: {fold_aucs[0]:.5f}  "
              f"total wall={time.time()-t0_total:.1f}s", flush=True)
        return

    # sort_back to original train.csv id order (same as p1_single_cb.py)
    order = train_S["id"].values
    sort_back = np.argsort(order)
    oof_aligned = oof[sort_back]
    order_te = test_S["id"].values
    id_to_pos = {tid: i for i, tid in enumerate(order_te)}
    orig_te = pd.read_csv(DATA / "test.csv", usecols=[ID_COL])[ID_COL].values
    test_aligned = np.array([test_pred[id_to_pos[t]] for t in orig_te])

    assert len(oof_aligned) == 439140
    assert len(test_aligned) == 188165

    np.save(ART / f"oof_{args.name}_strat.npy",
            np.column_stack([1 - oof_aligned, oof_aligned]).astype(np.float64))
    np.save(ART / f"test_{args.name}_strat.npy",
            np.column_stack([1 - test_aligned, test_aligned]).astype(np.float64))

    y_orig = pd.read_csv(DATA / "train.csv")[TARGET].astype(int).values
    standalone_auc = float(roc_auc_score(y_orig, oof_aligned))

    k18_oof = np.load(ART / "oof_K18_pathb_driverclass_stint_tau100000.npy")
    k18_test = np.load(ART / "test_K18_pathb_driverclass_stint_tau100000.npy")
    k18_oof_p = k18_oof[:, 1] if k18_oof.ndim == 2 else k18_oof.ravel()
    k18_test_p = k18_test[:, 1] if k18_test.ndim == 2 else k18_test.ravel()
    rho_oof = float(spearmanr(oof_aligned, k18_oof_p)[0])
    rho_test = float(spearmanr(test_aligned, k18_test_p)[0])
    k18_auc = float(roc_auc_score(y_orig, k18_oof_p))

    print(f"\n  === LR-kbins probe results ===", flush=True)
    print(f"  standalone OOF AUC : {standalone_auc:.5f} "
          f"(reference: lr_kbins20_ohe = 0.920)", flush=True)
    print(f"  K=18 PRIMARY AUC   : {k18_auc:.5f}", flush=True)
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
        fold_walls=fold_walls, k18_auc=k18_auc, c=args.c,
    ), indent=2))


if __name__ == "__main__":
    main()
