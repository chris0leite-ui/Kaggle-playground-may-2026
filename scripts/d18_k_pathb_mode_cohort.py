"""d18 K — Path-B hier-meta with TyreLife mode-id × Compound as cohort axis.

Per friction `path-b-amp-only-fires-on-meta-arch-not-base-add`, base-adds
realise ~1.0× amp; meta-arch redesigns realise 6-11.6×. Replacing the
Compound × Stint cohort with **CTGAN's actual discrete latent** (mode-id
of TyreLife from G's BGMM) is a true meta-arch redesign on the cohort
axis, not just a base-add.

E5 c1 already failed with chain_total_ll_q5 cohort (continuous-derived
quintile). K uses mode-id (BGMM-fitted on orig) which is the cleaner
CTGAN-aligned discrete latent.

Cohort axes tested:
  K1  Compound × mode_TyreLife       (5 × 11 = 55 cells)
  K2  Compound × mode_LapTime_Delta  (5 × 11 = 55 cells)
  K3  mode_TyreLife × Stint          (11 × 6 = 66 cells)

Pool: K=28 = K=21 + d16 + d18 + G + F2 + F5 + H + J (the strongest pool).
τ-sweep {5k, 20k, 100k}.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from d18_path_b import (
    POOL_KEEP, TOP_3_D9, FM_AB,
    expand, fit_lr_aug, predict_aug, load_pos,
)

ART = Path("scripts/artifacts")
DATA_OUT = Path("data")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_S = 0.95073
MIN_ROWS, MAX_ITER = 1000, 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["k1", "k2", "k3"], required=True)
    args = ap.parse_args()
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # K=28 pool (drops weakest I=mode_collapse)
    extras = ["d16_orig_continuous_only", "d18_chain_decomp",
              "d18_g_mode_id", "d18_f2_constraint", "d18_f5_class_cond_gmm",
              "d18_h_mode_lookup", "d18_j_cond_vector"]
    pool_names = POOL_KEEP + TOP_3_D9 + FM_AB + extras
    print(f"K={len(pool_names)} bases  cohort={args.cohort}")

    base_oofs, base_tests = [], []
    for name in pool_names:
        oo, te = load_pos(name)
        base_oofs.append(oo); base_tests.append(te)
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    print(f"F shape oof={F_oof.shape}")
    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy")[:, 1].astype(np.float64)

    # Load mode-id features
    p_tr = DATA_OUT / "mode_id_features_train.parquet"
    p_te = DATA_OUT / "mode_id_features_test.parquet"
    if not (p_tr.exists() and p_te.exists()):
        print(f"  ERROR: mode-id parquets not found; run d18_g first")
        return
    tr_M = pd.read_parquet(p_tr)
    te_M = pd.read_parquet(p_te)

    cmps = sorted(set(train["Compound"].astype(str)) | set(test["Compound"].astype(str)))
    cm = {c: i for i, c in enumerate(cmps)}
    c_tr = train["Compound"].astype(str).map(cm).astype(int).values
    c_te = test["Compound"].astype(str).map(cm).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    s_te = np.clip(test["Stint"].astype(int).values, 0, 5)

    if args.cohort == "k1":
        m_tr = tr_M["mode_TyreLife"].values; m_te = te_M["mode_TyreLife"].values
        # mode-ids range 0..N_MODES-1 with -1 sentinel; map to 0..N_MODES (11 buckets)
        mt = np.where(m_tr < 0, 10, m_tr).astype(int)
        me = np.where(m_te < 0, 10, m_te).astype(int)
        n_seg = len(cmps) * 11
        seg_train = c_tr * 11 + mt
        seg_test = c_te * 11 + me
        outname = "d18_k1_pathb_cmp_modeTL"
    elif args.cohort == "k2":
        m_tr = tr_M["mode_LapTime_Delta"].values; m_te = te_M["mode_LapTime_Delta"].values
        mt = np.where(m_tr < 0, 10, m_tr).astype(int)
        me = np.where(m_te < 0, 10, m_te).astype(int)
        n_seg = len(cmps) * 11
        seg_train = c_tr * 11 + mt
        seg_test = c_te * 11 + me
        outname = "d18_k2_pathb_cmp_modeLD"
    else:  # k3
        m_tr = tr_M["mode_TyreLife"].values; m_te = te_M["mode_TyreLife"].values
        mt = np.where(m_tr < 0, 10, m_tr).astype(int)
        me = np.where(m_te < 0, 10, m_te).astype(int)
        n_seg = 11 * 6
        seg_train = mt * 6 + s_tr
        seg_test = me * 6 + s_te
        outname = "d18_k3_pathb_modeTL_stint"

    sizes = np.bincount(seg_train, minlength=n_seg)
    print(f"  cohort={args.cohort} n_seg={n_seg}  ≥{MIN_ROWS} rows in "
          f"{int((sizes >= MIN_ROWS).sum())} cells "
          f"(min/med/max nonzero: {sizes[sizes>0].min()}/"
          f"{int(np.median(sizes[sizes>0]))}/{sizes.max()})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    print("\n--- Global LR baseline ---")
    meta_global = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_global[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc_global = float(roc_auc_score(y, meta_global))
    print(f"  Global meta OOF: {auc_global:.5f}")

    taus = [5000, 20000, 100000]
    oofs = {tau: np.zeros(len(y)) for tau in taus}
    print(f"\n--- Path-B {args.cohort} hier-meta ---")
    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        w_global = fit_lr_aug(F_oof[tr], y[tr])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_local[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            mask[s] = True
        for tau in taus:
            n_local = counts.astype(np.float64)
            alpha = n_local / (n_local + tau)
            W_shrunk = (alpha[:, None] * W_local +
                        (1 - alpha[:, None]) * w_global[None, :])
            for s in np.unique(seg_train[va]):
                idx = np.where(seg_train[va] == s)[0]
                w = W_shrunk[s] if mask[s] else w_global
                oofs[tau][va[idx]] = predict_aug(F_oof[va[idx]], w)
        print(f"  fold {fold}: {time.time()-t_fold:.1f}s "
              f"({int(mask.sum())}/{n_seg} fit)")

    print("\n--- Full-train test predictions ---")
    w_global_full = fit_lr_aug(F_oof, y)
    W_local_full = np.zeros((n_seg, len(w_global_full)))
    counts_full = np.zeros(n_seg, dtype=np.int64)
    mask_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        counts_full[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_local_full[s] = fit_lr_aug(F_oof[idx], y[idx])
        mask_full[s] = True
    test_preds = {}
    for tau in taus:
        n_local = counts_full.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local_full +
                    (1 - alpha[:, None]) * w_global_full[None, :])
        tp = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx = np.where(seg_test == s)[0]
            w = W_shrunk[s] if mask_full[s] else w_global_full
            tp[idx] = predict_aug(F_test[idx], w)
        test_preds[tau] = tp

    print("\n=== Path-B sweep ===")
    final = dict(cohort=args.cohort, k=len(pool_names), n_seg=n_seg,
                 global_oof=auc_global, taus={})
    for tau in taus:
        oof = oofs[tau]; tp = test_preds[tau]
        auc = float(roc_auc_score(y, oof))
        rho, _ = spearmanr(tp, primary_test)
        d_oof = (auc - PRIMARY_S) * 1e4
        d_oof_global = (auc - auc_global) * 1e4
        print(f"  τ={tau}: OOF {auc:.5f}  Δ vs PRIMARY_S {d_oof:+.2f} bp  "
              f"Δ vs global LR {d_oof_global:+.2f} bp  ρ={rho:.5f}")
        np.save(ART / f"oof_{outname}_tau{tau}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_{outname}_tau{tau}_strat.npy",
                np.column_stack([1 - tp, tp]))
        final["taus"][str(tau)] = dict(
            oof=auc, delta_oof_PRIMARY_S_bp=float(d_oof),
            delta_oof_global_bp=float(d_oof_global),
            rho_vs_d9f=float(rho))
    final["wall_s"] = time.time() - t0
    (ART / f"{outname}_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ {outname}_results.json  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
