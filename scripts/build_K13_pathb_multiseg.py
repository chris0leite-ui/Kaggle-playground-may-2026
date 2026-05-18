"""scripts/build_K13_pathb_multiseg.py — Round 7 Phase B

Multi-segmentation Path-B sweep on R5.2 K=13 pool. Default Path-B
uses Compound × Stint (24 segments); this script sweeps three new
segmentations to test whether different per-sub-population
shrinkage captures additional patterns:

- B.1 Year × Compound (4 × 5 = 20 segments)
- B.2 Driver-class × Stint (named-vs-D0XX × 6 = 12 segments)
- B.3 Compound × Stint × LapNumber-bucket (5 × 6 × 4 = 120 segments)

For each: run Path-B with τ=100k; compare OOF to R5.2 baseline.
If any Δ ≥ +0.1 bp, sweep τ ∈ {20k, 100k, 500k} on the winner.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from build_K11_full_pathb import (_pos, expand, fit_lr_aug, predict_aug,
                                   MIN_ROWS, MAX_ITER, ART, DATA, TARGET)

N_FOLDS = 5
SEED = 42

K13_FILES = [
    ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy",     "test_d17_h1d_yekenot_full_strat.npy"),
    ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy",      "test_p1_single_cb_v4_gpu_strat.npy"),
    ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy",             "test_f1_hgbc_deep_strat.npy"),
    ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ("qAT",       "dgp_v3_qAT_K1_oof.npy",                  "dgp_v3_qAT_K1_test.npy"),
    ("qAV",       "dgp_v3_qAV_K1_7feat_oof.npy",            "dgp_v3_qAV_K1_7feat_test.npy"),
    ("qAO",       "dgp_v3_qAO_knn_multi_oof.npy",           "dgp_v3_qAO_knn_multi_test.npy"),
    ("qAA",       "dgp_v3_qAA_stint_imputed_oof.npy",       "dgp_v3_qAA_stint_imputed_test.npy"),
    ("qAF",       "dgp_v3_qAF_d16plus_oof.npy",             "dgp_v3_qAF_d16plus_test.npy"),
    ("qAK",       "dgp_v3_qAK_knn3_oof.npy",                "dgp_v3_qAK_knn3_test.npy"),
    ("K27_100k",  "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
                  "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"),
    ("seg_fe",    "oof_r4_segment_fe_strat.npy",            "test_r4_segment_fe_strat.npy"),
    ("HMM",       "oof_r4_hmm_seq_strat.npy",               "test_r4_hmm_seq_strat.npy"),
]


def _is_named(d):
    return ~d.astype(str).str.match(r"^D\d{3}$").fillna(False)


def build_seg(name, train, test):
    """Return (seg_train, seg_test, n_seg, label) for a named scheme."""
    if name == "year_compound":
        cats_c = sorted(set(train["Compound"].astype(str).unique()) |
                        set(test["Compound"].astype(str).unique()))
        cmp = {c: i for i, c in enumerate(cats_c)}
        cats_y = sorted(set(train["Year"].astype(int).unique()) |
                        set(test["Year"].astype(int).unique()))
        ymap = {y: i for i, y in enumerate(cats_y)}
        seg_tr = (train["Year"].astype(int).map(ymap).values * len(cmp)
                  + train["Compound"].astype(str).map(cmp).values)
        seg_te = (test["Year"].astype(int).map(ymap).values * len(cmp)
                  + test["Compound"].astype(str).map(cmp).values)
        return seg_tr, seg_te, len(cmp) * len(cats_y), "Year×Compound"

    if name == "driverclass_stint":
        # named (1) vs D0XX (0) × 6 stint values
        named_tr = _is_named(train["Driver"]).astype(int).values
        named_te = _is_named(test["Driver"]).astype(int).values
        s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
        s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
        seg_tr = named_tr * 6 + s_tr
        seg_te = named_te * 6 + s_te
        return seg_tr, seg_te, 2 * 6, "DriverClass×Stint"

    if name == "compound_stint_lapbucket":
        cats_c = sorted(set(train["Compound"].astype(str).unique()) |
                        set(test["Compound"].astype(str).unique()))
        cmp = {c: i for i, c in enumerate(cats_c)}
        c_tr = train["Compound"].astype(str).map(cmp).values
        c_te = test["Compound"].astype(str).map(cmp).values
        s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
        s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
        # 4 LapNumber buckets: [1-15, 16-30, 31-45, 46+]
        lap_tr = train["LapNumber"].astype(int).values
        lap_te = test["LapNumber"].astype(int).values
        lb_tr = np.clip(lap_tr // 15, 0, 3)
        lb_te = np.clip(lap_te // 15, 0, 3)
        seg_tr = (c_tr * 24) + (s_tr * 4) + lb_tr
        seg_te = (c_te * 24) + (s_te * 4) + lb_te
        return seg_tr, seg_te, len(cmp) * 6 * 4, "Compound×Stint×LapBucket"

    raise ValueError(f"unknown seg name: {name}")


def run_pathb_segmented(base_oofs, base_tests, seg_train, seg_test,
                         n_seg, y, tau):
    """Generalized run_pathb that takes a precomputed segmentation."""
    F_oof = expand(np.column_stack(base_oofs))
    F_test = expand(np.column_stack(base_tests))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        t0 = time.time()
        w_g = fit_lr_aug(F_oof[tr], y[tr])
        W_l = np.zeros((n_seg, len(w_g)))
        cnt = np.zeros(n_seg, dtype=np.int64)
        msk = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr] == s)[0]
            cnt[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr][idx])) < 2:
                continue
            W_l[s] = fit_lr_aug(F_oof[tr][idx], y[tr][idx])
            msk[s] = True
        n_local = cnt.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_sh = alpha[:, None] * W_l + (1 - alpha[:, None]) * w_g[None, :]
        for s in np.unique(seg_train[va]):
            idx_v = np.where(seg_train[va] == s)[0]
            w = W_sh[s] if msk[s] else w_g
            oof[va[idx_v]] = predict_aug(F_oof[va[idx_v]], w)
        print(f"    fold {fold+1}: {time.time()-t0:.1f}s  "
              f"(segments used: {int(msk.sum())}/{n_seg})", flush=True)

    # Full-train test
    w_g_full = fit_lr_aug(F_oof, y)
    W_l_full = np.zeros((n_seg, len(w_g_full)))
    cnt_full = np.zeros(n_seg, dtype=np.int64)
    msk_full = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        idx = np.where(seg_train == s)[0]
        cnt_full[s] = len(idx)
        if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
            continue
        W_l_full[s] = fit_lr_aug(F_oof[idx], y[idx])
        msk_full[s] = True
    alpha_full = cnt_full.astype(np.float64) / (cnt_full.astype(np.float64) + tau)
    W_sh_full = alpha_full[:, None] * W_l_full + (1 - alpha_full[:, None]) * w_g_full[None, :]
    test_pred = np.zeros(len(seg_test))
    for s in np.unique(seg_test):
        idx_t = np.where(seg_test == s)[0]
        w = W_sh_full[s] if msk_full[s] else w_g_full
        test_pred[idx_t] = predict_aug(F_test[idx_t], w)

    return oof, test_pred, int(msk_full.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=100_000)
    ap.add_argument("--segs", nargs="+",
                    default=["year_compound", "driverclass_stint",
                             "compound_stint_lapbucket"])
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== R7 Phase B: multi-segmentation Path-B τ={args.tau} ===",
          flush=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values

    oofs = [_pos(ART / o) for _, o, _ in K13_FILES]
    tests = [_pos(ART / t) for _, _, t in K13_FILES]
    print(f"  K=13 pool: {[n for n, _, _ in K13_FILES]}", flush=True)

    # R5.2 baseline reference
    ref_oof = _pos(ART / "K13_seghmm_pathb_tau100000_oof.npy")
    ref_test = _pos(ART / "K13_seghmm_pathb_tau100000_test.npy")
    ref_auc = float(roc_auc_score(y, ref_oof))
    print(f"  R5.2 baseline (Compound×Stint τ=100k) OOF: {ref_auc:.5f}",
          flush=True)

    results = []
    for seg_name in args.segs:
        print(f"\n  -- segmentation: {seg_name} --", flush=True)
        seg_tr, seg_te, n_seg, label = build_seg(seg_name, train, test)
        # Diagnostics: per-segment counts
        cnts = np.bincount(seg_tr, minlength=n_seg)
        used = int((cnts >= MIN_ROWS).sum())
        print(f"     {label}: {n_seg} segments, {used} above MIN_ROWS={MIN_ROWS}",
              flush=True)
        if used == 0:
            print(f"     SKIP: no segments meet MIN_ROWS threshold", flush=True)
            continue
        t_s = time.time()
        oof, test_pred, n_used = run_pathb_segmented(
            oofs, tests, seg_tr, seg_te, n_seg, y, tau=args.tau)
        auc = float(roc_auc_score(y, oof))
        delta_bp = (auc - ref_auc) * 1e4
        rho_test, _ = spearmanr(test_pred, ref_test)
        results.append(dict(name=seg_name, label=label, n_seg=n_seg,
                            n_segments_used=n_used, auc=auc,
                            delta_bp=delta_bp, rho_test=float(rho_test),
                            wall=time.time() - t_s))
        print(f"     OOF AUC: {auc:.5f}  Δ vs R5.2 = {delta_bp:+.3f} bp",
              flush=True)
        print(f"     ρ_test vs R5.2: {rho_test:.6f}", flush=True)
        # Save artifact
        np.save(ART / f"oof_K13_pathb_{seg_name}_tau{args.tau}.npy", oof)
        np.save(ART / f"test_K13_pathb_{seg_name}_tau{args.tau}.npy", test_pred)

    # Summary
    print(f"\n=== Summary (R7 Phase B multi-segmentation Path-B) ===",
          flush=True)
    print(f"{'segmentation':<30s}{'OOF':>9s}{'Δ_bp':>9s}{'ρ_R52':>10s}",
          flush=True)
    print("-" * 60, flush=True)
    results.sort(key=lambda r: -r["delta_bp"])
    for r in results:
        marker = " ★" if r["delta_bp"] >= 0.10 else ("  " if r["delta_bp"] >= 0 else " ↓")
        print(f"{r['label']:<30s}{r['auc']:.5f}{r['delta_bp']:+9.3f}"
              f"{r['rho_test']:>10.6f}{marker}", flush=True)

    survivors = [r for r in results if r["delta_bp"] >= 0.10]
    print(f"\n  Survivors (Δ ≥ +0.10 bp): {[r['name'] for r in survivors]}",
          flush=True)
    print(f"  Total wall: {time.time()-t0:.1f}s", flush=True)

    Path("audit").mkdir(exist_ok=True)
    Path("audit/2026-05-18-round-7-phase-b.json").write_text(json.dumps({
        "tau": args.tau, "ref_auc": ref_auc, "results": results,
        "survivors": [r["name"] for r in survivors],
    }, indent=2))


if __name__ == "__main__":
    main()
