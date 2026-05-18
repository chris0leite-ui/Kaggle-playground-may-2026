"""scripts/probe_r6_operator_axis_retest.py — Round 6 Phase A

Operator-axis retest: re-gate prior LR-meta nulls under K=11+Path-B
operator. Same pool, different meta operator.

Round 5 finding: same OOF (K=11+seg+HMM, 0.95446) gives +5 bp LB
swing between LR-meta and Path-B C×Stint τ=100k operators. Past
mechanisms screened only under LR-meta may pass under Path-B.

Tests each candidate as a single addition to the K=13 pool (K=11 +
seg + HMM) at the Path-B operator. Compares to the R5.2 baseline
OOF 0.95446 (K=13+Path-B τ=100k).

Candidates (all with OOF/test on disk):
- K4_conformal_widths (R2 P2.2)
- K4_rrf_k60 (R2 P0.1)
- K4_meta_lgbm_rank (R2 P1.1)
- K4_trimmed_rank_t1_1 (R2 P0.2)
- r4_segment_fe_v2 (R4 variant)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from build_K11_full_pathb import (_pos, expand, lr_meta_oof, run_pathb,
                                   ART, DATA, TARGET)


K13_BASE = [
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

CANDIDATES = [
    ("conformal_widths", "oof_K4_conformal_widths_strat.npy",   "test_K4_conformal_widths_strat.npy"),
    ("rrf_k60",          "oof_K4_rrf_k60_strat.npy",            "test_K4_rrf_k60_strat.npy"),
    ("meta_lgbm_rank",   "oof_K4_meta_lgbm_rank_strat.npy",     "test_K4_meta_lgbm_rank_strat.npy"),
    ("trimmed_rank",     "oof_K4_trimmed_rank_t1_1_strat.npy",  "test_K4_trimmed_rank_t1_1_strat.npy"),
    ("seg_fe_v2",        "oof_r4_segment_fe_v2_strat.npy",      "test_r4_segment_fe_v2_strat.npy"),
]


def main():
    t0 = time.time()
    print(f"=== R6 Phase A: operator-axis retest ===")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values

    base_oofs = [_pos(ART / o) for _, o, _ in K13_BASE]
    base_tests = [_pos(ART / t) for _, _, t in K13_BASE]
    print(f"  K=13 base: {[n for n, _, _ in K13_BASE]}")

    # R5.2 baseline (K=13 + Path-B τ=100k) for Δ comparison
    r52_oof_path = ART / "K13_seghmm_pathb_tau100000_oof.npy"
    r52_test_path = ART / "K13_seghmm_pathb_tau100000_test.npy"
    if r52_oof_path.exists():
        r52_oof = _pos(r52_oof_path)
        r52_auc = float(roc_auc_score(y, r52_oof))
        r52_test = _pos(r52_test_path)
        print(f"  R5.2 baseline OOF (K=13+Path-B τ=100k): {r52_auc:.5f}")
    else:
        # Recompute baseline
        print(f"  Recomputing R5.2 baseline...")
        r52_oof, r52_test = run_pathb(base_oofs, base_tests, train, test, y, tau=100_000)
        r52_auc = float(roc_auc_score(y, r52_oof))
        np.save(r52_oof_path, r52_oof)
        np.save(r52_test_path, r52_test)
        print(f"  R5.2 baseline OOF: {r52_auc:.5f}")

    K27_pathb_test = _pos(ART / "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")

    results = []
    for cname, oof_file, test_file in CANDIDATES:
        if not (ART / oof_file).exists():
            print(f"  -- {cname}: MISSING {oof_file}; skip")
            continue
        t_c = time.time()
        cand_oof = _pos(ART / oof_file)
        cand_test = _pos(ART / test_file)

        pool_oofs = base_oofs + [cand_oof]
        pool_tests = base_tests + [cand_test]
        print(f"\n  -- {cname}: running K=14+Path-B (K=13 + {cname}) ...")
        oof, test_pred = run_pathb(pool_oofs, pool_tests, train, test, y, tau=100_000)
        auc = float(roc_auc_score(y, oof))
        delta_bp = (auc - r52_auc) * 1e4
        rho_test_r52, _ = spearmanr(test_pred, r52_test)
        rho_test_K27pb, _ = spearmanr(test_pred, K27_pathb_test)
        results.append(dict(name=cname, auc=auc, delta_bp=delta_bp,
                            rho_vs_r52=float(rho_test_r52),
                            rho_vs_K27pb=float(rho_test_K27pb),
                            wall=time.time() - t_c))
        print(f"     OOF AUC: {auc:.5f}  Δ vs R5.2 = {delta_bp:+.3f} bp")
        print(f"     ρ vs R5.2: {rho_test_r52:.6f}   ρ vs K27+Path-B: {rho_test_K27pb:.6f}")
        # Save artifact (allows downstream Phase D pickup)
        np.save(ART / f"oof_K14_r5plus_{cname}_pathb_strat.npy",
                np.column_stack([1 - oof, oof]).astype(np.float64))
        np.save(ART / f"test_K14_r5plus_{cname}_pathb_strat.npy",
                np.column_stack([1 - test_pred, test_pred]).astype(np.float64))

    # Summary
    print(f"\n=== Summary (R6 Phase A operator-axis retest) ===")
    print(f"{'candidate':<22s}{'OOF':>9s}{'Δ_bp':>9s}{'ρ_R52':>10s}{'ρ_K27pb':>10s}")
    print("-" * 65)
    results.sort(key=lambda r: -r["delta_bp"])
    for r in results:
        marker = " ★" if r["delta_bp"] >= 0.10 else ("  " if r["delta_bp"] >= 0 else " ↓")
        print(f"{r['name']:<22s}{r['auc']:.5f}{r['delta_bp']:+9.3f}"
              f"{r['rho_vs_r52']:>10.6f}{r['rho_vs_K27pb']:>10.6f}{marker}")

    survivors = [r for r in results if r["delta_bp"] >= 0.10]
    print(f"\n  Survivors (Δ ≥ +0.10 bp): {[r['name'] for r in survivors]}")
    print(f"  Total wall: {time.time()-t0:.1f}s")

    Path("audit").mkdir(exist_ok=True)
    Path("audit/2026-05-18-round-6-phase-a.json").write_text(json.dumps({
        "r52_baseline_auc": r52_auc, "results": results,
        "survivors": [r["name"] for r in survivors],
    }, indent=2))


if __name__ == "__main__":
    main()
