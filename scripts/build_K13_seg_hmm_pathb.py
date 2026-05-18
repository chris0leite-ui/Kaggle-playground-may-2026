"""Round 5 — K=13 (K=11 + seg_fe + HMM) + Path-B C×S τ=100k.

Applies the PRIMARY's per-segment shrinkage operator (Path-B) to the
K=11 pool augmented with Round 4's mechanism-orthogonal candidates.

Reuses `run_pathb` from build_K11_full_pathb.

The hypothesis: Path-B has better OOF→LB transfer than LR-meta
(empirical: K=4+Path-B transfer -5.2 bp vs K=4 LR-meta -5.1 bp; K=11
LR-meta transfer was -6.3 bp). Apply Path-B to the augmented pool
where today's seg+HMM lifted OOF by +0.245 bp; if Path-B preserves
this on LB, lands at LB ~0.95400.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, "scripts")
from build_K11_full_pathb import (_pos, expand, lr_meta_oof, run_pathb,
                                   TARGET, ART, SEED, N_FOLDS, TAU, DATA)

EXTRA_BASES = [
    ("seg_fe", "oof_r4_segment_fe_strat.npy",   "test_r4_segment_fe_strat.npy"),
    ("HMM",    "oof_r4_hmm_seq_strat.npy",      "test_r4_hmm_seq_strat.npy"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=100_000)
    args = ap.parse_args()
    tau = args.tau

    t0 = time.time()
    print(f"=== K=13 (K=11 + seg + HMM) + Path-B tau={tau} ===")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values

    K11_FILES = [
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
    ]
    all_files = K11_FILES + EXTRA_BASES
    oofs = [_pos(ART / o) for _, o, _ in all_files]
    tests = [_pos(ART / t) for _, _, t in all_files]
    names = [n for n, _, _ in all_files]
    print(f"  K={len(names)} bases: {names}")

    plain_oof = lr_meta_oof(expand(np.column_stack(oofs)), y)
    plain_auc = float(roc_auc_score(y, plain_oof))
    print(f"  K={len(names)} plain LR-meta OOF: {plain_auc:.5f}")

    print(f"  Running Path-B C x Stint tau={tau} ...")
    oof, test_pred = run_pathb(oofs, tests, train, test, y, tau=tau)
    auc = float(roc_auc_score(y, oof))
    print(f"  K={len(names)} + Path-B OOF AUC: {auc:.5f}")
    print(f"  Δ vs plain LR-meta: {(auc - plain_auc)*1e4:+.3f} bp")

    K27_pathb_test = _pos(ART / "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")
    K4_pathb_test = _pos(ART / "test_K4_fwd_pathb.npy")
    rho_test = float(spearmanr(test_pred, K27_pathb_test).statistic)
    rho_K4 = float(spearmanr(test_pred, K4_pathb_test).statistic)
    print(f"  ρ vs K=27+Path-B: {rho_test:.6f}")
    print(f"  ρ vs K=4+Path-B:  {rho_K4:.6f}")

    np.save(ART / f"K13_seghmm_pathb_tau{tau}_oof.npy", oof)
    np.save(ART / f"K13_seghmm_pathb_tau{tau}_test.npy", test_pred)

    sub = pd.DataFrame({"id": test["id"], "PitNextLap": np.clip(test_pred, 0.001, 0.999)})
    sub_path = Path(f"submissions/submission_K13_seghmm_pathb_tau{tau}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\n  Wrote {sub_path}")
    print(f"  Total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
