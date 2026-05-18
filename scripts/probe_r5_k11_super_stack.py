"""scripts/probe_r5_k11_super_stack.py — Round 5 Phases B + D combined

After Phase A (slim-kNN rebuild):
- Phase B: gate r4_segment_fe + r4_hmm_seq at the REAL K=11+1 anchor.
- Phase D: super-stack with K=11 + segment_fe + HMM (+ graph if available).

Builds K=11 from raw OOFs using the dgp_v3 naming convention:
  K=11 = 4 K=4 base trees + 6 slim-kNN (qAT,qAV,qAO,qAA,qAF,qAK)
         + 1 K=27 super-base (Path-B τ=100k)

Verifies K=11 LR-meta OOF AUC matches historical PRIMARY ~0.95443.

Outputs:
- oof_K11_super_stack_strat.npy + test pair (whichever variant best)
- submissions/submission_K11_super_stack.csv (if G2 passes)
- audit/2026-05-18-round-5-phase-bd.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
ART = Path("scripts/artifacts")

# 11 base files matching build_K11_full_pathb.py
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

CANDIDATES = [
    ("r4_segment_fe", "oof_r4_segment_fe_strat.npy",   "test_r4_segment_fe_strat.npy"),
    ("r4_hmm_seq",    "oof_r4_hmm_seq_strat.npy",      "test_r4_hmm_seq_strat.npy"),
]

OPTIONAL_GRAPH = ("r5_graph_pit_pressure",
                  "oof_r5_graph_pit_pressure_strat.npy",
                  "test_r5_graph_pit_pressure_strat.npy")


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def lr_meta(y, F_oof, F_test=None, n_folds=N_FOLDS, seed=SEED, C=1.0):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc = float(roc_auc_score(y, oof))
    test = None
    if F_test is not None:
        lr_full = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        lr_full.fit(F_oof, y)
        test = lr_full.predict_proba(F_test)[:, 1]
    return oof, auc, test


def main():
    print("=== R5 Phase B+D: K=11 super-stack ===")
    t0 = time.time()
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values

    print(f"\n  Loading K=11 bases...")
    K11_oofs = [_pos(ART / o) for _, o, _ in K11_FILES]
    K11_tests = [_pos(ART / t) for _, _, t in K11_FILES]
    K11_names = [n for n, _, _ in K11_FILES]
    print(f"  K=11 bases: {K11_names}")

    # Baseline K=11 LR-meta
    P_oof_K11 = np.column_stack(K11_oofs)
    P_test_K11 = np.column_stack(K11_tests)
    F_oof_K11 = expand(P_oof_K11)
    F_test_K11 = expand(P_test_K11)
    oof_K11, auc_K11, _ = lr_meta(y, F_oof_K11)
    print(f"\n  K=11 plain LR-meta OOF AUC: {auc_K11:.5f}")
    print(f"  (Historical PRIMARY OOF expected ~0.95443)")

    # K=4 baseline
    P_oof_K4 = np.column_stack(K11_oofs[:4])
    F_oof_K4 = expand(P_oof_K4)
    oof_K4, auc_K4, _ = lr_meta(y, F_oof_K4)
    print(f"  K=4 baseline OOF AUC: {auc_K4:.5f}")
    print(f"  K=11 vs K=4: {(auc_K11 - auc_K4)*1e4:+.3f} bp")

    # Phase B: K=11 + 2 (seg_fe + HMM)
    print(f"\n  Loading candidates...")
    cand_oofs, cand_tests, cand_names = [], [], []
    for name, oof_path, test_path in CANDIDATES:
        cand_oofs.append(_pos(ART / oof_path))
        cand_tests.append(_pos(ART / test_path))
        cand_names.append(name)

    P_oof_K11_2 = np.column_stack(K11_oofs + cand_oofs)
    P_test_K11_2 = np.column_stack(K11_tests + cand_tests)
    F_oof_K11_2 = expand(P_oof_K11_2)
    F_test_K11_2 = expand(P_test_K11_2)
    oof_K11_2, auc_K11_2, test_K11_2 = lr_meta(y, F_oof_K11_2, F_test_K11_2)
    delta_2 = (auc_K11_2 - auc_K11) * 1e4
    print(f"\n  === Phase B: K=11 + (seg_fe + HMM) ===")
    print(f"  OOF AUC: {auc_K11_2:.5f}  Δ vs K=11 = {delta_2:+.3f} bp")
    print(f"  Verdict: {'PASS G2' if delta_2 >= 0.3 else 'marginal/null'}")

    # Phase D: K=11 + 3 (add graph if available)
    graph_path = ART / OPTIONAL_GRAPH[1]
    test_K11_3 = None
    auc_K11_3 = None
    delta_3 = None
    if graph_path.exists():
        print(f"\n  Loading graph-class base...")
        graph_oof = _pos(graph_path)
        graph_test = _pos(ART / OPTIONAL_GRAPH[2])
        P_oof_K11_3 = np.column_stack([P_oof_K11_2, graph_oof])
        P_test_K11_3 = np.column_stack([P_test_K11_2, graph_test])
        F_oof_K11_3 = expand(P_oof_K11_3)
        F_test_K11_3 = expand(P_test_K11_3)
        oof_K11_3, auc_K11_3, test_K11_3 = lr_meta(y, F_oof_K11_3, F_test_K11_3)
        delta_3 = (auc_K11_3 - auc_K11) * 1e4
        print(f"\n  === Phase D: K=11 + (seg_fe + HMM + graph) ===")
        print(f"  OOF AUC: {auc_K11_3:.5f}  Δ vs K=11 = {delta_3:+.3f} bp")
        print(f"  Δ vs K=11+2 = {(auc_K11_3 - auc_K11_2)*1e4:+.3f} bp (graph marginal)")
        print(f"  Verdict: {'PASS G2' if delta_3 >= 0.3 else 'marginal/null'}")
    else:
        print(f"\n  Phase D: graph artifact missing — skipping")

    # rho diagnostics
    ref_K4pb = _pos(ART / "test_K4_fwd_pathb.npy")
    ref_K27pb = _pos(ART / "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")
    if test_K11_3 is not None:
        rho_K4pb, _ = spearmanr(test_K11_3, ref_K4pb)
        rho_K27pb, _ = spearmanr(test_K11_3, ref_K27pb)
        chosen = "Phase D K=11+3"
        chosen_test = test_K11_3
        chosen_oof = oof_K11_3
        chosen_auc = auc_K11_3
    else:
        rho_K4pb, _ = spearmanr(test_K11_2, ref_K4pb)
        rho_K27pb, _ = spearmanr(test_K11_2, ref_K27pb)
        chosen = "Phase B K=11+2"
        chosen_test = test_K11_2
        chosen_oof = oof_K11_2
        chosen_auc = auc_K11_2
    print(f"\n  ρ vs K=4+Path-B (LB 0.95351): {rho_K4pb:.6f}")
    print(f"  ρ vs K=27+Path-B (LB 0.95368): {rho_K27pb:.6f}")

    # Save artifacts
    np.save(ART / "oof_K11_super_stack_strat.npy",
            np.column_stack([1 - chosen_oof, chosen_oof]).astype(np.float64))
    np.save(ART / "test_K11_super_stack_strat.npy",
            np.column_stack([1 - chosen_test, chosen_test]).astype(np.float64))
    print(f"\n  -> oof+test_K11_super_stack_strat.npy (chosen: {chosen})")

    # Build submission CSV
    Path("submissions").mkdir(exist_ok=True)
    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = np.clip(chosen_test, 0.001, 0.999)
    sub_path = "submissions/submission_K11_super_stack.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  -> {sub_path}")

    # Save results JSON
    Path("audit").mkdir(exist_ok=True)
    results = dict(
        K11_OOF_auc=auc_K11, K4_OOF_auc=auc_K4,
        K11_plus_2_OOF_auc=auc_K11_2, delta_K11_plus_2_bp=delta_2,
        K11_plus_3_OOF_auc=auc_K11_3,
        delta_K11_plus_3_bp=delta_3,
        rho_test_vs_K4pathb=float(rho_K4pb),
        rho_test_vs_K27pathb=float(rho_K27pb),
        chosen=chosen, chosen_OOF_auc=chosen_auc,
        wall_seconds=time.time() - t0,
    )
    Path("audit/2026-05-18-round-5-phase-bd.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n  Total wall: {time.time()-t0:.1f}s")
    print(f"  Results: audit/2026-05-18-round-5-phase-bd.json")


if __name__ == "__main__":
    main()
