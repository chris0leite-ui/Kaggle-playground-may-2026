"""d18 — Combined synthesis: K=21 + v4 + h1d (mainline PRIMARY pool) + DGP-class.

Greedy stack-add starting from K=21 + v4 + h1d (current main PRIMARY pool).
Tests which of our DGP-class bases (d16, d18, G, F2, F5, H, I, J) add
real LR-meta signal on top — given main's friction
`pool-saturation-v4h1d-absorbs-d16d18` showed +1.3 bp at K=25.

Outputs:
  scripts/artifacts/d18_combined_synth_results.json
  scripts/artifacts/d18_combined_rho_matrix.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

K21_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
    "d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound",
    "d9f_FM_A", "d9f_FM_B",
]


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def _meta_oof(y, F):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values

    # Mainline anchors
    mainline = [
        ("p1_single_cb_v4_gpu",         "v4 CB yekenot (main +24bp K=21+1)"),
        ("d17_h1d_yekenot_full",        "h1d RealMLP yekenot (main load-bearing)"),
    ]
    # Our DGP-class bases + held HEDGE candidates from prior sessions
    dgp_class = [
        ("d16_orig_continuous_only",    "d16 cont_only"),
        ("d18_chain_decomp",            "d18 chain v1"),
        ("d18_g_mode_id",               "G CTGAN mode-id"),
        ("d18_f2_constraint",           "F2 constraint"),
        ("d18_f5_class_cond_gmm",       "F5 class-cond GMM"),
        ("d18_h_mode_lookup",           "H mode-lookup"),
        ("d18_i_mode_collapse",         "I mode-collapse"),
        ("d18_j_cond_vector",           "J cond-vector"),
        ("d15b_lgbm_dae_only",          "DAE-only (most-diverse base of session)"),
        ("d15b_lgbm_dae_full",          "DAE-full (raw+latent)"),
        ("d15_orig_transfer",           "orig-transfer full-feature LGBM"),
        ("d15_leak_lookup",             "EB leak_lookup univariate+bivariate"),
        ("d18b_chain_decomp",           "d18b chain v2 (causal+q10)"),
        ("p1_single_lgbm_v3_feA_te",    "p1 v3 fold-safe Rozen-style LGBM"),
        ("d18_e2_preimage_knn",         "E2 preimage kNN"),
    ]
    # Verify all artifacts exist
    for n, _ in mainline + dgp_class:
        p = ART / f"oof_{n}_strat.npy"
        if not p.exists():
            print(f"MISSING: {p}"); return

    # Load K=21 pool + extras
    pool_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]

    # Baselines: K=21 alone, K=21+v4, K=21+v4+h1d
    print("=== baselines ===")
    F_K21 = _expand(np.column_stack(pool_oofs))
    _, auc_K21 = _meta_oof(y, F_K21)
    print(f"K=21 LR-meta:                  OOF {auc_K21:.5f}")

    v4 = _pos(ART / "oof_p1_single_cb_v4_gpu_strat.npy")
    F_K22_v4 = _expand(np.column_stack(pool_oofs + [v4]))
    _, auc_K22_v4 = _meta_oof(y, F_K22_v4)
    print(f"K=22 + v4:                     OOF {auc_K22_v4:.5f}  Δ vs K=21 {(auc_K22_v4-auc_K21)*1e4:+.3f} bp")

    h1d = _pos(ART / "oof_d17_h1d_yekenot_full_strat.npy")
    F_K23_v4h1d = _expand(np.column_stack(pool_oofs + [v4, h1d]))
    _, auc_K23 = _meta_oof(y, F_K23_v4h1d)
    print(f"K=23 + v4 + h1d (BASELINE):    OOF {auc_K23:.5f}  Δ vs K=21 {(auc_K23-auc_K21)*1e4:+.3f} bp")
    print()

    # Greedy add over DGP-class
    print("=== greedy stack-add starting from K=23 + v4 + h1d ===")
    chosen = []
    cur_oofs = list(pool_oofs) + [v4, h1d]
    cur_auc = auc_K23
    log = []
    while True:
        best = None
        for name, label in dgp_class:
            if name in chosen:
                continue
            cand = _pos(ART / f"oof_{name}_strat.npy")
            F = _expand(np.column_stack(cur_oofs + [cand]))
            _, auc = _meta_oof(y, F)
            d = (auc - cur_auc) * 1e4
            if best is None or d > best[2]:
                best = (name, label, d, auc, cand)
        if best is None or best[2] < 0.05:
            break
        chosen.append(best[0])
        cur_oofs.append(best[4])
        cur_auc = best[3]
        log.append(dict(name=best[0], label=best[1], cum_oof=best[3],
                        marginal_bp=best[2],
                        total_bp=(best[3] - auc_K23) * 1e4))
        print(f"  + {best[0]:32s}  cum_OOF {best[3]:.5f}  +{best[2]:.3f} bp  "
              f"total {(best[3] - auc_K23) * 1e4:+.3f} bp")
    print(f"\nFinal: K=23+{len(chosen)} baseline → +{(cur_auc - auc_K23)*1e4:.3f} bp on top of K=23 v4+h1d")

    # Marginal scoring of each DGP candidate IF added solo on top of K=23 v4+h1d
    print(f"\n=== solo K=23+1 marginal ===")
    K23_solo = []
    for name, label in dgp_class:
        cand = _pos(ART / f"oof_{name}_strat.npy")
        F = _expand(np.column_stack(pool_oofs + [v4, h1d, cand]))
        _, auc = _meta_oof(y, F)
        d_K23 = (auc - auc_K23) * 1e4
        K23_solo.append(dict(name=name, label=label, k24_oof=auc,
                             delta_vs_K23=d_K23))
        print(f"  K=23+{name:32s}  OOF {auc:.5f}  Δ +{d_K23:+.3f} bp")
    K23_solo.sort(key=lambda r: -r["delta_vs_K23"])

    # Save
    summary = dict(
        K21_meta=auc_K21,
        K22_plus_v4=auc_K22_v4,
        K23_plus_v4_h1d=auc_K23,
        greedy_log=log,
        K23_solo_marginal=K23_solo,
    )
    (ART / "d18_combined_synth_results.json").write_text(json.dumps(summary, indent=2))
    print("\n→ d18_combined_synth_results.json")


if __name__ == "__main__":
    main()
