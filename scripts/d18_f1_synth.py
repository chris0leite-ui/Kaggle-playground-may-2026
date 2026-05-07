"""d18 F1 — Synthesis: ρ-matrix across DGP-class bases + greedy K=21+N panel.

Inputs: all d18-class OOF/test artifacts plus K=21 pool + d16. Computes:
  1. ρ matrix on TEST predictions (Spearman) — diversity heatmap
  2. K=21+1 LR-meta lift per d18-class candidate (already computed; just compile)
  3. Greedy K=21+N panel: add candidates one-by-one, ranked by marginal lift
  4. Joint K=21+full-d18-class with all candidates
  5. Final summary table for the audit note

Outputs:
  scripts/artifacts/d18_f1_synth_results.json
  scripts/artifacts/d18_f1_rho_matrix.csv
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

    # --- DGP-class candidates to synthesize ---
    dgp_candidates = [
        ("d16_orig_continuous_only",        "d16 cont_only (PRIMARY 22nd base)"),
        ("d18_chain_decomp",                "d18 v1 causal+gauss"),
        ("d18b_chain_decomp",               "d18b v2 causal+q10"),
        ("d18c_chain_decomp",               "d18c v3 reverse+q10"),
        ("d18_e2_preimage_knn",             "E2 preimage kNN"),
        ("d18_e5_pathb_C1_cmp_llq5_tau20000",  "E5 C1 Path-B Compound×llq5 τ=20k"),
        ("d18_e5_pathb_C2_cmp_stint_llq3_tau20000", "E5 C2 Path-B Compound×Stint×llq3 τ=20k"),
        ("d18_e5_pathb_C3_stint_llq5_tau20000", "E5 C3 Path-B Stint×llq5 τ=20k"),
    ]
    available = []
    for name, label in dgp_candidates:
        p_oof = ART / f"oof_{name}_strat.npy"
        p_test = ART / f"test_{name}_strat.npy"
        if p_oof.exists() and p_test.exists():
            available.append((name, label))
    print(f"available DGP candidates ({len(available)}/{len(dgp_candidates)}):")
    for name, label in available:
        print(f"  {name:50s}  {label}")

    # --- Load K=21 pool + DGP candidates ---
    pool_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K21_BASES]
    pool_tests = [_pos(ART / f"test_{b}_strat.npy") for b in K21_BASES]
    cand_oofs = {n: _pos(ART / f"oof_{n}_strat.npy") for n, _ in available}
    cand_tests = {n: _pos(ART / f"test_{n}_strat.npy") for n, _ in available}

    # --- ρ matrix on TEST (Spearman) ---
    all_names = ["K21_meta"] + [n for n, _ in available]
    F_test_K21 = _expand(np.column_stack(pool_tests))
    F_oof_K21 = _expand(np.column_stack(pool_oofs))
    _, auc_K21 = _meta_oof(y, F_oof_K21)
    print(f"\nK=21 LR-meta OOF: {auc_K21:.5f}")
    # Build K=21 meta TEST prediction
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof_K21, y)
    K21_test_meta = lr_full.predict_proba(F_test_K21)[:, 1]

    test_arr = np.column_stack([K21_test_meta] +
                               [cand_tests[n] for n, _ in available])
    rho = np.eye(len(all_names))
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            r, _ = spearmanr(test_arr[:, i], test_arr[:, j])
            rho[i, j] = rho[j, i] = r
    rho_df = pd.DataFrame(rho, index=all_names, columns=all_names).round(4)
    rho_df.to_csv(ART / "d18_f1_rho_matrix.csv")
    print("\nρ matrix (test):"); print(rho_df.to_string())

    # --- K=21+1 lift per DGP candidate ---
    deltas = []
    for name, label in available:
        F_oof_aug = _expand(np.column_stack(pool_oofs + [cand_oofs[name]]))
        _, auc = _meta_oof(y, F_oof_aug)
        d_bp = (auc - auc_K21) * 1e4
        deltas.append((name, label, d_bp, auc))
        print(f"  K=21+1 {name:45s} OOF={auc:.5f}  Δ={d_bp:+.3f} bp")
    deltas.sort(key=lambda r: -r[2])

    # --- Greedy stack-add panel (K=21+N) ---
    print("\n--- Greedy stack-add panel ---")
    chosen = []
    chosen_oofs = list(pool_oofs)
    auc_cum = auc_K21
    greedy_log = []
    while True:
        best = None
        for name, label in available:
            if name in chosen:
                continue
            F_aug = _expand(np.column_stack(chosen_oofs + [cand_oofs[name]]))
            _, auc = _meta_oof(y, F_aug)
            d = (auc - auc_cum) * 1e4
            if best is None or d > best[2]:
                best = (name, label, d, auc)
        if best is None or best[2] < 0.05:
            break
        chosen.append(best[0])
        chosen_oofs.append(cand_oofs[best[0]])
        auc_cum = best[3]
        greedy_log.append(dict(step=len(chosen), name=best[0], label=best[1],
                               cum_oof=best[3], marginal_bp=best[2],
                               total_bp=(best[3] - auc_K21) * 1e4))
        print(f"  +{len(chosen):>2}  {best[0]:45s} cum_OOF={best[3]:.5f}  "
              f"+{best[2]:.3f} bp  total {(best[3] - auc_K21) * 1e4:+.3f} bp")

    # --- Joint K=21+all-DGP ---
    print("\n--- Joint K=21+all DGP candidates ---")
    F_all = _expand(np.column_stack(pool_oofs + [cand_oofs[n] for n, _ in available]))
    _, auc_all = _meta_oof(y, F_all)
    print(f"  K=21+{len(available)} OOF={auc_all:.5f}  Δ vs K=21: "
          f"{(auc_all - auc_K21) * 1e4:+.3f} bp")

    summary = dict(
        n_candidates=len(available),
        K21_meta_oof=auc_K21,
        K21_plus_all_oof=auc_all,
        deltas_K21_plus_1=[
            dict(name=n, label=l, delta_bp=d, oof=a)
            for n, l, d, a in deltas
        ],
        greedy_log=greedy_log,
    )
    (ART / "d18_f1_synth_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ d18_f1_synth_results.json + d18_f1_rho_matrix.csv")


if __name__ == "__main__":
    main()
