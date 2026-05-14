"""Adaptive per-row blend driven by 11-base disagreement.

For each row, compute the standard deviation across K=11's 11 underlying
base predictions. High std means the bases disagree about that row;
those rows are where the K=11 stacker has the most uncertainty. We blend
with K=10 (slim-kNN-only, ablates the K=27 super-base) proportionally:

  alpha[i] = clip((std[i] - tau) * k, 0, max_alpha)
  new_pred[i] = (1 - alpha[i]) * K=11_pred[i] + alpha[i] * K=10_pred[i]

When std is low (bases agree), alpha=0 and we keep K=11 unchanged. When
std is high (bases disagree), alpha grows toward max_alpha and the
prediction moves toward K=10.

Reports cross-validation lift at several (tau, k, max_alpha) settings.
Submits if the best setting produces rho_test in [0.999, 0.9999].
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
DATA = Path("data")
TARGET = "PitNextLap"

# The 11 base prediction files that go into the K=11 stack.
BASE_FILES = [
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


def _pos(p):
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).values

    # Load 11 base predictions (rows in original train/test order)
    base_oofs = np.column_stack([_pos(ART / o) for _, o, _ in BASE_FILES])
    base_tests = np.column_stack([_pos(ART / t) for _, _, t in BASE_FILES])
    print(f"base predictions: train {base_oofs.shape}  test {base_tests.shape}",
          flush=True)

    # Standard deviation across bases per row
    std_oof = base_oofs.std(axis=1)
    std_test = base_tests.std(axis=1)
    print(f"OOF base std: mean={std_oof.mean():.4f}  p50={np.median(std_oof):.4f}  "
          f"p90={np.percentile(std_oof, 90):.4f}  max={std_oof.max():.4f}",
          flush=True)

    K11_oof = _pos(ART / "K11_full_pathb_tau100000_oof.npy")
    K11_test = _pos(ART / "K11_full_pathb_tau100000_test.npy")
    K10_oof = _pos(ART / "K10_slim_pathb_tau100000_oof.npy")
    K10_test = _pos(ART / "K10_slim_pathb_tau100000_test.npy")
    K11_auc = float(roc_auc_score(y, K11_oof))
    print(f"\nK=11 OOF AUC: {K11_auc:.5f}", flush=True)
    print(f"K=10 OOF AUC: {roc_auc_score(y, K10_oof):.5f}", flush=True)

    # Sweep adaptive-blend hyperparameters
    print("\nAdaptive blend sweep:", flush=True)
    print(f"  {'tau':>6}  {'k':>4}  {'max_a':>6}  {'OOF AUC':>9}  {'delta_bp':>9}  {'rho_oof':>9}  {'rho_test':>9}")
    results = []
    for tau in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        for k_slope in [1.0, 2.0, 5.0]:
            for max_alpha in [0.1, 0.2, 0.3, 0.5]:
                alpha_oof = np.clip((std_oof - tau) * k_slope, 0, max_alpha)
                new_oof = (1 - alpha_oof) * K11_oof + alpha_oof * K10_oof
                auc = float(roc_auc_score(y, new_oof))
                delta_bp = (auc - K11_auc) * 1e4

                alpha_test = np.clip((std_test - tau) * k_slope, 0, max_alpha)
                new_test = (1 - alpha_test) * K11_test + alpha_test * K10_test
                rho_oof = float(spearmanr(new_oof, K11_oof).statistic)
                rho_test = float(spearmanr(new_test, K11_test).statistic)
                results.append({
                    "tau": tau, "k_slope": k_slope, "max_alpha": max_alpha,
                    "oof_auc": auc, "delta_bp": delta_bp,
                    "rho_oof": rho_oof, "rho_test": rho_test,
                })
                print(f"  {tau:>6.2f}  {k_slope:>4.1f}  {max_alpha:>6.2f}  "
                      f"{auc:.6f}  {delta_bp:>+9.3f}  {rho_oof:.6f}  {rho_test:.6f}",
                      flush=True)

    df = pd.DataFrame(results)
    # Best lift candidate that's still in OK transfer zone (rho_test in [0.999, 0.9999])
    eligible = df[(df["rho_test"] >= 0.999) & (df["rho_test"] < 0.9999)]
    if not eligible.empty:
        best = eligible.sort_values("oof_auc", ascending=False).iloc[0]
        print(f"\nBest eligible candidate (rho_test in [0.999, 0.9999]):", flush=True)
        print(best, flush=True)

        # Build the test prediction CSV for this setting
        tau, k_slope, max_alpha = best["tau"], best["k_slope"], best["max_alpha"]
        alpha_test = np.clip((std_test - tau) * k_slope, 0, max_alpha)
        new_test_best = (1 - alpha_test) * K11_test + alpha_test * K10_test

        sub = pd.DataFrame({"id": test["id"], "PitNextLap": new_test_best})
        csv = ART / f"submission_adaptive_blend_tau{tau:.2f}_k{k_slope:.0f}_a{max_alpha:.1f}.csv"
        sub.to_csv(csv, index=False)
        np.save(ART / "adaptive_blend_test.npy", new_test_best)
        print(f"\nWrote {csv.name}", flush=True)
    else:
        max_lift = df["delta_bp"].max()
        print(f"\nNo eligible candidate. Max OOF lift = {max_lift:+.3f} bp;"
              " best candidates either tie (rho >= 0.9999) or regress (rho < 0.999).",
              flush=True)
        # Also list TIE_ZONE and REGRESSION best for reference
        tie = df[df["rho_test"] >= 0.9999].sort_values("oof_auc", ascending=False).head(1)
        reg = df[df["rho_test"] < 0.999].sort_values("oof_auc", ascending=False).head(1)
        if not tie.empty:
            print(f"  best TIE_ZONE: {tie.iloc[0].to_dict()}", flush=True)
        if not reg.empty:
            print(f"  best REGRESSION_RISK: {reg.iloc[0].to_dict()}", flush=True)

    (ART / "adaptive_blend_sweep.json").write_text(
        json.dumps({"K11_oof_auc": K11_auc, "results": results,
                    "elapsed_sec": time.time() - t0}, indent=2, default=str))
    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
