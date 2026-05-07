"""Day-16 results compiler — runs gates + multi-add min-meta and emits
a one-shot summary table for the audit.

Picks up whatever d16 artifacts have landed at run time. Skips missing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5

PRIMARY_OOF = ART / "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d15b_path_b_K22_dae_only_tau20000_strat.npy"

CANDIDATES = [
    ("h4_year_mask",       "d16_h4_year_mask"),
    ("h7_conformal",       None),  # multiple files; skip generic gate
    ("h10_two_stage",      "d16_h10_two_stage_stint"),
    ("h2_twin_meta",       "d16_h2_twin_meta"),
    ("epsilon4_deepgbm",   "d16_epsilon4_deepgbm"),
    ("h9_pseudo",          "d16_h9_transductive_pseudo"),
    ("h11_adv_weight",     "d16_h11_adv_weight"),
    ("h1_gru_seq",         "d16_gru_seq"),     # may not exist yet
]

K22_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
    "d15b_lgbm_dae_only",
]


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _expand(P):
    from scipy.stats import rankdata
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logit = np.log(Pc / (1 - Pc))
    return np.hstack([P, rk, logit])


def _meta_oof(y, F):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F[tr], y[tr])
        oof[va] = lr.predict_proba(F[va])[:, 1]
    return oof, float(roc_auc_score(y, oof))


def _meta_full(y, F_oof, F_test):
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr.fit(F_oof, y)
    return lr.predict_proba(F_test)[:, 1], lr


def main():
    t0 = time.time()
    y = pd.read_csv("data/train.csv", usecols=["PitNextLap"])["PitNextLap"].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))
    print(f"PRIMARY hier-meta OOF AUC = {auc_primary:.5f}\n", flush=True)

    available_cands = []
    print(f"{'name':<25s}  {'std OOF':>9s}  {'ρ vs PR':>9s}  {'pred LB':>9s}  {'flip':>6s}  verdict")
    print("-" * 82)
    rows = []
    for tag, prefix in CANDIDATES:
        if prefix is None:
            print(f"{tag:<25s}  (skipped: post-process)")
            continue
        oof_p = ART / f"oof_{prefix}_strat.npy"
        test_p = ART / f"test_{prefix}_strat.npy"
        if not oof_p.exists() or not test_p.exists():
            print(f"{tag:<25s}  (artifacts not yet present)")
            continue
        cand_oof = _pos(oof_p)
        cand_test = _pos(test_p)
        std_auc = float(roc_auc_score(y, cand_oof))
        rho = float(spearmanr(cand_test, primary_test)[0])
        d_oof_bp = (std_auc - auc_primary) * 1e4
        if rho >= 0.99996: pred_lb = d_oof_bp
        elif rho >= 0.999:  pred_lb = d_oof_bp - 0.5
        elif rho >= 0.995:  pred_lb = d_oof_bp - 1.5
        elif rho >= 0.99:   pred_lb = d_oof_bp - 3.0
        else:               pred_lb = d_oof_bp - 5.0

        rare_thr = float(np.quantile(primary_test, 0.99))
        primary_pos = primary_test >= rare_thr
        cand_pos = cand_test >= rare_thr
        flips_to_neg = int((primary_pos & ~cand_pos).sum())
        flips_to_pos = int((~primary_pos & cand_pos).sum())
        if max(flips_to_pos, flips_to_neg) > 0:
            flip_ratio = min(flips_to_pos, flips_to_neg) / max(flips_to_pos, flips_to_neg)
        else:
            flip_ratio = 1.0

        if std_auc < 0.93:
            verdict = "WEAK_BASE"
        elif rho >= 0.999:
            verdict = "TIE_EXPECTED"
        elif d_oof_bp < -3:
            verdict = "FAIL_OOF"
        elif d_oof_bp >= 0.3 and rho < 0.999:
            verdict = "PASS"
        elif d_oof_bp >= -1.5 and rho < 0.998:
            verdict = "MAYBE_DIVERSE"
        else:
            verdict = "FAIL"
        print(f"{tag:<25s}  {std_auc:>9.5f}  {rho:>9.4f}  {pred_lb:>+9.2f}  {flip_ratio:>6.3f}  {verdict}")
        rows.append(dict(tag=tag, prefix=prefix, std_oof=std_auc,
                         rho_vs_primary=rho, delta_oof_bp=d_oof_bp,
                         pred_lb_bp=pred_lb, flip_ratio=flip_ratio,
                         verdict=verdict))
        if verdict in ("PASS", "MAYBE_DIVERSE"):
            available_cands.append((tag, prefix, cand_oof, cand_test))

    if not available_cands:
        print("\nNo candidates clear gate; skipping multi-add min-meta.")
    else:
        print(f"\n=== K=22+N multi-add LR-meta gate (N={len(available_cands)}) ===",
              flush=True)
        pool_oofs = [_pos(ART / f"oof_{b}_strat.npy") for b in K22_BASES]
        pool_tests = [_pos(ART / f"test_{b}_strat.npy") for b in K22_BASES]
        P_oof_base = np.column_stack(pool_oofs)
        P_test_base = np.column_stack(pool_tests)
        F_oof_base = _expand(P_oof_base)
        F_test_base = _expand(P_test_base)

        oof_base, auc_base = _meta_oof(y, F_oof_base)
        print(f"LR-meta(K=22) OOF AUC = {auc_base:.5f}", flush=True)

        cand_oof_arr = np.column_stack([c[2] for c in available_cands])
        cand_test_arr = np.column_stack([c[3] for c in available_cands])
        P_oof_with = np.column_stack([P_oof_base, cand_oof_arr])
        P_test_with = np.column_stack([P_test_base, cand_test_arr])
        F_oof_with = _expand(P_oof_with)
        F_test_with = _expand(P_test_with)
        oof_with, auc_with = _meta_oof(y, F_oof_with)
        delta_lr = (auc_with - auc_base) * 1e4
        delta_pr = (auc_with - auc_primary) * 1e4
        print(f"LR-meta(K=22+{len(available_cands)}) OOF AUC = {auc_with:.5f}",
              flush=True)
        print(f"  Δ vs LR-meta(K=22): {delta_lr:+.3f}bp", flush=True)
        print(f"  Δ vs PRIMARY hier:  {delta_pr:+.3f}bp", flush=True)

        test_with, lr_full = _meta_full(y, F_oof_with, F_test_with)
        rho = float(spearmanr(test_with, primary_test)[0])
        print(f"  ρ vs PRIMARY: {rho:.6f}", flush=True)

        K = P_oof_with.shape[1]
        n_cand = len(available_cands)
        raw_w = lr_full.coef_.ravel()
        print("\n  Per-candidate L1 weight at K=22+N LR-meta:")
        for j, (tag, _, _, _) in enumerate(available_cands):
            col = K - n_cand + j
            l1 = abs(raw_w[col]) + abs(raw_w[K + col]) + abs(raw_w[2 * K + col])
            print(f"    {tag:<25s} |w| = {l1:.4f}")

    summary = dict(primary_oof_auc=auc_primary,
                   candidates=rows, wall_s=time.time() - t0)
    Path(ART / "d16_compile_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[compile] -> {ART/'d16_compile_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
