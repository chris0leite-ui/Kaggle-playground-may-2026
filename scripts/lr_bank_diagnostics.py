"""scripts/lr_bank_diagnostics.py — diversity/redundancy of the LR bank.

Mirrors lr_diag_e1_svd.py but operates on the LR bank built by
scripts/lr_bank.py. Reports:

  - Per-base standalone OOF AUC + ρ vs PRIMARY (d17_K24_d18pool_h1d).
  - SVD effective rank of the LR bank (logit space).
  - Eff_rank of LR-bank residualized after PRIMARY removal.
  - Eff_rank of K=24 GBDT pool ∪ LR-bank (combined-pool diversity).
  - Top-15 most-redundant pairs within the LR bank.
  - Top-15 most-orthogonal LR bases vs PRIMARY (lowest |ρ|).

Output: scripts/artifacts/lr_bank_diagnostics.json + console summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

K24_GBDT_BASES = [
    # K=21 PRIMARY pool
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B",
    # +3 Day-17 PM extras
    "d16_orig_continuous_only", "p1_single_cb_v3_gpu",
    "d17_h1d_yekenot_full",
]

LR_BANK_DEFAULT = [
    "lr_raw_std", "lr_raw_std_balanced",
    "lr_raw_freq", "lr_raw_te", "lr_raw_ohe",
    "lr_poly2_std", "lr_poly2_ohe", "lr_poly3_std",
    "lr_kbins5_ohe", "lr_kbins20_ohe", "lr_kbins50_uniform", "lr_kbins_yekenot",
    "lr_splines_5", "lr_splines_10",
    "lr_hash_2way_2k", "lr_hash_3way_8k",
    "lr_l1_lasso_kbins20", "lr_C_low_kbins20", "lr_C_high_kbins20", "lr_balanced_kbins20",
    "lr_perseg_compound", "lr_perseg_year",
    "lr_on_top_models",
]


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _eff_rank_entropy(s: np.ndarray) -> float:
    s2 = s ** 2
    p = s2 / s2.sum()
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def _cumvar_rank(s: np.ndarray, frac: float) -> int:
    s2 = s ** 2
    cum = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cum, frac) + 1)


def _spectrum(M: np.ndarray, label: str) -> dict:
    Mc = M - M.mean(axis=0)
    Mc = Mc / (Mc.std(axis=0) + 1e-12)
    s = np.linalg.svd(Mc, compute_uv=False)
    return dict(
        label=label,
        shape=list(Mc.shape),
        eff_rank_entropy=round(_eff_rank_entropy(s), 3),
        rank_90pct=_cumvar_rank(s, 0.90),
        rank_95pct=_cumvar_rank(s, 0.95),
        rank_99pct=_cumvar_rank(s, 0.99),
        var_top_5_pct=round(100 * (s[:5] ** 2).sum() / (s ** 2).sum(), 2),
        var_top_10_pct=round(100 * (s[:min(10, len(s))] ** 2).sum() / (s ** 2).sum(), 2),
        cond_number=round(float(s[0] / max(s[-1], 1e-12)), 1),
    )


def _logit(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-9, 1 - 1e-9)
    return np.log(P / (1 - P))


def _residualize(L: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    a = anchor - anchor.mean()
    aa = float((a * a).sum())
    R = np.zeros_like(L)
    for j in range(L.shape[1]):
        b = L[:, j] - L[:, j].mean()
        beta = float((a * b).sum() / aa)
        R[:, j] = b - beta * a
    return R


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].astype(int).values

    # Filter LR bases that exist
    lr_bases = [b for b in LR_BANK_DEFAULT if (ART / f"oof_{b}_strat.npy").exists()]
    print(f"LR bases found: {len(lr_bases)}/{len(LR_BANK_DEFAULT)}")
    print(f"  missing: {[b for b in LR_BANK_DEFAULT if b not in lr_bases]}")

    gbdt_bases = [b for b in K24_GBDT_BASES if (ART / f"oof_{b}_strat.npy").exists()]
    print(f"GBDT bases found: {len(gbdt_bases)}/{len(K24_GBDT_BASES)}")

    # Load matrices
    P_lr = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in lr_bases])
    P_gbdt = np.column_stack([_pos(ART / f"oof_{b}_strat.npy") for b in gbdt_bases])
    L_lr = _logit(P_lr)
    L_gbdt = _logit(P_gbdt)

    # PRIMARY (the meta-stacker output K=24+h1d)
    prim_oof = _pos(ART / "oof_d17_K24_d18pool_h1d_strat.npy")
    prim_log = _logit(prim_oof)

    # ---- Per-base summary
    per_base = []
    for b, p in zip(lr_bases, P_lr.T):
        auc = float(roc_auc_score(y, p))
        rho_prim, _ = spearmanr(p, prim_oof)
        per_base.append(dict(name=b, auc=round(auc, 5), rho_vs_primary=round(float(rho_prim), 4)))
    per_base.sort(key=lambda x: -x["auc"])

    # ---- SVD spectra
    spec_lr = _spectrum(L_lr, "LR_bank_logit")
    spec_gbdt = _spectrum(L_gbdt, "GBDT_K24_logit")
    L_combined = np.hstack([L_gbdt, L_lr])
    spec_combined = _spectrum(L_combined, "GBDT+LR_combined_logit")

    # Residualized after PRIMARY
    R_lr = _residualize(L_lr, prim_log)
    R_combined = _residualize(L_combined, prim_log)
    spec_R_lr = _spectrum(R_lr, "LR_bank_residualized_after_PRIMARY")
    spec_R_combined = _spectrum(R_combined, "GBDT+LR_residualized_after_PRIMARY")

    # ---- Top redundant pairs WITHIN LR bank
    K = L_lr.shape[1]
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            r, _ = spearmanr(L_lr[:, i], L_lr[:, j])
            pairs.append((float(r), lr_bases[i], lr_bases[j]))
    pairs.sort(key=lambda x: -abs(x[0]))
    top_redundant = [{"rho": round(r, 4), "a": a, "b": b} for r, a, b in pairs[:15]]

    # Most-orthogonal vs PRIMARY (lowest |ρ_PRIMARY|)
    orth = sorted(per_base, key=lambda x: abs(x["rho_vs_primary"]))[:8]

    out = dict(
        lr_bases=lr_bases,
        gbdt_bases=gbdt_bases,
        per_base=per_base,
        spec_lr_bank=spec_lr,
        spec_gbdt_K24=spec_gbdt,
        spec_combined=spec_combined,
        spec_residualized_lr=spec_R_lr,
        spec_residualized_combined=spec_R_combined,
        top_redundant_pairs_lr=top_redundant,
        most_orthogonal_vs_primary=orth,
    )
    out_json = ART / "lr_bank_diagnostics.json"
    out_json.write_text(json.dumps(out, indent=2))

    print("\n=== LR-bank standalone AUC + ρ vs PRIMARY ===")
    for r in per_base:
        print(f"  {r['name']:<26s}  AUC {r['auc']:.5f}   ρ_PRIM {r['rho_vs_primary']:+.4f}")

    print(f"\n=== SVD eff_rank ===")
    print(f"  LR-bank ({len(lr_bases)} cols)            eff_rank={spec_lr['eff_rank_entropy']}  "
          f"r95={spec_lr['rank_95pct']}  r99={spec_lr['rank_99pct']}  top5%={spec_lr['var_top_5_pct']}")
    print(f"  GBDT-K24 ({len(gbdt_bases)} cols)         eff_rank={spec_gbdt['eff_rank_entropy']}  "
          f"r95={spec_gbdt['rank_95pct']}  r99={spec_gbdt['rank_99pct']}  top5%={spec_gbdt['var_top_5_pct']}")
    print(f"  GBDT+LR combined ({L_combined.shape[1]} cols)  eff_rank={spec_combined['eff_rank_entropy']}  "
          f"r95={spec_combined['rank_95pct']}  r99={spec_combined['rank_99pct']}  top5%={spec_combined['var_top_5_pct']}")
    print(f"  LR-bank residualized          eff_rank={spec_R_lr['eff_rank_entropy']}  "
          f"r95={spec_R_lr['rank_95pct']}  r99={spec_R_lr['rank_99pct']}")
    print(f"  GBDT+LR residualized          eff_rank={spec_R_combined['eff_rank_entropy']}  "
          f"r95={spec_R_combined['rank_95pct']}  r99={spec_R_combined['rank_99pct']}")

    print(f"\n=== Top-15 most-redundant pairs WITHIN LR bank ===")
    for p in top_redundant:
        print(f"  ρ={p['rho']:+.4f}  {p['a']:<26s}  {p['b']}")

    print(f"\n=== Most-orthogonal-to-PRIMARY LR bases (lowest |ρ_PRIM|) ===")
    for r in orth:
        print(f"  {r['name']:<26s}  AUC {r['auc']:.5f}   ρ_PRIM {r['rho_vs_primary']:+.4f}")

    print(f"\n→ JSON saved: {out_json}")


if __name__ == "__main__":
    main()
