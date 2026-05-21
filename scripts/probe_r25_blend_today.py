"""scripts/probe_r25_blend_today.py — closed-form blend sweep of
today's K-add candidates after cohort-listwise axis saturation.

Inputs (5 ingredients):
  R15        — PRIMARY (LB 0.95397, OOF 0.954490)
  K=18_R17   — HEDGE 0 (LB 0.95398, OOF 0.954499)
  K=19_R20   — today's best K-add (OOF 0.954501, no LB yet)
  R21_std    — CB YetiRank standalone (OOF 0.953956, different model family)
  K=27_100k  — wide-pool diversity (LB 0.95368)

Sweep: 4 operators (arith / gmean / logit_mean / rank_mean) × simplex
grid (step depending on n_ingredients). Output: ranked list of OK-band
candidates with Δ vs R15 PRIMARY.

Rule 27 band:
  ρ_test ≥ 0.9999       → TIE_ZONE (likely LB-tie at 5-decimal quantization)
  0.999 ≤ ρ_test < 0.9999 → OK (target band; meaningful LB shift)
  ρ_test < 0.999        → REGRESSION_RISK (abort unless PI override)
"""
from __future__ import annotations
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"

INGREDIENTS = [
    ("R15_PRIMARY",         "oof_K17_xendcg_pathb_dcs_tau100000.npy",
                            "test_K17_xendcg_pathb_dcs_tau100000.npy",
                            0.95397),
    ("K18_R17_listwise",    "oof_K18_pathb_driverclass_stint_tau100000.npy",
                            "test_K18_pathb_driverclass_stint_tau100000.npy",
                            0.95398),
    ("K19_R20_lambdarank",  "oof_K19_pathb_dcs_tau100000_R20lambdarank.npy",
                            "test_K19_pathb_dcs_tau100000_R20lambdarank.npy",
                            None),
    ("R21_yetirank_std",    "oof_R21_yetirank_yrl_strat.npy",
                            "test_R21_yetirank_yrl_strat.npy",
                            None),
    ("K27_widepool",        "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
                            "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
                            0.95368),
]

PRIMARY_PROXY = "R15_PRIMARY"
RULE_27_TIE = 0.9999
RULE_27_OK_FLOOR = 0.999
OPERATORS = ("arith", "gmean", "logit_mean", "rank_mean")


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _from_logit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def blend(preds, weights, op):
    P = np.column_stack(preds)
    w = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    if op == "arith":
        return (P * w).sum(axis=1)
    if op == "gmean":
        return np.exp((np.log(np.clip(P, 1e-9, 1.0)) * w).sum(axis=1))
    if op == "logit_mean":
        Z = _logit(P)
        return _from_logit((Z * w).sum(axis=1))
    if op == "rank_mean":
        n = P.shape[0]
        R = np.column_stack([rankdata(c) / n for c in P.T])
        return (R * w).sum(axis=1)
    raise ValueError(op)


def simplex_grid(k, step):
    """Discrete simplex weights summing to 1 in 'step' increments."""
    pts = []
    n_per_axis = int(round(1.0 / step))

    def rec(remaining, depth, prefix):
        if depth == k - 1:
            w = np.array(prefix + [remaining * step / n_per_axis * n_per_axis])
            pts.append((np.array(prefix + [remaining]) * step))
            return
        for v in range(remaining + 1):
            rec(remaining - v, depth + 1, prefix + [v])

    rec(n_per_axis, 0, [])
    out = np.array(pts) * 1.0
    return out


def band(rho):
    if rho >= RULE_27_TIE:
        return "TIE_ZONE"
    if rho >= RULE_27_OK_FLOOR:
        return "OK"
    return "REGRESSION_RISK"


def main():
    t0 = time.time()
    print("== R25 blend sweep: today's K-candidates ==", flush=True)

    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    available = []
    for nm, of, tf, lb in INGREDIENTS:
        po, pt = ART / of, ART / tf
        if po.exists() and pt.exists():
            available.append(dict(name=nm, oof=_pos(po), test=_pos(pt), lb=lb))
            print(f"  loaded {nm}  oof={np.shape(available[-1]['oof'])}  "
                  f"test={np.shape(available[-1]['test'])}", flush=True)
        else:
            print(f"  SKIP {nm} (missing)", flush=True)

    primary = next(a for a in available if a["name"] == PRIMARY_PROXY)
    R15_oof = primary["oof"]
    R15_test = primary["test"]
    R15_auc = roc_auc_score(y, R15_oof)
    print(f"\n  R15 PRIMARY OOF AUC: {R15_auc:.6f}", flush=True)

    # Pairwise ρ_oof for diagnostics
    print("\n  pairwise ρ_oof:", flush=True)
    print(f"  {'':24s}" + "".join(f"  {a['name'][:18]:>18s}" for a in available),
          flush=True)
    for a in available:
        row = f"  {a['name'][:24]:24s}"
        for b in available:
            rho = spearmanr(a["oof"], b["oof"]).statistic
            row += f"  {rho:>18.4f}"
        print(row, flush=True)

    results = []
    # Run all 2-, 3-, 4-way blend combos
    for k_combo in (2, 3, 4):
        step = {2: 0.025, 3: 0.05, 4: 0.10}[k_combo]
        weights_grid = simplex_grid(k_combo, step)
        for combo in combinations(range(len(available)), k_combo):
            names = [available[i]["name"] for i in combo]
            if PRIMARY_PROXY not in names:
                continue  # focus on R15-anchored blends
            ingreds_oof = [available[i]["oof"] for i in combo]
            ingreds_test = [available[i]["test"] for i in combo]
            for op in OPERATORS:
                for w in weights_grid:
                    if w.min() < 1e-6:
                        continue  # skip degenerate
                    blend_oof = blend(ingreds_oof, w, op)
                    blend_test = blend(ingreds_test, w, op)
                    auc = roc_auc_score(y, blend_oof)
                    delta_bp = (auc - R15_auc) * 1e4
                    rho_t = spearmanr(blend_test, R15_test).statistic
                    results.append(dict(
                        combo="|".join(names),
                        op=op, weights=list(w),
                        oof_auc=auc, delta_bp=delta_bp,
                        rho_test=float(rho_t), band=band(rho_t)))

    print(f"\n  scanned {len(results)} blend candidates "
          f"in {time.time()-t0:.0f}s", flush=True)

    # Rank by OK-band first, then by Δ_bp
    results.sort(key=lambda r: (r["band"] != "OK", r["band"] != "TIE_ZONE",
                                -r["delta_bp"]))
    print(f"\n  Top-20 candidates (OK > TIE > REGRESSION):", flush=True)
    print(f"  {'combo':<55s}  {'op':<11s}  {'weights':<22s}  "
          f"{'Δ_bp':>7s}  {'ρ_test':>7s}  {'band':<10s}", flush=True)
    print("  " + "-" * 130, flush=True)
    for r in results[:20]:
        w_str = "[" + ", ".join(f"{w:.2f}" for w in r["weights"]) + "]"
        print(f"  {r['combo'][:55]:<55s}  {r['op']:<11s}  {w_str[:22]:<22s}  "
              f"{r['delta_bp']:+7.3f}  {r['rho_test']:>7.4f}  {r['band']:<10s}",
              flush=True)

    # Persist
    ART_AUDIT = Path("scripts/artifacts")
    out_json = Path("audit/2026-05-21-round-25-blend-today.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(
        primary=PRIMARY_PROXY, primary_auc=R15_auc,
        ingredient_count=len(available),
        candidates_total=len(results),
        top_ok=[r for r in results if r["band"] == "OK"][:10],
        top_tie=[r for r in results if r["band"] == "TIE_ZONE"][:5],
    )
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote {out_json}  wall {time.time()-t0:.0f}s", flush=True)

    # Save top OK-band candidate's submission CSV if found
    ok_top = [r for r in results if r["band"] == "OK"]
    if ok_top:
        top = ok_top[0]
        print(f"\n  Top-OK candidate: {top['combo']} {top['op']} "
              f"w={top['weights']} Δ={top['delta_bp']:+.3f} bp ρ={top['rho_test']:.4f}",
              flush=True)
        combo_names = top["combo"].split("|")
        ingreds_test = [next(a["test"] for a in available if a["name"] == n)
                        for n in combo_names]
        blend_test = blend(ingreds_test, np.array(top["weights"]), top["op"])
        test = pd.read_csv("data/test.csv")
        sub = pd.DataFrame({"id": test["id"].values,
                            TARGET: np.clip(blend_test, 0.001, 0.999)})
        Path("submissions").mkdir(exist_ok=True)
        sub_path = f"submissions/submission_R25_blend_top_ok.csv"
        sub.to_csv(sub_path, index=False)
        print(f"  Saved top-OK CSV: {sub_path}", flush=True)


if __name__ == "__main__":
    main()
