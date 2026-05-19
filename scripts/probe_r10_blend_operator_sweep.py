"""R10 hedge-prep: blend-operator sweep over R7-era LB-confirmed pool.

After R9 (NB4/C1 NULL) + R10 morning (4-constituent LR-meta alt-stack
NULL) confirmed structural rank-lock at the row-feature class, the
only EV-positive open axis at row-feature class is blend-operator
search over LB-confirmed submissions. Origin: 2026-05-12 K=11+K=9
70/30 rank-blend lifted +0.1 bp at rho=0.9998 — first
cross-mechanism error-cancellation in 4 days.

Pool (all LB-confirmed; OOF AUC verified 2026-05-19):
  R7.1 K=13 + Path-B DriverClass x Stint tau=100k    LB 0.95389
  R7.2 R7.1 + 5-seed fold-fit bag                    LB 0.95389
  R5.2 K=13 + Path-B Compound x Stint tau=100k       LB 0.95387
  R6.1 R5.2 + 5-seed fold-fit bag                    LB 0.95387
  K27  K=27 + Path-B tau=100k                        LB 0.95368

Operators: arith / gmean / logit_mean / rank_mean (rank-preserving).
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

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "scripts" / "artifacts"
DATA = ROOT / "data"
OUT_JSON = ART / "r10_blend_operator_sweep.json"

# (name, oof_npy, test_npy, LB, col_or_None_if_1d)
INGREDIENTS = [
    ("R7.1_DCS",     "oof_K13_pathb_driverclass_stint_tau100000.npy",
                     "test_K13_pathb_driverclass_stint_tau100000.npy", 0.95389, None),
    ("R7.2_DCS_bag", "oof_K13_dcs_pathb_foldbag_strat.npy",
                     "test_K13_dcs_pathb_foldbag_strat.npy",           0.95389, 1),
    ("R5.2_CxS",     "K13_seghmm_pathb_tau100000_oof.npy",
                     "K13_seghmm_pathb_tau100000_test.npy",            0.95387, None),
    ("R6.1_CxS_bag", "K13_seghmm_pathb_5seedbag_oof.npy",
                     "K13_seghmm_pathb_5seedbag_test.npy",             0.95387, None),
    ("K27_pool",     "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
                     "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy", 0.95368, 1),
]

PRIMARY = "R7.1_DCS"
WEIGHT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TIE_THRESHOLD = 0.9999
LIFT_FLOOR    = 0.999
OPERATORS = ("arith", "gmean", "logit_mean", "rank_mean")

EPS = 1e-15

def _to_unit(x): return np.clip(x, EPS, 1.0 - EPS)

def op_arith(arrs, ws):
    out = np.zeros_like(arrs[0])
    for a, w in zip(arrs, ws): out += w * a
    return out

def op_gmean(arrs, ws):
    out = np.zeros_like(arrs[0])
    for a, w in zip(arrs, ws): out += w * np.log(_to_unit(a))
    return np.exp(out)

def op_logit_mean(arrs, ws):
    out = np.zeros_like(arrs[0])
    for a, w in zip(arrs, ws):
        p = _to_unit(a)
        out += w * np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-out))

def op_rank_mean(arrs, ws):
    n = arrs[0].shape[0]
    out = np.zeros(n)
    for a, w in zip(arrs, ws):
        r = rankdata(a, method="average") / n
        out += w * r
    return out

OPS = {"arith": op_arith, "gmean": op_gmean, "logit_mean": op_logit_mean, "rank_mean": op_rank_mean}


def asym_flip_count(p_cand, p_ref, thresh=0.30):
    """Asymmetric class-decision flip count at p=thresh."""
    cand_pos = p_cand >= thresh
    ref_pos  = p_ref  >= thresh
    flip_to_pos = int(np.sum(cand_pos & ~ref_pos))
    flip_to_neg = int(np.sum(~cand_pos & ref_pos))
    return flip_to_pos, flip_to_neg


def load_ing(name, oof_f, test_f, lb, col):
    o = np.load(ART / oof_f); t = np.load(ART / test_f)
    if col is not None:
        o, t = o[:, col], t[:, col]
    assert o.shape == (439140,), f"{name}: oof shape {o.shape}"
    assert t.shape == (188165,), f"{name}: test shape {t.shape}"
    return o, t, lb


def main():
    t0 = time.time()
    y = pd.read_csv(DATA / "train.csv")["PitNextLap"].values
    ings = {}
    for name, oof_f, test_f, lb, col in INGREDIENTS:
        o, t, _lb = load_ing(name, oof_f, test_f, lb, col)
        ings[name] = {"oof": o, "test": t, "lb": _lb, "auc": float(roc_auc_score(y, o))}
        print(f"  {name:18s} OOF_AUC={ings[name]['auc']:.6f} LB={_lb}")

    prim_oof = ings[PRIMARY]["oof"]; prim_test = ings[PRIMARY]["test"]
    prim_auc = ings[PRIMARY]["auc"]
    assert abs(prim_auc - 0.954471) < 0.00001, f"PRIMARY drift: {prim_auc}"
    print(f"\nPRIMARY {PRIMARY} OOF_AUC = {prim_auc:.6f} (matches 0.954471 +- 0.00001)")
    print(f"baseline ref for delta: {prim_auc:.6f}\n")

    names = list(ings.keys())
    results = []

    # === 2-ingredient combos ===
    for a, b in combinations(names, 2):
        # higher-LB takes the heavier weight
        if ings[a]["lb"] < ings[b]["lb"]:
            a, b = b, a
        for op_name in OPERATORS:
            op = OPS[op_name]
            for w in WEIGHT_GRID:
                w_a, w_b = w, 1.0 - w
                oof_blend  = op([ings[a]["oof"],  ings[b]["oof"]],  [w_a, w_b])
                test_blend = op([ings[a]["test"], ings[b]["test"]], [w_a, w_b])
                auc = float(roc_auc_score(y, oof_blend))
                rho = float(spearmanr(test_blend, prim_test).statistic)
                fp, fn = asym_flip_count(test_blend, prim_test)
                results.append({
                    "kind": "2way",
                    "op": op_name,
                    "ings": [a, b],
                    "weights": [w_a, w_b],
                    "oof_auc": auc,
                    "delta_bp": (auc - prim_auc) * 1e4,
                    "rho_test": rho,
                    "flip_to_pos": fp,
                    "flip_to_neg": fn,
                    "flip_total": fp + fn,
                    "flip_asym": (max(fp, fn) - min(fp, fn)) / max(fp + fn, 1),
                })

    # === 3-ingredient combos (PRIMARY always present) ===
    others = [n for n in names if n != PRIMARY]
    for b, c in combinations(others, 2):
        # weight grid: PRIMARY weight in [0.5, 0.6, 0.7, 0.8], remainder split evenly OR 60/40 between b/c
        for w_prim in [0.50, 0.60, 0.70, 0.80]:
            for share_b in [0.50, 0.60, 0.40]:
                w_b = (1.0 - w_prim) * share_b
                w_c = (1.0 - w_prim) * (1.0 - share_b)
                for op_name in OPERATORS:
                    op = OPS[op_name]
                    oof_blend  = op([ings[PRIMARY]["oof"],  ings[b]["oof"],  ings[c]["oof"]],
                                    [w_prim, w_b, w_c])
                    test_blend = op([ings[PRIMARY]["test"], ings[b]["test"], ings[c]["test"]],
                                    [w_prim, w_b, w_c])
                    auc = float(roc_auc_score(y, oof_blend))
                    rho = float(spearmanr(test_blend, prim_test).statistic)
                    fp, fn = asym_flip_count(test_blend, prim_test)
                    results.append({
                        "kind": "3way",
                        "op": op_name,
                        "ings": [PRIMARY, b, c],
                        "weights": [w_prim, w_b, w_c],
                        "oof_auc": auc,
                        "delta_bp": (auc - prim_auc) * 1e4,
                        "rho_test": rho,
                        "flip_to_pos": fp,
                        "flip_to_neg": fn,
                        "flip_total": fp + fn,
                        "flip_asym": (max(fp, fn) - min(fp, fn)) / max(fp + fn, 1),
                    })

    # === reporting ===
    by_auc = sorted(results, key=lambda r: -r["oof_auc"])
    print(f"\nTop-20 by OOF AUC (out of {len(results)} configs):")
    print(f"{'kind':5s} {'op':10s} {'ings':45s} {'weights':32s} {'AUC':>9s} {'dbp':>7s} {'rho':>9s} {'flip+':>6s} {'flip-':>6s} {'asym':>5s} band")
    for r in by_auc[:20]:
        ws = ", ".join(f"{w:.2f}" for w in r["weights"])
        ig = "+".join(r["ings"])
        if r["rho_test"] >= TIE_THRESHOLD:
            band = "TIE"
        elif r["rho_test"] >= LIFT_FLOOR:
            band = "OK"
        else:
            band = "RISK"
        print(f"{r['kind']:5s} {r['op']:10s} {ig:45s} [{ws:30s}] {r['oof_auc']:.6f} {r['delta_bp']:+7.3f} {r['rho_test']:.6f} {r['flip_to_pos']:6d} {r['flip_to_neg']:6d} {r['flip_asym']:.2f} {band}")

    # Shortlist: delta_bp >= +0.05 AND rho in [0.999, 0.9999]
    SHORT = [r for r in results if r["delta_bp"] >= 0.05 and LIFT_FLOOR <= r["rho_test"] < TIE_THRESHOLD]
    SHORT.sort(key=lambda r: -r["delta_bp"])
    print(f"\nSHORTLIST (delta >= +0.05 bp AND rho in [0.999, 0.9999]): {len(SHORT)} configs")
    for r in SHORT[:15]:
        ws = ", ".join(f"{w:.2f}" for w in r["weights"])
        ig = "+".join(r["ings"])
        print(f"  {r['kind']:5s} {r['op']:10s} {ig:45s} [{ws:30s}] dbp={r['delta_bp']:+.3f} rho={r['rho_test']:.6f} flip+/-={r['flip_to_pos']}/{r['flip_to_neg']}")

    STRETCH = [r for r in results if r["delta_bp"] >= 0.10 and r["rho_test"] >= TIE_THRESHOLD]
    STRETCH.sort(key=lambda r: -r["delta_bp"])
    print(f"\nSTRETCH (delta >= +0.10 bp AND TIE_ZONE rho >= 0.9999): {len(STRETCH)} configs")
    for r in STRETCH[:15]:
        ws = ", ".join(f"{w:.2f}" for w in r["weights"])
        ig = "+".join(r["ings"])
        print(f"  {r['kind']:5s} {r['op']:10s} {ig:45s} [{ws:30s}] dbp={r['delta_bp']:+.3f} rho={r['rho_test']:.6f} flip+/-={r['flip_to_pos']}/{r['flip_to_neg']}")

    out = {
        "primary": PRIMARY,
        "primary_auc": prim_auc,
        "ingredients": {n: {"auc": d["auc"], "lb": d["lb"]} for n, d in ings.items()},
        "n_configs": len(results),
        "shortlist": SHORT,
        "stretch": STRETCH,
        "top20_by_auc": by_auc[:20],
        "wall_s": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWall: {out['wall_s']:.1f}s  ->  {OUT_JSON}")

if __name__ == "__main__":
    main()
