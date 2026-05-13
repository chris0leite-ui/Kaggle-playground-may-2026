"""Submission-level blend-weight harness.

Searches convex weights across 4 operators (arithmetic / geometric /
log-odds / rank means) over N leaderboard-confirmed predictions on
disk, reports OOF-best candidates with pre-submit diagnostics
(Spearman vs current PRIMARY proxy, asymmetric flip counts at the
rare-class operating point).

Origin: 2026-05-12 LB 0.95386 rank-blend 70/30 of K=11+K=27+Path-B
and K=9 qAX lifted +0.1 bp at rho=0.9998 — first cross-mechanism
error-cancellation lift in 4 days. The mechanism sits OUTSIDE the
LR-meta rank-lock because the blend operates on submission-level
probabilities, not on the [P, rank, logit] meta expansion.

Run: python scripts/probe_blend_harness.py
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

sys.path.insert(0, str(Path(__file__).parent))
from common import ART, SEED  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_JSON = ART / "probe_blend_harness.json"

# Each entry: (display_name, oof_npy, test_npy, lb_or_None).
# LB-confirmed predictions on disk in order of LB score (descending).
INGREDIENTS = [
    # K=27 super-base + Path-B C×S τ=100k — the best Path-B on disk
    ("K27_pathb_100k",
     "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
     "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
     0.95368),
    # K=27 super-base + Path-B τ=20k (different shrinkage)
    ("K27_pathb_20k",
     "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau20000_strat.npy",
     "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau20000_strat.npy",
     None),
    # K=4 forward-greedy + Path-B (sparse-pool PRIMARY-of-2026-05-08)
    ("K4_pathb",
     "oof_K4_fwd_pathb.npy",
     "test_K4_fwd_pathb.npy",
     0.95351),
    # K=23 v4+h1d + Path-B τ=100k (Day-17 PM)
    ("K23_v4h1d_pathb_100k",
     "oof_d17_path_b_K23_v4_h1d_tau100000_strat.npy",
     "test_d17_path_b_K23_v4_h1d_tau100000_strat.npy",
     0.95354),
    # K=22 DAE + Path-B τ=20k (Day-15 PRIMARY)
    ("K22_dae_pathb_20k",
     "oof_d15b_path_b_K22_dae_only_tau20000_strat.npy",
     "test_d15b_path_b_K22_dae_only_tau20000_strat.npy",
     0.95059),
    # K=22 continuous-only + Path-B τ=20k (Day-16 PRIMARY)
    ("K22_cont_pathb_20k",
     "oof_d16_path_b_K22_continuous_only_tau20000_strat.npy",
     "test_d16_path_b_K22_continuous_only_tau20000_strat.npy",
     0.95089),
]

# Reference PRIMARY for Rule 27 (we don't have current-LB-PRIMARY on
# disk; use K27_pathb_100k as the closest LB-confirmed proxy).
PRIMARY_PROXY = "K27_pathb_100k"
RULE_27_THRESHOLD = 0.999

OPERATORS = ("arith", "gmean", "logit_mean", "rank_mean")


def _pos(p: Path) -> np.ndarray:
    """Return the positive-class probability column from an npy file
    (handles both 1-D and 2-D layouts)."""
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def _clip_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _from_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def blend(preds: list[np.ndarray], weights: np.ndarray, op: str) -> np.ndarray:
    """Combine N predictions under one of four operators.

    The convex weights sum to 1 and live on the simplex.
    """
    P = np.column_stack(preds)
    w = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    assert P.shape[1] == w.shape[1]
    if op == "arith":
        return (P * w).sum(axis=1)
    if op == "gmean":
        return np.exp((np.log(np.clip(P, 1e-9, 1.0)) * w).sum(axis=1))
    if op == "logit_mean":
        Z = _clip_logit(P)
        return _from_logit((Z * w).sum(axis=1))
    if op == "rank_mean":
        n = P.shape[0]
        R = np.column_stack([rankdata(c) / n for c in P.T])
        return (R * w).sum(axis=1)
    raise ValueError(f"unknown op {op!r}")


def asymmetric_flip(p_a: np.ndarray, p_b: np.ndarray, base_rate: float) -> tuple[int, int]:
    """Top-k disagreement at the rare-class operating point.

    Returns (n_added_by_b, n_dropped_by_b) where k = base_rate * len.
    Uses A's top-k as the anchor set.
    """
    n = len(p_a)
    k = max(1, int(round(base_rate * n)))
    rank_a = rankdata(-p_a, method="ordinal")
    rank_b = rankdata(-p_b, method="ordinal")
    top_a = rank_a <= k
    top_b = rank_b <= k
    return int((top_b & ~top_a).sum()), int((top_a & ~top_b).sum())


def simplex_grid(d: int, step: float = 0.1) -> list[tuple[float, ...]]:
    """Enumerate the discrete simplex at the given step size.

    Returns all (w_1, ..., w_d) with sum=1, each w_i in {0, step, 2*step, ..., 1}.
    """
    n_steps = int(round(1 / step))
    out: list[tuple[float, ...]] = []

    def _rec(remaining: int, depth: int, prefix: list[int]) -> None:
        if depth == d - 1:
            prefix.append(remaining)
            out.append(tuple(s * step for s in prefix))
            prefix.pop()
            return
        for s in range(remaining + 1):
            prefix.append(s)
            _rec(remaining - s, depth + 1, prefix)
            prefix.pop()

    _rec(n_steps, 0, [])
    return out


def main() -> None:
    t0 = time.time()
    train_y = pd.read_csv(DATA / "train.csv")["PitNextLap"].astype(int).values
    base_rate = float(train_y.mean())
    print(f"y train: n={len(train_y)} base_rate={base_rate:.4f}", flush=True)

    available = []
    for name, oof_f, test_f, lb in INGREDIENTS:
        p_oof = ART / oof_f
        p_test = ART / test_f
        if not p_oof.exists() or not p_test.exists():
            print(f"  SKIP {name}: missing on disk", flush=True)
            continue
        try:
            oof = _pos(p_oof)
            test = _pos(p_test)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {name}: load error {e}", flush=True)
            continue
        if len(oof) != len(train_y):
            print(f"  SKIP {name}: shape mismatch oof={len(oof)} y={len(train_y)}", flush=True)
            continue
        auc = float(roc_auc_score(train_y, oof))
        available.append({"name": name, "oof": oof, "test": test, "lb": lb, "oof_auc": auc})
        print(f"  + {name:30s} OOF AUC={auc:.5f}  LB={lb}", flush=True)

    if len(available) < 2:
        print("ERROR: need >= 2 ingredients on disk; aborting", flush=True)
        return

    # ----------------- individual baselines + pairwise diagnostics -----------------
    primary_proxy = next((a for a in available if a["name"] == PRIMARY_PROXY), available[0])
    print(f"\nPRIMARY proxy: {primary_proxy['name']}  OOF={primary_proxy['oof_auc']:.5f}", flush=True)

    print("\nPairwise diagnostics:")
    print(f"  {'pair':<60s} {'rho':>7s} {'flips_add':>10s} {'flips_drop':>10s}")
    pair_diag: dict[str, dict[str, float]] = {}
    for a, b in combinations(available, 2):
        rho = float(spearmanr(a["oof"], b["oof"]).statistic)
        n_add, n_drop = asymmetric_flip(a["oof"], b["oof"], base_rate)
        key = f"{a['name']}_vs_{b['name']}"
        pair_diag[key] = {"rho": rho, "flips_add": n_add, "flips_drop": n_drop}
        print(f"  {key:<60s} {rho:7.4f} {n_add:>10d} {n_drop:>10d}")

    # ----------------- blend grid -----------------
    best_individual = max(a["oof_auc"] for a in available)
    print(f"\nBest individual OOF: {best_individual:.5f}  ({1e4*(best_individual-primary_proxy['oof_auc']):+.3f} bp vs proxy)")
    print("\nGrid-searching blends...", flush=True)

    rows: list[dict] = []
    # 2-way to 5-way blends.
    # Step schedule: coarser as dimension grows (keeps total evals near 2 min CPU).
    step_by_k = {2: 0.05, 3: 0.1, 4: 0.2, 5: 0.25}
    for k in range(2, min(5, len(available)) + 1):
        step = step_by_k[k]
        grid = simplex_grid(k, step=step)
        for combo in combinations(available, k):
            names = [c["name"] for c in combo]
            oof_list = [c["oof"] for c in combo]
            for w in grid:
                if any(wi >= 1.0 - 1e-9 for wi in w):
                    continue  # skip pure-single-source weights
                for op in OPERATORS:
                    p = blend(oof_list, np.asarray(w), op)
                    auc = float(roc_auc_score(train_y, p))
                    rows.append({
                        "k": k,
                        "ingredients": "+".join(names),
                        "weights": tuple(round(wi, 3) for wi in w),
                        "op": op,
                        "oof_auc": auc,
                        "delta_vs_proxy_bp": 1e4 * (auc - primary_proxy["oof_auc"]),
                        "delta_vs_best_indiv_bp": 1e4 * (auc - best_individual),
                    })

    df = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    print(f"\nGrid total: {len(df)} candidates. t={time.time()-t0:.1f}s")

    # ----------------- top-K -----------------
    top = df.head(10).copy()
    print("\nTop-10 blends by OOF AUC:")
    print(top[["k", "ingredients", "weights", "op", "oof_auc", "delta_vs_proxy_bp", "delta_vs_best_indiv_bp"]].to_string(index=False))

    # ----------------- pre-submit diagnostics on top-1 -----------------
    if not top.empty:
        top1 = top.iloc[0].to_dict()
        # Re-evaluate top1 to get test prediction + ρ vs proxy
        combo = [a for a in available if a["name"] in top1["ingredients"].split("+")]
        oof_list = [c["oof"] for c in combo]
        test_list = [c["test"] for c in combo]
        w = np.asarray(top1["weights"])
        oof_pred = blend(oof_list, w, top1["op"])
        test_pred = blend(test_list, w, top1["op"])
        rho_vs_proxy_oof = float(spearmanr(oof_pred, primary_proxy["oof"]).statistic)
        rho_vs_proxy_test = float(spearmanr(test_pred, primary_proxy["test"]).statistic)
        n_add, n_drop = asymmetric_flip(primary_proxy["oof"], oof_pred, base_rate)
        verdict = "ABORT_TIE" if rho_vs_proxy_test > RULE_27_THRESHOLD else "OK"
        print(f"\nTop-1 pre-submit:")
        print(f"  rho_OOF  vs proxy = {rho_vs_proxy_oof:.6f}")
        print(f"  rho_TEST vs proxy = {rho_vs_proxy_test:.6f} (Rule 27 threshold {RULE_27_THRESHOLD})")
        print(f"  flip_diff vs proxy (add / drop): {n_add} / {n_drop}")
        print(f"  verdict: {verdict}")

        # Save top-1 prediction CSV alongside artifacts for inspection.
        test_csv = ART / f"blend_harness_top1_{top1['op']}.csv"
        # We only need (id, PitNextLap) columns; use the test.csv id column.
        test_df = pd.read_csv(DATA / "test.csv", usecols=["id"])
        test_df["PitNextLap"] = test_pred
        test_df.to_csv(test_csv, index=False)
        print(f"  wrote {test_csv.name} ({len(test_df)} rows)")

    # ----------------- persist -----------------
    out = {
        "n_ingredients": len(available),
        "ingredients": [{"name": a["name"], "oof_auc": a["oof_auc"], "lb": a["lb"]} for a in available],
        "primary_proxy": primary_proxy["name"],
        "best_individual_oof": best_individual,
        "n_blend_candidates": len(df),
        "top_10_by_oof": top.to_dict(orient="records"),
        "pairwise_diag": pair_diag,
        "top1_diag": {
            "rho_oof_vs_proxy": rho_vs_proxy_oof if not top.empty else None,
            "rho_test_vs_proxy": rho_vs_proxy_test if not top.empty else None,
            "flips_add": n_add if not top.empty else None,
            "flips_drop": n_drop if not top.empty else None,
            "verdict": verdict if not top.empty else None,
        },
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_JSON.name}  t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
