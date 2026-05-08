"""M6 — Dirichlet random search blend (3000 candidates, alpha=1).

Per analyticaobscura recipe (cross-comp research Source 1 #2):
sample 3000 weight vectors from Dirichlet(alpha=1, K) over the K base
OOFs in BOTH raw-probability space and rank-normalised space; pick
best by held-out AUC on a 20% strip of OOF rows; report top-10
weights and OOF AUC for each mode.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
BASE_S, BASE_G = 0.94075, 0.92059
SEED = 42
N_CAND = 3000

POOL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("m3_catboost", "m3_catboost"),
    ("m4_relstate", "m4_relstate"),
]


def load(name: str, suffix: str):
    oof = np.load(ART / f"oof_{name}_{suffix}.npy")[:, 1]
    test = np.load(ART / f"test_{name}_{suffix}.npy")[:, 1]
    return oof.astype(np.float64), test.astype(np.float64)


def search_dirichlet(P_oof: np.ndarray, P_test: np.ndarray, y: np.ndarray,
                     mode: str, n_cand: int = N_CAND):
    """Sample n_cand weight vectors; pick best on 80/20 holdout; report.

    mode: 'raw' or 'rank'. In rank mode, P_oof / P_test are rank-normalised.
    Returns (best_w, best_auc_holdout, best_auc_full_oof, blend_test_predictions).
    """
    rng = np.random.default_rng(SEED)
    K = P_oof.shape[1]
    # 80/20 holdout split (random shuffle of rows)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(0.8 * len(y))
    tr_idx, ho_idx = idx[:cut], idx[cut:]

    weights = rng.dirichlet(np.ones(K), size=n_cand)  # (n_cand, K)

    best_w, best_auc_ho = None, -1.0
    top10 = []  # (auc_ho, weight)
    for i in range(n_cand):
        w = weights[i]
        blend = P_oof @ w  # (n,)
        auc_ho = roc_auc_score(y[ho_idx], blend[ho_idx])
        if auc_ho > best_auc_ho:
            best_auc_ho = auc_ho
            best_w = w
        top10.append((float(auc_ho), w.copy()))
    top10.sort(key=lambda t: -t[0])
    top10 = top10[:10]

    blend_oof = P_oof @ best_w
    auc_full = float(roc_auc_score(y, blend_oof))
    blend_test = P_test @ best_w
    return best_w, float(best_auc_ho), auc_full, blend_test, top10


def run_anchor(suffix: str, base_auc: float):
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values

    Xs_oof, Xs_test, names = [], [], []
    for label, name in POOL:
        oo, te = load(name, suffix)
        Xs_oof.append(oo)
        Xs_test.append(te)
        names.append(label)
    P_oof = np.column_stack(Xs_oof)
    P_test = np.column_stack(Xs_test)

    print(f"\n=== Dirichlet search ({suffix}, K={len(names)}, n_cand={N_CAND}) ===")

    # Raw mode
    print("Raw probability mode:")
    w_raw, ho_raw, full_raw, test_raw, top10_raw = search_dirichlet(
        P_oof, P_test, y, "raw")
    print(f"  best holdout AUC: {ho_raw:.5f}")
    print(f"  full-OOF AUC:     {full_raw:.5f}  Δ={(full_raw-base_auc)*1e4:+.1f}bp")
    print(f"  best weights: {dict(zip(names, [round(float(x),3) for x in w_raw]))}")

    # Rank-normalised mode
    P_oof_r = np.column_stack([rankdata(c) / len(c) for c in P_oof.T])
    P_test_r = np.column_stack([rankdata(c) / len(c) for c in P_test.T])
    print("Rank-normalised mode:")
    w_rk, ho_rk, full_rk, test_rk, top10_rk = search_dirichlet(
        P_oof_r, P_test_r, y, "rank")
    print(f"  best holdout AUC: {ho_rk:.5f}")
    print(f"  full-OOF AUC:     {full_rk:.5f}  Δ={(full_rk-base_auc)*1e4:+.1f}bp")
    print(f"  best weights: {dict(zip(names, [round(float(x),3) for x in w_rk]))}")

    # Pick the better mode
    if full_rk > full_raw:
        mode, best_w, best_auc, test_blend = "rank", w_rk, full_rk, test_rk
    else:
        mode, best_w, best_auc, test_blend = "raw", w_raw, full_raw, test_raw
    print(f"\nWinner ({suffix}): {mode} mode, OOF {best_auc:.5f}")

    return dict(
        suffix=suffix, mode=mode, best_w=best_w.tolist(), names=names,
        oof_auc_full=best_auc, holdout_raw=ho_raw, full_raw=full_raw,
        holdout_rank=ho_rk, full_rank=full_rk,
        delta_bp=(best_auc - base_auc) * 1e4,
        weights_raw=dict(zip(names, [round(float(x), 4) for x in w_raw])),
        weights_rank=dict(zip(names, [round(float(x), 4) for x in w_rk])),
    ), test_blend


def main():
    res_s, test_s = run_anchor("strat", BASE_S)
    res_g, test_g = run_anchor("groupkf", BASE_G)

    sub = pd.read_csv("data/sample_submission.csv")
    sub[TARGET] = test_s
    sub.to_csv("submissions/submission_m6_dirichlet.csv", index=False)

    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    np.save(ART / "test_m6_dirichlet_strat.npy",
            np.column_stack([1 - test_s, test_s]))

    (ART / "m6_dirichlet_results.json").write_text(json.dumps(
        dict(strat=res_s, groupkf=res_g), indent=2))

    body = (
        f"# M6 — Dirichlet random search blend ({N_CAND} candidates, α=1)\n\n"
        f"Pool: {res_s['names']}\n\n"
        f"## Two-anchor winners\n\n"
        f"| anchor | mode | OOF | Δ vs base |\n|---|---|---:|---:|\n"
        f"| Strat | {res_s['mode']} | {res_s['oof_auc_full']:.5f} | "
        f"{res_s['delta_bp']:+.1f}bp |\n"
        f"| GroupKF | {res_g['mode']} | {res_g['oof_auc_full']:.5f} | "
        f"{res_g['delta_bp']:+.1f}bp |\n\n"
        f"## Best weights — Strat\n\n"
        f"raw mode: {res_s['weights_raw']}\n\n"
        f"rank mode: {res_s['weights_rank']}\n\n"
        f"## Best weights — GroupKF\n\n"
        f"raw mode: {res_g['weights_raw']}\n\n"
        f"rank mode: {res_g['weights_rank']}\n\n"
        f"## Verdict\n\n"
        f"Submission file uses Strat winner ({res_s['mode']} mode).\n"
        f"Compare against M5 LR meta to choose the LB candidate.\n"
    )
    Path("audit/2026-05-04-m6-dirichlet.md").write_text(body)


if __name__ == "__main__":
    main()
