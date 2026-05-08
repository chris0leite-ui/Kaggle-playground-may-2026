"""scripts/probe_pool_structure.py — H1' + H2' combined.

H1': effective rank of K=27 pool (extends Day-18 PM K=24 SVD diagnostic
to the current pool); pairwise rho matrix; diversity-greedy selection.

H2': sparse-pool meta tests. For sparse subsets {K=3, K=5, K=7, K=10},
fit BOTH plain LR-meta and Path-B Compound x Stint hier-meta (tau=100k).
Compare to current PRIMARY (K=27 + Path-B Compound x Stint, tau=100k).

Cost: ~12-18 min (SVD <30s; 4 plain LR-metas + 4 Path-B sweeps).

Outputs scripts/artifacts/probe_pool_structure.json + sparse OOFs.
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

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS, MAX_ITER = 42, 5, 500
MIN_ROWS = 1000

K27_BASES = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub", "e3_hgbc",
    "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit", "f1_hgbc_deep",
    "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide", "cb_slow-wide-bag",
    "realmlp", "d6_rule_driver_compound", "d6_rule_year_race",
    "d9_R6_next_compound", "d9_R10_driver_eb", "d9_R7_prev_compound",
    "d9f_FM_A", "d9f_FM_B",
    "p1_single_cb_v4_gpu", "d17_h1d_yekenot_full",
    "d16_orig_continuous_only", "d18_chain_decomp",
    "d18_e2_preimage_knn", "d18_f2_constraint",
]
# E9-style forward-select K=10 (per t2_k10_primary.py)
K10_FORWARD_GREEDY = [
    "d17_h1d_yekenot_full", "p1_single_cb_v4_gpu", "f1_hgbc_deep",
    "d16_orig_continuous_only", "b_lapsuntilpit", "baseline_two_anchor",
    "d9_R6_next_compound", "cb_year-cat", "e5_optuna_lgbm", "d9f_FM_A",
]


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return (a[:, 1] if a.ndim == 2 else a.ravel()).astype(np.float64)


def load_pool() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    oofs, tests = [], []
    for n in K27_BASES:
        oofs.append(_pos(ART / f"oof_{n}_strat.npy"))
        tests.append(_pos(ART / f"test_{n}_strat.npy"))
    return (np.column_stack(oofs), np.column_stack(tests), y,
            train, test)


def expand(P: np.ndarray) -> np.ndarray:
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    return np.hstack([P, rk, np.log(Pc / (1 - Pc))])


def eff_rank(M: np.ndarray) -> dict:
    sv = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    p = (sv ** 2) / (sv ** 2).sum()
    eff = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    cs = np.cumsum(p)
    rk = lambda thr: int(np.searchsorted(cs, thr) + 1)
    stable = float((sv ** 2).sum() / (sv[0] ** 2))
    return {"eff_rank_entropy": eff, "rank_90": rk(0.90),
            "rank_95": rk(0.95), "rank_99": rk(0.99),
            "stable_rank": stable, "cond_number": float(sv[0] / sv[-1]),
            "var_top5": float(cs[min(4, len(cs) - 1)] * 100),
            "var_top10": float(cs[min(9, len(cs) - 1)] * 100),
            "singular_top10": [round(float(s), 4) for s in sv[:10]]}


def diversity_greedy(P: np.ndarray, y: np.ndarray, names: list[str],
                     k: int) -> list[int]:
    """Pick k bases: start with highest-AUC, then add max-min-rho."""
    aucs = [roc_auc_score(y, P[:, j]) for j in range(P.shape[1])]
    selected = [int(np.argmax(aucs))]
    print(f"  k=1: {names[selected[0]]} (auc={aucs[selected[0]]:.4f})")
    rho_matrix = np.corrcoef(P.T)
    while len(selected) < k:
        remaining = [j for j in range(len(names)) if j not in selected]
        # for each remaining, max abs rho to any selected
        best_j, best_score = -1, -np.inf
        for j in remaining:
            max_rho_to_sel = max(abs(rho_matrix[j, s]) for s in selected)
            # score: lower max-rho = more diverse
            score = -max_rho_to_sel
            if score > best_score:
                best_score, best_j = score, j
        selected.append(best_j)
        print(f"  k={len(selected)}: {names[best_j]} "
              f"(auc={aucs[best_j]:.4f}, max-rho-to-sel={-best_score:.4f})")
    return selected


def fit_plain_meta(F_oof: np.ndarray, y: np.ndarray,
                   splits: list) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    return oof


def fit_lr_aug(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
    lr.fit(F, y)
    return np.concatenate([lr.intercept_, lr.coef_.ravel()])


def fit_path_b(F_oof: np.ndarray, y: np.ndarray, seg_train: np.ndarray,
               splits: list, n_seg: int, tau: float) -> np.ndarray:
    """Per-segment LR with shrinkage tau toward global LR."""
    oof = np.zeros(len(y))
    for tr_idx, va_idx in splits:
        w_global = fit_lr_aug(F_oof[tr_idx], y[tr_idx])
        W_local = np.zeros((n_seg, len(w_global)))
        counts = np.zeros(n_seg, dtype=np.int64)
        mask = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train[tr_idx] == s)[0]
            counts[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[tr_idx][idx])) < 2:
                continue
            W_local[s] = fit_lr_aug(F_oof[tr_idx][idx], y[tr_idx][idx])
            mask[s] = True
        n_local = counts.astype(np.float64)
        alpha = n_local / (n_local + tau)
        W_shrunk = (alpha[:, None] * W_local +
                    (1 - alpha[:, None]) * w_global[None, :])
        for s in np.unique(seg_train[va_idx]):
            idx = np.where(seg_train[va_idx] == s)[0]
            w = W_shrunk[s] if mask[s] else w_global
            F_aug = np.column_stack([np.ones(len(idx)), F_oof[va_idx][idx]])
            oof[va_idx[idx]] = 1.0 / (1.0 + np.exp(
                -np.clip(F_aug @ w, -30, 30)))
    return oof


def eval_pool(name: str, idxs: list[int], P_oof: np.ndarray,
              y: np.ndarray, seg_train: np.ndarray, n_seg: int,
              splits: list, prim_oof: np.ndarray) -> dict:
    Psub = P_oof[:, idxs]
    F = expand(Psub)
    print(f"\n  --- pool {name} (k={len(idxs)}) ---")
    plain = fit_plain_meta(F, y, splits)
    auc_plain = roc_auc_score(y, plain)
    print(f"  plain LR-meta OOF: {auc_plain:.5f}")
    pathb = fit_path_b(F, y, seg_train, splits, n_seg, tau=100000)
    auc_pathb = roc_auc_score(y, pathb)
    print(f"  Path-B C×S tau=100k OOF: {auc_pathb:.5f}")
    rho_plain = float(spearmanr(plain, prim_oof)[0])
    rho_pathb = float(spearmanr(pathb, prim_oof)[0])
    return {"k": len(idxs), "indices": [int(i) for i in idxs],
            "names": [K27_BASES[i] for i in idxs],
            "plain_lr_meta_oof": float(auc_plain),
            "path_b_oof": float(auc_pathb),
            "delta_path_b_minus_plain_bp": float(
                (auc_pathb - auc_plain) * 1e4),
            "rho_plain_vs_K27_primary": rho_plain,
            "rho_pathb_vs_K27_primary": rho_pathb,
            "plain_oof_arr": plain, "pathb_oof_arr": pathb}


def main() -> None:
    t0 = time.time()
    print("Loading K=27 pool ...")
    P_oof, P_test, y, train, test = load_pool()
    K = len(K27_BASES)
    print(f"K={K} bases; OOF shape {P_oof.shape}")

    # 1) SVD diagnostic
    print("\n=== SVD eff-rank on K=27 ===")
    spec = {}
    for label, M in [("probability", P_oof),
                     ("logit", np.log(np.clip(P_oof, 1e-9, 1 - 1e-9) /
                                       (1 - np.clip(P_oof, 1e-9, 1 - 1e-9))))]:
        # Center each column for SVD diagnostic on directions
        Mc = M - M.mean(axis=0, keepdims=True)
        spec[label] = eff_rank(Mc)
        s = spec[label]
        print(f"  {label:>11s}: eff_rank={s['eff_rank_entropy']:.3f}  "
              f"rank95={s['rank_95']}/{K}  rank99={s['rank_99']}/{K}  "
              f"top5={s['var_top5']:.1f}%  top10={s['var_top10']:.1f}%  "
              f"cond={s['cond_number']:.1f}")

    # Pairwise rho (logit space, more sensitive)
    L = np.log(np.clip(P_oof, 1e-9, 1 - 1e-9) /
               (1 - np.clip(P_oof, 1e-9, 1 - 1e-9)))
    rho_mat = np.corrcoef(L.T)
    # Find redundant pairs
    ij = np.triu_indices(K, k=1)
    pairs = sorted(
        [(K27_BASES[i], K27_BASES[j], float(rho_mat[i, j]))
         for i, j in zip(ij[0], ij[1])],
        key=lambda x: -abs(x[2]))
    print("\n  Top 10 most-correlated pairs (logit Pearson rho):")
    for n1, n2, r in pairs[:10]:
        print(f"    {r:+.4f}  {n1:<28s} <-> {n2}")

    # 2) Diversity-greedy at K=3, 5, 7
    print("\n=== Diversity-greedy selection (max-min-rho) ===")
    sel_3 = diversity_greedy(P_oof, y, K27_BASES, 3)
    print()
    sel_5 = diversity_greedy(P_oof, y, K27_BASES, 5)
    print()
    sel_7 = diversity_greedy(P_oof, y, K27_BASES, 7)

    # K=10 forward-greedy from existing E9 pick
    sel_10 = [K27_BASES.index(n) for n in K10_FORWARD_GREEDY]

    # 3) Build segmentation Compound x Stint
    cats = sorted(set(train["Compound"].astype(str).unique()) |
                  set(test["Compound"].astype(str).unique()))
    cmp = {c: i for i, c in enumerate(cats)}
    c_tr = train["Compound"].astype(str).map(cmp).astype(int).values
    s_tr = np.clip(train["Stint"].astype(int).values, 0, 5)
    n_seg = len(cats) * 6
    seg_train = c_tr * 6 + s_tr

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # 4) PRIMARY reference (K=27 Path-B C×S τ=100k from saved artifact)
    prim_oof = np.load(
        ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy")[:, 1]
    auc_prim = float(roc_auc_score(y, prim_oof))
    print(f"\nReference: K=27 PRIMARY OOF = {auc_prim:.5f}")

    # 5) Eval each pool
    pools = [
        ("K3_div", sel_3),
        ("K5_div", sel_5),
        ("K7_div", sel_7),
        ("K10_fwd", sel_10),
        ("K27_full", list(range(K))),
    ]
    results = {}
    for name, idxs in pools:
        r = eval_pool(name, idxs, P_oof, y, seg_train, n_seg, splits,
                      prim_oof)
        # Save sparse OOF + test arrays
        np.save(ART / f"oof_{name}_plain_strat.npy", r.pop("plain_oof_arr"))
        np.save(ART / f"oof_{name}_pathb_strat.npy", r.pop("pathb_oof_arr"))
        # Build test predictions: re-fit on full y
        Psub_test = P_test[:, idxs]
        F_train_full = expand(P_oof[:, idxs])
        F_test_full = expand(Psub_test)
        # Plain LR-meta full-train
        lr = LogisticRegression(C=1.0, max_iter=MAX_ITER, solver="lbfgs")
        lr.fit(F_train_full, y)
        test_plain = lr.predict_proba(F_test_full)[:, 1]
        np.save(ART / f"test_{name}_plain_strat.npy",
                np.column_stack([1 - test_plain, test_plain]))
        # Path-B full-train test
        cmp_te = test["Compound"].astype(str).map(cmp).astype(int).values
        s_te = np.clip(test["Stint"].astype(int).values, 0, 5)
        seg_test = cmp_te * 6 + s_te
        w_global_full = fit_lr_aug(F_train_full, y)
        W_local_full = np.zeros((n_seg, len(w_global_full)))
        counts_full = np.zeros(n_seg, dtype=np.int64)
        mask_full = np.zeros(n_seg, dtype=bool)
        for s in range(n_seg):
            idx = np.where(seg_train == s)[0]
            counts_full[s] = len(idx)
            if len(idx) < MIN_ROWS or len(np.unique(y[idx])) < 2:
                continue
            W_local_full[s] = fit_lr_aug(F_train_full[idx], y[idx])
            mask_full[s] = True
        n_local = counts_full.astype(np.float64)
        alpha = n_local / (n_local + 100000.0)
        W_shrunk = (alpha[:, None] * W_local_full +
                    (1 - alpha[:, None]) * w_global_full[None, :])
        tp = np.zeros(len(test))
        for s in np.unique(seg_test):
            idx = np.where(seg_test == s)[0]
            w = W_shrunk[s] if mask_full[s] else w_global_full
            F_aug = np.column_stack([np.ones(len(idx)),
                                     F_test_full[idx]])
            tp[idx] = 1.0 / (1.0 + np.exp(-np.clip(F_aug @ w, -30, 30)))
        np.save(ART / f"test_{name}_pathb_strat.npy",
                np.column_stack([1 - tp, tp]))
        # Also write submission CSVs for sparse PathB pools (held)
        sub = pd.read_csv("data/sample_submission.csv")
        sub[TARGET] = tp
        Path("submissions").mkdir(exist_ok=True)
        sub.to_csv(f"submissions/submission_{name}_pathb.csv", index=False)
        results[name] = r

    # 6) Summary
    print("\n=== SUMMARY ===")
    print(f"  PRIMARY (K=27 Path-B C×S τ=100k) OOF: {auc_prim:.5f}\n")
    print(f"{'pool':>10s}  {'plain':>8s}  {'PathB':>8s}  "
          f"{'Δ(B-P)':>8s}  {'ρ_pathb_vs_PRI':>15s}  {'Δ_vs_PRIMARY':>13s}")
    for name in ["K3_div", "K5_div", "K7_div", "K10_fwd", "K27_full"]:
        r = results[name]
        print(f"  {name:>8s}  {r['plain_lr_meta_oof']:.5f}  "
              f"{r['path_b_oof']:.5f}  "
              f"{r['delta_path_b_minus_plain_bp']:+8.2f}  "
              f"{r['rho_pathb_vs_K27_primary']:>15.5f}  "
              f"{(r['path_b_oof'] - auc_prim) * 1e4:+8.2f}")

    out = {
        "K27_eff_rank": spec,
        "top_redundant_pairs": [
            {"a": p[0], "b": p[1], "rho_logit": p[2]} for p in pairs[:30]
        ],
        "pools": results,
        "primary_oof_K27_path_b_cs_tau100k": auc_prim,
        "wall_s": time.time() - t0,
    }
    (ART / "probe_pool_structure.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote scripts/artifacts/probe_pool_structure.json. "
          f"Wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
