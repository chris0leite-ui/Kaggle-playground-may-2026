"""Day-12 — Compact LambdaRank meta over K=21 PRIMARY pool.

Compute-budget-aware version (≤30 min): smaller boost rounds and
more aggressive early stopping vs the slow `d12_lambdarank_meta.py`.

Tests:
  1. LR meta (LR anchor; reuse existing d9f K=21 OOF builder via re-run).
  2. LambdaMART meta with **Race groups** (matches task spec).
  3. LambdaMART meta with **random groups of ~1000** (better global-AUC alignment).
  4. CatBoost YetiRank meta with **Race groups**.

For the LR anchor we don't refit; we reuse PRIMARY's OOF/test
(already saved as test_d9f_K21_swap_strat.npy) and compute its OOF AUC
from the d9f cache.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_S = 0.95073
PRIMARY_LB = 0.95031
MAX_QUERY_LEN = 8000

POOL_KEEP = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"), ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"), ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"), ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"), ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
    ("realmlp", "realmlp"),
    ("rule_driver_compound", "d6_rule_driver_compound"),
    ("rule_year_race", "d6_rule_year_race"),
]
TOP_3_D9 = [
    ("R6_next_compound", "d9_R6_next_compound"),
    ("R10_driver_eb", "d9_R10_driver_eb"),
    ("R7_prev_compound", "d9_R7_prev_compound"),
]
PARTITION_FMS = [
    ("FM_A", "d9f_FM_A"),
    ("FM_B", "d9f_FM_B"),
]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def load_pool(names_files):
    Xs_oof, Xs_test, names = [], [], []
    for label, fname in names_files:
        oo = np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64)
        Xs_oof.append(oo); Xs_test.append(te); names.append(label)
    return Xs_oof, Xs_test, names


def build_groups_chunked(group_ids, idx, max_query_len=MAX_QUERY_LEN):
    g_sub = group_ids[idx]
    order = np.argsort(g_sub, kind="stable")
    g_sorted = g_sub[order]
    boundaries = np.where(np.diff(g_sorted) != 0)[0] + 1
    counts_raw = np.diff(np.concatenate([[0], boundaries, [len(g_sorted)]]))
    counts_split = []
    for c in counts_raw:
        if c <= max_query_len:
            counts_split.append(int(c))
        else:
            n_chunks = int(np.ceil(c / max_query_len))
            base = c // n_chunks; rem = c - base * n_chunks
            for j in range(n_chunks):
                counts_split.append(int(base + (1 if j < rem else 0)))
    return np.asarray(counts_split, dtype=np.int32), idx[order]


def build_random_groups(idx, group_size=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    perm_local = rng.permutation(len(idx))
    sorted_idx = idx[perm_local]
    n = len(sorted_idx)
    n_groups = max(1, n // group_size)
    base = n // n_groups; rem = n - base * n_groups
    counts = np.array([base + (1 if g < rem else 0) for g in range(n_groups)],
                      dtype=np.int32)
    return counts, sorted_idx


def fit_lr_meta(F_oof, F_test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1]


def fit_lambdarank_meta_fast(F_oof, F_test, y, group_builder, label="LM"):
    """Compact lambdarank: lr=0.1, num_leaves=8, max_rounds=300, ES=20."""
    params = dict(
        objective="lambdarank", metric=["auc"],
        learning_rate=0.1, num_leaves=8, min_data_in_leaf=200,
        lambda_l2=1.0, feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=5, verbose=-1, seed=SEED,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(F_test.shape[0], dtype=np.float64)
    biters = []
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr_counts, tr_perm = group_builder(tr)
        va_counts, va_perm = group_builder(va)
        Xtr = F_oof[tr_perm]; ytr = y[tr_perm]
        Xva = F_oof[va_perm]; yva = y[va_perm]
        dtrain = lgb.Dataset(Xtr, label=ytr, group=tr_counts)
        dval = lgb.Dataset(Xva, label=yva, group=va_counts, reference=dtrain)
        booster = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20, verbose=False,
                                          first_metric_only=True),
                       lgb.log_evaluation(0)],
        )
        p_va = booster.predict(Xva, num_iteration=booster.best_iteration)
        meta_oof[va_perm] = p_va
        test_pred += booster.predict(F_test,
                                     num_iteration=booster.best_iteration) / N_FOLDS
        biters.append(int(booster.best_iteration))
        print(f"  [{label}] fold {k}: iters={biters[-1]} "
              f"AUC={roc_auc_score(yva, p_va):.5f}", flush=True)
    return meta_oof, test_pred, biters


def fit_yetirank_fast(F_oof, F_test, y, group_ids, label="YT"):
    """Compact YetiRank: 200 iters, lr=0.1, depth=4."""
    from catboost import CatBoost, Pool
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(F_test.shape[0], dtype=np.float64)
    biters = []
    F_oof32 = F_oof.astype(np.float32)
    F_test32 = F_test.astype(np.float32)
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr_perm = tr[np.argsort(group_ids[tr], kind="stable")]
        va_perm = va[np.argsort(group_ids[va], kind="stable")]
        tr_pool = Pool(F_oof32[tr_perm], y[tr_perm],
                       group_id=group_ids[tr_perm])
        va_pool = Pool(F_oof32[va_perm], y[va_perm],
                       group_id=group_ids[va_perm])
        m = CatBoost({
            "loss_function": "YetiRank",
            "iterations": 200, "learning_rate": 0.1, "depth": 4,
            "l2_leaf_reg": 3.0,
            "od_type": "Iter", "od_wait": 30,
            "random_seed": SEED, "verbose": False, "thread_count": -1,
        })
        m.fit(tr_pool, eval_set=va_pool, verbose=False)
        meta_oof[va] = m.predict(F_oof32[va])
        test_pred += m.predict(F_test32) / N_FOLDS
        biters.append(int(m.tree_count_))
        print(f"  [{label}] fold {k}: iters={biters[-1]} "
              f"AUC={roc_auc_score(y[va], meta_oof[va]):.5f}", flush=True)
    return meta_oof, test_pred, biters


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    race_ids_train = pd.Categorical(train["Race"]).codes.astype(np.int32)

    base_oof, base_test, base_names = load_pool(POOL_KEEP)
    d9_oof, d9_test, d9_names = load_pool(TOP_3_D9)
    fm_oof, fm_test, fm_names = load_pool(PARTITION_FMS)
    Xs_oof = base_oof + d9_oof + fm_oof
    Xs_test = base_test + d9_test + fm_test
    assert len(Xs_oof) == 21
    P_oof = np.column_stack(Xs_oof); P_test = np.column_stack(Xs_test)
    F_oof = expand(P_oof); F_test = expand(P_test)
    print(f"K=21  F_oof={F_oof.shape}  F_test={F_test.shape}", flush=True)
    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy"
                           )[:, 1].astype(np.float64)

    def gb_race(idx): return build_groups_chunked(race_ids_train, idx)
    def gb_random(idx): return build_random_groups(idx, 1000, SEED)

    results = {}

    # === LR meta (anchor) — reuse existing PRIMARY OOF if cached, else recompute ===
    print("\n=== LR meta (anchor) ===", flush=True)
    t = time.time()
    lr_oof_path = ART / "oof_d12_lr_meta_strat.npy"
    if lr_oof_path.exists():
        lr_oof = np.load(lr_oof_path)[:, 1].astype(np.float64)
        lr_test = np.load(ART / "test_d12_lr_meta_strat.npy")[:, 1].astype(np.float64)
        print(f"  reusing cached d12 LR meta")
    else:
        lr_oof, lr_test = fit_lr_meta(F_oof, F_test, y)
        np.save(lr_oof_path, np.column_stack([1 - lr_oof, lr_oof]))
        np.save(ART / "test_d12_lr_meta_strat.npy",
                np.column_stack([1 - lr_test, lr_test]))
    lr_auc = float(roc_auc_score(y, lr_oof))
    lr_rho_prim, _ = spearmanr(lr_test, primary_test)
    print(f"  Strat OOF: {lr_auc:.5f}  Δ PRIMARY {(lr_auc - PRIMARY_S)*1e4:+.2f}bp"
          f"  ρ vs PRIMARY: {lr_rho_prim:.5f}  wall {time.time()-t:.1f}s", flush=True)
    results["lr_meta"] = dict(strat_oof=lr_auc,
                              delta_primary_bp=(lr_auc - PRIMARY_S) * 1e4,
                              rho_vs_primary_test=float(lr_rho_prim))

    # === LambdaMART (Race-grouped, fast) ===
    print("\n=== LambdaMART meta (Race-grouped, fast) ===", flush=True)
    t = time.time()
    lm_oof, lm_test, biters_lm = fit_lambdarank_meta_fast(
        F_oof, F_test, y, gb_race, "LM-Race")
    lm_auc = float(roc_auc_score(y, lm_oof))
    lm_rho_prim, _ = spearmanr(lm_test, primary_test)
    lm_rho_lr, _ = spearmanr(lm_test, lr_test)
    print(f"  Strat OOF: {lm_auc:.5f}  Δ PRIMARY {(lm_auc - PRIMARY_S)*1e4:+.2f}bp",
          flush=True)
    print(f"  ρ vs PRIMARY: {lm_rho_prim:.5f}  ρ vs LR-meta: {lm_rho_lr:.5f}", flush=True)
    print(f"  best_iters: {biters_lm}  wall {time.time()-t:.1f}s", flush=True)
    results["lambdamart_race"] = dict(
        strat_oof=lm_auc, delta_primary_bp=(lm_auc - PRIMARY_S) * 1e4,
        rho_vs_primary_test=float(lm_rho_prim),
        rho_vs_lr_test=float(lm_rho_lr), best_iters=biters_lm)

    # === LambdaMART (random groups ~1000, fast) ===
    print("\n=== LambdaMART meta (random groups ~1000, fast) ===", flush=True)
    t = time.time()
    lmr_oof, lmr_test, biters_lmr = fit_lambdarank_meta_fast(
        F_oof, F_test, y, gb_random, "LM-Rand")
    lmr_auc = float(roc_auc_score(y, lmr_oof))
    lmr_rho_prim, _ = spearmanr(lmr_test, primary_test)
    lmr_rho_lr, _ = spearmanr(lmr_test, lr_test)
    print(f"  Strat OOF: {lmr_auc:.5f}  Δ PRIMARY {(lmr_auc - PRIMARY_S)*1e4:+.2f}bp",
          flush=True)
    print(f"  ρ vs PRIMARY: {lmr_rho_prim:.5f}  ρ vs LR-meta: {lmr_rho_lr:.5f}",
          flush=True)
    print(f"  best_iters: {biters_lmr}  wall {time.time()-t:.1f}s", flush=True)
    results["lambdamart_rand1000"] = dict(
        strat_oof=lmr_auc, delta_primary_bp=(lmr_auc - PRIMARY_S) * 1e4,
        rho_vs_primary_test=float(lmr_rho_prim),
        rho_vs_lr_test=float(lmr_rho_lr), best_iters=biters_lmr)

    # === YetiRank (Race-grouped, fast) ===
    print("\n=== YetiRank meta (Race-grouped, fast) ===", flush=True)
    t = time.time()
    yt_oof, yt_test, biters_yt = fit_yetirank_fast(
        F_oof, F_test, y, race_ids_train, "YT-Race")
    yt_auc = float(roc_auc_score(y, yt_oof))
    yt_rho_prim, _ = spearmanr(yt_test, primary_test)
    yt_rho_lr, _ = spearmanr(yt_test, lr_test)
    print(f"  Strat OOF: {yt_auc:.5f}  Δ PRIMARY {(yt_auc - PRIMARY_S)*1e4:+.2f}bp",
          flush=True)
    print(f"  ρ vs PRIMARY: {yt_rho_prim:.5f}  ρ vs LR-meta: {yt_rho_lr:.5f}",
          flush=True)
    print(f"  best_iters: {biters_yt}  wall {time.time()-t:.1f}s", flush=True)
    results["yetirank_race"] = dict(
        strat_oof=yt_auc, delta_primary_bp=(yt_auc - PRIMARY_S) * 1e4,
        rho_vs_primary_test=float(yt_rho_prim),
        rho_vs_lr_test=float(yt_rho_lr), best_iters=biters_yt)

    # Save per-meta artifacts
    saves = [
        ("lambdarank_meta", lm_oof, lm_test),
        ("lambdarank_rand1000", lmr_oof, lmr_test),
        ("yetirank_meta", yt_oof, yt_test),
    ]
    for nm, oof, te in saves:
        np.save(ART / f"oof_d12_{nm}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d12_{nm}_strat.npy",
                np.column_stack([1 - te, te]))

    sub = sample_sub.copy(); sub[TARGET] = lm_test
    sub.to_csv("submissions/submission_d12_lambdarank_meta.csv", index=False)
    sub2 = sample_sub.copy(); sub2[TARGET] = lmr_test
    sub2.to_csv("submissions/submission_d12_lambdarank_rand1000.csv", index=False)
    sub3 = sample_sub.copy(); sub3[TARGET] = yt_test
    sub3.to_csv("submissions/submission_d12_yetirank_meta.csv", index=False)

    best_name = max(results.keys(), key=lambda k: results[k]["strat_oof"])
    print(f"\n→ Best meta by OOF: {best_name} "
          f"(OOF {results[best_name]['strat_oof']:.5f})", flush=True)

    final = dict(results=results, best_meta_by_oof=best_name,
                 primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB,
                              name="d9f_K21_swap_partA_partB"),
                 K=21, total_wall_s=time.time() - t0)
    (ART / "d12_lambdarank_meta_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d12_lambdarank_meta_results.json  "
          f"(wall {time.time()-t0:.0f}s)", flush=True)
    print("\n" + "=" * 78)
    print(f"{'meta':<24s} {'OOF':>8s} {'Δprim':>8s} {'ρ_PRIM':>8s} {'ρ_LR':>8s}")
    print("-" * 78)
    for nm in ("lr_meta", "lambdamart_race", "lambdamart_rand1000", "yetirank_race"):
        r = results[nm]
        rho_lr = r.get("rho_vs_lr_test", float("nan"))
        print(f"{nm:<24s} {r['strat_oof']:>8.5f} "
              f"{r['delta_primary_bp']:>+7.2f} "
              f"{r['rho_vs_primary_test']:>8.5f} "
              f"{rho_lr:>8.5f}")


if __name__ == "__main__":
    main()
