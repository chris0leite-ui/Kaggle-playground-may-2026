"""Day-9d — FM hyperparameter sweep + 3-seed bag of the winner.

Runs 5 FM variants then a 3-seed bag of the best config. Per-variant:
std OOF, ρ vs PRIMARY (d9c K=20 swap+FM), min-meta vs PRIMARY.

Variants:
  v0_k4              embed_dim=4 (lower rank)
  v1_k8_baseline     embed_dim=8 (= d9c)
  v2_k16             embed_dim=16 (higher rank)
  v3_k8_wd1e5        embed_dim=8, weight_decay=1e-5 (heavier reg via L2 on dense bias)
  v4_k8_ep10         embed_dim=8, 10 epochs (longer training)

Then a 3-seed bag of the best variant (rank-averaged test).

Strat-only 5-fold SEED=42. PyTorch CPU. Wall ≈ 5 × 56s + 3 × 56s = ~7 min.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
N_HASH_BITS = 18

# d9c K=20 swap+FM is the new PRIMARY (LB 0.95029)
PRIMARY_OOF_PATH = ART / "oof_d9c_Sd_K20_swap_FM_strat.npy"
PRIMARY_TEST_PATH = ART / "test_d9c_Sd_K20_swap_FM_strat.npy"
PRIMARY_S = 0.95070  # Strat OOF of d9c Sd
PRIMARY_LB = 0.95029
RHO_TIE = 0.999

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


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


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


def _quantile_bin(arr_train, arr_query, n_bins):
    edges = np.quantile(arr_train, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                   0, n_bins - 1).astype(int)


def build_main_effect_hashes(train, test, tr_idx):
    D_tr, D_te = train["Driver"].astype(str).values, test["Driver"].astype(str).values
    C_tr, C_te = train["Compound"].astype(str).values, test["Compound"].astype(str).values
    R_tr, R_te = train["Race"].astype(str).values, test["Race"].astype(str).values
    Y_tr, Y_te = train["Year"].astype(str).values, test["Year"].astype(str).values
    S_tr = train["Stint"].clip(upper=5).astype(int).astype(str).values
    S_te = test["Stint"].clip(upper=5).astype(int).astype(str).values
    Tq_tr = _quantile_bin(train["TyreLife"].values[tr_idx],
                          train["TyreLife"].values, 5).astype(str)
    Tq_te = _quantile_bin(train["TyreLife"].values[tr_idx],
                          test["TyreLife"].values, 5).astype(str)
    Rq_tr = _quantile_bin(train["RaceProgress"].values[tr_idx],
                          train["RaceProgress"].values, 5).astype(str)
    Rq_te = _quantile_bin(train["RaceProgress"].values[tr_idx],
                          test["RaceProgress"].values, 5).astype(str)
    Pq_tr = _quantile_bin(train["Position"].values[tr_idx],
                          train["Position"].values, 5).astype(str)
    Pq_te = _quantile_bin(train["Position"].values[tr_idx],
                          test["Position"].values, 5).astype(str)

    def rows(arrs):
        out = []
        for i in range(len(arrs[0])):
            out.append([f"{prefix}={a[i]}" for prefix, a in
                        zip(["D", "C", "R", "Y", "S", "T", "Rp", "P"], arrs)])
        return out
    h = FeatureHasher(n_features=2**N_HASH_BITS, input_type="string",
                      alternate_sign=False)
    Xtr = h.transform(rows([D_tr, C_tr, R_tr, Y_tr, S_tr, Tq_tr, Rq_tr, Pq_tr]))
    Xte = h.transform(rows([D_te, C_te, R_te, Y_te, S_te, Tq_te, Rq_te, Pq_te]))
    return Xtr, Xte


def csr_to_index_array(csr: csr_matrix) -> np.ndarray:
    n = csr.shape[0]
    indptr, indices = csr.indptr, csr.indices
    expected = 8
    out = np.zeros((n, expected), dtype=np.int64)
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        row = indices[s:e]
        if len(row) >= expected:
            out[i] = row[:expected]
        else:
            out[i, :len(row)] = row
            out[i, len(row):] = row[0] if len(row) else 0
    return out


class FMModel(nn.Module):
    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        nn.init.zeros_(self.linear.weight)
        self.embed = nn.Embedding(n_features, embed_dim, sparse=True)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, idx_batch):
        lin = self.linear(idx_batch).sum(dim=1).squeeze(-1)
        v = self.embed(idx_batch)
        sum_v = v.sum(dim=1)
        sum_v_sq = (v * v).sum(dim=1)
        inter = 0.5 * (sum_v.pow(2).sum(dim=1) - sum_v_sq.sum(dim=1))
        return self.bias + lin + inter


def fit_fm_one_fold(idx_tr_full, idx_te, y, tr_idx, va_idx, params, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n_features = 2 ** N_HASH_BITS
    model = FMModel(n_features, params["embed_dim"])
    opt_dense = torch.optim.Adam([model.bias], lr=params["lr"])
    opt_sparse = torch.optim.SparseAdam(
        [model.linear.weight, model.embed.weight], lr=params["lr"])
    bce = nn.BCEWithLogitsLoss()
    idx_tr = torch.from_numpy(idx_tr_full[tr_idx])
    y_tr = torch.from_numpy(y[tr_idx].astype(np.float32))
    idx_va = torch.from_numpy(idx_tr_full[va_idx])
    idx_te_t = torch.from_numpy(idx_te)
    n_tr = len(idx_tr)
    perm = np.arange(n_tr)
    BATCH = params["batch"]
    wd = params.get("wd", 0.0)
    for ep in range(params["epochs"]):
        np.random.shuffle(perm)
        for s in range(0, n_tr, BATCH):
            b_idx = perm[s:s + BATCH]
            xb = idx_tr[b_idx]; yb = y_tr[b_idx]
            logits = model(xb)
            loss = bce(logits, yb)
            if wd > 0:
                loss = loss + wd * model.bias.pow(2).sum()
            opt_dense.zero_grad(); opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step(); opt_sparse.step()
    with torch.no_grad():
        chunks = []
        for s in range(0, len(idx_te_t), BATCH):
            chunks.append(torch.sigmoid(model(idx_te_t[s:s + BATCH])).numpy())
        p_te = np.concatenate(chunks)
        chunks = []
        for s in range(0, len(idx_va), BATCH):
            chunks.append(torch.sigmoid(model(idx_va[s:s + BATCH])).numpy())
        p_va = np.concatenate(chunks)
    return p_va.astype(np.float64), p_te.astype(np.float64)


def train_fm(train, test, y, splits, params, seed):
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        Xtr_csr, Xte_csr = build_main_effect_hashes(train, test, tr)
        idx_tr_full = csr_to_index_array(Xtr_csr)
        idx_te = csr_to_index_array(Xte_csr)
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va, params,
                                     seed=seed + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
    return oof, test_avg


def evaluate_base(label, oof, test_avg, y, primary_oof, primary_test, results):
    auc = float(roc_auc_score(y, oof))
    rho_test, _ = spearmanr(test_avg, primary_test)
    F_min = expand(np.column_stack([primary_oof, oof]))
    F_min_t = expand(np.column_stack([primary_test, test_avg]))
    mo, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo))
    delta = (auc_min - PRIMARY_S) * 1e4
    print(f"  {label:<22s} std OOF {auc:.5f}  ρ vs PRIMARY {rho_test:.5f}  "
          f"min-meta {auc_min:.5f}  Δ {delta:+.2f}bp  "
          f"{'PASS ✓' if auc_min >= PRIMARY_S else 'FAIL ✗'}")
    results[label] = dict(std_oof=auc, rho_vs_primary=float(rho_test),
                          min_meta_oof=auc_min, delta_primary_bp=float(delta),
                          min_meta_pass=bool(auc_min >= PRIMARY_S))
    return oof, test_avg


def stack_eval(label, fm_oof, fm_test, y, primary_test, base_oof, base_test,
               base_names, d9_oof, d9_test, d9_names, results):
    Xs = base_oof + d9_oof + [fm_oof]
    Ts = base_test + d9_test + [fm_test]
    Ns = base_names + d9_names + ["FM_variant"]
    P_oof = np.column_stack(Xs); P_test = np.column_stack(Ts)
    F_oof = expand(P_oof); F_test = expand(P_test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    test_pred = lr_full.predict_proba(F_test)[:, 1]
    auc = float(roc_auc_score(y, meta_oof))
    rho, _ = spearmanr(test_pred, primary_test)
    delta_oof = (auc - PRIMARY_S) * 1e4
    print(f"  {label:<22s} K=20 swap OOF {auc:.5f}  Δ PRIMARY {delta_oof:+.2f}bp  "
          f"ρ vs PRIMARY {rho:.5f}")
    results[f"stack_{label}"] = dict(K=20, strat_oof=auc,
                                     delta_primary_bp=float(delta_oof),
                                     rho_vs_primary=float(rho))
    return meta_oof, test_pred


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = np.load(PRIMARY_OOF_PATH)[:, 1].astype(np.float64)
    primary_test = np.load(PRIMARY_TEST_PATH)[:, 1].astype(np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    base_oof, base_test, base_names = [], [], []
    for label, fname in POOL_KEEP:
        base_oof.append(np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64))
        base_test.append(np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64))
        base_names.append(label)
    d9_oof, d9_test, d9_names = [], [], []
    for label, fname in TOP_3_D9:
        d9_oof.append(np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64))
        d9_test.append(np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64))
        d9_names.append(label)

    BATCH = 8192
    LR = 0.05
    variants = [
        ("v0_k4",         dict(embed_dim=4,  epochs=6,  lr=LR, batch=BATCH, wd=0.0)),
        ("v1_k8_baseline", dict(embed_dim=8,  epochs=6,  lr=LR, batch=BATCH, wd=0.0)),
        ("v2_k16",        dict(embed_dim=16, epochs=6,  lr=LR, batch=BATCH, wd=0.0)),
        ("v3_k8_wd1e5",   dict(embed_dim=8,  epochs=6,  lr=LR, batch=BATCH, wd=1e-5)),
        ("v4_k8_ep10",    dict(embed_dim=8,  epochs=10, lr=LR, batch=BATCH, wd=0.0)),
    ]

    results = {}
    bases = {}
    print("\n=== FM hyperparameter sweep ===\n")
    for label, params in variants:
        t_v = time.time()
        oof, test_avg = train_fm(train, test, y, splits, params, seed=42)
        evaluate_base(label, oof, test_avg, y, primary_oof, primary_test, results)
        np.save(ART / f"oof_d9d_{label}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / f"test_d9d_{label}_strat.npy",
                np.column_stack([1 - test_avg, test_avg]))
        bases[label] = (oof, test_avg)
        print(f"  ({time.time()-t_v:.0f}s wall)\n")

    # Pick best by std OOF (best generalization signal)
    best_label = max(bases.keys(), key=lambda lbl: results[lbl]["std_oof"])
    best_params = dict(variants)[best_label]
    print(f"\n=== Best single variant: {best_label} (std OOF "
          f"{results[best_label]['std_oof']:.5f}) ===")

    # Bag the best across 3 seeds (rank-average)
    print(f"\n=== 3-seed rank-average bag of {best_label} ===\n")
    all_oof_seeds = [bases[best_label][0]]
    all_test_seeds = [bases[best_label][1]]
    for seed in [123, 456]:
        t_v = time.time()
        oof, test_avg = train_fm(train, test, y, splits, best_params, seed=seed)
        all_oof_seeds.append(oof); all_test_seeds.append(test_avg)
        s = float(roc_auc_score(y, oof))
        print(f"  seed={seed}: std OOF {s:.5f}  ({time.time()-t_v:.0f}s)")
    # Rank-average across seeds
    oof_bag = np.mean([rankdata(o) / len(o) for o in all_oof_seeds], axis=0)
    test_bag = np.mean([rankdata(t) / len(t) for t in all_test_seeds], axis=0)
    evaluate_base("bag3_seeds", oof_bag, test_bag, y, primary_oof, primary_test,
                  results)
    np.save(ART / "oof_d9d_bag3_strat.npy",
            np.column_stack([1 - oof_bag, oof_bag]))
    np.save(ART / "test_d9d_bag3_strat.npy",
            np.column_stack([1 - test_bag, test_bag]))

    # K=20 swap stacks for each FM variant + bag
    print(f"\n=== K=20 swap stacks (PRIMARY-keep + R6/R10/R7 + FM_variant) ===\n")
    stack_outputs = {}
    for label, (oof, test_avg) in bases.items():
        mo, tp = stack_eval(label, oof, test_avg, y, primary_test,
                            base_oof, base_test, base_names,
                            d9_oof, d9_test, d9_names, results)
        stack_outputs[label] = (mo, tp)
    mo_bag, tp_bag = stack_eval("bag3_seeds", oof_bag, test_bag, y, primary_test,
                                base_oof, base_test, base_names,
                                d9_oof, d9_test, d9_names, results)
    stack_outputs["bag3_seeds"] = (mo_bag, tp_bag)

    # Build submission CSVs for the top stack candidates
    # Best stack by OOF
    stack_results_only = {k: v for k, v in results.items() if k.startswith("stack_")}
    best_stack = max(stack_results_only.keys(),
                     key=lambda k: stack_results_only[k]["strat_oof"])
    best_stack_label = best_stack.replace("stack_", "")
    print(f"\n=== Best stack: {best_stack} (OOF "
          f"{stack_results_only[best_stack]['strat_oof']:.5f}, "
          f"Δ +{stack_results_only[best_stack]['delta_primary_bp']:.2f}bp) ===")
    mo_b, tp_b = stack_outputs[best_stack_label]
    np.save(ART / f"oof_d9d_K20_swap_{best_stack_label}_strat.npy",
            np.column_stack([1 - mo_b, mo_b]))
    np.save(ART / f"test_d9d_K20_swap_{best_stack_label}_strat.npy",
            np.column_stack([1 - tp_b, tp_b]))
    sub = sample_sub.copy(); sub[TARGET] = tp_b
    sub_path = f"submissions/submission_d9d_K20_swap_{best_stack_label}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"→ wrote {sub_path}")

    # Also write the bag stack if not already best
    if best_stack_label != "bag3_seeds":
        mo_b, tp_b = stack_outputs["bag3_seeds"]
        sub = sample_sub.copy(); sub[TARGET] = tp_b
        sub.to_csv("submissions/submission_d9d_K20_swap_bag3_seeds.csv", index=False)
        print("→ wrote submissions/submission_d9d_K20_swap_bag3_seeds.csv")

    final = dict(
        results=results,
        best_single=best_label,
        best_stack=best_stack,
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        total_wall_s=time.time() - t0,
    )
    (ART / "d9d_fm_sweep_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9d_fm_sweep_results.json  "
          f"(total wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
