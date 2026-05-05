"""Day-9f — Multi-FM with disjoint feature partitions.

Two FMs trained on different feature subsets, both stacked. Tests
whether feature-partition diversity earns FM-class L1 weight beyond
the d9c FM's unified 8-feature model.

Partition (motivated by domain semantics):
  FM_A "driver-dynamics":  D, C, S, T_q5  (driver/compound/stint/tyre)
  FM_B "race-context":     R, Y, Rp_q5, P_q5  (race/year/race-progress/position)

Two stacks:
  S_swap K=21: PRIMARY-keep + R6/R10/R7 + FM_A + FM_B  (drop d9c FM)
  S_add  K=22: PRIMARY-keep + R6/R10/R7 + FM_d9c + FM_A + FM_B  (keep all 3)

Reuses d9c FM code via build_main_effect_hashes-style partition hashes.
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
EMBED_DIM = 8       # same as d9c FM
EPOCHS = 6
BATCH = 8192
LR = 0.05
PRIMARY_S = 0.95070
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

# Partition definitions — (name, list_of_field_specs)
PARTITION_A = ["D", "C", "S", "T"]      # driver-dynamics
PARTITION_B = ["R", "Y", "Rp", "P"]     # race-context


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
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def _quantile_bin(arr_train, arr_query, n_bins):
    edges = np.quantile(arr_train, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                   0, n_bins - 1).astype(int)


def build_partition_hashes(train, test, tr_idx, fields):
    """Build (n_rows, len(fields)) hashed indices for the named field subset.

    field name → array source mapping:
      D → Driver, C → Compound, R → Race, Y → Year,
      S → Stint (clip 5), T → TyreLife q5, Rp → RaceProgress q5,
      P → Position q5.
    """
    n_tr, n_te = len(train), len(test)
    arrays_tr, arrays_te = [], []
    for f in fields:
        if f == "D":
            arrays_tr.append(train["Driver"].astype(str).values)
            arrays_te.append(test["Driver"].astype(str).values)
        elif f == "C":
            arrays_tr.append(train["Compound"].astype(str).values)
            arrays_te.append(test["Compound"].astype(str).values)
        elif f == "R":
            arrays_tr.append(train["Race"].astype(str).values)
            arrays_te.append(test["Race"].astype(str).values)
        elif f == "Y":
            arrays_tr.append(train["Year"].astype(str).values)
            arrays_te.append(test["Year"].astype(str).values)
        elif f == "S":
            arrays_tr.append(train["Stint"].clip(upper=5).astype(int).astype(str).values)
            arrays_te.append(test["Stint"].clip(upper=5).astype(int).astype(str).values)
        elif f == "T":
            arrays_tr.append(_quantile_bin(train["TyreLife"].values[tr_idx],
                                            train["TyreLife"].values, 5).astype(str))
            arrays_te.append(_quantile_bin(train["TyreLife"].values[tr_idx],
                                            test["TyreLife"].values, 5).astype(str))
        elif f == "Rp":
            arrays_tr.append(_quantile_bin(train["RaceProgress"].values[tr_idx],
                                            train["RaceProgress"].values, 5).astype(str))
            arrays_te.append(_quantile_bin(train["RaceProgress"].values[tr_idx],
                                            test["RaceProgress"].values, 5).astype(str))
        elif f == "P":
            arrays_tr.append(_quantile_bin(train["Position"].values[tr_idx],
                                            train["Position"].values, 5).astype(str))
            arrays_te.append(_quantile_bin(train["Position"].values[tr_idx],
                                            test["Position"].values, 5).astype(str))
        else:
            raise ValueError(f"Unknown field {f}")

    def rows(arrs, n):
        return [[f"{p}={a[i]}" for p, a in zip(fields, arrs)] for i in range(n)]
    h = FeatureHasher(n_features=2**N_HASH_BITS, input_type="string",
                      alternate_sign=False)
    return (h.transform(rows(arrays_tr, n_tr)),
            h.transform(rows(arrays_te, n_te)))


def csr_to_index_array(csr, n_active):
    n = csr.shape[0]
    indptr, indices = csr.indptr, csr.indices
    out = np.zeros((n, n_active), dtype=np.int64)
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        row = indices[s:e]
        if len(row) >= n_active:
            out[i] = row[:n_active]
        else:
            out[i, :len(row)] = row
            out[i, len(row):] = row[0] if len(row) else 0
    return out


class FMModel(nn.Module):
    def __init__(self, n_features, embed_dim):
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


def fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n_features = 2 ** N_HASH_BITS
    model = FMModel(n_features, EMBED_DIM)
    opt_dense = torch.optim.Adam([model.bias], lr=LR)
    opt_sparse = torch.optim.SparseAdam(
        [model.linear.weight, model.embed.weight], lr=LR)
    bce = nn.BCEWithLogitsLoss()
    idx_tr = torch.from_numpy(idx_tr_full[tr])
    y_tr = torch.from_numpy(y[tr].astype(np.float32))
    idx_va = torch.from_numpy(idx_tr_full[va])
    idx_te_t = torch.from_numpy(idx_te)
    n_tr = len(idx_tr)
    perm = np.arange(n_tr)
    for ep in range(EPOCHS):
        np.random.shuffle(perm)
        for s in range(0, n_tr, BATCH):
            b = perm[s:s + BATCH]
            xb = idx_tr[b]; yb = y_tr[b]
            logits = model(xb)
            loss = bce(logits, yb)
            opt_dense.zero_grad(); opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step(); opt_sparse.step()
    with torch.no_grad():
        chunks = [torch.sigmoid(model(idx_te_t[s:s+BATCH])).numpy()
                  for s in range(0, len(idx_te_t), BATCH)]
        p_te = np.concatenate(chunks)
        chunks = [torch.sigmoid(model(idx_va[s:s+BATCH])).numpy()
                  for s in range(0, len(idx_va), BATCH)]
        p_va = np.concatenate(chunks)
    return p_va.astype(np.float64), p_te.astype(np.float64)


def train_partition_fm(train, test, y, splits, fields, label):
    print(f"\n--- {label} (fields={fields}, n_active={len(fields)}) ---")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        Xtr_csr, Xte_csr = build_partition_hashes(train, test, tr, fields)
        idx_tr_full = csr_to_index_array(Xtr_csr, len(fields))
        idx_te = csr_to_index_array(Xte_csr, len(fields))
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=42 + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        print(f"  fold {k}: val AUC {s:.5f}  wall {time.time()-t_fold:.1f}s")
    return oof, test_avg


def stack_eval(label, fm_extras_oof, fm_extras_test, fm_extras_names,
               y, primary_test, base_oof, base_test, base_names,
               d9_oof, d9_test, d9_names, results,
               include_pool_fm=False, fm_pool_oof=None, fm_pool_test=None):
    Xs = base_oof + d9_oof
    Ts = base_test + d9_test
    Ns = list(base_names) + list(d9_names)
    if include_pool_fm:
        Xs = Xs + [fm_pool_oof]; Ts = Ts + [fm_pool_test]; Ns = Ns + ["FM_d9c"]
    for o, t, n in zip(fm_extras_oof, fm_extras_test, fm_extras_names):
        Xs.append(o); Ts.append(t); Ns.append(n)
    K = len(Ns)
    P_oof = np.column_stack(Xs); P_test = np.column_stack(Ts)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, primary_test)
    delta = (auc - PRIMARY_S) * 1e4
    l1 = {Ns[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"\n=== {label} (K={K}) ===")
    print(f"  Strat OOF: {auc:.5f}  Δ PRIMARY {delta:+.2f}bp  "
          f"ρ vs PRIMARY {rho:.5f}")
    print(f"  L1 top-15:")
    for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:15]:
        marker = ""
        if n_.startswith("FM"): marker = "  ← FM-class"
        elif n_.startswith("R") and "_" in n_: marker = "  ← d9-base"
        elif n_.startswith("rule_"): marker = "  ← existing rule"
        print(f"    {n_:<24s} L1={v:.3f}{marker}")
    results[label] = dict(K=K, strat_oof=auc, delta_primary_bp=delta,
                          rho_vs_primary=float(rho), l1_ranking=l1)
    return mo, tp


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = np.load(ART / "oof_d9c_Sd_K20_swap_FM_strat.npy"
                          )[:, 1].astype(np.float64)
    primary_test = np.load(ART / "test_d9c_Sd_K20_swap_FM_strat.npy"
                           )[:, 1].astype(np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Train both partition FMs
    oof_A, test_A = train_partition_fm(train, test, y, splits,
                                       PARTITION_A, "FM_A_driver_dynamics")
    oof_B, test_B = train_partition_fm(train, test, y, splits,
                                       PARTITION_B, "FM_B_race_context")
    auc_A = float(roc_auc_score(y, oof_A))
    auc_B = float(roc_auc_score(y, oof_B))
    rho_A_prim, _ = spearmanr(test_A, primary_test)
    rho_B_prim, _ = spearmanr(test_B, primary_test)
    rho_A_B, _ = spearmanr(test_A, test_B)
    print(f"\n  FM_A std OOF: {auc_A:.5f}  ρ vs PRIMARY {rho_A_prim:.5f}")
    print(f"  FM_B std OOF: {auc_B:.5f}  ρ vs PRIMARY {rho_B_prim:.5f}")
    print(f"  ρ FM_A vs FM_B: {rho_A_B:.5f}  (low = orthogonal partitions)")

    # Min-meta vs PRIMARY for each
    F_min = expand(np.column_stack([primary_oof, oof_A]))
    F_min_t = expand(np.column_stack([primary_test, test_A]))
    mo_A, _, _ = fit_lr_meta(F_min, F_min_t, y)
    mm_A = float(roc_auc_score(y, mo_A)); dA = (mm_A - PRIMARY_S) * 1e4
    F_min = expand(np.column_stack([primary_oof, oof_B]))
    F_min_t = expand(np.column_stack([primary_test, test_B]))
    mo_B, _, _ = fit_lr_meta(F_min, F_min_t, y)
    mm_B = float(roc_auc_score(y, mo_B)); dB = (mm_B - PRIMARY_S) * 1e4
    print(f"  FM_A min-meta vs PRIMARY: {mm_A:.5f}  Δ {dA:+.2f}bp  "
          f"{'PASS ✓' if mm_A >= PRIMARY_S else 'FAIL ✗'}")
    print(f"  FM_B min-meta vs PRIMARY: {mm_B:.5f}  Δ {dB:+.2f}bp  "
          f"{'PASS ✓' if mm_B >= PRIMARY_S else 'FAIL ✗'}")

    # Save partition FM bases
    np.save(ART / "oof_d9f_FM_A_strat.npy", np.column_stack([1 - oof_A, oof_A]))
    np.save(ART / "test_d9f_FM_A_strat.npy", np.column_stack([1 - test_A, test_A]))
    np.save(ART / "oof_d9f_FM_B_strat.npy", np.column_stack([1 - oof_B, oof_B]))
    np.save(ART / "test_d9f_FM_B_strat.npy", np.column_stack([1 - test_B, test_B]))

    # Load PRIMARY pool components
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
    fm_pool_oof = np.load(ART / "oof_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    fm_pool_test = np.load(ART / "test_d9c_fm_strat.npy")[:, 1].astype(np.float64)

    results = dict(
        FM_A=dict(std_oof=auc_A, rho_vs_primary=float(rho_A_prim),
                  min_meta_oof=mm_A, delta_primary_bp=float(dA),
                  min_meta_pass=bool(mm_A >= PRIMARY_S)),
        FM_B=dict(std_oof=auc_B, rho_vs_primary=float(rho_B_prim),
                  min_meta_oof=mm_B, delta_primary_bp=float(dB),
                  min_meta_pass=bool(mm_B >= PRIMARY_S)),
        rho_A_B=float(rho_A_B),
    )

    # K=21 swap (drop d9c FM, add FM_A + FM_B)
    mo_swap, tp_swap = stack_eval(
        "K21_swap_partA_partB", [oof_A, oof_B], [test_A, test_B],
        ["FM_A", "FM_B"], y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results)
    # K=22 add (keep d9c FM, add FM_A + FM_B)
    mo_add, tp_add = stack_eval(
        "K22_add_partA_partB", [oof_A, oof_B], [test_A, test_B],
        ["FM_A", "FM_B"], y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results,
        include_pool_fm=True, fm_pool_oof=fm_pool_oof,
        fm_pool_test=fm_pool_test)

    # Save submissions
    np.save(ART / "test_d9f_K21_swap_strat.npy",
            np.column_stack([1 - tp_swap, tp_swap]))
    sub = sample_sub.copy(); sub[TARGET] = tp_swap
    sub.to_csv("submissions/submission_d9f_K21_swap_partA_partB.csv", index=False)
    print("→ wrote submissions/submission_d9f_K21_swap_partA_partB.csv")
    np.save(ART / "test_d9f_K22_add_strat.npy",
            np.column_stack([1 - tp_add, tp_add]))
    sub = sample_sub.copy(); sub[TARGET] = tp_add
    sub.to_csv("submissions/submission_d9f_K22_add_partA_partB.csv", index=False)
    print("→ wrote submissions/submission_d9f_K22_add_partA_partB.csv")

    final = dict(results=results,
                 partitions=dict(A=PARTITION_A, B=PARTITION_B),
                 params=dict(embed_dim=EMBED_DIM, epochs=EPOCHS, batch=BATCH, lr=LR),
                 primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
                 total_wall_s=time.time() - t0)
    (ART / "d9f_multi_fm_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9f_multi_fm_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
