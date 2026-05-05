"""Day-9i — Augmented 2-way multi-FM partition.

Combines d9f's 2-way partition (which gave the +2bp LB lift) with
d9h's feature augmentation (which boosted standalone +4.7bp but
didn't help the stack as a unified FM).

Hypothesis: feeding the new features (Cd, Ld, Nx, Pv) to the
partitioned FMs preserves ρ-diversity (d9f advantage) while
extracting the new-feature signal (d9h advantage).

Partition (each FM has 6 features → 15 pairwise interactions):
  FM_A_aug "driver+degradation": D, C, S, T_q5, Cd, Ld
  FM_B_aug "race+neighbor":      R, Y, Rp_q5, P_q5, Nx, Pv

Two stacks vs PRIMARY (d9f K=21 swap, OOF 0.95073, LB 0.95031):
  S1 K=21 swap: drop d9f FM_A + FM_B, replace with FM_A_aug + FM_B_aug
  S2 K=23 add:  keep d9f FM_A + FM_B, ADD FM_A_aug + FM_B_aug
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
EMBED_DIM = 8
EPOCHS = 6
BATCH = 8192
LR = 0.05

PRIMARY_S = 0.95073
PRIMARY_LB = 0.95031

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

# Augmented partitions
PART_A_AUG = ["D", "C", "S", "T", "Cd", "Ld"]
PART_B_AUG = ["R", "Y", "Rp", "P", "Nx", "Pv"]


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


def _compute_neighbour_compounds(df):
    sort_idx = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).index
    s = df.loc[sort_idx].copy()
    grp = s.groupby(["Year", "Race", "Driver"], sort=False)
    s["next_compound"] = grp["Compound"].shift(-1).fillna("UNK").astype(str)
    s["prev_compound"] = grp["Compound"].shift(+1).fillna("UNK").astype(str)
    s = s.sort_index()
    return s["next_compound"].values, s["prev_compound"].values


def build_partition_hashes(train, test, tr_idx, fields,
                            next_train, next_test,
                            prev_train, prev_test):
    """Build (n_rows, len(fields)) hashed indices for any field subset."""
    n_tr, n_te = len(train), len(test)
    arrs_tr, arrs_te = [], []
    for f in fields:
        if f == "D":
            arrs_tr.append(train["Driver"].astype(str).values)
            arrs_te.append(test["Driver"].astype(str).values)
        elif f == "C":
            arrs_tr.append(train["Compound"].astype(str).values)
            arrs_te.append(test["Compound"].astype(str).values)
        elif f == "R":
            arrs_tr.append(train["Race"].astype(str).values)
            arrs_te.append(test["Race"].astype(str).values)
        elif f == "Y":
            arrs_tr.append(train["Year"].astype(str).values)
            arrs_te.append(test["Year"].astype(str).values)
        elif f == "S":
            arrs_tr.append(train["Stint"].clip(upper=5).astype(int).astype(str).values)
            arrs_te.append(test["Stint"].clip(upper=5).astype(int).astype(str).values)
        elif f == "T":
            arrs_tr.append(_quantile_bin(train["TyreLife"].values[tr_idx],
                                          train["TyreLife"].values, 5).astype(str))
            arrs_te.append(_quantile_bin(train["TyreLife"].values[tr_idx],
                                          test["TyreLife"].values, 5).astype(str))
        elif f == "Rp":
            arrs_tr.append(_quantile_bin(train["RaceProgress"].values[tr_idx],
                                          train["RaceProgress"].values, 5).astype(str))
            arrs_te.append(_quantile_bin(train["RaceProgress"].values[tr_idx],
                                          test["RaceProgress"].values, 5).astype(str))
        elif f == "P":
            arrs_tr.append(_quantile_bin(train["Position"].values[tr_idx],
                                          train["Position"].values, 5).astype(str))
            arrs_te.append(_quantile_bin(train["Position"].values[tr_idx],
                                          test["Position"].values, 5).astype(str))
        elif f == "Cd":
            arrs_tr.append(_quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                                          train["Cumulative_Degradation"].values, 5).astype(str))
            arrs_te.append(_quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                                          test["Cumulative_Degradation"].values, 5).astype(str))
        elif f == "Ld":
            arrs_tr.append(_quantile_bin(train["LapTime_Delta"].values[tr_idx],
                                          train["LapTime_Delta"].values, 5).astype(str))
            arrs_te.append(_quantile_bin(train["LapTime_Delta"].values[tr_idx],
                                          test["LapTime_Delta"].values, 5).astype(str))
        elif f == "Nx":
            arrs_tr.append(next_train.astype(str))
            arrs_te.append(next_test.astype(str))
        elif f == "Pv":
            arrs_tr.append(prev_train.astype(str))
            arrs_te.append(prev_test.astype(str))
        else:
            raise ValueError(f"Unknown field {f}")

    def rows(arrs, n):
        return [[f"{p}={a[i]}" for p, a in zip(fields, arrs)] for i in range(n)]
    h = FeatureHasher(n_features=2**N_HASH_BITS, input_type="string",
                      alternate_sign=False)
    return (h.transform(rows(arrs_tr, n_tr)),
            h.transform(rows(arrs_te, n_te)))


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


def train_partition_fm(train, test, y, splits, fields, label,
                       next_train, next_test, prev_train, prev_test):
    print(f"\n--- {label} (fields={fields}, n_active={len(fields)}) ---")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        Xtr_csr, Xte_csr = build_partition_hashes(train, test, tr, fields,
                                                   next_train, next_test,
                                                   prev_train, prev_test)
        idx_tr_full = csr_to_index_array(Xtr_csr, len(fields))
        idx_te = csr_to_index_array(Xte_csr, len(fields))
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=42 + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        print(f"  fold {k}: val AUC {s:.5f}  wall {time.time()-t_fold:.1f}s")
    return oof, test_avg


def stack_eval(label, extra_oof, extra_test, extra_names, y, primary_test,
               base_oof, base_test, base_names, d9_oof, d9_test, d9_names,
               results, include_d9f_AB=False, fma_oof=None, fma_test=None,
               fmb_oof=None, fmb_test=None):
    Xs = list(base_oof) + list(d9_oof)
    Ts = list(base_test) + list(d9_test)
    Ns = list(base_names) + list(d9_names)
    if include_d9f_AB:
        Xs.extend([fma_oof, fmb_oof])
        Ts.extend([fma_test, fmb_test])
        Ns.extend(["FM_A_d9f", "FM_B_d9f"])
    for o, t, n in zip(extra_oof, extra_test, extra_names):
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
        if n_.startswith("FM_A_aug") or n_.startswith("FM_B_aug"): marker = "  ← d9i-FM"
        elif n_.startswith("FM_A_d9f") or n_.startswith("FM_B_d9f"): marker = "  ← d9f-FM"
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
    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy"
                           )[:, 1].astype(np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Compute next/prev compounds
    print("Computing next/prev compound on train+test ...")
    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    nx, pv = _compute_neighbour_compounds(full)
    next_train = nx[full["__src"].values == "tr"]
    next_test = nx[full["__src"].values == "te"]
    prev_train = pv[full["__src"].values == "tr"]
    prev_test = pv[full["__src"].values == "te"]

    # Train both augmented partition FMs
    print(f"\n=== d9i augmented 2-way multi-FM ===\n")
    oof_a, test_a = train_partition_fm(train, test, y, splits,
                                       PART_A_AUG, "FM_A_aug_driver_deg",
                                       next_train, next_test,
                                       prev_train, prev_test)
    oof_b, test_b = train_partition_fm(train, test, y, splits,
                                       PART_B_AUG, "FM_B_aug_race_neighbor",
                                       next_train, next_test,
                                       prev_train, prev_test)
    auc_a = float(roc_auc_score(y, oof_a))
    auc_b = float(roc_auc_score(y, oof_b))
    rho_ap, _ = spearmanr(test_a, primary_test)
    rho_bp, _ = spearmanr(test_b, primary_test)
    rho_ab, _ = spearmanr(test_a, test_b)
    print(f"\n  FM_A_aug std OOF: {auc_a:.5f}  ρ vs PRIMARY {rho_ap:.5f}")
    print(f"  FM_B_aug std OOF: {auc_b:.5f}  ρ vs PRIMARY {rho_bp:.5f}")
    print(f"  ρ FM_A_aug vs FM_B_aug: {rho_ab:.5f}")

    np.save(ART / "oof_d9i_FM_A_aug_strat.npy",
            np.column_stack([1 - oof_a, oof_a]))
    np.save(ART / "test_d9i_FM_A_aug_strat.npy",
            np.column_stack([1 - test_a, test_a]))
    np.save(ART / "oof_d9i_FM_B_aug_strat.npy",
            np.column_stack([1 - oof_b, oof_b]))
    np.save(ART / "test_d9i_FM_B_aug_strat.npy",
            np.column_stack([1 - test_b, test_b]))

    # Load PRIMARY pool
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
    fma_d9f_oof = np.load(ART / "oof_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    fma_d9f_test = np.load(ART / "test_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    fmb_d9f_oof = np.load(ART / "oof_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)
    fmb_d9f_test = np.load(ART / "test_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)

    results = dict(
        FM_A_aug=dict(std_oof=auc_a, rho_vs_primary=float(rho_ap)),
        FM_B_aug=dict(std_oof=auc_b, rho_vs_primary=float(rho_bp)),
        rho_A_B=float(rho_ab),
    )

    # S1: K=21 swap (drop d9f FM_A+FM_B, add FM_A_aug + FM_B_aug)
    mo_s1, tp_s1 = stack_eval(
        "S1_K21_swap_aug2way",
        [oof_a, oof_b], [test_a, test_b],
        ["FM_A_aug", "FM_B_aug"],
        y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results)

    # S2: K=23 add (keep d9f FM_A+FM_B, ADD FM_A_aug + FM_B_aug)
    mo_s2, tp_s2 = stack_eval(
        "S2_K23_add_aug2way_to_d9f",
        [oof_a, oof_b], [test_a, test_b],
        ["FM_A_aug", "FM_B_aug"],
        y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results,
        include_d9f_AB=True, fma_oof=fma_d9f_oof, fma_test=fma_d9f_test,
        fmb_oof=fmb_d9f_oof, fmb_test=fmb_d9f_test)

    for name, mo, tp in [("S1_K21_swap_aug2way", mo_s1, tp_s1),
                         ("S2_K23_add_aug2way_to_d9f", mo_s2, tp_s2)]:
        np.save(ART / f"test_d9i_{name}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_d9i_{name}.csv", index=False)
        print(f"→ wrote submissions/submission_d9i_{name}.csv")

    final = dict(
        results=results,
        partitions=dict(A=PART_A_AUG, B=PART_B_AUG),
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        total_wall_s=time.time() - t0,
    )
    (ART / "d9i_aug2way_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9i_aug2way_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
