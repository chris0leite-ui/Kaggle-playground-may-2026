"""Day-9h — Feature-augmented FM (12 fields).

d9c FM (LB 0.95029, demoted by d9f) used 8 main-effect features.
d9f 2-way partition (LB 0.95031, NEW PRIMARY) split those 8 across
two FMs.  d9g 3-way partition regressed.

This script tests whether *richer* features in a *unified* FM beat
the 2-way partition. Adds 4 new field types to the 8 original:

  Original 8: D, C, R, Y, S, T_q5, Rp_q5, P_q5
  + next_compound: 1-step look-ahead (P5: 68% test coverage)
  + prev_compound: 1-step look-back
  + CumDeg_q5: 5-quantile of Cumulative_Degradation
  + LapDelta_q5: 5-quantile of LapTime_Delta

Each FM row now has 12 active features → 66 pairwise interactions
(vs unified 8-field FM's 28).

PRIMARY = d9f K=21 swap+multi-FM (Strat OOF 0.95073, LB 0.95031).
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
N_FIELDS = 12

PRIMARY_S = 0.95073
PRIMARY_LB = 0.95031
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
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def _quantile_bin(arr_train, arr_query, n_bins):
    edges = np.quantile(arr_train, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                   0, n_bins - 1).astype(int)


def _compute_neighbour_compounds(df):
    """Within (Year, Race, Driver) sorted by LapNumber, look up
    next/prev Compound. 'UNK' where unavailable. Leak-free."""
    sort_idx = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).index
    s = df.loc[sort_idx].copy()
    grp = s.groupby(["Year", "Race", "Driver"], sort=False)
    s["next_compound"] = grp["Compound"].shift(-1).fillna("UNK").astype(str)
    s["prev_compound"] = grp["Compound"].shift(+1).fillna("UNK").astype(str)
    s = s.sort_index()
    return s["next_compound"].values, s["prev_compound"].values


def build_aug_hashes(train, test, tr_idx, next_train, next_test,
                    prev_train, prev_test):
    """Build (n_rows, 12) hashed indices for all 12 fields."""
    n_tr, n_te = len(train), len(test)
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
    # New
    Nx_tr = next_train.astype(str)
    Nx_te = next_test.astype(str)
    Pv_tr = prev_train.astype(str)
    Pv_te = prev_test.astype(str)
    Cd_tr = _quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                          train["Cumulative_Degradation"].values, 5).astype(str)
    Cd_te = _quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                          test["Cumulative_Degradation"].values, 5).astype(str)
    Ld_tr = _quantile_bin(train["LapTime_Delta"].values[tr_idx],
                          train["LapTime_Delta"].values, 5).astype(str)
    Ld_te = _quantile_bin(train["LapTime_Delta"].values[tr_idx],
                          test["LapTime_Delta"].values, 5).astype(str)

    fields = ["D", "C", "R", "Y", "S", "T", "Rp", "P", "Nx", "Pv", "Cd", "Ld"]
    arrs_tr = [D_tr, C_tr, R_tr, Y_tr, S_tr, Tq_tr, Rq_tr, Pq_tr,
               Nx_tr, Pv_tr, Cd_tr, Ld_tr]
    arrs_te = [D_te, C_te, R_te, Y_te, S_te, Tq_te, Rq_te, Pq_te,
               Nx_te, Pv_te, Cd_te, Ld_te]

    def rows(arrs, n):
        return [[f"{p}={a[i]}" for p, a in zip(fields, arrs)] for i in range(n)]
    h = FeatureHasher(n_features=2**N_HASH_BITS, input_type="string",
                      alternate_sign=False)
    return h.transform(rows(arrs_tr, n_tr)), h.transform(rows(arrs_te, n_te))


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


def fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va, seed, epochs=EPOCHS):
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
    for ep in range(epochs):
        np.random.shuffle(perm)
        ep_loss = 0.0; n_seen = 0
        for s in range(0, n_tr, BATCH):
            b = perm[s:s + BATCH]
            xb = idx_tr[b]; yb = y_tr[b]
            logits = model(xb)
            loss = bce(logits, yb)
            opt_dense.zero_grad(); opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step(); opt_sparse.step()
            ep_loss += float(loss.item()) * len(b); n_seen += len(b)
        with torch.no_grad():
            chunks = [torch.sigmoid(model(idx_va[s:s+BATCH])).numpy()
                      for s in range(0, len(idx_va), BATCH)]
            pv = np.concatenate(chunks)
            try: auc = roc_auc_score(y[va], pv)
            except Exception: auc = float("nan")
        print(f"      epoch {ep}: train_loss={ep_loss/n_seen:.5f}  val_auc={auc:.5f}")
    with torch.no_grad():
        chunks = [torch.sigmoid(model(idx_te_t[s:s+BATCH])).numpy()
                  for s in range(0, len(idx_te_t), BATCH)]
        p_te = np.concatenate(chunks)
        chunks = [torch.sigmoid(model(idx_va[s:s+BATCH])).numpy()
                  for s in range(0, len(idx_va), BATCH)]
        p_va = np.concatenate(chunks)
    return p_va.astype(np.float64), p_te.astype(np.float64)


def stack_eval(label, fm_oof, fm_test, y, primary_test, base_oof, base_test,
               base_names, d9_oof, d9_test, d9_names, results,
               include_d9f_AB=False, fma_oof=None, fma_test=None,
               fmb_oof=None, fmb_test=None):
    Xs = list(base_oof) + list(d9_oof)
    Ts = list(base_test) + list(d9_test)
    Ns = list(base_names) + list(d9_names)
    if include_d9f_AB:
        Xs.extend([fma_oof, fmb_oof])
        Ts.extend([fma_test, fmb_test])
        Ns.extend(["FM_A_d9f", "FM_B_d9f"])
    Xs.append(fm_oof); Ts.append(fm_test); Ns.append("FM_aug12")
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
        if n_ == "FM_aug12": marker = "  ← FM_aug12"
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

    # Compute next/prev compounds on train+test union
    print("Computing next/prev compound on train+test ...")
    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    nx, pv = _compute_neighbour_compounds(full)
    next_train = nx[full["__src"].values == "tr"]
    next_test = nx[full["__src"].values == "te"]
    prev_train = pv[full["__src"].values == "tr"]
    prev_test = pv[full["__src"].values == "te"]
    print(f"  next_compound test coverage: "
          f"{(next_test != 'UNK').mean():.3f}")
    print(f"  prev_compound test coverage: "
          f"{(prev_test != 'UNK').mean():.3f}")

    # Train FM_aug12
    print(f"\n=== d9h FM_aug12 (12 fields, k={EMBED_DIM}, "
          f"epochs={EPOCHS}) ===\n")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        Xtr_csr, Xte_csr = build_aug_hashes(train, test, tr,
                                             next_train, next_test,
                                             prev_train, prev_test)
        idx_tr_full = csr_to_index_array(Xtr_csr, N_FIELDS)
        idx_te = csr_to_index_array(Xte_csr, N_FIELDS)
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=42 + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        print(f"    fold {k} done: val AUC {s:.5f}  wall {time.time()-t_fold:.1f}s")

    auc = float(roc_auc_score(y, oof))
    rho_test, _ = spearmanr(test_avg, primary_test)
    # Min-meta uses d9c K=20 swap OOF as anchor (closest available to
    # d9f K=21 swap; OOF differs by 0.0003).
    primary_oof_anchor = np.load(
        ART / "oof_d9c_Sd_K20_swap_FM_strat.npy")[:, 1].astype(np.float64)
    F_min = expand(np.column_stack([primary_oof_anchor, oof]))
    F_min_t = expand(np.column_stack([primary_test, test_avg]))
    mo, _, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo))
    delta_mm = (auc_min - PRIMARY_S) * 1e4
    print(f"\n  FM_aug12 std OOF: {auc:.5f}")
    print(f"  ρ vs PRIMARY (d9f K=21) test: {rho_test:.5f}")
    print(f"  Min-meta vs PRIMARY: {auc_min:.5f}  Δ {delta_mm:+.2f}bp  "
          f"{'PASS ✓' if auc_min >= PRIMARY_S else 'FAIL ✗'}")

    np.save(ART / "oof_d9h_FM_aug12_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d9h_FM_aug12_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))

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
    fma_oof = np.load(ART / "oof_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    fma_test = np.load(ART / "test_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    fmb_oof = np.load(ART / "oof_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)
    fmb_test = np.load(ART / "test_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)

    results = dict(
        FM_aug12=dict(std_oof=auc, rho_vs_primary=float(rho_test),
                      min_meta_oof=auc_min,
                      delta_primary_bp=float(delta_mm),
                      min_meta_pass=bool(auc_min >= PRIMARY_S)),
        coverages=dict(next=float((next_test != "UNK").mean()),
                       prev=float((prev_test != "UNK").mean())),
    )

    # S1 K=20 swap (drop d9f's FM_A+FM_B, replace with single FM_aug12)
    mo_s1, tp_s1 = stack_eval(
        "S1_K20_swap_aug12_replaces_2way",
        oof, test_avg, y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results)
    # S2 K=22 add (keep d9f's FM_A+FM_B, ADD FM_aug12)
    mo_s2, tp_s2 = stack_eval(
        "S2_K22_add_aug12_to_2way",
        oof, test_avg, y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results,
        include_d9f_AB=True, fma_oof=fma_oof, fma_test=fma_test,
        fmb_oof=fmb_oof, fmb_test=fmb_test)

    for name, mo, tp in [("S1_K20_swap_aug12", mo_s1, tp_s1),
                         ("S2_K22_add_aug12", mo_s2, tp_s2)]:
        np.save(ART / f"test_d9h_{name}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_d9h_{name}.csv", index=False)
        print(f"→ wrote submissions/submission_d9h_{name}.csv")

    final = dict(
        results=results,
        params=dict(embed_dim=EMBED_DIM, epochs=EPOCHS, batch=BATCH,
                    lr=LR, n_fields=N_FIELDS),
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        total_wall_s=time.time() - t0,
    )
    (ART / "d9h_aug12_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9h_aug12_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
