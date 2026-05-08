"""H1 — FM aug13 with Compound × TyreLife_q5 × RaceProgress_q5 3-way field.

Source: `audit/2026-05-13-eda-deep-dive-synthesis.md` H1.
Phase C of the EDA found 3-way cells (Compound × TL_q × RP_q) with
3.35× lift (SOFT TL_d=5 RP_d=2 high-pit; HARD TL_d=9 RP_d=8). Existing
d9h_aug12 has 12 fields but no explicit 3-way concatenation field.

Adds a 13th field: `CTRq = f"{Compound}_{TyreLife_q5}_{RaceProgress_q5}"`
(5×5×5 = 125 levels). Same FM architecture as d9h_aug12 (k=8, 6 epochs,
hashed at 2^18 bits).

Min-meta gate against NEW PRIMARY (Path B Stint τ=100000, LB 0.95041,
std OOF 0.95082).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn

os.environ.setdefault("OMP_NUM_THREADS", "6")
torch.set_num_threads(6)

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
N_HASH_BITS = 18
EMBED_DIM = 8
EPOCHS = 6
BATCH = 8192
LR = 0.05
N_FIELDS = 13
NAME = "d14_h1_fm_aug13_3way"

# NEW PRIMARY (Day-13 Path B Stint τ=100000)
PRIMARY_S = 0.95082
PRIMARY_LB = 0.95041
PRIMARY_OOF_FILE = "oof_d13_path_b_stint_tau100000_strat.npy"
PRIMARY_TEST_FILE = "test_d13_path_b_stint_tau100000_strat.npy"
RHO_TIE = 0.999


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y, splits):
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1]


def predicted_lb(auc_min, rho):
    base_lb = PRIMARY_LB + (auc_min - PRIMARY_S)
    if rho >= RHO_TIE:
        return base_lb
    if rho >= 0.995:
        return base_lb - 0.0001
    if rho >= 0.99:
        return base_lb - 0.00025
    return base_lb - 0.0004


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


def build_aug13_hashes(train, test, tr_idx, next_train, next_test,
                      prev_train, prev_test):
    """13 fields: 12 from d9h_aug12 + CTRq 3-way concat."""
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
    Nx_tr = next_train.astype(str); Nx_te = next_test.astype(str)
    Pv_tr = prev_train.astype(str); Pv_te = prev_test.astype(str)
    Cd_tr = _quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                          train["Cumulative_Degradation"].values, 5).astype(str)
    Cd_te = _quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                          test["Cumulative_Degradation"].values, 5).astype(str)
    Ld_tr = _quantile_bin(train["LapTime_Delta"].values[tr_idx],
                          train["LapTime_Delta"].values, 5).astype(str)
    Ld_te = _quantile_bin(train["LapTime_Delta"].values[tr_idx],
                          test["LapTime_Delta"].values, 5).astype(str)

    # H1 — 3-way concatenation
    CTRq_tr = np.array(
        [f"{c}_{t}_{r}" for c, t, r in zip(C_tr, Tq_tr, Rq_tr)], dtype=object)
    CTRq_te = np.array(
        [f"{c}_{t}_{r}" for c, t, r in zip(C_te, Tq_te, Rq_te)], dtype=object)

    fields = ["D", "C", "R", "Y", "S", "T", "Rp", "P",
              "Nx", "Pv", "Cd", "Ld", "CTRq"]
    arrs_tr = [D_tr, C_tr, R_tr, Y_tr, S_tr, Tq_tr, Rq_tr, Pq_tr,
               Nx_tr, Pv_tr, Cd_tr, Ld_tr, CTRq_tr]
    arrs_te = [D_te, C_te, R_te, Y_te, S_te, Tq_te, Rq_te, Pq_te,
               Nx_te, Pv_te, Cd_te, Ld_te, CTRq_te]

    def rows(arrs, n):
        return [[f"{p}={a[i]}" for p, a in zip(fields, arrs)] for i in range(n)]
    h = FeatureHasher(n_features=2**N_HASH_BITS, input_type="string",
                      alternate_sign=False)
    return (h.transform(rows(arrs_tr, n_tr)),
            h.transform(rows(arrs_te, n_te)),
            CTRq_tr)


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


def main():
    t0 = time.time()
    print(f"=== {NAME} (FM aug13 with H1 CTRq 3-way) ===")
    print(f"PRIMARY = Path B Stint τ=100000 (LB {PRIMARY_LB}, "
          f"std OOF {PRIMARY_S})")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = np.load(ART / PRIMARY_OOF_FILE)[:, 1].astype(np.float64)
    primary_test = np.load(ART / PRIMARY_TEST_FILE)[:, 1].astype(np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    print("Computing next/prev compound on train+test ...")
    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    nx, pv = _compute_neighbour_compounds(full)
    next_train = nx[full["__src"].values == "tr"]
    next_test = nx[full["__src"].values == "te"]
    prev_train = pv[full["__src"].values == "tr"]
    prev_test = pv[full["__src"].values == "te"]

    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []
    ctrq_levels_seen = None
    print(f"\n=== Training FM_aug13 (13 fields, k={EMBED_DIM}, "
          f"epochs={EPOCHS}) ===\n")
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        Xtr_csr, Xte_csr, CTRq_tr = build_aug13_hashes(
            train, test, tr, next_train, next_test, prev_train, prev_test)
        if ctrq_levels_seen is None:
            ctrq_levels_seen = len(set(CTRq_tr))
            print(f"  CTRq distinct levels (train): {ctrq_levels_seen} "
                  f"(target ≤ 125 = 5×5×5)")
        idx_tr_full = csr_to_index_array(Xtr_csr, N_FIELDS)
        idx_te = csr_to_index_array(Xte_csr, N_FIELDS)
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=42 + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        a = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(a)
        print(f"    fold {k} done: val AUC {a:.5f}  "
              f"wall {time.time()-t_fold:.1f}s")

    std_auc = float(roc_auc_score(y, oof))
    rho, _ = spearmanr(test_avg, primary_test)

    # Min-meta gate vs NEW PRIMARY
    F_min = expand(np.column_stack([primary_oof, oof]))
    F_min_t = expand(np.column_stack([primary_test, test_avg]))
    mo, _ = fit_lr_meta(F_min, F_min_t, y, splits)
    auc_min = float(roc_auc_score(y, mo))
    delta_mm = (auc_min - PRIMARY_S) * 1e4
    pred_lb = predicted_lb(auc_min, float(rho))

    print(f"\n--- Results ---")
    print(f"  std OOF      : {std_auc:.5f}")
    print(f"  ρ vs PRIMARY : {rho:.5f}")
    print(f"  min-meta OOF : {auc_min:.5f}  Δ {delta_mm:+.2f}bp  "
          f"{'PASS ✓' if delta_mm > 0 else 'FAIL ✗'}")
    print(f"  pred LB      : {pred_lb:.5f}  "
          f"Δ {(pred_lb - PRIMARY_LB)*1e4:+.2f}bp")

    np.save(ART / f"oof_{NAME}_strat.npy",
            np.column_stack([1 - oof, oof]).astype(np.float32))
    np.save(ART / f"test_{NAME}_strat.npy",
            np.column_stack([1 - test_avg, test_avg]).astype(np.float32))
    sub = sample_sub.copy(); sub[TARGET] = test_avg
    sub.to_csv(f"submissions/submission_{NAME}.csv", index=False)
    info = dict(
        name=NAME,
        std_oof=std_auc,
        rho_vs_primary=float(rho),
        min_meta_oof=auc_min,
        min_meta_delta_bp=float(delta_mm),
        pred_lb=float(pred_lb),
        pred_lb_delta_bp=float((pred_lb - PRIMARY_LB) * 1e4),
        fold_aucs=fold_aucs,
        ctrq_levels=int(ctrq_levels_seen),
        wall_seconds=time.time() - t0,
    )
    (ART / f"{NAME}_results.json").write_text(json.dumps(info, indent=2))
    print(f"\n→ saved oof_{NAME}_strat.npy, test_{NAME}_strat.npy, "
          f"submissions/submission_{NAME}.csv")
    print(f"  wall: {info['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
