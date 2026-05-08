"""Day-12 — Rebuild missing GroupKF OOF + test predictions for the K=21 pool.

P6 found 80.1% within-group leakage in StratifiedKFold(5). We rebuild
each pool base (that has no GroupKF artifact yet) under
GroupKFold(5) by (Race, Driver, Year, Stint) and save matching
`oof_<name>_groupkf.npy` / `test_<name>_groupkf.npy`.

Bases rebuilt:
  - d6_rule_driver_compound (HGBC residual on Bayesian-smoothed lookup)
  - d6_rule_year_race        (HGBC residual on Bayesian-smoothed lookup)
  - d9_R6_next_compound      (HGBC residual on next-compound lookup)
  - d9_R7_prev_compound      (HGBC residual on prev-compound lookup)
  - d9_R10_driver_eb         (HGBC residual on driver Beta-Binom EB)
  - d9b_R14_L4               (sparse-LR with 3-way interactions)
  - d9c_fm                   (Factorization Machine, PyTorch CPU)

Skipped: realmlp (GPU only).

Strict GroupKFold by (Race, Driver, Year, Stint) — same key as P6.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
TARGET, ID_COL = "PitNextLap", "id"
import os
SEED = 42
# N_FOLDS default 5; allow override via env to drop to 3 under CPU contention.
N_FOLDS = int(os.environ.get("D12_NFOLDS", "5"))
ALPHA = 50.0
N_HASH_BITS_FM = 18
EMBED_DIM_FM = 8
EPOCHS_FM = 6
BATCH_FM = 8192
LR_FM = 0.05


# ---- shared helpers ------------------------------------------------

def encode_features(X, X_test):
    HIGH_CARD = ["Driver"]
    LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True
                             ).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mp).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")
    return X, X_test


def make_hgbc_regressor():
    # NOTE: param-reduced from d6_multi_rule for CPU-budget under contention.
    # max_iter 400 vs 1500, max_leaf_nodes 31 vs 63, n_iter_no_change 25 vs 50.
    # On STRAT this would lose ~0.5-1bp; we accept the trade for time.
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=SEED,
        categorical_features="from_dtype",
    )


def _clip(p):
    return np.clip(p, 1e-9, 1.0 - 1e-9)


def _smoothed_lookup(keys_tr, y_tr, keys_apply_lists, alpha=ALPHA):
    df = pd.DataFrame({"k": keys_tr, "y": y_tr})
    g = df.groupby("k", observed=True)["y"]
    counts = g.count(); means = g.mean()
    glob = float(np.mean(y_tr))
    smoothed = ((means * counts + glob * alpha) / (counts + alpha)).to_dict()
    return [np.array([smoothed.get(k, glob) for k in keys], dtype=np.float64)
            for keys in keys_apply_lists]


def _decile_edges(arr, n_bins=10):
    edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return edges


def _bin(arr, edges):
    return np.clip(np.searchsorted(edges, arr, side="right") - 1,
                   0, len(edges) - 2).astype(np.int32)


def _quantile_bin(arr_train, arr_query, n_bins):
    edges = np.quantile(arr_train, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                   0, n_bins - 1).astype(int)


# ---- rule kernels (mirror existing scripts) ------------------------

def rule_driver_compound(train, test, tr, va, y_tr):
    """d6 rule: (Driver, Compound) Bayesian-smoothed lookup."""
    keys_tr = list(zip(train["Driver"].astype(str).values[tr],
                       train["Compound"].astype(str).values[tr]))
    keys_va = list(zip(train["Driver"].astype(str).values[va],
                       train["Compound"].astype(str).values[va]))
    keys_te = list(zip(test["Driver"].astype(str).values,
                       test["Compound"].astype(str).values))
    return _smoothed_lookup(keys_tr, y_tr, [keys_tr, keys_va, keys_te])


def rule_year_race(train, test, tr, va, y_tr):
    """d6 rule: (Year, Race) Bayesian-smoothed lookup."""
    keys_tr = list(zip(train["Year"].astype(str).values[tr],
                       train["Race"].astype(str).values[tr]))
    keys_va = list(zip(train["Year"].astype(str).values[va],
                       train["Race"].astype(str).values[va]))
    keys_te = list(zip(test["Year"].astype(str).values,
                       test["Race"].astype(str).values))
    return _smoothed_lookup(keys_tr, y_tr, [keys_tr, keys_va, keys_te])


def _compute_neighbour_compounds(df):
    sort_idx = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).index
    s = df.loc[sort_idx].copy()
    grp = s.groupby(["Year", "Race", "Driver"], sort=False)
    s["next_compound"] = grp["Compound"].shift(-1).fillna("UNK").astype(str)
    s["prev_compound"] = grp["Compound"].shift(+1).fillna("UNK").astype(str)
    s = s.sort_index()
    return s["next_compound"].values, s["prev_compound"].values


def rule_next_compound(train, test, tr, va, y_tr,
                       next_train, next_test):
    sq_train = train["Stint"].clip(upper=4).astype(int).values
    sq_test = test["Stint"].clip(upper=4).astype(int).values
    keys_tr = list(zip(train["Compound"].values[tr], next_train[tr], sq_train[tr]))
    keys_va = list(zip(train["Compound"].values[va], next_train[va], sq_train[va]))
    keys_te = list(zip(test["Compound"].values, next_test, sq_test))
    return _smoothed_lookup(keys_tr, y_tr, [keys_tr, keys_va, keys_te])


def rule_prev_compound(train, test, tr, va, y_tr,
                       prev_train, prev_test):
    edges = _decile_edges(train["TyreLife"].values[tr], n_bins=5)
    lis_train = _bin(train["TyreLife"].values, edges)
    lis_test = _bin(test["TyreLife"].values, edges)
    keys_tr = list(zip(prev_train[tr], train["Compound"].values[tr], lis_train[tr]))
    keys_va = list(zip(prev_train[va], train["Compound"].values[va], lis_train[va]))
    keys_te = list(zip(prev_test, test["Compound"].values, lis_test))
    return _smoothed_lookup(keys_tr, y_tr, [keys_tr, keys_va, keys_te])


def rule_driver_eb(train, test, tr, va, y_tr, alpha_eb=20.0):
    df_tr = pd.DataFrame({"Driver": train["Driver"].values[tr], "y": y_tr})
    g = df_tr.groupby("Driver", observed=True)["y"]
    counts = g.count(); sums = g.sum()
    glob = float(np.mean(y_tr))
    eb = ((sums + alpha_eb * glob) / (counts + alpha_eb)).to_dict()
    def score(d_arr):
        return np.array([eb.get(d, glob) for d in d_arr], dtype=np.float64)
    return (score(train["Driver"].values[tr]),
            score(train["Driver"].values[va]),
            score(test["Driver"].values))


# ---- residual builder ----------------------------------------------

def build_with_residual(name, rule_fn, train, test, X_enc, X_test_enc,
                        y, splits, has_residual=True, **rule_kwargs):
    print(f"\n--- {name} (residual={has_residual}, GroupKF) ---")
    n_train, n_test = len(train), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    walls, fold_aucs = [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        rp_tr, rp_va, rp_te = rule_fn(train, test, tr, va, y[tr], **rule_kwargs)
        if has_residual:
            m = make_hgbc_regressor()
            m.fit(X_enc.iloc[tr], y[tr].astype(np.float64) - rp_tr)
            resid_va = m.predict(X_enc.iloc[va])
            resid_te = m.predict(X_test_enc)
            pred_va = _clip(rp_va + resid_va)
            pred_te = _clip(rp_te + resid_te)
        else:
            pred_va = _clip(rp_va); pred_te = _clip(rp_te)
        oof[va] = pred_va
        test_avg += pred_te / N_FOLDS
        s = float(roc_auc_score(y[va], pred_va))
        fold_aucs.append(s); walls.append(time.time() - t0)
        print(f"  f{k}: AUC={s:.5f}  wall={walls[-1]:.1f}s")
    auc = float(roc_auc_score(y, oof))
    print(f"  GroupKF std OOF: {auc:.5f}  total wall={sum(walls):.1f}s")
    return oof, test_avg, auc, sum(walls)


# ---- R14_L4 features (mirror d9b_r14_ladder L4 build) ---------------

def build_r14_l4_features(train, test, tr_idx):
    n_tr, n_te = len(train), len(test)
    D = train["Driver"].astype(str).values; D_te = test["Driver"].astype(str).values
    C = train["Compound"].astype(str).values; C_te = test["Compound"].astype(str).values
    R = train["Race"].astype(str).values; R_te = test["Race"].astype(str).values
    Y = train["Year"].astype(str).values; Y_te = test["Year"].astype(str).values
    S = train["Stint"].clip(upper=5).astype(int).astype(str).values
    S_te = test["Stint"].clip(upper=5).astype(int).astype(str).values

    def qbin_str(col):
        edges = np.quantile(train[col].values[tr_idx], np.linspace(0, 1, 6))
        edges[0] = -np.inf; edges[-1] = np.inf
        a_tr = np.clip(np.searchsorted(edges, train[col].values, side="right") - 1, 0, 4).astype(str)
        a_te = np.clip(np.searchsorted(edges, test[col].values, side="right") - 1, 0, 4).astype(str)
        return a_tr, a_te
    T_str, T_te_str = qbin_str("TyreLife")
    Rp_str, Rp_te_str = qbin_str("RaceProgress")
    P_str, P_te_str = qbin_str("Position")

    def feats_for(idx_arr=None, te=False):
        if te:
            d, c, r, y_, s = D_te, C_te, R_te, Y_te, S_te
            T, Rp, P = T_te_str, Rp_te_str, P_te_str
        else:
            d = D[idx_arr]; c = C[idx_arr]; r = R[idx_arr]
            y_ = Y[idx_arr]; s = S[idx_arr]
            T = T_str[idx_arr]; Rp = Rp_str[idx_arr]; P = P_str[idx_arr]
        out = []
        for i in range(len(d)):
            row = [
                f"D={d[i]}", f"C={c[i]}", f"S={s[i]}",
                f"DC={d[i]}|{c[i]}", f"CS={c[i]}|{s[i]}",
                f"DS={d[i]}|{s[i]}",
                f"DCS={d[i]}|{c[i]}|{s[i]}",
                # L1
                f"R={r[i]}", f"Y={y_[i]}",
                f"DR={d[i]}|{r[i]}", f"DY={d[i]}|{y_[i]}",
                f"CR={c[i]}|{r[i]}", f"CY={c[i]}|{y_[i]}",
                f"RY={r[i]}|{y_[i]}",
                # L2
                f"T={T[i]}", f"Rp={Rp[i]}", f"P={P[i]}",
                # L3
                f"CT={c[i]}|{T[i]}", f"CRp={c[i]}|{Rp[i]}",
                f"CP={c[i]}|{P[i]}",
                f"CSint_T={c[i]}|{s[i]}|{T[i]}",
                # L4
                f"DT={d[i]}|{T[i]}", f"DRp={d[i]}|{Rp[i]}",
                f"DP={d[i]}|{P[i]}",
            ]
            out.append(row)
        return out

    h = FeatureHasher(n_features=2**18, input_type="string", alternate_sign=False)
    X_tr = h.transform(feats_for(np.arange(n_tr)))
    X_te = h.transform(feats_for(None, te=True))
    return X_tr, X_te


def build_r14_l4(train, test, y, splits):
    print(f"\n--- d9b_R14_L4 (sparse LR, GroupKF) ---")
    n_tr, n_te = len(train), len(test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_avg = np.zeros(n_te, dtype=np.float64)
    walls, fold_aucs = [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        X_tr_full, X_te = build_r14_l4_features(train, test, tr)
        lr = LogisticRegression(C=1.0, max_iter=300, solver="liblinear")
        lr.fit(X_tr_full[tr], y[tr])
        pred_va = lr.predict_proba(X_tr_full[va])[:, 1]
        pred_te = lr.predict_proba(X_te)[:, 1]
        oof[va] = pred_va
        test_avg += pred_te / N_FOLDS
        s = float(roc_auc_score(y[va], pred_va))
        fold_aucs.append(s); walls.append(time.time() - t0)
        print(f"  f{k}: AUC={s:.5f}  wall={walls[-1]:.1f}s")
    auc = float(roc_auc_score(y, oof))
    print(f"  GroupKF std OOF: {auc:.5f}  total wall={sum(walls):.1f}s")
    return oof, test_avg, auc, sum(walls)


# ---- FM (8-field, same as d9c_fm) -----------------------------------

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
    h = FeatureHasher(n_features=2**N_HASH_BITS_FM, input_type="string",
                      alternate_sign=False)
    Xtr = h.transform(rows([D_tr, C_tr, R_tr, Y_tr, S_tr, Tq_tr, Rq_tr, Pq_tr]))
    Xte = h.transform(rows([D_te, C_te, R_te, Y_te, S_te, Tq_te, Rq_te, Pq_te]))
    return Xtr, Xte


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


def csr_to_index_array(csr, n_active=8):
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


def fit_fm_one_fold(idx_tr_full, idx_te, y, tr_idx, va_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n_features = 2 ** N_HASH_BITS_FM
    model = FMModel(n_features, EMBED_DIM_FM)
    opt_dense = torch.optim.Adam([model.bias], lr=LR_FM)
    opt_sparse = torch.optim.SparseAdam(
        [model.linear.weight, model.embed.weight], lr=LR_FM)
    bce = nn.BCEWithLogitsLoss()
    idx_tr = torch.from_numpy(idx_tr_full[tr_idx])
    y_tr = torch.from_numpy(y[tr_idx].astype(np.float32))
    idx_va = torch.from_numpy(idx_tr_full[va_idx])
    idx_te_t = torch.from_numpy(idx_te)
    n_tr = len(idx_tr)
    perm = np.arange(n_tr)
    for ep in range(EPOCHS_FM):
        np.random.shuffle(perm)
        for s in range(0, n_tr, BATCH_FM):
            b = perm[s:s + BATCH_FM]
            xb = idx_tr[b]; yb = y_tr[b]
            logits = model(xb)
            loss = bce(logits, yb)
            opt_dense.zero_grad(); opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step(); opt_sparse.step()
    with torch.no_grad():
        chunks = [torch.sigmoid(model(idx_va[s:s + BATCH_FM])).numpy()
                  for s in range(0, len(idx_va), BATCH_FM)]
        p_va = np.concatenate(chunks)
        chunks = [torch.sigmoid(model(idx_te_t[s:s + BATCH_FM])).numpy()
                  for s in range(0, len(idx_te_t), BATCH_FM)]
        p_te = np.concatenate(chunks)
    return p_va.astype(np.float64), p_te.astype(np.float64)


def build_fm(train, test, y, splits):
    print(f"\n--- d9c_fm (PyTorch CPU FM, GroupKF) ---")
    n_tr, n_te = len(train), len(test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_avg = np.zeros(n_te, dtype=np.float64)
    walls, fold_aucs = [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        Xtr_csr, Xte_csr = build_main_effect_hashes(train, test, tr)
        idx_tr_full = csr_to_index_array(Xtr_csr)
        idx_te = csr_to_index_array(Xte_csr)
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=SEED + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        fold_aucs.append(s); walls.append(time.time() - t0)
        print(f"  f{k}: AUC={s:.5f}  wall={walls[-1]:.1f}s")
    auc = float(roc_auc_score(y, oof))
    print(f"  GroupKF std OOF: {auc:.5f}  total wall={sum(walls):.1f}s")
    return oof, test_avg, auc, sum(walls)


# ---- main ----------------------------------------------------------

def main():
    import sys
    t_total = time.time()

    # Allow running specific bases via CLI args, else run all
    valid = {"rule_driver_compound", "rule_year_race",
             "R6_next_compound", "R7_prev_compound", "R10_driver_eb",
             "R14_L4", "FM"}
    selected = set(sys.argv[1:]) & valid if len(sys.argv) > 1 else valid
    print(f"Bases to build: {sorted(selected)}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # Group IDs by (Race, Driver, Year, Stint) — strict P6 key
    grp = train.groupby(["Race", "Driver", "Year", "Stint"], sort=False).ngroup().values
    print(f"GroupKF group key (Race, Driver, Year, Stint) — n_groups={len(np.unique(grp))}")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.zeros(len(y)), y, groups=grp))
    for k, (tr, va) in enumerate(splits):
        print(f"  fold {k}: n_tr={len(tr)}, n_va={len(va)}, "
              f"val pos rate={y[va].mean():.4f}")

    # Pre-compute next/prev compound (no leakage; sequential lookup)
    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    nx, pv = _compute_neighbour_compounds(full)
    next_train = nx[full["__src"].values == "tr"]
    next_test = nx[full["__src"].values == "te"]
    prev_train = pv[full["__src"].values == "tr"]
    prev_test = pv[full["__src"].values == "te"]

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    X_test = test.drop(columns=[ID_COL], errors="ignore").copy()
    X_enc, X_test_enc = encode_features(X.copy(), X_test.copy())

    results = {}

    if "rule_driver_compound" in selected:
        oof, tp, auc, wall = build_with_residual(
            "d6_rule_driver_compound", rule_driver_compound, train, test,
            X_enc, X_test_enc, y, splits, has_residual=True)
        np.save(ART / "oof_d6_rule_driver_compound_groupkf.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / "test_d6_rule_driver_compound_groupkf.npy",
                np.column_stack([1 - tp, tp]))
        results["d6_rule_driver_compound"] = dict(groupkf_auc=auc, wall_s=wall)

    if "rule_year_race" in selected:
        oof, tp, auc, wall = build_with_residual(
            "d6_rule_year_race", rule_year_race, train, test,
            X_enc, X_test_enc, y, splits, has_residual=True)
        np.save(ART / "oof_d6_rule_year_race_groupkf.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / "test_d6_rule_year_race_groupkf.npy",
                np.column_stack([1 - tp, tp]))
        results["d6_rule_year_race"] = dict(groupkf_auc=auc, wall_s=wall)

    if "R6_next_compound" in selected:
        oof, tp, auc, wall = build_with_residual(
            "d9_R6_next_compound", rule_next_compound, train, test,
            X_enc, X_test_enc, y, splits, has_residual=True,
            next_train=next_train, next_test=next_test)
        np.save(ART / "oof_d9_R6_next_compound_groupkf.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / "test_d9_R6_next_compound_groupkf.npy",
                np.column_stack([1 - tp, tp]))
        results["d9_R6_next_compound"] = dict(groupkf_auc=auc, wall_s=wall)

    if "R7_prev_compound" in selected:
        oof, tp, auc, wall = build_with_residual(
            "d9_R7_prev_compound", rule_prev_compound, train, test,
            X_enc, X_test_enc, y, splits, has_residual=True,
            prev_train=prev_train, prev_test=prev_test)
        np.save(ART / "oof_d9_R7_prev_compound_groupkf.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / "test_d9_R7_prev_compound_groupkf.npy",
                np.column_stack([1 - tp, tp]))
        results["d9_R7_prev_compound"] = dict(groupkf_auc=auc, wall_s=wall)

    if "R10_driver_eb" in selected:
        oof, tp, auc, wall = build_with_residual(
            "d9_R10_driver_eb", rule_driver_eb, train, test,
            X_enc, X_test_enc, y, splits, has_residual=True)
        np.save(ART / "oof_d9_R10_driver_eb_groupkf.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / "test_d9_R10_driver_eb_groupkf.npy",
                np.column_stack([1 - tp, tp]))
        results["d9_R10_driver_eb"] = dict(groupkf_auc=auc, wall_s=wall)

    if "R14_L4" in selected:
        oof, tp, auc, wall = build_r14_l4(train, test, y, splits)
        np.save(ART / "oof_d9b_R14_L4_groupkf.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / "test_d9b_R14_L4_groupkf.npy",
                np.column_stack([1 - tp, tp]))
        results["d9b_R14_L4"] = dict(groupkf_auc=auc, wall_s=wall)

    if "FM" in selected:
        oof, tp, auc, wall = build_fm(train, test, y, splits)
        np.save(ART / "oof_d9c_fm_groupkf.npy",
                np.column_stack([1 - oof, oof]))
        np.save(ART / "test_d9c_fm_groupkf.npy",
                np.column_stack([1 - tp, tp]))
        results["d9c_fm"] = dict(groupkf_auc=auc, wall_s=wall)

    final = dict(results=results, total_wall_s=time.time() - t_total)
    out_name = "d12_groupkf_rebuild_results.json"
    if len(selected) < len(valid):
        out_name = f"d12_groupkf_rebuild_partial_{'_'.join(sorted(selected))[:80]}.json"
    (ART / out_name).write_text(json.dumps(final, indent=2))
    print(f"\n→ {ART / out_name}  (wall {time.time()-t_total:.0f}s)")


if __name__ == "__main__":
    main()
