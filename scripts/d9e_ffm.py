"""Day-9e — Field-aware Factorization Machine (FFM).

FFM extends FM by giving each feature F embeddings (one per "target
field" it interacts with). For features i in field f_i and j in
field f_j the interaction uses:
  ⟨v_{i, f_j}, v_{j, f_i}⟩
i.e., feature i's embedding "intended for" field f_j times feature
j's embedding "intended for" field f_i.

For our 8-field setup (D, C, R, Y, S, T, Rp, P) and k=4:
  embed param count: n_features × n_fields × k = 2^18 × 8 × 4 ≈ 8.4M
  per-row interaction cost: O(n_fields² × k) = O(256) ops

Reuses d9c's hash + index-array feature builder.

Hypothesis: FFM provides an embedding structure FM cannot — one
feature's expression of "what it means to interact with this field"
varies per partner field. Predicted std OOF: 0.92–0.94 (FM's 0.92
plus 0–2bp from the richer embedding).
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
N_FIELDS = 8
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
PRIMARY_FM_FNAME = "d9c_fm"


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
    out = np.zeros((n, N_FIELDS), dtype=np.int64)
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        row = indices[s:e]
        if len(row) >= N_FIELDS:
            out[i] = row[:N_FIELDS]
        else:
            out[i, :len(row)] = row
            out[i, len(row):] = row[0] if len(row) else 0
    return out


class FFMModel(nn.Module):
    """Field-aware FM. Each feature has n_fields embeddings of size k.
    Per-row idx_batch shape (B, F): one feature index per field."""
    def __init__(self, n_features: int, n_fields: int, embed_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        nn.init.zeros_(self.linear.weight)
        self.embed = nn.Embedding(n_features, n_fields * embed_dim, sparse=True)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.n_fields = n_fields
        self.embed_dim = embed_dim

    def forward(self, idx_batch):
        # idx_batch: (B, n_fields) — feature index in each field
        B, F = idx_batch.shape
        K = self.embed_dim
        # Linear part
        lin = self.linear(idx_batch).sum(dim=1).squeeze(-1)
        # Embedding lookup → (B, F, F*K) → (B, F, F, K)
        # E[b, i, j, :] = embedding for feature in field i when interacting with field j
        E = self.embed(idx_batch).view(B, F, F, K)
        # FFM interaction: 0.5 × Σ_{i ≠ j} <E[b, i, j], E[b, j, i]>
        # E_T[b, i, j, :] = E[b, j, i, :]
        E_T = E.transpose(1, 2)
        # Element-wise product, sum over k → (B, F, F) of pairwise dot products
        prod = (E * E_T).sum(dim=-1)
        full_sum = prod.sum(dim=(-2, -1))                     # (B,)
        diag_sum = prod.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
        inter = 0.5 * (full_sum - diag_sum)
        return self.bias + lin + inter


def fit_ffm_one_fold(idx_tr_full, idx_te, y, tr_idx, va_idx, params, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n_features = 2 ** N_HASH_BITS
    model = FFMModel(n_features, N_FIELDS, params["embed_dim"])
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
    for ep in range(params["epochs"]):
        np.random.shuffle(perm)
        ep_loss = 0.0; n_seen = 0
        for s in range(0, n_tr, BATCH):
            b_idx = perm[s:s + BATCH]
            xb = idx_tr[b_idx]; yb = y_tr[b_idx]
            logits = model(xb)
            loss = bce(logits, yb)
            opt_dense.zero_grad(); opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step(); opt_sparse.step()
            ep_loss += float(loss.item()) * len(b_idx); n_seen += len(b_idx)
        with torch.no_grad():
            chunks = []
            for s in range(0, len(idx_va), BATCH):
                chunks.append(torch.sigmoid(model(idx_va[s:s + BATCH])).numpy())
            p_va = np.concatenate(chunks)
            try:
                auc = roc_auc_score(y[va_idx], p_va)
            except Exception:
                auc = float("nan")
        print(f"      epoch {ep}: train_loss={ep_loss/n_seen:.5f}  val_auc={auc:.5f}")
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


def train_ffm(train, test, y, splits, params, seed):
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        Xtr_csr, Xte_csr = build_main_effect_hashes(train, test, tr)
        idx_tr_full = csr_to_index_array(Xtr_csr)
        idx_te = csr_to_index_array(Xte_csr)
        p_va, p_te = fit_ffm_one_fold(idx_tr_full, idx_te, y, tr, va, params,
                                      seed=seed + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        print(f"    fold {k} done: val AUC {s:.5f}  wall {time.time()-t_fold:.1f}s")
    return oof, test_avg


def stack_eval(label, fm_like_oof, fm_like_test, y, primary_test,
               base_oof, base_test, base_names, d9_oof, d9_test, d9_names,
               results, also_keep_pool_fm=False, fm_pool_oof=None,
               fm_pool_test=None):
    """Build K=20 (replace FM) or K=21 (keep FM, add new) stack."""
    if also_keep_pool_fm:
        Xs = base_oof + d9_oof + [fm_pool_oof, fm_like_oof]
        Ts = base_test + d9_test + [fm_pool_test, fm_like_test]
        Ns = base_names + d9_names + ["FM_d9c", "FFM"]
        K = len(Ns)
    else:
        Xs = base_oof + d9_oof + [fm_like_oof]
        Ts = base_test + d9_test + [fm_like_test]
        Ns = base_names + d9_names + ["FFM"]
        K = len(Ns)
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
    coef = lr_full.coef_.ravel()
    auc = float(roc_auc_score(y, meta_oof))
    rho, _ = spearmanr(test_pred, primary_test)
    delta = (auc - PRIMARY_S) * 1e4
    l1 = {Ns[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"\n=== {label} (K={K}) ===")
    print(f"  Strat OOF: {auc:.5f}  Δ PRIMARY {delta:+.2f}bp  "
          f"ρ vs PRIMARY {rho:.5f}")
    print(f"  L1 top-12:")
    for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:12]:
        marker = ""
        if n_ == "FFM" or n_ == "FM_d9c": marker = "  ← FM-class"
        elif n_.startswith("R") and "_" in n_: marker = "  ← d9-base"
        elif n_.startswith("rule_"): marker = "  ← existing rule"
        print(f"    {n_:<24s} L1={v:.3f}{marker}")
    results[label] = dict(K=K, strat_oof=auc, delta_primary_bp=delta,
                          rho_vs_primary=float(rho), l1_ranking=l1)
    return meta_oof, test_pred


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

    # Train FFM
    print("\n=== FFM 5-fold (k=4, n_fields=8, 6 epochs) ===\n")
    params = dict(embed_dim=4, epochs=6, lr=0.05, batch=8192)
    oof_ffm, test_ffm = train_ffm(train, test, y, splits, params, seed=42)
    auc_ffm = float(roc_auc_score(y, oof_ffm))
    rho_ffm_prim, _ = spearmanr(test_ffm, primary_test)
    F_min = expand(np.column_stack([primary_oof, oof_ffm]))
    F_min_t = expand(np.column_stack([primary_test, test_ffm]))
    mo, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo))
    delta_mm = (auc_min - PRIMARY_S) * 1e4
    print(f"\n  FFM std OOF: {auc_ffm:.5f}")
    print(f"  ρ vs PRIMARY test: {rho_ffm_prim:.5f}")
    print(f"  Min-meta vs PRIMARY OOF: {auc_min:.5f}  Δ {delta_mm:+.2f}bp  "
          f"{'PASS ✓' if auc_min >= PRIMARY_S else 'FAIL ✗'}")

    np.save(ART / "oof_d9e_ffm_strat.npy",
            np.column_stack([1 - oof_ffm, oof_ffm]))
    np.save(ART / "test_d9e_ffm_strat.npy",
            np.column_stack([1 - test_ffm, test_ffm]))

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
    fm_pool_oof = np.load(ART / f"oof_{PRIMARY_FM_FNAME}_strat.npy"
                          )[:, 1].astype(np.float64)
    fm_pool_test = np.load(ART / f"test_{PRIMARY_FM_FNAME}_strat.npy"
                           )[:, 1].astype(np.float64)

    results = dict(
        ffm_standalone=dict(std_oof=auc_ffm, rho_vs_primary=float(rho_ffm_prim),
                            min_meta_oof=auc_min,
                            delta_primary_bp=float(delta_mm),
                            min_meta_pass=bool(auc_min >= PRIMARY_S)),
    )

    # Stack experiments
    # (a) K=20 swap FM → FFM
    mo_swap, tp_swap = stack_eval(
        "K20_swap_FM_to_FFM", oof_ffm, test_ffm, y, primary_test,
        base_oof, base_test, base_names, d9_oof, d9_test, d9_names, results)
    # (b) K=21 keep FM AND add FFM
    mo_add, tp_add = stack_eval(
        "K21_add_FFM", oof_ffm, test_ffm, y, primary_test,
        base_oof, base_test, base_names, d9_oof, d9_test, d9_names, results,
        also_keep_pool_fm=True, fm_pool_oof=fm_pool_oof,
        fm_pool_test=fm_pool_test)

    # Save submission CSVs
    np.save(ART / "oof_d9e_K20_swap_FFM_strat.npy",
            np.column_stack([1 - mo_swap, mo_swap]))
    np.save(ART / "test_d9e_K20_swap_FFM_strat.npy",
            np.column_stack([1 - tp_swap, tp_swap]))
    sub = sample_sub.copy(); sub[TARGET] = tp_swap
    sub.to_csv("submissions/submission_d9e_K20_swap_FFM.csv", index=False)
    print("→ wrote submissions/submission_d9e_K20_swap_FFM.csv")

    np.save(ART / "oof_d9e_K21_add_FFM_strat.npy",
            np.column_stack([1 - mo_add, mo_add]))
    np.save(ART / "test_d9e_K21_add_FFM_strat.npy",
            np.column_stack([1 - tp_add, tp_add]))
    sub = sample_sub.copy(); sub[TARGET] = tp_add
    sub.to_csv("submissions/submission_d9e_K21_add_FFM.csv", index=False)
    print("→ wrote submissions/submission_d9e_K21_add_FFM.csv")

    final = dict(
        results=results, params=params,
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        total_wall_s=time.time() - t0,
    )
    (ART / "d9e_ffm_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9e_ffm_results.json  "
          f"(total wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
