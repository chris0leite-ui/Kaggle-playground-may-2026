"""Factorization Machine (FM) baseline — Tier-1 idea 2.

Mathematically:
  ŷ = w₀ + Σᵢ wᵢxᵢ + Σᵢ<ⱼ ⟨vᵢ, vⱼ⟩ xᵢxⱼ
where each feature i has a low-rank embedding vᵢ ∈ R^k. Computed
efficiently as
  Σᵢ<ⱼ ⟨vᵢ, vⱼ⟩ xᵢxⱼ = (1/2) Σf [(Σᵢ vᵢ,f xᵢ)² − Σᵢ vᵢ,f² xᵢ²].

Distinct mechanism vs both LR (which needs hand-engineered
interactions) and GBDT (which partitions feature space). Same FEATURE
set as R14_L2 (the d9b sweet spot) — main effects only — so FM's job
is to *learn* the cross-feature low-rank interactions.

Features (main effects only):
  D, C, R, Y, S, T_q5, Rp_q5, P_q5

PyTorch implementation, CPU, batch SGD with Adam.
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
PRIMARY_S, PRIMARY_LB = 0.95065, 0.95026
RHO_TIE = 0.999
N_HASH_BITS = 18  # 2^18 = 262144 hash buckets
EMBED_DIM = 8     # FM rank
EPOCHS = 6
BATCH = 8192
LR = 0.05
WD = 1e-6


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
    """Return (csr_train, csr_test) of one-hot main-effect features.
    Bin edges fitted on tr_idx slice only. 8 fields."""
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


class FMModel(nn.Module):
    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        # Linear weights — one per hashed feature
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        nn.init.zeros_(self.linear.weight)
        # Factor embeddings
        self.embed = nn.Embedding(n_features, embed_dim, sparse=True)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, idx_batch: torch.Tensor) -> torch.Tensor:
        """idx_batch: LongTensor (batch, n_active_per_row=8)."""
        # Linear: sum embedding lookups
        lin = self.linear(idx_batch).sum(dim=1).squeeze(-1)
        # FM 2-way: 0.5 * (sum_v² − sum_v_squared)
        v = self.embed(idx_batch)             # (B, m, k)
        sum_v = v.sum(dim=1)                  # (B, k)
        sum_v_sq = (v * v).sum(dim=1)         # (B, k)
        inter = 0.5 * (sum_v.pow(2).sum(dim=1) - sum_v_sq.sum(dim=1))
        return self.bias + lin + inter


def csr_to_index_array(csr: csr_matrix) -> np.ndarray:
    """Convert CSR with constant nnz=8 per row to (n, 8) int32 indices."""
    n = csr.shape[0]
    indptr, indices = csr.indptr, csr.indices
    nnz_per_row = np.diff(indptr)
    expected = 8
    if nnz_per_row.max() > expected:
        # FeatureHasher hash collisions can collapse two identical strings to
        # one bucket within a row. Pad to width=expected by repeating last idx.
        pass
    out = np.zeros((n, expected), dtype=np.int64)
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        row = indices[s:e]
        if len(row) >= expected:
            out[i] = row[:expected]
        else:
            out[i, :len(row)] = row
            # Pad with the first index (FM linear is just a sum so duplicates
            # don't change the "main effect" contribution by much; or use 0
            # as a no-op since linear[0] starts at zero)
            out[i, len(row):] = row[0] if len(row) else 0
    return out


def fit_fm_one_fold(idx_tr_full: np.ndarray, idx_te: np.ndarray, y: np.ndarray,
                   tr_idx: np.ndarray, va_idx: np.ndarray, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_features = 2 ** N_HASH_BITS
    model = FMModel(n_features, EMBED_DIM)
    # Split: dense bias uses Adam; sparse embeddings use SparseAdam.
    opt_dense = torch.optim.Adam([model.bias], lr=LR)
    opt_sparse = torch.optim.SparseAdam(
        [model.linear.weight, model.embed.weight], lr=LR)
    bce = nn.BCEWithLogitsLoss()

    idx_tr = torch.from_numpy(idx_tr_full[tr_idx])
    y_tr = torch.from_numpy(y[tr_idx].astype(np.float32))
    idx_va = torch.from_numpy(idx_tr_full[va_idx])
    idx_te_t = torch.from_numpy(idx_te)
    n_tr = len(idx_tr)
    perm = np.arange(n_tr)
    for ep in range(EPOCHS):
        np.random.shuffle(perm)
        ep_loss = 0.0
        n_seen = 0
        for s in range(0, n_tr, BATCH):
            b_idx = perm[s:s + BATCH]
            xb = idx_tr[b_idx]
            yb = y_tr[b_idx]
            logits = model(xb)
            loss = bce(logits, yb)
            opt_dense.zero_grad()
            opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step()
            opt_sparse.step()
            ep_loss += float(loss.item()) * len(b_idx)
            n_seen += len(b_idx)
        # End-of-epoch quick eval
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
    # Final test predictions
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


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_oof = np.load(ART / "oof_d6_k18_multi_rule_strat.npy"
                          )[:, 1].astype(np.float64)
    primary_test = np.load(ART / "test_d6_k18_multi_rule_strat.npy"
                           )[:, 1].astype(np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    n_train = len(y); n_test = len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)

    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        print(f"\n--- fold {k} ---")
        Xtr_csr, Xte_csr = build_main_effect_hashes(train, test, tr)
        idx_tr_full = csr_to_index_array(Xtr_csr)
        idx_te = csr_to_index_array(Xte_csr)
        print(f"    feat-build wall {time.time()-t_fold:.1f}s; "
              f"shapes train_idx {idx_tr_full.shape}, test_idx {idx_te.shape}")
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=SEED + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        wall = time.time() - t_fold
        s = float(roc_auc_score(y[va], p_va))
        print(f"    fold {k} done: val AUC {s:.5f}  wall {wall:.1f}s")

    auc = float(roc_auc_score(y, oof))
    rho_test, _ = spearmanr(test_avg, primary_test)
    F_min = expand(np.column_stack([primary_oof, oof]))
    F_min_t = expand(np.column_stack([primary_test, test_avg]))
    mo, _ = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo))
    delta = (auc_min - PRIMARY_S) * 1e4
    print(f"\n=== FM standalone ===")
    print(f"  std OOF: {auc:.5f}")
    print(f"  ρ vs PRIMARY test: {rho_test:.5f}")
    print(f"  Min-meta OOF: {auc_min:.5f}  Δ PRIMARY {delta:+.2f}bp  "
          f"{'PASS ✓' if auc_min >= PRIMARY_S else 'FAIL ✗'}")

    np.save(ART / "oof_d9c_fm_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d9c_fm_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))
    final = dict(
        std_oof=auc, rho_vs_primary_test=float(rho_test),
        min_meta_oof=auc_min, delta_primary_bp=float(delta),
        min_meta_pass=bool(auc_min >= PRIMARY_S),
        params=dict(embed_dim=EMBED_DIM, epochs=EPOCHS, batch=BATCH,
                    lr=LR, wd=WD, n_hash_bits=N_HASH_BITS),
        total_wall_s=time.time() - t0,
    )
    (ART / "d9c_fm_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d9c_fm_results.json  "
          f"(total wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
