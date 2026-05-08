"""Day-10 — GroupKF stack rebuild as private-LB proxy.

The d10 audit confirmed FM bases are leakage-robust: under strict
GroupKF by (Race, Driver, Year, Stint), FM bases drop only 2.5–54bp
vs GBDTs dropping 200+bp under weaker Race-only GKF. This says
the *bases* don't exploit leakage — but the *stack* OOF (0.95073)
is built on Strat splits and could still be 200bp inflated through
the GBDT pool members.

This script rebuilds the K=N LR-meta stack on Race-only GroupKF
artifacts (which we have for the GBDT/baseline pool). We freshly
train d9f's FM_A and FM_B under Race-only GKF (matching partition)
so all bases share a leakage-blocking partition. Then we compare:

  - K=13 GKF stack: GBDT/baseline GKF pool only
  - K=15 GKF stack: + FM_A + FM_B (Race-only GKF)
  - K=21 Strat stack reference: 0.95073 (existing PRIMARY)

The diagnostic question:
  Δ(K=15 GKF − K=13 GKF) ≈ Δ(K=21 Strat − K=19 Strat)?

If yes → FM-class lift is private-robust; the +2bp K=21 swap that
landed +2bp on public LB is real and should transfer.

If no (K=13 → K=15 GKF flat) → FM lift came from interaction with
leakage-inflated GBDT predictions on Strat splits; private LB will
likely revert to the GBDT-only baseline.

Race-only GKF is the available partition (existing artifacts) and
is *weaker* than strict (Race, Driver, Year, Stint), so this is a
conservative test — if FM lifts under weaker GKF, it almost
certainly lifts under stricter GKF too.
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
from sklearn.model_selection import GroupKFold

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

# Pool members that have Race-only GroupKF artifacts already built.
# These bases all use GroupKFold(5) on Race (the comp-context default).
POOL_GKF_AVAILABLE = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"),
    ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
]

PARTITION_A = ["D", "C", "S", "T"]
PARTITION_B = ["R", "Y", "Rp", "P"]


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, y, splits):
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.coef_.ravel()


def _quantile_bin(arr_train, arr_query, n_bins):
    edges = np.quantile(arr_train, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                   0, n_bins - 1).astype(int)


def build_partition_hashes(train, test, tr_idx, fields):
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
            arrs_tr.append(train["Stint"].clip(upper=5).astype(int)
                            .astype(str).values)
            arrs_te.append(test["Stint"].clip(upper=5).astype(int)
                            .astype(str).values)
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


def fit_fm_one_fold(idx_tr_full, idx_te_arr, y, tr, va, seed):
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
    idx_te = torch.from_numpy(idx_te_arr)
    n_tr = len(idx_tr)
    perm = np.arange(n_tr)
    for ep in range(EPOCHS):
        np.random.shuffle(perm)
        for s in range(0, n_tr, BATCH):
            b = perm[s:s + BATCH]
            logits = model(idx_tr[b])
            loss = bce(logits, y_tr[b])
            opt_dense.zero_grad(); opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step(); opt_sparse.step()
    with torch.no_grad():
        p_va = np.concatenate([
            torch.sigmoid(model(idx_va[s:s+BATCH])).numpy()
            for s in range(0, len(idx_va), BATCH)])
        p_te = np.concatenate([
            torch.sigmoid(model(idx_te[s:s+BATCH])).numpy()
            for s in range(0, len(idx_te), BATCH)])
    return p_va.astype(np.float64), p_te.astype(np.float64)


def train_fm_groupkf(train, test, y, group_ids, fields, label):
    print(f"\n--- {label} (fields={fields}, GKF on Race) ---")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.zeros(n_train), y, groups=group_ids))
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
    auc = float(roc_auc_score(y, oof))
    print(f"  → {label} GKF OOF: {auc:.5f}")
    return oof, test_avg, auc, splits


def stack_under_gkf(label, base_oofs, base_tests, names, y, splits):
    K = len(names)
    P_oof = np.column_stack(base_oofs)
    F_oof = expand(P_oof)
    meta_oof, coef = fit_lr_meta(F_oof, y, splits)
    auc = float(roc_auc_score(y, meta_oof))
    l1 = {names[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"\n=== {label} (K={K}) ===")
    print(f"  GKF stack OOF: {auc:.5f}")
    print(f"  L1 ranking (top-15):")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:15]:
        marker = "  ← FM-class" if n.startswith("FM") else ""
        print(f"    {n:<24s} L1={v:.3f}{marker}")
    return meta_oof, auc, l1


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # Race-only GKF group key (matches existing pool artifacts)
    grp = train["Race"].astype(str).values
    print(f"Race-only GKF: {len(np.unique(grp))} groups, "
          f"{N_FOLDS} folds")

    # Build the LR-meta CV splits using Race-only GKF (matching base
    # OOFs' partition).
    gkf = GroupKFold(n_splits=N_FOLDS)
    meta_splits = list(gkf.split(np.zeros(len(y)), y, groups=grp))

    # ---- Train FM_A + FM_B under Race-only GKF (matches pool) ----
    oof_A, test_A, auc_A, _ = train_fm_groupkf(
        train, test, y, grp, PARTITION_A, "FM_A_driver_dynamics")
    np.save(ART / "oof_d9f_FM_A_groupkf_race.npy",
            np.column_stack([1 - oof_A, oof_A]))
    oof_B, test_B, auc_B, _ = train_fm_groupkf(
        train, test, y, grp, PARTITION_B, "FM_B_race_context")
    np.save(ART / "oof_d9f_FM_B_groupkf_race.npy",
            np.column_stack([1 - oof_B, oof_B]))

    # ---- Load existing GBDT/baseline GKF pool ----
    print(f"\n--- Loading {len(POOL_GKF_AVAILABLE)} GBDT/baseline GKF bases ---")
    base_oofs, base_tests, base_names = [], [], []
    for label, fname in POOL_GKF_AVAILABLE:
        oo = np.load(ART / f"oof_{fname}_groupkf.npy")[:, 1].astype(np.float64)
        # test arrays exist for most but not all; we don't need them for
        # stack OOF comparison — only OOF.
        s = float(roc_auc_score(y, oo))
        print(f"  {label:<24s} GKF AUC {s:.5f}")
        base_oofs.append(oo); base_names.append(label)

    # ---- K=13 GKF stack (GBDT/baseline only) ----
    base_only_oofs = list(base_oofs)
    base_only_names = list(base_names)
    _, auc_K13, l1_K13 = stack_under_gkf(
        "K=13 GKF baseline (no FM)", base_only_oofs, None,
        base_only_names, y, meta_splits)

    # ---- K=15 GKF stack (+ FM_A + FM_B Race-GKF) ----
    full_oofs = base_oofs + [oof_A, oof_B]
    full_names = base_names + ["FM_A_driver_dynamics", "FM_B_race_context"]
    _, auc_K15, l1_K15 = stack_under_gkf(
        "K=15 GKF + FM_A + FM_B", full_oofs, None, full_names,
        y, meta_splits)

    # ---- Lift analysis ----
    fm_lift_gkf_bp = (auc_K15 - auc_K13) * 1e4
    print("\n" + "=" * 72)
    print("FM-class lift comparison: GKF vs Strat reference")
    print("-" * 72)
    print(f"  K=13 GKF baseline (no FM):    {auc_K13:.5f}")
    print(f"  K=15 GKF + FM_A + FM_B:       {auc_K15:.5f}")
    print(f"  → FM-class lift under GKF:    {fm_lift_gkf_bp:+.2f}bp")
    print()
    print(f"  K=21 Strat swap (PRIMARY):    0.95073")
    print(f"  K=19 Strat swap (no FM ref):  estimated 0.95065 (d6_k18 + R6/7/10)")
    print(f"  → FM-class lift under Strat:  ~+0.7-0.8bp (estimate)")
    print()
    if fm_lift_gkf_bp >= 0.5:
        print("  VERDICT: FM-class lift transfers under GKF (≥+0.5bp).")
        print("  PRIMARY (d9f K=21 swap) is private-LB robust.")
    elif fm_lift_gkf_bp >= 0:
        print("  VERDICT: FM-class lift partly transfers (0–0.5bp).")
        print("  Some Strat lift is real; expect ~half on private.")
    else:
        print("  VERDICT: FM-class lift does NOT transfer under GKF.")
        print("  PRIMARY's Strat lift was leakage-amplified.")
        print("  Reconsider HEDGE selection.")
    print("=" * 72)

    # ---- Save full results ----
    final = dict(
        race_only_gkf=dict(
            n_groups=int(len(np.unique(grp))),
            fm_A_std_oof=auc_A, fm_B_std_oof=auc_B,
            stack_K13_oof=auc_K13, stack_K15_oof=auc_K15,
            fm_class_lift_bp=float(fm_lift_gkf_bp),
            l1_K13=l1_K13, l1_K15=l1_K15,
        ),
        strat_reference=dict(
            primary_K21_swap_oof=0.95073,
            primary_lb=0.95031,
        ),
        wall_s=time.time() - t0,
    )
    (ART / "d10b_groupkf_stack_rebuild.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d10b_groupkf_stack_rebuild.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
