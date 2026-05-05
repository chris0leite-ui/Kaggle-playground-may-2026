"""Day-10 evening — GroupKFold audit on FM bases.

Purpose: confirm or refute the hypothesis that FM-class submissions
(d9c, d9f, d9h, d9i — all +2-3bp public LB lifts) are real, not
leakage-amplified noise. Per P6, StratifiedKFold has 80% within-
group leakage on (Race, Driver, Year, Stint). We re-train the FM
bases under strict GroupKFold by that key and compare AUCs.

If FM AUC under strict GroupKF ≈ FM AUC under Strat → FM doesn't
exploit leakage; the public-LB lifts are private-LB-robust.

If FM AUC under strict GroupKF << FM AUC under Strat → FM gain
came from within-group leakage; lifts likely don't transfer to
private. Should de-emphasize FM PRIMARY.

Bases tested:
  d9c FM (8 features unified)
  d9f FM_A (D, C, S, T_q5)
  d9f FM_B (R, Y, Rp_q5, P_q5)

For comparison, also report Strat→GroupKF AUC drop for one GBDT
base (e3_hgbc — already has GroupKF artifact). If FM drops less
than GBDT, FM is more robust → lifts real.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
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


def _quantile_bin(arr_train, arr_query, n_bins):
    edges = np.quantile(arr_train, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                   0, n_bins - 1).astype(int)


def build_partition_hashes(train, test, tr_idx, fields):
    """Same as d9f's, but train and test."""
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
        chunks = [torch.sigmoid(model(idx_va[s:s+BATCH])).numpy()
                  for s in range(0, len(idx_va), BATCH)]
        p_va = np.concatenate(chunks)
    return p_va.astype(np.float64)


def train_groupkf(train, test, y, group_ids, fields, label):
    print(f"\n--- {label} (fields={fields}, GroupKF strict) ---")
    n_train = len(y)
    oof = np.zeros(n_train, dtype=np.float64)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (tr, va) in enumerate(gkf.split(np.zeros(n_train), y, groups=group_ids)):
        t_fold = time.time()
        Xtr_csr, _ = build_partition_hashes(train, test, tr, fields)
        idx_tr_full = csr_to_index_array(Xtr_csr, len(fields))
        # Need test array too; pass empty test and skip
        p_va = fit_fm_one_fold(idx_tr_full, idx_tr_full[:1], y, tr, va,
                               seed=42 + k)
        oof[va] = p_va
        s = float(roc_auc_score(y[va], p_va))
        print(f"  fold {k}: val AUC {s:.5f}  wall {time.time()-t_fold:.1f}s "
              f"(n_tr={len(tr)}, n_va={len(va)})")
    auc = float(roc_auc_score(y, oof))
    print(f"  GroupKF std OOF: {auc:.5f}")
    return oof, auc


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # Build group IDs by (Race, Driver, Year, Stint) — strict P6 key
    grp = train.groupby(["Race", "Driver", "Year", "Stint"], sort=False).ngroup().values
    print(f"GroupKF group key: (Race, Driver, Year, Stint)")
    print(f"  n_groups: {len(np.unique(grp))}")

    # Existing Strat AUCs for reference
    strat_aucs = {
        "d9c_FM": 0.92069,
        "d9f_FM_A": 0.82505,
        "d9f_FM_B": 0.88438,
        "e3_hgbc": 0.94876,
        "cb_slow-wide-bag": 0.94790,
    }
    # Existing GroupKF AUCs (Race-only) from comp-context calibration ladder
    groupkf_race_aucs = {
        "e3_hgbc": 0.92785,
        "cb_slow-wide-bag": 0.92322,
    }

    results = {}
    # FM bases under strict GroupKF
    for label, fields in [
        ("d9c_FM_unified_8", ["D", "C", "R", "Y", "S", "T", "Rp", "P"]),
        ("d9f_FM_A_4", ["D", "C", "S", "T"]),
        ("d9f_FM_B_4", ["R", "Y", "Rp", "P"]),
    ]:
        oof, auc = train_groupkf(train, test, y, grp, fields, label)
        np.save(ART / f"oof_{label}_groupkf_strict.npy",
                np.column_stack([1 - oof, oof]))
        strat_auc = strat_aucs.get(label.split("_strict")[0].split("_")[0]) or {
            "d9c_FM_unified_8": 0.92069,
            "d9f_FM_A_4": 0.82505,
            "d9f_FM_B_4": 0.88438,
        }[label]
        # Strat reference comparison
        ref = {
            "d9c_FM_unified_8": 0.92069,
            "d9f_FM_A_4": 0.82505,
            "d9f_FM_B_4": 0.88438,
        }[label]
        drop_bp = (ref - auc) * 1e4
        results[label] = dict(strat_auc=ref, groupkf_strict_auc=auc,
                              drop_bp=float(drop_bp))
        print(f"  Δ vs Strat: {drop_bp:+.2f}bp")

    # Comparison table
    print("\n" + "=" * 72)
    print(f"{'base':<24s} {'Strat':>8s} {'GKF-Race':>9s} {'GKF-strict':>10s}  Δstrat→strict")
    print("-" * 72)
    for label, r in results.items():
        gkf_race = groupkf_race_aucs.get(label.split("_")[0], None)
        gkf_race_str = f"{gkf_race:.5f}" if gkf_race else "n/a"
        print(f"{label:<24s} {r['strat_auc']:>8.5f} {gkf_race_str:>9s} "
              f"{r['groupkf_strict_auc']:>10.5f}  {-r['drop_bp']:+.2f}bp")
    # GBDT bases for context
    for label in ["e3_hgbc", "cb_slow-wide-bag"]:
        gkf = groupkf_race_aucs[label]
        s = strat_aucs[label]
        print(f"{label:<24s} {s:>8.5f} {gkf:>9.5f} {'n/a':>10s}  "
              f"{(gkf - s)*1e4:+.2f}bp (GKF-Race only)")
    print("=" * 72)

    final = dict(
        results=results,
        strat_reference=strat_aucs,
        groupkf_race_reference=groupkf_race_aucs,
        total_wall_s=time.time() - t0,
    )
    (ART / "d10_groupkf_audit_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d10_groupkf_audit_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
