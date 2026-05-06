"""Day-10 — leak-corrected LR meta.

d10b/c found that under Race-only GKF, FM_B becomes the #1 L1
component (L1=6.96, 2× the next base) — but our PRIMARY (d9f K=21
swap) is fit with an LR meta on STRAT OOFs, where FMs are mid-pack
(L1=0.138 for FM_B). The Strat meta under-weights FM precisely
because the GBDT bases produce leakage-inflated OOFs that "look"
more informative.

This script builds a leak-corrected LR meta:
  1. Refit LR on K=15 GKF OOFs (13 GBDT + FM_A + FM_B, all under
     Race-only GroupKFold). FM gets correct weight.
  2. Apply the GKF-trained LR coefficients to test predictions
     produced by GKF base training (5-fold-averaged).
  3. Compare to existing PRIMARY (Strat-meta) test prediction:
     ρ, predicted-LB, pre_submit_diff gate.

The bases are still fit on full train data (5-fold averaged from
GKF training); only the meta weights are leak-corrected. This is
not a private-LB oracle — but it should up-weight FM_B without
needing leakier validation signal.

Builder + submission writer combined; ~5 min wall (FM training
dominates).
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
PRIMARY_S = 0.95073   # d9f K=21 swap Strat OOF
PRIMARY_LB = 0.95031  # current public PRIMARY

POOL_GKF = [
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
        sum_v = v.sum(dim=1); sum_v_sq = (v * v).sum(dim=1)
        inter = 0.5 * (sum_v.pow(2).sum(dim=1) - sum_v_sq.sum(dim=1))
        return self.bias + lin + inter


def fit_fm_one_fold(idx_tr_full, idx_te_arr, y, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FMModel(2 ** N_HASH_BITS, EMBED_DIM)
    opt_dense = torch.optim.Adam([model.bias], lr=LR)
    opt_sparse = torch.optim.SparseAdam(
        [model.linear.weight, model.embed.weight], lr=LR)
    bce = nn.BCEWithLogitsLoss()
    idx_tr = torch.from_numpy(idx_tr_full[tr])
    y_tr = torch.from_numpy(y[tr].astype(np.float32))
    idx_va = torch.from_numpy(idx_tr_full[va])
    idx_te = torch.from_numpy(idx_te_arr)
    perm = np.arange(len(idx_tr))
    for ep in range(EPOCHS):
        np.random.shuffle(perm)
        for s in range(0, len(idx_tr), BATCH):
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
    print(f"\n--- {label} (fields={fields}) ---")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (tr, va) in enumerate(gkf.split(np.zeros(n_train), y, groups=group_ids)):
        t_fold = time.time()
        Xtr_csr, Xte_csr = build_partition_hashes(train, test, tr, fields)
        idx_tr_full = csr_to_index_array(Xtr_csr, len(fields))
        idx_te = csr_to_index_array(Xte_csr, len(fields))
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va, seed=42+k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        print(f"  fold {k}: val AUC {roc_auc_score(y[va], p_va):.5f}  "
              f"wall {time.time()-t_fold:.1f}s")
    return oof, test_avg


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    # Race-only GKF (matches existing pool partition)
    grp = train["Race"].astype(str).values
    print(f"Race-only GKF: {len(np.unique(grp))} groups, {N_FOLDS} folds")

    # ---- Train FM_A + FM_B under Race-only GKF ----
    oof_A, test_A = train_fm_groupkf(train, test, y, grp,
                                      PARTITION_A, "FM_A_driver_dynamics")
    np.save(ART / "test_d9f_FM_A_groupkf_race.npy",
            np.column_stack([1 - test_A, test_A]))
    oof_B, test_B = train_fm_groupkf(train, test, y, grp,
                                      PARTITION_B, "FM_B_race_context")
    np.save(ART / "test_d9f_FM_B_groupkf_race.npy",
            np.column_stack([1 - test_B, test_B]))
    print(f"\nFM_A GKF OOF AUC: {roc_auc_score(y, oof_A):.5f}")
    print(f"FM_B GKF OOF AUC: {roc_auc_score(y, oof_B):.5f}")

    # ---- Load 13-base GKF pool (OOF + test) ----
    print(f"\n--- Loading {len(POOL_GKF)} GBDT/baseline GKF bases ---")
    base_oofs, base_tests, base_names = [], [], []
    for label, fname in POOL_GKF:
        oo = np.load(ART / f"oof_{fname}_groupkf.npy")[:, 1].astype(np.float64)
        te = np.load(ART / f"test_{fname}_groupkf.npy")[:, 1].astype(np.float64)
        base_oofs.append(oo); base_tests.append(te); base_names.append(label)

    # ---- Stack with K=15 (13 GBDT + FM_A + FM_B) ----
    full_oofs = base_oofs + [oof_A, oof_B]
    full_tests = base_tests + [test_A, test_B]
    full_names = base_names + ["FM_A_driver_dynamics", "FM_B_race_context"]
    K = len(full_names)
    P_oof = np.column_stack(full_oofs)
    P_test = np.column_stack(full_tests)
    F_oof = expand(P_oof); F_test = expand(P_test)

    # GKF CV splits for the meta itself
    gkf = GroupKFold(n_splits=N_FOLDS)
    meta_splits = list(gkf.split(np.zeros(len(y)), y, groups=grp))
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in meta_splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    auc_gkf = float(roc_auc_score(y, meta_oof))

    # Full LR meta fit on GKF OOFs → applied to GKF-averaged test preds
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    test_pred = lr_full.predict_proba(F_test)[:, 1]
    coef = lr_full.coef_.ravel()
    l1 = {full_names[i]: float(abs(coef[i]) + abs(coef[K + i]) +
                               abs(coef[2*K + i])) for i in range(K)}

    print(f"\n=== K=15 GKF leak-corrected meta ===")
    print(f"  GKF stack OOF: {auc_gkf:.5f}")
    print(f"  L1 ranking (top-15):")
    for n, v in sorted(l1.items(), key=lambda kv: -kv[1])[:15]:
        marker = "  ← FM-class" if n.startswith("FM") else ""
        print(f"    {n:<24s} L1={v:.3f}{marker}")

    # ---- Compare to PRIMARY ----
    primary_test = np.load(ART / "test_d9f_K21_swap_strat.npy"
                            )[:, 1].astype(np.float64)
    rho_primary, _ = spearmanr(test_pred, primary_test)
    print(f"\n  ρ vs PRIMARY (d9f K=21 swap) test: {rho_primary:.5f}")

    # Same comparison vs the K=22 add (d9h) for context
    if (ART / "test_d9h_K22_add_strat.npy").exists():
        d9h_test = np.load(ART / "test_d9h_K22_add_strat.npy"
                            )[:, 1].astype(np.float64)
        rho_d9h, _ = spearmanr(test_pred, d9h_test)
        print(f"  ρ vs d9h K=22 add test:           {rho_d9h:.5f}")

    # Direction-flip count vs PRIMARY (rare-class flip ratio gate)
    rare_thr = float(np.quantile(primary_test, 0.99))
    primary_pos = primary_test >= rare_thr
    new_pos = test_pred >= rare_thr
    flips_to_pos = int(np.sum(~primary_pos & new_pos))
    flips_to_neg = int(np.sum(primary_pos & ~new_pos))
    flip_ratio = (min(flips_to_pos, flips_to_neg) /
                  max(flips_to_pos, flips_to_neg) if max(flips_to_pos,
                                                         flips_to_neg) > 0
                  else 1.0)
    print(f"  rare-class flips vs PRIMARY: +→− {flips_to_neg}, "
          f"−→+ {flips_to_pos}, ratio {flip_ratio:.3f}")

    # ---- Save submission + artifacts ----
    np.save(ART / "oof_d10d_leak_corrected_meta_strat.npy",
            np.column_stack([1 - meta_oof, meta_oof]))
    np.save(ART / "test_d10d_leak_corrected_meta_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))
    sub = sample_sub.copy(); sub[TARGET] = test_pred
    sub.to_csv("submissions/submission_d10d_leak_corrected_meta.csv",
               index=False)

    # Predicted LB heuristic
    # GKF stack OOF 0.92764 corresponds to Strat 0.95052 in d10c.
    # PRIMARY Strat 0.95073 → LB 0.95031. So the "GKF→Strat→LB" anchor
    # is ~+229bp (Strat is +229 above GKF) and Strat→LB is -3.8bp.
    # If our GKF stack OOF is X, the implied Strat-equivalent is
    # X + 0.0023, and the implied LB is X + 0.0023 - 0.00038.
    pred_strat_eq = auc_gkf + 0.00229
    pred_lb_strat = pred_strat_eq - 0.00038
    if rho_primary >= 0.999:
        pred_lb = PRIMARY_LB
    elif rho_primary >= 0.995:
        pred_lb = PRIMARY_LB - 0.0001
    else:
        pred_lb = PRIMARY_LB - 0.0003
    print(f"\n  pred-Strat-equivalent OOF: {pred_strat_eq:.5f}")
    print(f"  pred-LB (from GKF→Strat→LB chain): {pred_lb_strat:.5f}")
    print(f"  pred-LB (from ρ vs PRIMARY): {pred_lb:.5f}")

    final = dict(
        gkf_stack_oof=auc_gkf,
        rho_vs_primary_test=float(rho_primary),
        flips=dict(to_neg=flips_to_neg, to_pos=flips_to_pos,
                   ratio=float(flip_ratio)),
        l1_ranking=l1,
        pred_strat_equivalent_oof=float(pred_strat_eq),
        pred_lb_from_chain=float(pred_lb_strat),
        pred_lb_from_rho=float(pred_lb),
        wall_s=time.time() - t0,
    )
    (ART / "d10d_leak_corrected_meta.json").write_text(
        json.dumps(final, indent=2))
    print(f"\n→ submissions/submission_d10d_leak_corrected_meta.csv")
    print(f"→ scripts/artifacts/d10d_leak_corrected_meta.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
