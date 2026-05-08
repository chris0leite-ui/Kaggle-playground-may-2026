"""Day-13b — Full GroupKF stack: rebuild d9f FM test predictions + 4-FM stack analysis.

Closes the gap from d13a probe: d9f_FM_A/B have GroupKF OOFs but no
matching test predictions, blocking a full GKF stack with all FM
candidates.

Steps:
  1. Re-train d9f_FM_A (D,C,S,T) and d9f_FM_B (R,Y,Rp,P) under GroupKF
     to produce both OOF and test predictions; save as
     test_d9f_FM_{A,B}_4_groupkf_strict.npy. (OOF reproduced — should
     match within seed noise; we save fresh.)

  2. Build five GKF stacks:
       BASE_18:   17 GKF bases + d9c_FM             (no d9f, no d13a)
       PLUS_d9f:  ... + d9f_FM_A + d9f_FM_B          (K=20)
       PLUS_d13a: ... + FM_A_53  + FM_B_53           (K=20, prior result)
       FULL_22:   ... + all 4 partition FMs          (K=22, MAIN TEST)
       SWAP_22:   ... + drop d9c_FM, all 4 partition (K=21)

  3. Report L1 reshuffles + GKF AUC delta per variant.

Tells us: do d13a FMs *replace* d9f FMs in L1, or do they *stack*?
Decides Move C refactor design (drop GBDT leakage-eaters, replace
with which FM combination).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

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

# d9f partitions (mirror scripts/d9f_multi_fm.py)
PART_A_d9f = ["D", "C", "S", "T"]
PART_B_d9f = ["R", "Y", "Rp", "P"]


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


def train_fm_gkf(train, test, y, splits, fields, label):
    print(f"\n--- {label} (fields={fields}, GKF) ---")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        Xtr_csr, Xte_csr = build_partition_hashes(train, test, tr, fields)
        idx_tr_full = csr_to_index_array(Xtr_csr, len(fields))
        idx_te = csr_to_index_array(Xte_csr, len(fields))
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=42 + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        print(f"  f{k}: AUC={s:.5f} wall={time.time()-t:.1f}s")
    return oof, test_avg


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


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    grp = train.groupby(["Race", "Driver", "Year", "Stint"], sort=False).ngroup().values
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.zeros(len(y)), y, groups=grp))

    # Step 1: rebuild d9f FM_A and FM_B under GKF (produce test predictions)
    print(f"\n=== Step 1: rebuild d9f FM GKF artifacts ===")
    oof_fa, test_fa = train_fm_gkf(train, test, y, splits, PART_A_d9f, "d9f_FM_A_4")
    oof_fb, test_fb = train_fm_gkf(train, test, y, splits, PART_B_d9f, "d9f_FM_B_4")
    auc_fa = float(roc_auc_score(y, oof_fa))
    auc_fb = float(roc_auc_score(y, oof_fb))
    print(f"\n  d9f_FM_A_4 GKF OOF: {auc_fa:.5f}  (Day-12 reported: 0.81964)")
    print(f"  d9f_FM_B_4 GKF OOF: {auc_fb:.5f}  (Day-12 reported: 0.88413)")

    np.save(ART / "oof_d9f_FM_A_4_groupkf_strict.npy",
            np.column_stack([1 - oof_fa, oof_fa]))
    np.save(ART / "test_d9f_FM_A_4_groupkf_strict.npy",
            np.column_stack([1 - test_fa, test_fa]))
    np.save(ART / "oof_d9f_FM_B_4_groupkf_strict.npy",
            np.column_stack([1 - oof_fb, oof_fb]))
    np.save(ART / "test_d9f_FM_B_4_groupkf_strict.npy",
            np.column_stack([1 - test_fb, test_fb]))

    # Step 2: build five GKF stacks
    print(f"\n=== Step 2: GKF stack matrix ===")
    GKF_BASE = [
        ("baseline", "oof_baseline_two_anchor_groupkf.npy", "test_baseline_two_anchor_groupkf.npy"),
        ("d2a_te", "oof_d2a_te_groupkf.npy", "test_d2a_te_groupkf.npy"),
        ("e3_hgbc", "oof_e3_hgbc_groupkf.npy", "test_e3_hgbc_groupkf.npy"),
        ("e5_optuna_lgbm", "oof_e5_optuna_lgbm_groupkf.npy", "test_e5_optuna_lgbm_groupkf.npy"),
        ("a_horizon", "oof_a_horizon_groupkf.npy", "test_a_horizon_groupkf.npy"),
        ("b_lapsuntilpit", "oof_b_lapsuntilpit_groupkf.npy", "test_b_lapsuntilpit_groupkf.npy"),
        ("f1_hgbc_deep", "oof_f1_hgbc_deep_groupkf.npy", "test_f1_hgbc_deep_groupkf.npy"),
        ("f2_hgbc_shallow", "oof_f2_hgbc_shallow_groupkf.npy", "test_f2_hgbc_shallow_groupkf.npy"),
        ("cb_year-cat", "oof_cb_year-cat_groupkf.npy", "test_cb_year-cat_groupkf.npy"),
        ("cb_lossguide", "oof_cb_lossguide_groupkf.npy", "test_cb_lossguide_groupkf.npy"),
        ("cb_slow-wide-bag", "oof_cb_slow-wide-bag_groupkf.npy", "test_cb_slow-wide-bag_groupkf.npy"),
        ("e1_cb_sub", "oof_e1_catboost_sub_groupkf.npy", "test_e1_catboost_sub_groupkf.npy"),
        ("rule_driver_compound", "oof_d6_rule_driver_compound_groupkf.npy", "test_d6_rule_driver_compound_groupkf.npy"),
        ("rule_year_race", "oof_d6_rule_year_race_groupkf.npy", "test_d6_rule_year_race_groupkf.npy"),
        ("R6_next_compound", "oof_d9_R6_next_compound_groupkf.npy", "test_d9_R6_next_compound_groupkf.npy"),
        ("R10_driver_eb", "oof_d9_R10_driver_eb_groupkf.npy", "test_d9_R10_driver_eb_groupkf.npy"),
        ("R7_prev_compound", "oof_d9_R7_prev_compound_groupkf.npy", "test_d9_R7_prev_compound_groupkf.npy"),
    ]
    base_oof, base_test, names = [], [], []
    for label, of, tf in GKF_BASE:
        base_oof.append(np.load(ART / of)[:, 1].astype(np.float64))
        base_test.append(np.load(ART / tf)[:, 1].astype(np.float64))
        names.append(label)

    # Load FM bases
    fm_d9c_oof = np.load(ART / "oof_d9c_fm_groupkf.npy")[:, 1].astype(np.float64)
    fm_d9c_test = np.load(ART / "test_d9c_fm_groupkf.npy")[:, 1].astype(np.float64)
    fm_d13a_a_oof = np.load(ART / "oof_d13a_FM_A_53_groupkf.npy")[:, 1].astype(np.float64)
    fm_d13a_a_test = np.load(ART / "test_d13a_FM_A_53_groupkf.npy")[:, 1].astype(np.float64)
    fm_d13a_b_oof = np.load(ART / "oof_d13a_FM_B_53_groupkf.npy")[:, 1].astype(np.float64)
    fm_d13a_b_test = np.load(ART / "test_d13a_FM_B_53_groupkf.npy")[:, 1].astype(np.float64)

    def stack(label, fm_oof, fm_test, fm_names):
        Xs = list(base_oof) + list(fm_oof)
        Ts = list(base_test) + list(fm_test)
        Ns = list(names) + list(fm_names)
        K = len(Ns)
        F_oof = expand(np.column_stack(Xs))
        F_test = expand(np.column_stack(Ts))
        mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
        auc = float(roc_auc_score(y, mo))
        l1 = {Ns[i]: float(abs(coef[i]) + abs(coef[K+i]) + abs(coef[2*K+i]))
              for i in range(K)}
        print(f"\n--- {label} (K={K}) ---")
        print(f"  GKF meta OOF: {auc:.5f}")
        print(f"  L1 top-12:")
        for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:12]:
            mk = ""
            if n_.startswith("FM_A_53") or n_.startswith("FM_B_53"): mk = "  ← d13a"
            elif n_ in ("d9f_FM_A", "d9f_FM_B"): mk = "  ← d9f"
            elif n_ == "d9c_FM": mk = "  ← d9c"
            elif n_.startswith("rule_"): mk = "  ← rule"
            print(f"    {n_:<24s} L1={v:.3f}{mk}")
        return auc, l1, mo, tp

    auc_base, l1_base, _, tp_base = stack(
        "BASE_18 (17 + d9c_FM)",
        [fm_d9c_oof], [fm_d9c_test], ["d9c_FM"])

    auc_d9f, l1_d9f, _, tp_d9f = stack(
        "PLUS_d9f K=20 (... + d9f_FM_A + d9f_FM_B)",
        [fm_d9c_oof, oof_fa, oof_fb],
        [fm_d9c_test, test_fa, test_fb],
        ["d9c_FM", "d9f_FM_A", "d9f_FM_B"])

    auc_d13a, l1_d13a, _, tp_d13a = stack(
        "PLUS_d13a K=20 (... + FM_A_53 + FM_B_53)",
        [fm_d9c_oof, fm_d13a_a_oof, fm_d13a_b_oof],
        [fm_d9c_test, fm_d13a_a_test, fm_d13a_b_test],
        ["d9c_FM", "FM_A_53", "FM_B_53"])

    auc_full, l1_full, mo_full, tp_full = stack(
        "FULL_22 K=22 (... + d9c + d9f A+B + d13a A+B)",
        [fm_d9c_oof, oof_fa, oof_fb, fm_d13a_a_oof, fm_d13a_b_oof],
        [fm_d9c_test, test_fa, test_fb, fm_d13a_a_test, fm_d13a_b_test],
        ["d9c_FM", "d9f_FM_A", "d9f_FM_B", "FM_A_53", "FM_B_53"])

    auc_swap, l1_swap, mo_swap, tp_swap = stack(
        "SWAP_21 K=21 (... + d9f A+B + d13a A+B; drop d9c)",
        [oof_fa, oof_fb, fm_d13a_a_oof, fm_d13a_b_oof],
        [test_fa, test_fb, fm_d13a_a_test, fm_d13a_b_test],
        ["d9f_FM_A", "d9f_FM_B", "FM_A_53", "FM_B_53"])

    print(f"\n=== Stack matrix summary ===")
    print(f"  BASE_18  (d9c only):              {auc_base:.5f}")
    print(f"  PLUS_d9f (K=20):                  {auc_d9f:.5f}  Δ vs base "
          f"{(auc_d9f-auc_base)*1e4:+.2f}bp")
    print(f"  PLUS_d13a (K=20):                 {auc_d13a:.5f}  Δ vs base "
          f"{(auc_d13a-auc_base)*1e4:+.2f}bp")
    print(f"  FULL_22  (all 4 FMs):             {auc_full:.5f}  Δ vs base "
          f"{(auc_full-auc_base)*1e4:+.2f}bp")
    print(f"  SWAP_21  (d9f+d13a, drop d9c):    {auc_swap:.5f}  Δ vs base "
          f"{(auc_swap-auc_base)*1e4:+.2f}bp")

    # Save the FULL_22 GKF stack as a HEDGE candidate (R5 final-3-day rule)
    np.save(ART / "test_d13b_FULL_22_gkf_strat.npy",
            np.column_stack([1 - tp_full, tp_full]))
    np.save(ART / "test_d13b_SWAP_21_gkf_strat.npy",
            np.column_stack([1 - tp_swap, tp_swap]))

    final = dict(
        d9f_rebuild=dict(
            FM_A_gkf_oof=auc_fa, FM_B_gkf_oof=auc_fb,
            day12_FM_A_reported=0.81964, day12_FM_B_reported=0.88413,
        ),
        stacks=dict(
            BASE_18=dict(K=18, gkf_auc=auc_base, l1=l1_base),
            PLUS_d9f=dict(K=20, gkf_auc=auc_d9f, l1=l1_d9f,
                           delta_bp=(auc_d9f-auc_base)*1e4),
            PLUS_d13a=dict(K=20, gkf_auc=auc_d13a, l1=l1_d13a,
                            delta_bp=(auc_d13a-auc_base)*1e4),
            FULL_22=dict(K=22, gkf_auc=auc_full, l1=l1_full,
                          delta_bp=(auc_full-auc_base)*1e4),
            SWAP_21=dict(K=21, gkf_auc=auc_swap, l1=l1_swap,
                          delta_bp=(auc_swap-auc_base)*1e4),
        ),
        total_wall_s=time.time() - t0,
    )
    (ART / "d13b_full_gkf_stack_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13b_full_gkf_stack_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
