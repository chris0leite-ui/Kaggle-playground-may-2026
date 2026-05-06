"""Day-13 Move B — Three FM-class diversification variants.

Builds on d9f's 2-way 4/4 sweet spot. Tests 3 alternative partition
schemes searching for the next FM-class diversification slot in the
K=22 stack pool. All FMs are PyTorch CPU (mirrors d9c/d9f/d9h/d9i).

Variants:
  V1: 5/3 multi-FM partition (asymmetric)
      FM_5fA: D, C, S, T_q5, LapNumber_q5  (5 fields, 10 pairs)
      FM_3fB: R, Y, Rp_q5                  (3 fields,  3 pairs)
  V2: 4/4 alternative split (axis-rotated from d9f)
      FM_4fC: C, T_q5, S, Rp_q5           (compound × tyre × stint × race-progress)
      FM_4fD: D, R, Y, P_q5               (entity axis: driver/race/year/position)
  V3: Augmented 6/6 alt split (different from d9i; physical/state vs identity/context)
      FM_6fE: C, T_q5, S, LapNumber_q5, Rp_q5, P_q5     (physical/state)
      FM_6fF: D, R, Y, Cd_q5, Ld_q5, Nx                 (identity + degradation context)

For each variant: standalone Strat OOF, ρ vs PRIMARY, ρ FM_A vs FM_B,
min-meta vs PRIMARY, K=22 add stack (Strat) + L1 ranking.

GroupKF gate (R5/Day-12 secondary): for each variant pair, also build
GroupKFold OOFs and compute GKF stack OOF. If GKF stack OOF < d12
GroupKF baseline (0.94776 from d12_groupkf_meta), candidate fails GKF.

PRIMARY OOF anchor: oof_d9c_Sd_K20_swap_FM_strat.npy (closest available
to d9h K=22 OOF, mirrors d9h/d9i precedent).
PRIMARY test:       test_d9h_S2_K22_add_aug12_strat.npy (current LB
0.95034 PRIMARY).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

import torch
import torch.nn as nn

# Limit threads to be polite (8-core box)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
torch.set_num_threads(4)

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SUBM = Path("submissions")
SUBM.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
N_HASH_BITS = 18
EMBED_DIM = 8
EPOCHS = 6
BATCH = 8192
LR = 0.05

# PRIMARY anchors (mirrors d9h/d9i)
PRIMARY_S = 0.95073        # d9f K=21 swap Strat OOF; closest to PRIMARY
PRIMARY_LB = 0.95034       # d9h K=22 / d9i K=21 swap LB (TIED)
PRIMARY_OOF_FILE = "oof_d9c_Sd_K20_swap_FM_strat.npy"
PRIMARY_TEST_FILE = "test_d9h_S2_K22_add_aug12_strat.npy"
GKF_BASELINE_AUC = 0.94776  # d12_groupkf_meta K=21 GKF-CV baseline
RHO_TIE = 0.999

# K=22 add pool for each variant: PRIMARY-keep (16) + 3 d9 + R14_L4 + FM_d9c + 2 NEW FMs
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
EXTRA = [
    ("R14_L4", "d9b_R14_L4"),
    ("FM_d9c", "d9c_fm"),  # base FM that K=22 stack keeps
]

# Variant partition definitions (field names → field-builder keys)
VARIANTS = {
    "V1_5_3": dict(
        A=("FM_5fA", ["D", "C", "S", "T", "Ln"]),     # 5 driver-state fields
        B=("FM_3fB", ["R", "Y", "Rp"]),                # 3 race-context
    ),
    "V2_4_4_alt": dict(
        A=("FM_4fC", ["C", "T", "S", "Rp"]),           # compound x tyre axis
        B=("FM_4fD", ["D", "R", "Y", "P"]),            # entity axis
    ),
    "V3_6_6_aug_alt": dict(
        A=("FM_6fE", ["C", "T", "S", "Ln", "Rp", "P"]),   # physical/state
        B=("FM_6fF", ["D", "R", "Y", "Cd", "Ld", "Nx"]),  # identity + degradation
    ),
}


def expand(P):
    n = len(P)
    rk = np.column_stack([rankdata(c) / n for c in P.T])
    logit = np.log(np.clip(P, 1e-9, 1 - 1e-9) /
                   (1 - np.clip(P, 1e-9, 1 - 1e-9)))
    return np.hstack([P, rk, logit])


def fit_lr_meta(F_oof, F_test, y, splits=None):
    """splits: list of (tr, va) for OOF meta. If None, StratifiedKFold."""
    if splits is None:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(skf.split(np.zeros(len(y)), y))
    meta_oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in splits:
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(F_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(F_oof[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr_full.fit(F_oof, y)
    return meta_oof, lr_full.predict_proba(F_test)[:, 1], lr_full.coef_.ravel()


def predicted_lb(auc, rho):
    base_lb = PRIMARY_LB + (auc - PRIMARY_S)
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


def _field_arrays(field, train, test, tr_idx, neighbours):
    """Return (train_str_array, test_str_array) for a field code.
    `neighbours` carries (next_train, next_test, prev_train, prev_test)."""
    nx_tr, nx_te, pv_tr, pv_te = neighbours
    if field == "D":
        return (train["Driver"].astype(str).values,
                test["Driver"].astype(str).values)
    if field == "C":
        return (train["Compound"].astype(str).values,
                test["Compound"].astype(str).values)
    if field == "R":
        return (train["Race"].astype(str).values,
                test["Race"].astype(str).values)
    if field == "Y":
        return (train["Year"].astype(str).values,
                test["Year"].astype(str).values)
    if field == "S":
        return (train["Stint"].clip(upper=5).astype(int).astype(str).values,
                test["Stint"].clip(upper=5).astype(int).astype(str).values)
    if field == "T":
        return (_quantile_bin(train["TyreLife"].values[tr_idx],
                              train["TyreLife"].values, 5).astype(str),
                _quantile_bin(train["TyreLife"].values[tr_idx],
                              test["TyreLife"].values, 5).astype(str))
    if field == "Rp":
        return (_quantile_bin(train["RaceProgress"].values[tr_idx],
                              train["RaceProgress"].values, 5).astype(str),
                _quantile_bin(train["RaceProgress"].values[tr_idx],
                              test["RaceProgress"].values, 5).astype(str))
    if field == "P":
        return (_quantile_bin(train["Position"].values[tr_idx],
                              train["Position"].values, 5).astype(str),
                _quantile_bin(train["Position"].values[tr_idx],
                              test["Position"].values, 5).astype(str))
    if field == "Ln":
        return (_quantile_bin(train["LapNumber"].values[tr_idx],
                              train["LapNumber"].values, 5).astype(str),
                _quantile_bin(train["LapNumber"].values[tr_idx],
                              test["LapNumber"].values, 5).astype(str))
    if field == "Cd":
        return (_quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                              train["Cumulative_Degradation"].values, 5).astype(str),
                _quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                              test["Cumulative_Degradation"].values, 5).astype(str))
    if field == "Ld":
        return (_quantile_bin(train["LapTime_Delta"].values[tr_idx],
                              train["LapTime_Delta"].values, 5).astype(str),
                _quantile_bin(train["LapTime_Delta"].values[tr_idx],
                              test["LapTime_Delta"].values, 5).astype(str))
    if field == "Nx":
        return (nx_tr.astype(str), nx_te.astype(str))
    if field == "Pv":
        return (pv_tr.astype(str), pv_te.astype(str))
    raise ValueError(f"Unknown field code: {field}")


def build_partition_hashes(train, test, tr_idx, fields, neighbours):
    n_tr, n_te = len(train), len(test)
    arrs_tr, arrs_te = [], []
    for f in fields:
        a_tr, a_te = _field_arrays(f, train, test, tr_idx, neighbours)
        arrs_tr.append(a_tr); arrs_te.append(a_te)

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


def train_partition_fm(train, test, y, splits, fields, label, neighbours,
                       cv_label="strat"):
    print(f"\n--- {label} (fields={fields}, n_active={len(fields)}, cv={cv_label}) ---")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    fold_walls = []
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        Xtr_csr, Xte_csr = build_partition_hashes(
            train, test, tr, fields, neighbours)
        idx_tr_full = csr_to_index_array(Xtr_csr, len(fields))
        idx_te = csr_to_index_array(Xte_csr, len(fields))
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=42 + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        fw = time.time() - t_fold
        fold_walls.append(fw)
        print(f"  f{k}: AUC={s:.5f}  wall={fw:.1f}s")
    auc = float(roc_auc_score(y, oof))
    print(f"  {cv_label} std OOF: {auc:.5f}  wall={sum(fold_walls):.1f}s")
    return oof, test_avg, auc, sum(fold_walls)


def k22_stack_eval(label, fm_a_oof, fm_a_test, fm_a_name,
                   fm_b_oof, fm_b_test, fm_b_name,
                   y, primary_test, base_oof, base_test, base_names,
                   d9_oof, d9_test, d9_names,
                   r14_oof, r14_test, fm_d9c_oof, fm_d9c_test,
                   results, splits=None, cv_label="strat"):
    """K=22 add stack: POOL_KEEP + d9-3 + R14_L4 + FM_d9c + (2 new FMs).
    Replaces the d9f-FM_A/B with the variant's (FM_A_var, FM_B_var)."""
    Xs = list(base_oof) + list(d9_oof) + [r14_oof, fm_d9c_oof,
                                            fm_a_oof, fm_b_oof]
    Ts = list(base_test) + list(d9_test) + [r14_test, fm_d9c_test,
                                              fm_a_test, fm_b_test]
    Ns = list(base_names) + list(d9_names) + ["R14_L4", "FM_d9c",
                                                fm_a_name, fm_b_name]
    K = len(Ns)
    P_oof = np.column_stack(Xs); P_test = np.column_stack(Ts)
    F_oof = expand(P_oof); F_test = expand(P_test)
    mo, tp, coef = fit_lr_meta(F_oof, F_test, y, splits=splits)
    auc = float(roc_auc_score(y, mo))
    rho, _ = spearmanr(tp, primary_test)
    pred_lb = predicted_lb(auc, rho) if cv_label == "strat" else None
    delta = (auc - PRIMARY_S) * 1e4
    l1 = {Ns[i]: float(abs(coef[i]) + abs(coef[K + i]) + abs(coef[2*K + i]))
          for i in range(K)}
    print(f"\n=== {label} ({cv_label}, K={K}) ===")
    print(f"  {cv_label} OOF: {auc:.5f}  Δ PRIMARY {delta:+.2f}bp  "
          f"ρ vs PRIMARY: {rho:.5f}")
    if pred_lb is not None:
        print(f"  pred-LB: {pred_lb:.5f}  "
              f"(Δ {(pred_lb - PRIMARY_LB)*1e4:+.2f}bp)")
    print(f"  L1 top-15:")
    for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:15]:
        marker = ""
        if n_.startswith("FM_") and n_ not in ("FM_d9c",):
            marker = "  ← d13-FM"
        elif n_ == "FM_d9c":
            marker = "  ← FM-class"
        elif n_.startswith("R") and "_" in n_:
            marker = "  ← d9-base"
        elif n_.startswith("rule_"):
            marker = "  ← existing rule"
        print(f"    {n_:<24s} L1={v:.3f}{marker}")
    info = dict(K=K, auc=auc, delta_primary_bp=delta,
                rho_vs_primary=float(rho),
                pred_lb=float(pred_lb) if pred_lb is not None else None,
                l1_ranking=l1)
    results[f"{label}_{cv_label}"] = info
    return mo, tp


def min_meta_gate(primary_oof_anchor, primary_test, fm_oof, fm_test, y,
                  splits=None):
    """Min-meta with PRIMARY anchor. Returns (auc_min, delta_bp)."""
    F_min = expand(np.column_stack([primary_oof_anchor, fm_oof]))
    F_min_t = expand(np.column_stack([primary_test, fm_test]))
    mo, _, _ = fit_lr_meta(F_min, F_min_t, y, splits=splits)
    auc_min = float(roc_auc_score(y, mo))
    return auc_min, (auc_min - PRIMARY_S) * 1e4


def main():
    t0 = time.time()
    smoke = "--smoke" in sys.argv

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof_anchor = np.load(ART / PRIMARY_OOF_FILE)[:, 1].astype(np.float64)
    primary_test = np.load(ART / PRIMARY_TEST_FILE)[:, 1].astype(np.float64)

    print(f"Train n={len(y)}, Test n={len(test)}")
    print(f"PRIMARY anchor: OOF {PRIMARY_OOF_FILE} ({PRIMARY_S:.5f}), "
          f"test {PRIMARY_TEST_FILE} (LB {PRIMARY_LB})")
    print(f"GKF baseline: {GKF_BASELINE_AUC:.5f} (d12_groupkf_meta K=21)")

    # Splits
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_strat = list(skf.split(np.zeros(len(y)), y))
    grp = train.groupby(["Race", "Driver", "Year", "Stint"],
                        sort=False).ngroup().values
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_gkf = list(gkf.split(np.zeros(len(y)), y, groups=grp))

    # Compute next/prev compounds (needed by V3 Nx/Pv)
    print("\nComputing next/prev compounds on train+test ...")
    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    nx, pv = _compute_neighbour_compounds(full)
    next_train = nx[full["__src"].values == "tr"]
    next_test = nx[full["__src"].values == "te"]
    prev_train = pv[full["__src"].values == "tr"]
    prev_test = pv[full["__src"].values == "te"]
    neighbours = (next_train, next_test, prev_train, prev_test)
    print(f"  next test coverage: {(next_test != 'UNK').mean():.3f}, "
          f"prev: {(prev_test != 'UNK').mean():.3f}")

    # Load PRIMARY pool
    print("\nLoading pool components ...")
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
    r14_oof = np.load(ART / "oof_d9b_R14_L4_strat.npy")[:, 1].astype(np.float64)
    r14_test = np.load(ART / "test_d9b_R14_L4_strat.npy")[:, 1].astype(np.float64)
    fm_d9c_oof = np.load(ART / "oof_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    fm_d9c_test = np.load(ART / "test_d9c_fm_strat.npy")[:, 1].astype(np.float64)

    # GKF pool: load _groupkf where available, fall back to strat
    base_oof_g, base_test_g = [], []
    g_skipped = []
    for label, fname in POOL_KEEP:
        p = ART / f"oof_{fname}_groupkf.npy"
        pt = ART / f"test_{fname}_groupkf.npy"
        if p.exists() and pt.exists():
            base_oof_g.append(np.load(p)[:, 1].astype(np.float64))
            base_test_g.append(np.load(pt)[:, 1].astype(np.float64))
        else:
            base_oof_g.append(np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64))
            base_test_g.append(np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64))
            g_skipped.append(label)
    d9_oof_g, d9_test_g = [], []
    for label, fname in TOP_3_D9:
        p = ART / f"oof_{fname}_groupkf.npy"
        pt = ART / f"test_{fname}_groupkf.npy"
        if p.exists() and pt.exists():
            d9_oof_g.append(np.load(p)[:, 1].astype(np.float64))
            d9_test_g.append(np.load(pt)[:, 1].astype(np.float64))
        else:
            d9_oof_g.append(d9_oof[len(d9_oof_g)])
            d9_test_g.append(d9_test[len(d9_test_g)])
            g_skipped.append(label)
    r14_oof_g = np.load(ART / "oof_d9b_R14_L4_groupkf.npy")[:, 1].astype(np.float64)
    r14_test_g = np.load(ART / "test_d9b_R14_L4_groupkf.npy")[:, 1].astype(np.float64)
    fm_d9c_oof_g = np.load(ART / "oof_d9c_fm_groupkf.npy")[:, 1].astype(np.float64)
    fm_d9c_test_g = np.load(ART / "test_d9c_fm_groupkf.npy")[:, 1].astype(np.float64)
    print(f"  GKF pool components: {len(POOL_KEEP) + len(TOP_3_D9) + 2}; "
          f"strat-fallback for: {g_skipped}")

    if smoke:
        print("\n*** SMOKE MODE: 1-fold each variant, full data, "
              "report std-only AUC ***")
        smoke_results = {}
        for vkey, vdef in VARIANTS.items():
            for side in ("A", "B"):
                fm_label, fields = vdef[side]
                tr0, va0 = splits_strat[0]
                t_fold = time.time()
                Xtr_csr, Xte_csr = build_partition_hashes(
                    train, test, tr0, fields, neighbours)
                idx_tr_full = csr_to_index_array(Xtr_csr, len(fields))
                idx_te = csr_to_index_array(Xte_csr, len(fields))
                p_va, _ = fit_fm_one_fold(idx_tr_full, idx_te, y, tr0, va0,
                                          seed=42)
                auc = float(roc_auc_score(y[va0], p_va))
                wall = time.time() - t_fold
                proj_5fold = wall * 5
                smoke_results[f"{vkey}/{fm_label}"] = dict(
                    fold0_auc=auc, fold0_wall=wall, proj_5fold=proj_5fold)
                print(f"  smoke {vkey}/{fm_label}: f0 AUC={auc:.5f}  "
                      f"wall={wall:.1f}s  proj 5-fold={proj_5fold:.0f}s")
        (ART / "d13_move_b_smoke_results.json").write_text(
            json.dumps(smoke_results, indent=2))
        print(f"\n→ smoke wrote scripts/artifacts/d13_move_b_smoke_results.json"
              f" (wall {time.time()-t0:.0f}s)")
        return

    # Full run
    results = {"variants": {}, "k22_stacks": {}}

    for vkey, vdef in VARIANTS.items():
        print(f"\n========== Variant {vkey} ==========")
        a_label, a_fields = vdef["A"]
        b_label, b_fields = vdef["B"]

        # Strat training
        oof_a_s, test_a_s, auc_a_s, wall_a = train_partition_fm(
            train, test, y, splits_strat, a_fields,
            f"{vkey}/{a_label}", neighbours, cv_label="strat")
        oof_b_s, test_b_s, auc_b_s, wall_b = train_partition_fm(
            train, test, y, splits_strat, b_fields,
            f"{vkey}/{b_label}", neighbours, cv_label="strat")

        # Standalone diagnostics (Strat)
        rho_a_prim, _ = spearmanr(test_a_s, primary_test)
        rho_b_prim, _ = spearmanr(test_b_s, primary_test)
        rho_a_b, _ = spearmanr(test_a_s, test_b_s)
        mm_a, dA = min_meta_gate(primary_oof_anchor, primary_test,
                                 oof_a_s, test_a_s, y, splits=splits_strat)
        mm_b, dB = min_meta_gate(primary_oof_anchor, primary_test,
                                 oof_b_s, test_b_s, y, splits=splits_strat)
        print(f"\n  -- Standalone Strat ({vkey}) --")
        print(f"  {a_label}: std OOF {auc_a_s:.5f}  ρ vs PRIMARY {rho_a_prim:.5f}  "
              f"min-meta Δ {dA:+.2f}bp  {'PASS' if mm_a >= PRIMARY_S else 'FAIL'}")
        print(f"  {b_label}: std OOF {auc_b_s:.5f}  ρ vs PRIMARY {rho_b_prim:.5f}  "
              f"min-meta Δ {dB:+.2f}bp  {'PASS' if mm_b >= PRIMARY_S else 'FAIL'}")
        print(f"  ρ {a_label} vs {b_label}: {rho_a_b:.5f}  "
              f"(d9f sweet spot 0.41; d9g over-frag 0.65+; d9i aug-2way 0.66)")

        # Save Strat bases
        np.save(ART / f"oof_d13_{a_label.lower()}_strat.npy",
                np.column_stack([1 - oof_a_s, oof_a_s]))
        np.save(ART / f"test_d13_{a_label.lower()}_strat.npy",
                np.column_stack([1 - test_a_s, test_a_s]))
        np.save(ART / f"oof_d13_{b_label.lower()}_strat.npy",
                np.column_stack([1 - oof_b_s, oof_b_s]))
        np.save(ART / f"test_d13_{b_label.lower()}_strat.npy",
                np.column_stack([1 - test_b_s, test_b_s]))

        # K=22 add stack — Strat
        mo_strat, tp_strat = k22_stack_eval(
            f"{vkey}_K22_add", oof_a_s, test_a_s, a_label,
            oof_b_s, test_b_s, b_label,
            y, primary_test, base_oof, base_test, base_names,
            d9_oof, d9_test, d9_names, r14_oof, r14_test,
            fm_d9c_oof, fm_d9c_test, results["k22_stacks"],
            splits=splits_strat, cv_label="strat")

        # GroupKF training of new FMs
        oof_a_g, test_a_g, auc_a_g, _ = train_partition_fm(
            train, test, y, splits_gkf, a_fields,
            f"{vkey}/{a_label}", neighbours, cv_label="groupkf")
        oof_b_g, test_b_g, auc_b_g, _ = train_partition_fm(
            train, test, y, splits_gkf, b_fields,
            f"{vkey}/{b_label}", neighbours, cv_label="groupkf")
        np.save(ART / f"oof_d13_{a_label.lower()}_groupkf.npy",
                np.column_stack([1 - oof_a_g, oof_a_g]))
        np.save(ART / f"test_d13_{a_label.lower()}_groupkf.npy",
                np.column_stack([1 - test_a_g, test_a_g]))
        np.save(ART / f"oof_d13_{b_label.lower()}_groupkf.npy",
                np.column_stack([1 - oof_b_g, oof_b_g]))
        np.save(ART / f"test_d13_{b_label.lower()}_groupkf.npy",
                np.column_stack([1 - test_b_g, test_b_g]))

        # K=22 add stack — GroupKF (Track B: pool & meta both GKF)
        mo_gkf, tp_gkf = k22_stack_eval(
            f"{vkey}_K22_add", oof_a_g, test_a_g, a_label,
            oof_b_g, test_b_g, b_label,
            y, primary_test, base_oof_g, base_test_g, base_names,
            d9_oof_g, d9_test_g, d9_names, r14_oof_g, r14_test_g,
            fm_d9c_oof_g, fm_d9c_test_g, results["k22_stacks"],
            splits=splits_gkf, cv_label="groupkf")
        gkf_auc = results["k22_stacks"][f"{vkey}_K22_add_groupkf"]["auc"]
        gkf_pass = gkf_auc >= GKF_BASELINE_AUC
        results["k22_stacks"][f"{vkey}_K22_add_groupkf"]["gkf_pass"] = bool(gkf_pass)
        results["k22_stacks"][f"{vkey}_K22_add_groupkf"]["gkf_baseline"] = GKF_BASELINE_AUC
        print(f"  GKF gate: AUC {gkf_auc:.5f} vs baseline {GKF_BASELINE_AUC:.5f} "
              f"→ {'PASS' if gkf_pass else 'FAIL'}")

        # Verdict
        strat_info = results["k22_stacks"][f"{vkey}_K22_add_strat"]
        strat_pass = strat_info["delta_primary_bp"] >= 0  # OOF lift over PRIMARY
        verdict = ("PASS_BOTH_GATES" if (strat_pass and gkf_pass)
                   else "PASS_STRAT_ONLY" if strat_pass
                   else "PASS_GKF_ONLY" if gkf_pass
                   else "FAIL")
        print(f"\n  ===== VERDICT {vkey}: {verdict} =====")
        print(f"  Strat OOF Δ {strat_info['delta_primary_bp']:+.2f}bp, "
              f"pred-LB Δ {(strat_info['pred_lb']-PRIMARY_LB)*1e4:+.2f}bp; "
              f"GKF OOF Δ {(gkf_auc-GKF_BASELINE_AUC)*1e4:+.2f}bp")

        results["variants"][vkey] = dict(
            partition_a=dict(label=a_label, fields=a_fields,
                             std_oof_strat=auc_a_s, std_oof_groupkf=auc_a_g,
                             rho_vs_primary=float(rho_a_prim),
                             min_meta_strat=mm_a,
                             min_meta_delta_bp=float(dA),
                             min_meta_pass=bool(mm_a >= PRIMARY_S)),
            partition_b=dict(label=b_label, fields=b_fields,
                             std_oof_strat=auc_b_s, std_oof_groupkf=auc_b_g,
                             rho_vs_primary=float(rho_b_prim),
                             min_meta_strat=mm_b,
                             min_meta_delta_bp=float(dB),
                             min_meta_pass=bool(mm_b >= PRIMARY_S)),
            rho_a_b=float(rho_a_b),
            verdict=verdict,
            wall_s=wall_a + wall_b,
        )

        # Save Strat-meta as candidate submission CSV (we keep all 3; pick best later)
        sub = sample_sub.copy(); sub[TARGET] = tp_strat
        sub.to_csv(SUBM / f"submission_d13_{vkey}_K22_add.csv", index=False)
        print(f"  → wrote submissions/submission_d13_{vkey}_K22_add.csv "
              f"(HELD pending PI approval per Rule 1)")

    # Pick best variant by Strat OOF (then GKF gate as tiebreaker)
    best = None
    for vkey in VARIANTS:
        info = results["k22_stacks"][f"{vkey}_K22_add_strat"]
        gkf_info = results["k22_stacks"][f"{vkey}_K22_add_groupkf"]
        score = info["delta_primary_bp"]
        if gkf_info["gkf_pass"]:
            score += 0.05  # tiebreak nudge
        if best is None or score > best[1]:
            best = (vkey, score, info, gkf_info)
    print(f"\n========== BEST VARIANT: {best[0]} (score {best[1]:.2f}) ==========")
    print(f"  Strat Δ PRIMARY: {best[2]['delta_primary_bp']:+.2f}bp")
    print(f"  Strat pred-LB: {best[2]['pred_lb']:.5f} "
          f"(Δ {(best[2]['pred_lb']-PRIMARY_LB)*1e4:+.2f}bp)")
    print(f"  GKF Δ baseline: {(best[3]['auc']-GKF_BASELINE_AUC)*1e4:+.2f}bp  "
          f"({'PASS' if best[3]['gkf_pass'] else 'FAIL'})")

    final = dict(
        results=results,
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB,
                     oof_anchor=PRIMARY_OOF_FILE, test_file=PRIMARY_TEST_FILE),
        gkf_baseline=GKF_BASELINE_AUC,
        params=dict(embed_dim=EMBED_DIM, epochs=EPOCHS, batch=BATCH, lr=LR),
        best_variant=best[0],
        gkf_skipped_strat_fallback=g_skipped,
        total_wall_s=time.time() - t0,
    )
    (ART / "d13_move_b_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13_move_b_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
