"""Day-13a — 5/3 multi-FM partition (Move B variant 1).

Tests a 5-driver-state / 3-race-context partition as an alternative
shape to d9f's 4/4 (won +2bp LB) and d9i's 6/6 aug (won +3bp LB).
Different operating point on the partition-shape axis: more capacity
in driver-dynamics FM, less in race-context FM.

Partition (each FM has its own pairwise-interaction lattice):
  FM_A_53 "driver+degradation":  D, C, S, T_q5, Cd_q5     (5 fields,  10 pairs)
  FM_B_53 "race-context":        R, Y, Rp_q5              (3 fields,   3 pairs)

  P (Position) and Ld (LapTime_Delta) excluded — both are
  driver-state outcomes that overlap T/Cd; including them in d9i's
  6/6 may have absorbed signal redundantly. 5/3 isolates the
  driver-degradation lattice from the macro race-context lattice.

Three stacks vs PRIMARY (d9i_S1_K21_swap_aug2way, OOF 0.95071,
LB 0.95034):
  S1 K=21 swap:    drop d9c FM, add FM_A_53 + FM_B_53          (mirrors d9f shape)
  S2 K=23 add:     keep d9f FM_A+B, ADD FM_A_53 + FM_B_53      (mirrors d9i shape)
  S3 K=24 add+:    keep d9c FM, d9f FM_A+B, ADD FM_A_53 + FM_B_53

Min-meta gate vs PRIMARY: each FM_A_53 / FM_B_53 alone with PRIMARY,
must Δ ≥ 0bp on Strat OOF. (Day-12: GroupKF as secondary gate; held
to Strat-only for first probe — extend to GKF if Strat passes.)
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

# PRIMARY = d9i S1 K21 swap aug2way (TIED LB 0.95034 with d9h K22)
PRIMARY_S = 0.95071
PRIMARY_LB = 0.95034
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

PART_A_53 = ["D", "C", "S", "T", "Cd"]
PART_B_53 = ["R", "Y", "Rp"]


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
        elif f == "Cd":
            arrs_tr.append(_quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                                          train["Cumulative_Degradation"].values, 5).astype(str))
            arrs_te.append(_quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                                          test["Cumulative_Degradation"].values, 5).astype(str))
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


def train_partition_fm(train, test, y, splits, fields, label):
    print(f"\n--- {label} (fields={fields}, n_active={len(fields)}) ---")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
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
    return oof, test_avg


def stack_eval(label, extra_oof, extra_test, extra_names, y, primary_test,
               base_oof, base_test, base_names, d9_oof, d9_test, d9_names,
               results, include_d9c_FM=False, fm_d9c_oof=None, fm_d9c_test=None,
               include_d9f_AB=False, fma_oof=None, fma_test=None,
               fmb_oof=None, fmb_test=None):
    Xs = list(base_oof) + list(d9_oof)
    Ts = list(base_test) + list(d9_test)
    Ns = list(base_names) + list(d9_names)
    if include_d9c_FM:
        Xs.append(fm_d9c_oof); Ts.append(fm_d9c_test); Ns.append("FM_d9c")
    if include_d9f_AB:
        Xs.extend([fma_oof, fmb_oof])
        Ts.extend([fma_test, fmb_test])
        Ns.extend(["FM_A_d9f", "FM_B_d9f"])
    for o, t, n in zip(extra_oof, extra_test, extra_names):
        Xs.append(o); Ts.append(t); Ns.append(n)
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
        if n_.startswith("FM_A_53") or n_.startswith("FM_B_53"): marker = "  ← d13a-FM"
        elif n_.startswith("FM_A_d9f") or n_.startswith("FM_B_d9f"): marker = "  ← d9f-FM"
        elif n_ == "FM_d9c": marker = "  ← d9c-FM"
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
    primary_test = np.load(ART / "test_d9i_S1_K21_swap_aug2way_strat.npy"
                           )[:, 1].astype(np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Build the d9i S1 OOF predictions for delta-vs-primary baseline.
    # Use it for min-meta gate (need OOF, not just test).
    # (LR meta OOF for d9i not saved separately — we approximate by
    # using the strongest available stand-in: m5q OOF as fallback only
    # for the min-meta sanity check; if absent, skip min-meta in OOF
    # space and rely on K-stack delta vs PRIMARY_S below.)
    primary_oof_path = ART / "oof_d9i_S1_K21_swap_aug2way_strat.npy"
    has_primary_oof = primary_oof_path.exists()
    if has_primary_oof:
        primary_oof = np.load(primary_oof_path)[:, 1].astype(np.float64)
    else:
        primary_oof = None
        print("[note] d9i S1 OOF not saved; min-meta uses primary_test ρ only")

    # Train both 5/3 partition FMs
    print(f"\n=== d13a 5/3 multi-FM ===\n")
    oof_a, test_a = train_partition_fm(train, test, y, splits,
                                       PART_A_53, "FM_A_53_driver_deg")
    oof_b, test_b = train_partition_fm(train, test, y, splits,
                                       PART_B_53, "FM_B_53_race")
    auc_a = float(roc_auc_score(y, oof_a))
    auc_b = float(roc_auc_score(y, oof_b))
    rho_ap, _ = spearmanr(test_a, primary_test)
    rho_bp, _ = spearmanr(test_b, primary_test)
    rho_ab, _ = spearmanr(test_a, test_b)
    print(f"\n  FM_A_53 std OOF: {auc_a:.5f}  ρ vs PRIMARY {rho_ap:.5f}")
    print(f"  FM_B_53 std OOF: {auc_b:.5f}  ρ vs PRIMARY {rho_bp:.5f}")
    print(f"  ρ FM_A_53 vs FM_B_53: {rho_ab:.5f}  (low = orthogonal partitions)")

    # Save partition FM bases
    np.save(ART / "oof_d13a_FM_A_53_strat.npy",
            np.column_stack([1 - oof_a, oof_a]))
    np.save(ART / "test_d13a_FM_A_53_strat.npy",
            np.column_stack([1 - test_a, test_a]))
    np.save(ART / "oof_d13a_FM_B_53_strat.npy",
            np.column_stack([1 - oof_b, oof_b]))
    np.save(ART / "test_d13a_FM_B_53_strat.npy",
            np.column_stack([1 - test_b, test_b]))

    # Min-meta gate: each FM alone with PRIMARY (if PRIMARY OOF available)
    mm_results = {}
    if has_primary_oof:
        for name, oof_x, test_x in [("FM_A_53", oof_a, test_a),
                                     ("FM_B_53", oof_b, test_b)]:
            F_min = expand(np.column_stack([primary_oof, oof_x]))
            F_min_t = expand(np.column_stack([primary_test, test_x]))
            mo_x, _, _ = fit_lr_meta(F_min, F_min_t, y)
            mm = float(roc_auc_score(y, mo_x))
            d = (mm - PRIMARY_S) * 1e4
            verdict = "PASS ✓" if mm >= PRIMARY_S else "FAIL ✗"
            print(f"  {name} min-meta vs PRIMARY: {mm:.5f}  Δ {d:+.2f}bp  {verdict}")
            mm_results[name] = dict(mm_oof=mm, delta_bp=d, pass_=mm >= PRIMARY_S)
    else:
        print("  [min-meta gate skipped — no PRIMARY OOF]")

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
    fm_d9c_oof = np.load(ART / "oof_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    fm_d9c_test = np.load(ART / "test_d9c_fm_strat.npy")[:, 1].astype(np.float64)
    fma_d9f_oof = np.load(ART / "oof_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    fma_d9f_test = np.load(ART / "test_d9f_FM_A_strat.npy")[:, 1].astype(np.float64)
    fmb_d9f_oof = np.load(ART / "oof_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)
    fmb_d9f_test = np.load(ART / "test_d9f_FM_B_strat.npy")[:, 1].astype(np.float64)

    results = dict(
        FM_A_53=dict(std_oof=auc_a, rho_vs_primary=float(rho_ap)),
        FM_B_53=dict(std_oof=auc_b, rho_vs_primary=float(rho_bp)),
        rho_A_B=float(rho_ab),
        min_meta=mm_results,
    )

    # S1: K=21 swap (drop d9c FM, add FM_A_53 + FM_B_53)
    mo_s1, tp_s1 = stack_eval(
        "S1_K21_swap_partA53_partB53",
        [oof_a, oof_b], [test_a, test_b],
        ["FM_A_53", "FM_B_53"],
        y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results)

    # S2: K=23 add (keep d9f FM_A+B, ADD FM_A_53 + FM_B_53)
    mo_s2, tp_s2 = stack_eval(
        "S2_K23_add_53_to_d9f",
        [oof_a, oof_b], [test_a, test_b],
        ["FM_A_53", "FM_B_53"],
        y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results,
        include_d9f_AB=True, fma_oof=fma_d9f_oof, fma_test=fma_d9f_test,
        fmb_oof=fmb_d9f_oof, fmb_test=fmb_d9f_test)

    # S3: K=24 add (keep d9c FM and d9f FM_A+B, ADD FM_A_53 + FM_B_53)
    mo_s3, tp_s3 = stack_eval(
        "S3_K24_add_all_FMs",
        [oof_a, oof_b], [test_a, test_b],
        ["FM_A_53", "FM_B_53"],
        y, primary_test, base_oof, base_test, base_names,
        d9_oof, d9_test, d9_names, results,
        include_d9c_FM=True, fm_d9c_oof=fm_d9c_oof, fm_d9c_test=fm_d9c_test,
        include_d9f_AB=True, fma_oof=fma_d9f_oof, fma_test=fma_d9f_test,
        fmb_oof=fmb_d9f_oof, fmb_test=fmb_d9f_test)

    # Save submission CSVs (held; don't submit until PI approves)
    for name, mo, tp in [("S1_K21_swap_partA53_partB53", mo_s1, tp_s1),
                         ("S2_K23_add_53_to_d9f", mo_s2, tp_s2),
                         ("S3_K24_add_all_FMs", mo_s3, tp_s3)]:
        np.save(ART / f"test_d13a_{name}_strat.npy",
                np.column_stack([1 - tp, tp]))
        sub = sample_sub.copy(); sub[TARGET] = tp
        sub.to_csv(f"submissions/submission_d13a_{name}.csv", index=False)
        print(f"→ wrote submissions/submission_d13a_{name}.csv")

    final = dict(
        results=results,
        partitions=dict(A=PART_A_53, B=PART_B_53),
        params=dict(embed_dim=EMBED_DIM, epochs=EPOCHS, batch=BATCH, lr=LR),
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB,
                     ref_artifact="test_d9i_S1_K21_swap_aug2way_strat.npy"),
        total_wall_s=time.time() - t0,
    )
    (ART / "d13a_multi_fm_53_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13a_multi_fm_53_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
