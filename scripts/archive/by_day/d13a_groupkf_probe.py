"""Day-13a GroupKF probe — d13a 5/3 multi-FM under GroupKF(Race,Driver,Year,Stint).

Day-12 secondary-gate test (per CLAUDE.md Rule 5 of Critical Operating
Rules in HANDOVER.md). Each new candidate must either:
  - Pass Strat AND not regress GroupKF, OR
  - Pass GroupKF if it is leakage-robust class (FM/rule/sparse-LR)

d13a 5/3 was TIE_EXPECTED on Strat (S1 +0.09bp, S2 +0.19bp, S3 +0.20bp
all ρ ≈ 0.9997-0.9998). This probe asks: are FM_A_53 and FM_B_53
leakage-robust enough to qualify for Move C pool refactor (drop GBDT
leakage-eaters, replace with FM/rule)?

Day-12 calibration:
  d9c_FM (8-field unified):    Δ -9.1bp   (leakage-robust ✓)
  d9f_FM_A (D,C,S,T):          Δ -54.1bp  (leakage-robust ✓)
  d9f_FM_B (R,Y,Rp,P):         Δ -2.5bp   (leakage-robust ✓)
  rule_driver_compound:        Δ -40.2bp  (leakage-robust ✓)
  e3_hgbc / e5 / cb-bag:       Δ -209 to -247bp (leakage eaters ✗)

Threshold (rough): |Δ| ≤ 100bp = leakage-robust; >150bp = eater.

Prediction (this script writes a pre-registration, then trains):
  FM_A_53 (D,C,S,T,Cd):  expect Δ -20 to -60bp (Cd adds some risk
                          but should be leakage-robust)
  FM_B_53 (R,Y,Rp):      expect Δ -2 to -15bp  (3 fields, all macro
                          context; cleanest expected)

If both pass, slot into a K=21 GroupKF-meta stack alongside d9c_FM,
d9f_FM_A, d9f_FM_B and the rules — predict GKF OOF lift over d12's
0.94776 baseline.
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

PART_A_53 = ["D", "C", "S", "T", "Cd"]
PART_B_53 = ["R", "Y", "Rp"]

# Day-12 K=21 GroupKF baseline (from d12_groupkf_rebuild_partial)
D12_GKF_META_BASELINE = 0.94776


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


def train_partition_fm_gkf(train, test, y, splits, fields, label):
    print(f"\n--- {label} (fields={fields}, n_active={len(fields)}, GroupKF) ---")
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
        print(f"  f{k}: AUC={s:.5f}  wall={time.time()-t_fold:.1f}s")
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
    print(f"GroupKF key (Race, Driver, Year, Stint), n_groups={len(np.unique(grp))}")
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.zeros(len(y)), y, groups=grp))

    # Train both partition FMs under GroupKF
    oof_a, test_a = train_partition_fm_gkf(train, test, y, splits,
                                            PART_A_53, "FM_A_53")
    oof_b, test_b = train_partition_fm_gkf(train, test, y, splits,
                                            PART_B_53, "FM_B_53")
    auc_a_gkf = float(roc_auc_score(y, oof_a))
    auc_b_gkf = float(roc_auc_score(y, oof_b))

    # Compare to Strat (already trained)
    s_a = float(roc_auc_score(y, np.load(ART / "oof_d13a_FM_A_53_strat.npy")[:, 1]))
    s_b = float(roc_auc_score(y, np.load(ART / "oof_d13a_FM_B_53_strat.npy")[:, 1]))
    d_a_bp = (auc_a_gkf - s_a) * 1e4
    d_b_bp = (auc_b_gkf - s_b) * 1e4

    print("\n=== d13a leakage profile ===")
    print(f"  FM_A_53 (D,C,S,T,Cd):  Strat {s_a:.5f}  GKF {auc_a_gkf:.5f}  "
          f"Δ {d_a_bp:+7.1f}bp")
    print(f"  FM_B_53 (R,Y,Rp):      Strat {s_b:.5f}  GKF {auc_b_gkf:.5f}  "
          f"Δ {d_b_bp:+7.1f}bp")
    print("\n  Reference (Day-12):")
    print("    d9c_FM        Δ  -9.1bp   (leakage-robust ✓)")
    print("    d9f_FM_A      Δ -54.1bp   (leakage-robust ✓)")
    print("    d9f_FM_B      Δ  -2.5bp   (leakage-robust ✓)")
    print("    e3_hgbc       Δ -209.1bp  (leakage eater ✗)")
    print("    e5_optuna     Δ -215.1bp  (leakage eater ✗)")
    print("    cb_swb        Δ -246.9bp  (leakage eater ✗)")

    verdict_a = "ROBUST ✓" if d_a_bp >= -100 else ("MIXED" if d_a_bp >= -150 else "EATER ✗")
    verdict_b = "ROBUST ✓" if d_b_bp >= -100 else ("MIXED" if d_b_bp >= -150 else "EATER ✗")
    print(f"\n  FM_A_53 verdict: {verdict_a}")
    print(f"  FM_B_53 verdict: {verdict_b}")

    # Save GroupKF artifacts
    np.save(ART / "oof_d13a_FM_A_53_groupkf.npy",
            np.column_stack([1 - oof_a, oof_a]))
    np.save(ART / "test_d13a_FM_A_53_groupkf.npy",
            np.column_stack([1 - test_a, test_a]))
    np.save(ART / "oof_d13a_FM_B_53_groupkf.npy",
            np.column_stack([1 - oof_b, oof_b]))
    np.save(ART / "test_d13a_FM_B_53_groupkf.npy",
            np.column_stack([1 - test_b, test_b]))

    # GroupKF stack: load existing GKF leakage-robust pool + add d13a FMs
    # Day-12 K=21 GKF baseline = 0.94776 (with d12 pool composition)
    # We test: does adding d13a FMs improve over the d12 baseline?
    GKF_POOL = [
        # (label, oof_file, test_file)
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
        ("d9c_FM", "oof_d9c_fm_groupkf.npy", "test_d9c_fm_groupkf.npy"),
        ("d9f_FM_A", "oof_d9f_FM_A_4_groupkf_strict.npy", "test_d9f_FM_A_4_groupkf_strict.npy"),
        ("d9f_FM_B", "oof_d9f_FM_B_4_groupkf_strict.npy", "test_d9f_FM_B_4_groupkf_strict.npy"),
    ]

    base_oof, base_test, names = [], [], []
    for label, of, tf in GKF_POOL:
        if (ART / of).exists() and (ART / tf).exists():
            base_oof.append(np.load(ART / of)[:, 1].astype(np.float64))
            base_test.append(np.load(ART / tf)[:, 1].astype(np.float64))
            names.append(label)
        else:
            print(f"  [skip] {label} GKF artifact missing")

    def gkf_stack(label, extra_oof, extra_test, extra_names):
        Xs = list(base_oof) + list(extra_oof)
        Ts = list(base_test) + list(extra_test)
        Ns = list(names) + list(extra_names)
        K = len(Ns)
        F_oof = expand(np.column_stack(Xs))
        F_test = expand(np.column_stack(Ts))
        mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
        auc = float(roc_auc_score(y, mo))
        d_bp = (auc - D12_GKF_META_BASELINE) * 1e4
        l1 = {Ns[i]: float(abs(coef[i]) + abs(coef[K+i]) + abs(coef[2*K+i]))
              for i in range(K)}
        print(f"\n=== GKF stack: {label} (K={K}) ===")
        print(f"  GKF OOF: {auc:.5f}  Δ vs d12 baseline (0.94776): {d_bp:+.2f}bp")
        print(f"  L1 top-12:")
        for n_, v in sorted(l1.items(), key=lambda kv: -kv[1])[:12]:
            mk = "  ← d13a-FM" if n_.startswith("FM_") and "53" in n_ else \
                 "  ← d9-FM" if "FM" in n_ else \
                 "  ← rule" if n_.startswith("rule_") else ""
            print(f"    {n_:<24s} L1={v:.3f}{mk}")
        return auc, d_bp, l1

    auc_baseline, _, l1_baseline = gkf_stack(
        f"baseline_K{len(names)}_no_d13a", [], [], [])
    auc_with, d_with, l1_with = gkf_stack(
        f"K{len(names)+2}_add_d13a_53", [oof_a, oof_b], [test_a, test_b],
        ["FM_A_53", "FM_B_53"])
    auc_swap, d_swap, l1_swap = gkf_stack(
        f"K{len(names)}_swap_d9c_for_d13a_FM_A_53",
        [oof_a], [test_a], ["FM_A_53"]) if False else (None, None, None)
    # (swap variant deferred — first see if add-only helps)

    final = dict(
        leakage_profile=dict(
            FM_A_53=dict(strat=s_a, gkf=auc_a_gkf, delta_bp=d_a_bp, verdict=verdict_a),
            FM_B_53=dict(strat=s_b, gkf=auc_b_gkf, delta_bp=d_b_bp, verdict=verdict_b),
        ),
        gkf_pool_size=len(names),
        gkf_baseline_auc=auc_baseline,
        gkf_with_d13a_auc=auc_with,
        gkf_lift_bp=d_with - (auc_baseline - D12_GKF_META_BASELINE) * 1e4,
        d12_baseline_auc=D12_GKF_META_BASELINE,
        params=dict(embed_dim=EMBED_DIM, epochs=EPOCHS, batch=BATCH, lr=LR),
        total_wall_s=time.time() - t0,
    )
    (ART / "d13a_groupkf_probe_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13a_groupkf_probe_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
