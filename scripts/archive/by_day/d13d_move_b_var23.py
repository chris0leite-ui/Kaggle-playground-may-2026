"""Day-13d — Move B variants 2 and 3 (FM-class diversification continued).

Continues the FM partition-shape exploration after d13a (5/3) showed
within-class diversification works structurally (GKF +2.34bp) but
TIE_EXPECTED on Strat (+0.19bp).

Per HANDOVER Day-13 Move B plan, two more partition shapes:

  Variant 2 — 4/4 alt-split, Compound × TyreLife axis
    FM_A_CT:   C, T, Cd, Ld    (4 fields — wheel-physics axis)
    FM_B_DR:   D, R, S, Y      (4 fields — categorical-identity axis)
    Different orthogonalisation than d9f (D,C,S,T vs R,Y,Rp,P) — moves
    Compound to the wheel side, drops Rp/P/Nx/Pv entirely.

  Variant 3 — 6/6 alt-split, neighbours-with-driver
    FM_A_DH:   D, C, Cd, Ld, Nx, Pv   (driver + tire-history + neighbours)
    FM_B_RT:   R, Y, S, T, Rp, P      (race + position + tire-life)
    Different from d9i's 6/6 (D,C,S,T,Cd,Ld vs R,Y,Rp,P,Nx,Pv): T moves
    to FM_B, Nx/Pv move to FM_A. Tests whether neighbour-compound signal
    is better paired with driver/compound or with race-context.

For each variant, four stacks vs PRIMARY (d9i S1 K21 swap, OOF 0.95071):
  S_solo_K23:  T1_drop_d9c (K=23) + var FMs replace one slot pair?
                — actually run as ADD not SWAP since T1 is already minimal
  S_add_K25:   T1_drop_d9c (K=23) + var.FM_A + var.FM_B  (clean add)
  S_combo_K27: T1 (K=23) + d13a (var.1) + var.X            (compound)

Best-of variants reported and submission CSV staged for PI.
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

PRIMARY_S = 0.95071
PRIMARY_LB = 0.95034
RHO_TIE = 0.9995

# T1 baseline pool (from d13c — drop d9c_FM, K=23)
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
EXISTING_FMS = [
    ("d9f_FM_A", "oof_d9f_FM_A_strat.npy", "test_d9f_FM_A_strat.npy"),
    ("d9f_FM_B", "oof_d9f_FM_B_strat.npy", "test_d9f_FM_B_strat.npy"),
    ("FM_A_53", "oof_d13a_FM_A_53_strat.npy", "test_d13a_FM_A_53_strat.npy"),
    ("FM_B_53", "oof_d13a_FM_B_53_strat.npy", "test_d13a_FM_B_53_strat.npy"),
]

# Variant 2: Compound × TyreLife axis (4/4)
PART_V2_A = ["C", "T", "Cd", "Ld"]
PART_V2_B = ["D", "R", "S", "Y"]

# Variant 3: 6/6 alt — driver+history+neighbours vs race+position+tirelife
PART_V3_A = ["D", "C", "Cd", "Ld", "Nx", "Pv"]
PART_V3_B = ["R", "Y", "S", "T", "Rp", "P"]


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


def _compute_neighbour_compounds(df):
    sort_idx = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).index
    s = df.loc[sort_idx].copy()
    grp = s.groupby(["Year", "Race", "Driver"], sort=False)
    s["next_compound"] = grp["Compound"].shift(-1).fillna("UNK").astype(str)
    s["prev_compound"] = grp["Compound"].shift(+1).fillna("UNK").astype(str)
    s = s.sort_index()
    return s["next_compound"].values, s["prev_compound"].values


def build_partition_hashes(train, test, tr_idx, fields,
                            next_train=None, next_test=None,
                            prev_train=None, prev_test=None):
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
        elif f == "Cd":
            arrs_tr.append(_quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                                          train["Cumulative_Degradation"].values, 5).astype(str))
            arrs_te.append(_quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                                          test["Cumulative_Degradation"].values, 5).astype(str))
        elif f == "Ld":
            arrs_tr.append(_quantile_bin(train["LapTime_Delta"].values[tr_idx],
                                          train["LapTime_Delta"].values, 5).astype(str))
            arrs_te.append(_quantile_bin(train["LapTime_Delta"].values[tr_idx],
                                          test["LapTime_Delta"].values, 5).astype(str))
        elif f == "Nx":
            arrs_tr.append(next_train.astype(str))
            arrs_te.append(next_test.astype(str))
        elif f == "Pv":
            arrs_tr.append(prev_train.astype(str))
            arrs_te.append(prev_test.astype(str))
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


def train_fm(train, test, y, splits, fields, label, **nbr):
    print(f"\n--- {label} (fields={fields}, n={len(fields)}) ---")
    n_train, n_test = len(y), len(test)
    oof = np.zeros(n_train, dtype=np.float64)
    test_avg = np.zeros(n_test, dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        Xtr_csr, Xte_csr = build_partition_hashes(train, test, tr, fields, **nbr)
        idx_tr_full = csr_to_index_array(Xtr_csr, len(fields))
        idx_te = csr_to_index_array(Xte_csr, len(fields))
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va, seed=42 + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        print(f"  f{k}: AUC={s:.5f}  wall={time.time()-t:.1f}s")
    return oof, test_avg


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

    # Pre-compute neighbour compounds (for V3)
    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    nx, pv = _compute_neighbour_compounds(full)
    next_train = nx[full["__src"].values == "tr"]
    next_test = nx[full["__src"].values == "te"]
    prev_train = pv[full["__src"].values == "tr"]
    prev_test = pv[full["__src"].values == "te"]
    nbr_kwargs = dict(next_train=next_train, next_test=next_test,
                       prev_train=prev_train, prev_test=prev_test)

    # ----- Variant 2: 4/4 Compound × TyreLife axis -----
    print(f"\n=== Variant 2 — 4/4 Compound×TyreLife axis ===")
    oof_v2a, test_v2a = train_fm(train, test, y, splits, PART_V2_A, "FM_A_CT")
    oof_v2b, test_v2b = train_fm(train, test, y, splits, PART_V2_B, "FM_B_DR")

    auc_v2a = float(roc_auc_score(y, oof_v2a))
    auc_v2b = float(roc_auc_score(y, oof_v2b))
    rho_v2ap, _ = spearmanr(test_v2a, primary_test)
    rho_v2bp, _ = spearmanr(test_v2b, primary_test)
    rho_v2ab, _ = spearmanr(test_v2a, test_v2b)
    print(f"\n  FM_A_CT std OOF: {auc_v2a:.5f}  ρ vs PRIMARY {rho_v2ap:.5f}")
    print(f"  FM_B_DR std OOF: {auc_v2b:.5f}  ρ vs PRIMARY {rho_v2bp:.5f}")
    print(f"  ρ FM_A_CT vs FM_B_DR: {rho_v2ab:.5f}")

    np.save(ART / "oof_d13d_FM_A_CT_strat.npy", np.column_stack([1-oof_v2a, oof_v2a]))
    np.save(ART / "test_d13d_FM_A_CT_strat.npy", np.column_stack([1-test_v2a, test_v2a]))
    np.save(ART / "oof_d13d_FM_B_DR_strat.npy", np.column_stack([1-oof_v2b, oof_v2b]))
    np.save(ART / "test_d13d_FM_B_DR_strat.npy", np.column_stack([1-test_v2b, test_v2b]))

    # ----- Variant 3: 6/6 alt-split -----
    print(f"\n=== Variant 3 — 6/6 alt-split (Nx/Pv with driver) ===")
    oof_v3a, test_v3a = train_fm(train, test, y, splits, PART_V3_A, "FM_A_DH",
                                  **nbr_kwargs)
    oof_v3b, test_v3b = train_fm(train, test, y, splits, PART_V3_B, "FM_B_RT",
                                  **nbr_kwargs)

    auc_v3a = float(roc_auc_score(y, oof_v3a))
    auc_v3b = float(roc_auc_score(y, oof_v3b))
    rho_v3ap, _ = spearmanr(test_v3a, primary_test)
    rho_v3bp, _ = spearmanr(test_v3b, primary_test)
    rho_v3ab, _ = spearmanr(test_v3a, test_v3b)
    print(f"\n  FM_A_DH std OOF: {auc_v3a:.5f}  ρ vs PRIMARY {rho_v3ap:.5f}")
    print(f"  FM_B_RT std OOF: {auc_v3b:.5f}  ρ vs PRIMARY {rho_v3bp:.5f}")
    print(f"  ρ FM_A_DH vs FM_B_RT: {rho_v3ab:.5f}")

    np.save(ART / "oof_d13d_FM_A_DH_strat.npy", np.column_stack([1-oof_v3a, oof_v3a]))
    np.save(ART / "test_d13d_FM_A_DH_strat.npy", np.column_stack([1-test_v3a, test_v3a]))
    np.save(ART / "oof_d13d_FM_B_RT_strat.npy", np.column_stack([1-oof_v3b, oof_v3b]))
    np.save(ART / "test_d13d_FM_B_RT_strat.npy", np.column_stack([1-test_v3b, test_v3b]))

    # ----- Stack matrix -----
    # T1 baseline = POOL_KEEP + TOP_3_D9 + d9f FM_A,B + d13a FM_A_53,B_53 (K=23)
    print(f"\n=== Stack matrix on T1 K=23 baseline ===")
    base_oof, base_test, base_names = [], [], []
    for label, fname in POOL_KEEP + TOP_3_D9:
        base_oof.append(np.load(ART / f"oof_{fname}_strat.npy")[:, 1].astype(np.float64))
        base_test.append(np.load(ART / f"test_{fname}_strat.npy")[:, 1].astype(np.float64))
        base_names.append(label)
    for label, of, tf in EXISTING_FMS:
        base_oof.append(np.load(ART / of)[:, 1].astype(np.float64))
        base_test.append(np.load(ART / tf)[:, 1].astype(np.float64))
        base_names.append(label)

    def stack(label, extras_oof, extras_test, extras_names):
        Xs = list(base_oof) + list(extras_oof)
        Ts = list(base_test) + list(extras_test)
        Ns = list(base_names) + list(extras_names)
        K = len(Ns)
        F_oof = expand(np.column_stack(Xs))
        F_test = expand(np.column_stack(Ts))
        mo, tp, coef = fit_lr_meta(F_oof, F_test, y)
        auc = float(roc_auc_score(y, mo))
        rho, _ = spearmanr(tp, primary_test)
        delta_bp = (auc - PRIMARY_S) * 1e4
        l1 = {Ns[i]: float(abs(coef[i]) + abs(coef[K+i]) + abs(coef[2*K+i]))
              for i in range(K)}
        if auc >= PRIMARY_S and rho < RHO_TIE:
            verdict = "PRIMARY-CANDIDATE ★"
        elif auc >= PRIMARY_S:
            verdict = "TIE_EXPECTED (hold)"
        else:
            verdict = "REGRESS / HEDGE-only"
        print(f"\n--- {label} (K={K}) ---")
        print(f"  Strat OOF: {auc:.5f}  Δ {delta_bp:+.2f}bp  ρ {rho:.5f}  {verdict}")
        l1_top = sorted(l1.items(), key=lambda kv: -kv[1])[:8]
        for n_, v in l1_top:
            mk = ""
            if n_ in ("FM_A_CT", "FM_B_DR"): mk = "  ← v2"
            elif n_ in ("FM_A_DH", "FM_B_RT"): mk = "  ← v3"
            elif n_ in ("FM_A_53", "FM_B_53"): mk = "  ← d13a"
            elif n_ in ("d9f_FM_A", "d9f_FM_B"): mk = "  ← d9f"
            elif n_.startswith("rule_"): mk = "  ← rule"
            print(f"    {n_:<24s} L1={v:.3f}{mk}")
        return dict(K=K, strat_oof=auc, delta_bp=delta_bp,
                    rho=float(rho), verdict=verdict, l1=l1, tp=tp)

    r_t1 = stack("T1_baseline_K23", [], [], [])  # baseline reproduction
    r_v2 = stack("V2_K25_add_CT_DR", [oof_v2a, oof_v2b], [test_v2a, test_v2b],
                 ["FM_A_CT", "FM_B_DR"])
    r_v3 = stack("V3_K25_add_DH_RT", [oof_v3a, oof_v3b], [test_v3a, test_v3b],
                 ["FM_A_DH", "FM_B_RT"])
    r_v23 = stack("V2+V3_K27_add_all_4_new",
                   [oof_v2a, oof_v2b, oof_v3a, oof_v3b],
                   [test_v2a, test_v2b, test_v3a, test_v3b],
                   ["FM_A_CT", "FM_B_DR", "FM_A_DH", "FM_B_RT"])

    print(f"\n=== Move B var.2/3 summary ===")
    rows = [("T1_K23", r_t1), ("V2_K25", r_v2), ("V3_K25", r_v3),
            ("V2+V3_K27", r_v23)]
    for name, r in rows:
        marker = " ★" if r["strat_oof"] >= PRIMARY_S and r["rho"] < RHO_TIE else ""
        print(f"  {name:<14s} K={r['K']}  Strat {r['strat_oof']:.5f}  "
              f"Δ {r['delta_bp']:+.2f}bp  ρ {r['rho']:.5f}  {r['verdict']}{marker}")

    # Save the highest-OOF candidate's submission
    candidates = [r_v2, r_v3, r_v23]
    leader = max(candidates, key=lambda r: r["strat_oof"])
    leader_name = {id(r_v2): "V2_K25_add_CT_DR",
                   id(r_v3): "V3_K25_add_DH_RT",
                   id(r_v23): "V2plusV3_K27_add_all"}[id(leader)]
    tp_leader = leader["tp"]
    np.save(ART / f"test_d13d_{leader_name}_strat.npy",
            np.column_stack([1 - tp_leader, tp_leader]))
    sub = sample_sub.copy(); sub[TARGET] = tp_leader
    sub.to_csv(f"submissions/submission_d13d_{leader_name}.csv", index=False)
    print(f"\n→ leader: {leader_name} (Strat {leader['strat_oof']:.5f}, "
          f"ρ {leader['rho']:.5f})")
    print(f"  wrote submissions/submission_d13d_{leader_name}.csv")

    serial = {n: {k: v for k, v in r.items() if k != "tp"} for n, r in rows}
    final = dict(
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        leader=leader_name,
        variants=serial,
        standalone=dict(
            v2=dict(FM_A_CT=auc_v2a, FM_B_DR=auc_v2b,
                    rho_A_primary=float(rho_v2ap), rho_B_primary=float(rho_v2bp),
                    rho_A_B=float(rho_v2ab)),
            v3=dict(FM_A_DH=auc_v3a, FM_B_RT=auc_v3b,
                    rho_A_primary=float(rho_v3ap), rho_B_primary=float(rho_v3bp),
                    rho_A_B=float(rho_v3ab)),
        ),
        total_wall_s=time.time() - t0,
    )
    (ART / "d13d_var23_results.json").write_text(json.dumps(final, indent=2))
    print(f"\n→ scripts/artifacts/d13d_var23_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
