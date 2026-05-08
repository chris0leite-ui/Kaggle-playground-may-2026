"""H1 (combined) — 15-field augmented FM with H1 + H4 + H5 fields added.

Builds on d9h's 12-field hashed FM by adding 3 new fields:
  Field 13 (CRT):  Compound × TL_q5 × RP_q5         (H1 — 3-way EDA finding)
  Field 14 (Cdpl): cumdeg_per_lap_q5                (H4 — independent signal)
  Field 15 (Ldz):  LapTime_Delta_zr_q5              (H5 — race-z normalization)

Predicted: by analogy to d9h aug12 (12 fields → +3 bp LB), an aug15 FM with
these orthogonal signals should pass min-meta with EV +1-3 bp LB.

Min-meta gate vs PRIMARY K=22.  Single fold for speed; CPU-only.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED = 42
N_HASH_BITS = 18
EMBED_DIM = 8
EPOCHS = 6
BATCH = 8192
LR_FM = 0.05
N_FIELDS = 15

K22 = [
    "baseline_two_anchor", "d2a_te", "m2_xgb", "e1_catboost_sub",
    "e3_hgbc", "e5_optuna_lgbm", "a_horizon", "b_lapsuntilpit",
    "f1_hgbc_deep", "f2_hgbc_shallow", "cb_year-cat", "cb_lossguide",
    "cb_slow-wide-bag", "realmlp", "d6_rule_driver_compound",
    "d6_rule_year_race", "d9_R6_next_compound", "d9_R10_driver_eb",
    "d9_R7_prev_compound", "d9f_FM_A", "d9f_FM_B", "d9h_FM_aug12",
]


def _qbin(arr_train_fold, arr_query, n_bins):
    edges = np.quantile(arr_train_fold, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    return np.clip(np.searchsorted(edges, arr_query, side="right") - 1,
                   0, n_bins - 1).astype(int)


def _zr(train, test, key, by, fold_train_idx):
    """Race-Year-Compound z-score, fit on fold_train_idx, applied to all."""
    sub = train.iloc[fold_train_idx]
    stats = sub.groupby(by)[key].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].clip(lower=1e-6)
    stats.columns = list(by) + ["__mu", "__sd"]
    glob_mu = float(sub[key].mean())
    glob_sd = float(sub[key].std() + 1e-6)
    def apply(df):
        merged = df.merge(stats, on=by, how="left")
        mu = merged["__mu"].fillna(glob_mu).values
        sd = merged["__sd"].fillna(glob_sd).values
        return ((df[key].values - mu) / sd).astype(np.float64)
    return apply(train), apply(test)


def _neighbour_compounds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sort_idx = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).index
    s = df.loc[sort_idx].copy()
    grp = s.groupby(["Year", "Race", "Driver"], sort=False)
    s["next_compound"] = grp["Compound"].shift(-1).fillna("UNK").astype(str)
    s["prev_compound"] = grp["Compound"].shift(+1).fillna("UNK").astype(str)
    s = s.sort_index()
    return s["next_compound"].values, s["prev_compound"].values


def build_fields(train, test, fold_tr_idx):
    """Build 15-field categorical strings for FeatureHasher."""
    nx_tr, pv_tr = _neighbour_compounds(train)
    nx_te, pv_te = _neighbour_compounds(test)

    n_tr, n_te = len(train), len(test)
    D = ("Driver", "D"); C = ("Compound", "C"); R = ("Race", "R")
    Y = ("Year", "Y")

    # Quintile fields fit on fold-train slice
    Tq_tr = _qbin(train["TyreLife"].values[fold_tr_idx],
                   train["TyreLife"].values, 5)
    Tq_te = _qbin(train["TyreLife"].values[fold_tr_idx],
                   test["TyreLife"].values, 5)
    Rq_tr = _qbin(train["RaceProgress"].values[fold_tr_idx],
                   train["RaceProgress"].values, 5)
    Rq_te = _qbin(train["RaceProgress"].values[fold_tr_idx],
                   test["RaceProgress"].values, 5)
    Pq_tr = _qbin(train["Position"].values[fold_tr_idx],
                   train["Position"].values, 5)
    Pq_te = _qbin(train["Position"].values[fold_tr_idx],
                   test["Position"].values, 5)
    Cd_tr = _qbin(train["Cumulative_Degradation"].values[fold_tr_idx],
                   train["Cumulative_Degradation"].values, 5)
    Cd_te = _qbin(train["Cumulative_Degradation"].values[fold_tr_idx],
                   test["Cumulative_Degradation"].values, 5)
    Ld_tr = _qbin(train["LapTime_Delta"].values[fold_tr_idx],
                   train["LapTime_Delta"].values, 5)
    Ld_te = _qbin(train["LapTime_Delta"].values[fold_tr_idx],
                   test["LapTime_Delta"].values, 5)

    # H4 cumdeg_per_lap (already added before call; quintile-bin)
    cdpl_tr_full = (train["Cumulative_Degradation"].values
                     / np.clip(train["TyreLife"].values, 1, None))
    cdpl_te_full = (test["Cumulative_Degradation"].values
                     / np.clip(test["TyreLife"].values, 1, None))
    Cdpl_tr = _qbin(cdpl_tr_full[fold_tr_idx], cdpl_tr_full, 5)
    Cdpl_te = _qbin(cdpl_tr_full[fold_tr_idx], cdpl_te_full, 5)

    # H5 LapTime_Delta z-score, then quintile
    ldz_tr, ldz_te = _zr(train, test, "LapTime_Delta",
                          ["Race", "Year", "Compound"], fold_tr_idx)
    Ldz_tr = _qbin(ldz_tr[fold_tr_idx], ldz_tr, 5)
    Ldz_te = _qbin(ldz_tr[fold_tr_idx], ldz_te, 5)

    S_tr = train["Stint"].clip(upper=5).astype(int).values
    S_te = test["Stint"].clip(upper=5).astype(int).values

    # H1 3-way concat: Compound × TL_q5 × RP_q5
    crt_tr = (train["Compound"].astype(str).values + "|"
              + Tq_tr.astype(str) + "|" + Rq_tr.astype(str))
    crt_te = (test["Compound"].astype(str).values + "|"
              + Tq_te.astype(str) + "|" + Rq_te.astype(str))

    fields = ["D","C","R","Y","S","T","Rp","P","Nx","Pv","Cd","Ld",
              "CRT","Cdpl","Ldz"]
    arrs_tr = [
        train["Driver"].astype(str).values,
        train["Compound"].astype(str).values,
        train["Race"].astype(str).values,
        train["Year"].astype(str).values,
        S_tr.astype(str),
        Tq_tr.astype(str),
        Rq_tr.astype(str),
        Pq_tr.astype(str),
        nx_tr.astype(str),
        pv_tr.astype(str),
        Cd_tr.astype(str),
        Ld_tr.astype(str),
        crt_tr,
        Cdpl_tr.astype(str),
        Ldz_tr.astype(str),
    ]
    arrs_te = [
        test["Driver"].astype(str).values,
        test["Compound"].astype(str).values,
        test["Race"].astype(str).values,
        test["Year"].astype(str).values,
        S_te.astype(str),
        Tq_te.astype(str),
        Rq_te.astype(str),
        Pq_te.astype(str),
        nx_te.astype(str),
        pv_te.astype(str),
        Cd_te.astype(str),
        Ld_te.astype(str),
        crt_te,
        Cdpl_te.astype(str),
        Ldz_te.astype(str),
    ]
    def rows(arrs, n):
        return [[f"{p}={a[i]}" for p, a in zip(fields, arrs)] for i in range(n)]
    fh = FeatureHasher(n_features=2 ** N_HASH_BITS, input_type="string",
                      alternate_sign=False)
    return fh.transform(rows(arrs_tr, n_tr)), fh.transform(rows(arrs_te, n_te))


def csr_to_idx(csr, n_active):
    n = csr.shape[0]
    out = np.zeros((n, n_active), dtype=np.int64)
    indptr = csr.indptr; indices = csr.indices
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
    def __init__(self, n_features, k):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        nn.init.zeros_(self.linear.weight)
        self.embed = nn.Embedding(n_features, k, sparse=True)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, x):
        lin = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embed(x)
        sum_v = v.sum(dim=1)
        sum_v_sq = (v * v).sum(dim=1)
        inter = 0.5 * (sum_v.pow(2).sum(dim=1) - sum_v_sq.sum(dim=1))
        return self.bias + lin + inter


def fit_fm_fold(idx_tr_all, idx_te, y, tr, va, seed=SEED, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FMModel(2 ** N_HASH_BITS, EMBED_DIM)
    od = torch.optim.Adam([model.bias], lr=LR_FM)
    os_ = torch.optim.SparseAdam([model.linear.weight, model.embed.weight], lr=LR_FM)
    bce = nn.BCEWithLogitsLoss()
    Xtr = torch.from_numpy(idx_tr_all[tr])
    ytr = torch.from_numpy(y[tr].astype(np.float32))
    Xva = torch.from_numpy(idx_tr_all[va])
    Xte = torch.from_numpy(idx_te)
    n = len(Xtr)
    perm = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(perm)
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            xb, yb = Xtr[b], ytr[b]
            logits = model(xb)
            loss = bce(logits, yb)
            od.zero_grad(); os_.zero_grad()
            loss.backward()
            od.step(); os_.step()
        with torch.no_grad():
            preds = []
            for s in range(0, len(Xva), BATCH):
                preds.append(torch.sigmoid(model(Xva[s:s + BATCH])).numpy())
            pv = np.concatenate(preds)
            try:
                auc = roc_auc_score(y[va], pv)
            except Exception:
                auc = float("nan")
            print(f"    ep {ep}: val auc {auc:.5f}")
    with torch.no_grad():
        pv = np.concatenate([torch.sigmoid(model(Xva[s:s+BATCH])).numpy()
                              for s in range(0, len(Xva), BATCH)])
        pt = np.concatenate([torch.sigmoid(model(Xte[s:s+BATCH])).numpy()
                              for s in range(0, len(Xte), BATCH)])
    return pv, pt


def main() -> None:
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).to_numpy()

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof_fm = np.zeros(len(y), dtype=np.float64)
    test_fm = np.zeros(len(test), dtype=np.float64)
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"\n=== fold {fold} ===")
        Xtr_csr, Xte_csr = build_fields(train, test, tr)
        idx_tr = csr_to_idx(Xtr_csr, N_FIELDS)
        idx_te = csr_to_idx(Xte_csr, N_FIELDS)
        pv, pt = fit_fm_fold(idx_tr, idx_te, y, tr, va)
        oof_fm[va] = pv
        test_fm += pt / 5
    print(f"\nFM_aug15 OOF AUC = {roc_auc_score(y, oof_fm):.5f}")
    np.save(ART / "oof_H1_FM_aug15_strat.npy", oof_fm.astype(np.float32))
    np.save(ART / "test_H1_FM_aug15_strat.npy", test_fm.astype(np.float32))

    # ----- gate -----
    primary_oof = np.load(ART / "oof_PRIMARY_K22_strat.npy").astype(np.float64)
    primary_test = np.load(ART / "test_PRIMARY_K22_strat.npy").astype(np.float64)
    primary_auc = roc_auc_score(y, primary_oof)
    base_auc = roc_auc_score(y, oof_fm)
    rho = spearmanr(oof_fm, primary_oof).correlation

    def expand(M):
        n = len(M)
        rk = np.column_stack([rankdata(c) / n for c in M.T])
        logit = np.log(np.clip(M, 1e-9, 1 - 1e-9) /
                       (1 - np.clip(M, 1e-9, 1 - 1e-9)))
        return np.hstack([M, rk, logit])

    # min-meta gate: PRIMARY + base
    Z = np.column_stack([primary_oof, oof_fm])
    Zt = np.column_stack([primary_test, test_fm])
    F_oof = expand(Z); F_te = expand(Zt)
    skf2 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    moo = np.zeros(len(y))
    for tr, va in skf2.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(F_oof[tr], y[tr])
        moo[va] = lr.predict_proba(F_oof[va])[:, 1]
    mm = roc_auc_score(y, moo)
    print(f"\n--- min-meta gate H1_FM_aug15 ---")
    print(f"  base OOF AUC      = {base_auc:.5f}")
    print(f"  ρ vs PRIMARY      = {rho:.5f}")
    print(f"  min-meta OOF AUC  = {mm:.5f}  Δ = {(mm - primary_auc)*1e4:+.2f} bp")

    # K=23 add gate
    P_oof = []; P_te = []
    for fn in K22:
        oo = np.load(ART / f"oof_{fn}_strat.npy")
        if oo.ndim == 2: oo = oo[:, 1]
        P_oof.append(oo)
        tt = np.load(ART / f"test_{fn}_strat.npy")
        if tt.ndim == 2: tt = tt[:, 1]
        P_te.append(tt)
    P_oof = np.column_stack(P_oof).astype(np.float64)
    P_te = np.column_stack(P_te).astype(np.float64)

    K23_oof_arr = np.column_stack([P_oof, oof_fm])
    K23_te_arr = np.column_stack([P_te, test_fm])
    F_oof23 = expand(K23_oof_arr); F_te23 = expand(K23_te_arr)
    skf3 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    moo23 = np.zeros(len(y))
    for tr, va in skf3.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(F_oof23[tr], y[tr])
        moo23[va] = lr.predict_proba(F_oof23[va])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=2000)
    lr_full.fit(F_oof23, y)
    test23 = lr_full.predict_proba(F_te23)[:, 1]
    mm23 = roc_auc_score(y, moo23)
    print(f"  K=23 add OOF AUC = {mm23:.5f}  Δ = {(mm23 - primary_auc)*1e4:+.2f} bp")

    np.save(ART / "oof_K23_add_H1_FM_aug15_strat.npy",
            moo23.astype(np.float32))
    np.save(ART / "test_K23_add_H1_FM_aug15_strat.npy",
            test23.astype(np.float32))

    # K=23 swap (drop d9h_FM_aug12 since H1 supersedes it)
    keep_idx = [i for i, fn in enumerate(K22) if fn != "d9h_FM_aug12"]
    P_oof_swap = np.column_stack([P_oof[:, i] for i in keep_idx] + [oof_fm])
    P_te_swap = np.column_stack([P_te[:, i] for i in keep_idx] + [test_fm])
    F_oof_sw = expand(P_oof_swap); F_te_sw = expand(P_te_swap)
    skf4 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    moo_sw = np.zeros(len(y))
    for tr, va in skf4.split(np.zeros(len(y)), y):
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(F_oof_sw[tr], y[tr])
        moo_sw[va] = lr.predict_proba(F_oof_sw[va])[:, 1]
    lr_full_sw = LogisticRegression(C=1.0, max_iter=2000)
    lr_full_sw.fit(F_oof_sw, y)
    test_sw = lr_full_sw.predict_proba(F_te_sw)[:, 1]
    mm_sw = roc_auc_score(y, moo_sw)
    print(f"  K=22 swap OOF AUC = {mm_sw:.5f}  Δ = {(mm_sw - primary_auc)*1e4:+.2f} bp"
          " (drops d9h_FM_aug12 in favor of FM_aug15)")
    np.save(ART / "oof_K22_swap_H1_FM_aug15_strat.npy",
            moo_sw.astype(np.float32))
    np.save(ART / "test_K22_swap_H1_FM_aug15_strat.npy",
            test_sw.astype(np.float32))

    print(f"\nwall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
