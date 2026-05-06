"""Day-13 Move F-2 — 16-field FM (d9h_aug12 + F1-F4 from d13_move_f_features).

Tests whether the 4 new FM-input fields (F1_PitWindow_q5, F2_HazardDecay_q5,
F3_CompoundPress_q5, F4_RaceStage) built in d13_move_f_features.py add signal
to the existing 12-field FM (d9h_aug12).

New fields:
    F1_PitWindow_q5     laps_since_last_pit → quantile bin
    F2_HazardDecay_q5   exp(-TL/μ_C) → quantile bin  (monotone, Δrate=0.25)
    F3_CompoundPress_q5 TL − μ_C → quantile bin       (monotone, Δrate=0.22)
    F4_RaceStage        non-uniform RaceProgress bins (mid_b=0.38, opening=0.06)

Prerequisite: scripts/artifacts/d13_move_f_features.csv.gz must exist.
Run scripts/d13_move_f_features.py first if missing.

Gate:
    G1 std OOF vs PRIMARY Strat OOF (0.95083)
    G2 ρ vs PRIMARY test preds < 0.999 (genuinely diverse)
    G3 min-meta vs PRIMARY OOF > 0 (additive signal)

To run:
    python scripts/d13_move_f_fm_aug16.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix  # noqa: F401
from scipy.stats import spearmanr
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
N_FIELDS = 16  # 12 original + 4 new

PRIMARY_S = 0.95083   # d13e Compound×Stint τ=20000 Strat OOF
PRIMARY_LB = 0.95049  # d13e LB


def expand(P):
    n = len(P)
    rk = np.column_stack([np.argsort(np.argsort(c)) / n for c in P.T])
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


def _quantile_bin(arr_train, arr_query, n_bins=5):
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


def build_aug16_hashes(train, test, tr_idx,
                       next_train, next_test,
                       prev_train, prev_test,
                       f_tr: pd.DataFrame, f_te: pd.DataFrame):
    """Build (n_rows, 16) hashed indices: 12 d9h fields + 4 Move-F fields."""
    n_tr, n_te = len(train), len(test)

    # ---- 12 original d9h fields ----------------------------------------
    D_tr = train["Driver"].astype(str).values
    D_te = test["Driver"].astype(str).values
    C_tr = train["Compound"].astype(str).values
    C_te = test["Compound"].astype(str).values
    R_tr = train["Race"].astype(str).values
    R_te = test["Race"].astype(str).values
    Y_tr = train["Year"].astype(str).values
    Y_te = test["Year"].astype(str).values
    S_tr = train["Stint"].clip(upper=5).astype(int).astype(str).values
    S_te = test["Stint"].clip(upper=5).astype(int).astype(str).values
    Tq_tr = _quantile_bin(train["TyreLife"].values[tr_idx],
                          train["TyreLife"].values).astype(str)
    Tq_te = _quantile_bin(train["TyreLife"].values[tr_idx],
                          test["TyreLife"].values).astype(str)
    Rq_tr = _quantile_bin(train["RaceProgress"].values[tr_idx],
                          train["RaceProgress"].values).astype(str)
    Rq_te = _quantile_bin(train["RaceProgress"].values[tr_idx],
                          test["RaceProgress"].values).astype(str)
    Pq_tr = _quantile_bin(train["Position"].values[tr_idx],
                          train["Position"].values).astype(str)
    Pq_te = _quantile_bin(train["Position"].values[tr_idx],
                          test["Position"].values).astype(str)
    Nx_tr = next_train.astype(str)
    Nx_te = next_test.astype(str)
    Pv_tr = prev_train.astype(str)
    Pv_te = prev_test.astype(str)
    Cd_tr = _quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                          train["Cumulative_Degradation"].values).astype(str)
    Cd_te = _quantile_bin(train["Cumulative_Degradation"].values[tr_idx],
                          test["Cumulative_Degradation"].values).astype(str)
    Ld_tr = _quantile_bin(train["LapTime_Delta"].values[tr_idx],
                          train["LapTime_Delta"].values).astype(str)
    Ld_te = _quantile_bin(train["LapTime_Delta"].values[tr_idx],
                          test["LapTime_Delta"].values).astype(str)

    # ---- 4 Move-F fields (train-fit binner already applied in d13_move_f_features) --
    F1_tr = f_tr["F1_PitWindow_q5"].astype(str).values
    F1_te = f_te["F1_PitWindow_q5"].astype(str).values
    F2_tr = f_tr["F2_HazardDecay_q5"].astype(str).values
    F2_te = f_te["F2_HazardDecay_q5"].astype(str).values
    F3_tr = f_tr["F3_CompoundPress_q5"].astype(str).values
    F3_te = f_te["F3_CompoundPress_q5"].astype(str).values
    F4_tr = f_tr["F4_RaceStage"].astype(str).values
    F4_te = f_te["F4_RaceStage"].astype(str).values

    fields = ["D", "C", "R", "Y", "S", "T", "Rp", "P", "Nx", "Pv", "Cd", "Ld",
              "F1", "F2", "F3", "F4"]
    arrs_tr = [D_tr, C_tr, R_tr, Y_tr, S_tr, Tq_tr, Rq_tr, Pq_tr,
               Nx_tr, Pv_tr, Cd_tr, Ld_tr, F1_tr, F2_tr, F3_tr, F4_tr]
    arrs_te = [D_te, C_te, R_te, Y_te, S_te, Tq_te, Rq_te, Pq_te,
               Nx_te, Pv_te, Cd_te, Ld_te, F1_te, F2_te, F3_te, F4_te]

    def rows(arrs, n):
        return [[f"{p}={a[i]}" for p, a in zip(fields, arrs)] for i in range(n)]

    h = FeatureHasher(n_features=2**N_HASH_BITS, input_type="string",
                      alternate_sign=False)
    return h.transform(rows(arrs_tr, n_tr)), h.transform(rows(arrs_te, n_te))


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


def fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va, seed, epochs=EPOCHS):
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
    for ep in range(epochs):
        np.random.shuffle(perm)
        ep_loss = 0.0; n_seen = 0
        for s in range(0, n_tr, BATCH):
            b = perm[s:s + BATCH]
            xb = idx_tr[b]; yb = y_tr[b]
            logits = model(xb)
            loss = bce(logits, yb)
            opt_dense.zero_grad(); opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step(); opt_sparse.step()
            ep_loss += float(loss.item()) * len(b); n_seen += len(b)
        with torch.no_grad():
            chunks = [torch.sigmoid(model(idx_va[s:s+BATCH])).numpy()
                      for s in range(0, len(idx_va), BATCH)]
            pv = np.concatenate(chunks)
            try: auc = roc_auc_score(y[va], pv)
            except Exception: auc = float("nan")
        print(f"      epoch {ep}: train_loss={ep_loss/n_seen:.5f}  val_auc={auc:.5f}")
    with torch.no_grad():
        chunks = [torch.sigmoid(model(idx_te_t[s:s+BATCH])).numpy()
                  for s in range(0, len(idx_te_t), BATCH)]
        p_te = np.concatenate(chunks)
        chunks = [torch.sigmoid(model(idx_va[s:s+BATCH])).numpy()
                  for s in range(0, len(idx_va), BATCH)]
        p_va = np.concatenate(chunks)
    return p_va.astype(np.float64), p_te.astype(np.float64)


def main():
    t0 = time.time()
    art_path = ART / "d13_move_f_features.csv.gz"
    if not art_path.exists():
        raise SystemExit(f"Missing {art_path}. Run scripts/d13_move_f_features.py first.")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    n_tr, n_te = len(train), len(test)

    # Load Move-F features (train-fit quantile bins already applied)
    feats = pd.read_csv(art_path)
    f_tr = feats[feats["__src"] == "tr"].drop(columns=["__src"]).reset_index(drop=True)
    f_te = feats[feats["__src"] == "te"].drop(columns=["__src"]).reset_index(drop=True)
    assert len(f_tr) == n_tr and len(f_te) == n_te, \
        f"Shape mismatch: f_tr={len(f_tr)} f_te={len(f_te)}"
    print(f"Loaded Move-F features: train={len(f_tr)}  test={len(f_te)}")
    print(f"  columns: {list(feats.columns)}")

    # PRIMARY test preds for ρ check (d13e Compound×Stint τ=20k)
    primary_test = np.load(
        ART / "test_d13e_compound_stint_tau20000_strat.npy")[:, 1].astype(np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n_tr), y))

    # next/prev compound (d9h precedent)
    print("Computing next/prev compound on train+test ...")
    full = pd.concat([train.assign(__src="tr"), test.assign(__src="te")],
                     ignore_index=True)
    nx, pv_comp = _compute_neighbour_compounds(full)
    next_train = nx[full["__src"].values == "tr"]
    next_test = nx[full["__src"].values == "te"]
    prev_train = pv_comp[full["__src"].values == "tr"]
    prev_test = pv_comp[full["__src"].values == "te"]

    print(f"\n=== FM_aug16 (16 fields = 12 d9h + 4 Move-F, "
          f"k={EMBED_DIM}, epochs={EPOCHS}) ===\n")
    oof = np.zeros(n_tr, dtype=np.float64)
    test_avg = np.zeros(n_te, dtype=np.float64)
    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        Xtr_csr, Xte_csr = build_aug16_hashes(
            train, test, tr,
            next_train, next_test,
            prev_train, prev_test,
            f_tr, f_te,
        )
        idx_tr_full = csr_to_index_array(Xtr_csr, N_FIELDS)
        idx_te = csr_to_index_array(Xte_csr, N_FIELDS)
        p_va, p_te = fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va,
                                     seed=42 + k)
        oof[va] = p_va
        test_avg += p_te / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        print(f"    fold {k} done: val AUC {s:.5f}  wall {time.time()-t_fold:.1f}s")

    std_oof = float(roc_auc_score(y, oof))
    rho_test, _ = spearmanr(test_avg, primary_test)

    # Compare vs d9h_aug12 standalone
    aug12_oof = np.load(ART / "oof_d9h_FM_aug12_strat.npy")[:, 1].astype(np.float64)
    aug12_test = np.load(ART / "test_d9h_FM_aug12_strat.npy")[:, 1].astype(np.float64)
    aug12_std = float(roc_auc_score(y, aug12_oof))
    rho_12_vs_16, _ = spearmanr(test_avg, aug12_test)

    print(f"\n  FM_aug12 std OOF (baseline): {aug12_std:.5f}")
    print(f"  FM_aug16 std OOF:            {std_oof:.5f}  "
          f"Δ aug12 {(std_oof-aug12_std)*1e4:+.2f}bp")
    print(f"  ρ FM_aug16 vs PRIMARY test:  {rho_test:.5f}  "
          f"{'diverse ✓' if rho_test < 0.999 else 'TIE band'}")
    print(f"  ρ FM_aug16 vs FM_aug12 test: {rho_12_vs_16:.5f}")

    # Min-meta gate: PRIMARY OOF (d13e τ=20k) + FM_aug16
    primary_oof = np.load(
        ART / "oof_d13e_compound_stint_tau20000_strat.npy")[:, 1].astype(np.float64)
    F_min = expand(np.column_stack([primary_oof, oof]))
    F_min_t = expand(np.column_stack([primary_test, test_avg]))
    mo_mm, _, coef_mm = fit_lr_meta(F_min, F_min_t, y)
    auc_min = float(roc_auc_score(y, mo_mm))
    delta_mm = (auc_min - PRIMARY_S) * 1e4
    gate = auc_min >= PRIMARY_S
    print(f"\n  Min-meta OOF: {auc_min:.5f}  Δ PRIMARY {delta_mm:+.2f}bp  "
          f"{'PASS ✓' if gate else 'FAIL ✗'}")

    np.save(ART / "oof_d13_FM_aug16_strat.npy", np.column_stack([1 - oof, oof]))
    np.save(ART / "test_d13_FM_aug16_strat.npy",
            np.column_stack([1 - test_avg, test_avg]))

    results = dict(
        FM_aug16=dict(
            std_oof=std_oof,
            aug12_std_oof=aug12_std,
            delta_aug12_bp=float((std_oof - aug12_std) * 1e4),
            rho_vs_primary=float(rho_test),
            rho_vs_aug12=float(rho_12_vs_16),
            min_meta_oof=auc_min,
            delta_primary_bp=float(delta_mm),
            min_meta_pass=gate,
        ),
        params=dict(embed_dim=EMBED_DIM, epochs=EPOCHS, batch=BATCH,
                    lr=LR, n_fields=N_FIELDS),
        primary=dict(strat_oof=PRIMARY_S, lb=PRIMARY_LB),
        total_wall_s=time.time() - t0,
    )
    (ART / "d13_move_f_fm_aug16_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n→ scripts/artifacts/d13_move_f_fm_aug16_results.json  "
          f"(wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
