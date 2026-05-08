"""scripts/probe_multi_target_nn.py — multi-target NN with auxiliary head.

Same architecture as probe_nn_embeddings.py (Driver/Race/Compound/Year/
Stint embeds + numerics → MLP) but with TWO output heads:

  Head 1: pit_next_lap (BCE) — primary target, used for OOF + LB.
  Head 2: inv_laps_until_pit (MSE) — auxiliary, regularizes shared trunk.

Joint loss: BCE + 0.3 * MSE. The auxiliary head SHARES the embedding
trunk so the embedding learning signal benefits from both targets.
Only Head 1 is consumed at inference.

This is the synth-data-aware variant of SOSTA's cascaded loss
(but using only features available in our dataset).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"

torch.manual_seed(SEED); np.random.seed(SEED)


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def compute_inv_laps_until_pit(df, y):
    df = df.copy(); df["_y"] = y; df["_idx"] = np.arange(len(df))
    out = np.zeros(len(df), dtype=np.float32)
    for keys, grp in df.groupby(["Driver", "Race", "Year"], sort=False):
        gs = grp.sort_values("LapNumber")
        laps = gs["LapNumber"].values
        ys = gs["_y"].values
        idxs = gs["_idx"].values
        next_pit_lap = np.full(len(gs), 999, dtype=np.int32)
        last = 999
        for i in range(len(gs) - 1, -1, -1):
            if ys[i] == 1:
                last = laps[i]; next_pit_lap[i] = 0
            else:
                next_pit_lap[i] = max(0, last - laps[i])
        out[idxs] = 1.0 / (1.0 + next_pit_lap)
    return out


class MultiHeadEmbMLP(nn.Module):
    def __init__(self, n_drv, n_race, n_cmp, n_year, n_stint, n_num):
        super().__init__()
        self.e_drv = nn.Embedding(n_drv, 8)
        self.e_race = nn.Embedding(n_race, 4)
        self.e_cmp = nn.Embedding(n_cmp, 3)
        self.e_year = nn.Embedding(n_year, 3)
        self.e_stint = nn.Embedding(n_stint, 3)
        self.bn = nn.BatchNorm1d(n_num)
        d = 8 + 4 + 3 + 3 + 3 + n_num
        self.trunk1 = nn.Linear(d, 64)
        self.trunk2 = nn.Linear(64, 32)
        self.drop = nn.Dropout(0.2)
        self.head_cls = nn.Linear(32, 1)   # pit_next_lap
        self.head_reg = nn.Linear(32, 1)   # inv_laps_until_pit

    def forward(self, drv, race, cmp, yr, st, num):
        x = torch.cat([self.e_drv(drv), self.e_race(race), self.e_cmp(cmp),
                       self.e_year(yr), self.e_stint(st), self.bn(num)], dim=1)
        h = torch.relu(self.trunk1(x))
        h = self.drop(h)
        h = torch.relu(self.trunk2(h))
        return self.head_cls(h).squeeze(-1), self.head_reg(h).squeeze(-1)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))

    print("Computing aux target inv_laps_until_pit...")
    t_aux = time.time()
    inv_y = compute_inv_laps_until_pit(train, y)
    print(f"  aux build wall: {time.time()-t_aux:.1f}s, mean={inv_y.mean():.4f}")

    drv_levels = sorted(set(train["Driver"].astype(str).unique()) |
                        set(test["Driver"].astype(str).unique()))
    drv_map = {v: i for i, v in enumerate(drv_levels)}
    race_levels = sorted(set(train["Race"].astype(str).unique()) |
                         set(test["Race"].astype(str).unique()))
    race_map = {v: i for i, v in enumerate(race_levels)}
    cmp_levels = sorted(set(train["Compound"].astype(str).unique()) |
                        set(test["Compound"].astype(str).unique()))
    cmp_map = {v: i for i, v in enumerate(cmp_levels)}
    year_levels = sorted(set(train["Year"].astype(int).unique()) |
                         set(test["Year"].astype(int).unique()))
    year_map = {v: i for i, v in enumerate(year_levels)}
    n_stint = 8

    def encode(df):
        return dict(
            drv=df["Driver"].astype(str).map(drv_map).astype(np.int64).values,
            race=df["Race"].astype(str).map(race_map).astype(np.int64).values,
            cmp=df["Compound"].astype(str).map(cmp_map).astype(np.int64).values,
            yr=df["Year"].astype(int).map(year_map).astype(np.int64).values,
            st=np.clip(df["Stint"].astype(int).values, 0, n_stint-1).astype(np.int64),
            num=df[["TyreLife", "RaceProgress", "LapTime_Delta",
                     "Cumulative_Degradation", "Position",
                     "LapTime (s)", "Position_Change"]].values.astype(np.float32),
        )

    enc_tr = encode(train); enc_te = encode(test)
    num_mean = enc_tr["num"].mean(axis=0)
    num_std = enc_tr["num"].std(axis=0) + 1e-6
    enc_tr["num"] = (enc_tr["num"] - num_mean) / num_std
    enc_te["num"] = (enc_te["num"] - num_mean) / num_std

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    oof = np.zeros(len(y), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    BATCH = 4096; EPOCHS = 6; AUX_W = 0.3

    def to_tensor(idx, src):
        return [torch.from_numpy(src[k][idx]) for k in
                ["drv", "race", "cmp", "yr", "st", "num"]]

    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        model = MultiHeadEmbMLP(len(drv_levels), len(race_levels), len(cmp_levels),
                                 len(year_levels), n_stint, enc_tr["num"].shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        for epoch in range(EPOCHS):
            model.train()
            perm = np.random.RandomState(SEED + fold*100 + epoch).permutation(len(tr))
            for s in range(0, len(tr), BATCH):
                idx = tr[perm[s:s+BATCH]]
                drv, race, cmp, yr, st, num = to_tensor(idx, enc_tr)
                yb = torch.from_numpy(y[idx]).float()
                aux_b = torch.from_numpy(inv_y[idx])
                logits, reg = model(drv, race, cmp, yr, st, num)
                loss = bce(logits, yb) + AUX_W * mse(reg, aux_b)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            for s in range(0, len(va), BATCH):
                idx = va[s:s+BATCH]
                drv, race, cmp, yr, st, num = to_tensor(idx, enc_tr)
                logits, _ = model(drv, race, cmp, yr, st, num)
                oof[idx] = torch.sigmoid(logits).numpy().astype(np.float32)
            tp = np.zeros(len(test), dtype=np.float32)
            for s in range(0, len(test), BATCH):
                idx = np.arange(s, min(s+BATCH, len(test)))
                drv, race, cmp, yr, st, num = to_tensor(idx, enc_te)
                logits, _ = model(drv, race, cmp, yr, st, num)
                tp[idx] = torch.sigmoid(logits).numpy().astype(np.float32)
            test_pred += tp / N_FOLDS
        s = float(roc_auc_score(y[va], oof[va]))
        print(f"  fold {fold}: AUC {s:.5f} wall {time.time()-t_fold:.1f}s")

    auc = float(roc_auc_score(y, oof))
    rho, _ = spearmanr(test_pred, primary_test)
    print(f"\n=== multi_target_nn ===")
    print(f"  std OOF: {auc:.5f}  Δ vs PRIMARY {(auc-auc_primary)*1e4:+.2f} bp")
    print(f"  ρ vs PRIMARY: {rho:.6f}")
    np.save(ART / "oof_multi_target_nn_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_multi_target_nn_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))
    summary = dict(std_oof=auc, delta_vs_primary_bp=(auc-auc_primary)*1e4,
                   rho_vs_primary=float(rho), aux_weight=AUX_W,
                   wall_s=time.time() - t0)
    (ART / "probe_multi_target_nn.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ {ART / 'probe_multi_target_nn.json'} (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
