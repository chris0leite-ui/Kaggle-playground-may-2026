"""scripts/probe_nn_embeddings.py — small NN with embedding layers.

Synthetic-data lens: a small NN with learned embeddings for
high-cardinality categoricals (Driver, Race, Compound, Year, Stint)
+ numeric features → MLP → sigmoid. CPU-only. No focal loss
(SOSTA's relevance unclear without raw lap times). Standard BCE.

Architecture:
  Driver   embed(887, 8)
  Race     embed(26, 4)
  Compound embed(5, 3)
  Year     embed(4, 3)
  Stint    embed(8, 3)
  numerics: TyreLife, RaceProgress, LapTime_Delta, Cumulative_Degradation,
            Position, LapTime, Position_Change → 7 dims, BatchNorm
  concat → 8+4+3+3+3+7 = 28 dim → Linear(64) → ReLU → Dropout(0.2)
            → Linear(32) → ReLU → Linear(1)
  Loss: BCE
  5-fold StratKF, 6 epochs, batch 4096, AdamW lr=2e-3 wd=1e-4

Saves OOF + test under oof_nn_emb_lgbm_strat.npy / test_*.npy
(file naming kept consistent with rest of pool).
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

torch.manual_seed(SEED)
np.random.seed(SEED)


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


class EmbMLP(nn.Module):
    def __init__(self, n_drv, n_race, n_cmp, n_year, n_stint, n_num):
        super().__init__()
        self.e_drv = nn.Embedding(n_drv, 8)
        self.e_race = nn.Embedding(n_race, 4)
        self.e_cmp = nn.Embedding(n_cmp, 3)
        self.e_year = nn.Embedding(n_year, 3)
        self.e_stint = nn.Embedding(n_stint, 3)
        self.bn = nn.BatchNorm1d(n_num)
        d = 8 + 4 + 3 + 3 + 3 + n_num
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, drv, race, cmp, yr, st, num):
        x = torch.cat([self.e_drv(drv), self.e_race(race), self.e_cmp(cmp),
                       self.e_year(yr), self.e_stint(st), self.bn(num)], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    # Encode categoricals
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
    n_stint_levels = 8

    def encode(df):
        return dict(
            drv=df["Driver"].astype(str).map(drv_map).astype(np.int64).values,
            race=df["Race"].astype(str).map(race_map).astype(np.int64).values,
            cmp=df["Compound"].astype(str).map(cmp_map).astype(np.int64).values,
            yr=df["Year"].astype(int).map(year_map).astype(np.int64).values,
            st=np.clip(df["Stint"].astype(int).values, 0, n_stint_levels-1).astype(np.int64),
            num=df[["TyreLife", "RaceProgress", "LapTime_Delta",
                     "Cumulative_Degradation", "Position",
                     "LapTime (s)", "Position_Change"]].values.astype(np.float32),
        )

    enc_tr = encode(train)
    enc_te = encode(test)

    # Numeric standardization (using train stats only)
    num_mean = enc_tr["num"].mean(axis=0)
    num_std = enc_tr["num"].std(axis=0) + 1e-6
    enc_tr["num"] = (enc_tr["num"] - num_mean) / num_std
    enc_te["num"] = (enc_te["num"] - num_mean) / num_std

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    BATCH = 4096
    EPOCHS = 6

    for fold, (tr, va) in enumerate(splits):
        t_fold = time.time()
        model = EmbMLP(len(drv_levels), len(race_levels), len(cmp_levels),
                       len(year_levels), n_stint_levels, enc_tr["num"].shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        # Tensors
        def to_tensor(idx, src):
            return [torch.from_numpy(src[k][idx]) for k in
                    ["drv", "race", "cmp", "yr", "st", "num"]]

        for epoch in range(EPOCHS):
            model.train()
            perm = np.random.RandomState(SEED + fold * 100 + epoch).permutation(len(tr))
            for s in range(0, len(tr), BATCH):
                idx = tr[perm[s:s+BATCH]]
                drv, race, cmp, yr, st, num = to_tensor(idx, enc_tr)
                yb = torch.from_numpy(y[idx]).float()
                logits = model(drv, race, cmp, yr, st, num)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            # Val OOF
            for s in range(0, len(va), BATCH):
                idx = va[s:s+BATCH]
                drv, race, cmp, yr, st, num = to_tensor(idx, enc_tr)
                logits = model(drv, race, cmp, yr, st, num)
                oof[idx] = torch.sigmoid(logits).numpy().astype(np.float32)
            # Test
            tp = np.zeros(len(test), dtype=np.float32)
            for s in range(0, len(test), BATCH):
                idx = np.arange(s, min(s+BATCH, len(test)))
                drv, race, cmp, yr, st, num = to_tensor(idx, enc_te)
                logits = model(drv, race, cmp, yr, st, num)
                tp[idx] = torch.sigmoid(logits).numpy().astype(np.float32)
            test_pred += tp / N_FOLDS

        s = float(roc_auc_score(y[va], oof[va]))
        print(f"  fold {fold}: AUC {s:.5f} wall {time.time()-t_fold:.1f}s")

    auc = float(roc_auc_score(y, oof))
    primary_test = _pos(PRIMARY_TEST)
    rho, _ = spearmanr(test_pred, primary_test)
    primary_oof = _pos(PRIMARY_OOF)
    auc_primary = float(roc_auc_score(y, primary_oof))
    print(f"\n=== nn_embeddings base ===")
    print(f"  std OOF: {auc:.5f}  Δ vs PRIMARY {(auc-auc_primary)*1e4:+.2f} bp")
    print(f"  ρ vs PRIMARY: {rho:.6f}")

    np.save(ART / "oof_nn_embeddings_strat.npy",
            np.column_stack([1 - oof, oof]))
    np.save(ART / "test_nn_embeddings_strat.npy",
            np.column_stack([1 - test_pred, test_pred]))
    summary = dict(
        std_oof=auc, delta_vs_primary_bp=(auc - auc_primary)*1e4,
        rho_vs_primary=float(rho),
        wall_s=time.time() - t0,
    )
    (ART / "probe_nn_embeddings.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/probe_nn_embeddings.json (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
