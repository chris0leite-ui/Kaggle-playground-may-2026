"""Stage-2A: train DAE on full (train+test), save 768-d latent.

Architecture: input_dim -> 256 -> 512 -> 256 -> input_dim, ReLU, Adam lr=1e-3
Swap-noise: 15% per cell, in-batch column resample.
20 epochs, batch=512.

Latent extraction: concat h2 (512) + h3 (256) -> 768-d.
Save:
    scripts/artifacts/d15b_dae_X_train_latent.npy   (439140, 768)
    scripts/artifacts/d15b_dae_X_test_latent.npy    (188165, 768)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED = 42

DAE_HIDDEN = (256, 512, 256)
SWAP_NOISE_FRAC = 0.15
DAE_EPOCHS = 20
DAE_BATCH = 512
LR = 1e-3

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
HIGH_CARD = ["Driver", "Race"]
LOW_CARD = ["Compound", "Year"]


def target_mean_encode(s_train, y_train, s_test, alpha=20.0):
    glob = float(np.mean(y_train))
    df = pd.DataFrame({"k": s_train, "y": y_train})
    grp = df.groupby("k")["y"]
    counts = grp.count()
    means = grp.mean()
    smoothed = (means * counts + glob * alpha) / (counts + alpha)
    mp = smoothed.to_dict()
    enc_train = pd.Series(s_train).map(mp).fillna(glob).values.astype(np.float32)
    enc_test = pd.Series(s_test).map(mp).fillna(glob).values.astype(np.float32)
    return enc_train, enc_test


def encode_for_dae(train_df, test_df, y):
    parts_train, parts_test, names = [], [], []
    num_train = train_df[NUMERICS].astype(np.float32).values
    num_test = test_df[NUMERICS].astype(np.float32).values
    combined = np.vstack([num_train, num_test])
    mu = combined.mean(axis=0)
    sd = combined.std(axis=0) + 1e-6
    parts_train.append((num_train - mu) / sd)
    parts_test.append((num_test - mu) / sd)
    names += NUMERICS
    for c in HIGH_CARD:
        et, ev = target_mean_encode(
            train_df[c].astype(str).values, y,
            test_df[c].astype(str).values,
        )
        all_e = np.concatenate([et, ev])
        mu_e, sd_e = all_e.mean(), all_e.std() + 1e-6
        parts_train.append(((et - mu_e) / sd_e).reshape(-1, 1))
        parts_test.append(((ev - mu_e) / sd_e).reshape(-1, 1))
        names.append(f"{c}_te")
    for c in LOW_CARD:
        cats = sorted(set(train_df[c].astype(str).unique()) |
                      set(test_df[c].astype(str).unique()))
        for cat in cats:
            tr_col = (train_df[c].astype(str).values == cat).astype(np.float32)
            te_col = (test_df[c].astype(str).values == cat).astype(np.float32)
            parts_train.append(tr_col.reshape(-1, 1))
            parts_test.append(te_col.reshape(-1, 1))
            names.append(f"{c}={cat}")
    X_train = np.hstack(parts_train).astype(np.float32)
    X_test = np.hstack(parts_test).astype(np.float32)
    return X_train, X_test, names


class DAE(nn.Module):
    def __init__(self, input_dim, hidden=DAE_HIDDEN):
        super().__init__()
        h1, h2, h3 = hidden
        self.enc1 = nn.Linear(input_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.enc3 = nn.Linear(h2, h3)
        self.dec = nn.Linear(h3, input_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        h1 = self.act(self.enc1(x))
        h2 = self.act(self.enc2(h1))
        h3 = self.act(self.enc3(h2))
        out = self.dec(h3)
        return out, h1, h2, h3


def swap_noise_batch(batch, frac=SWAP_NOISE_FRAC, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n, d = batch.shape
    mask = (rng.random((n, d)) < frac)
    out = batch.copy()
    rows = rng.integers(0, n, size=(n, d))
    out[mask] = batch[rows[mask], np.tile(np.arange(d), (n, 1))[mask]]
    return out


def train_dae(X_all, input_dim, epochs=DAE_EPOCHS, batch=DAE_BATCH,
              hidden=DAE_HIDDEN, lr=LR, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DAE(input_dim, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    n = X_all.shape[0]
    print(f"  DAE train: n={n} d={input_dim} hidden={hidden} epochs={epochs} batch={batch}")
    for ep in range(epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            clean = X_all[b]
            noisy = swap_noise_batch(clean, rng=rng)
            x_clean = torch.from_numpy(clean).float()
            x_noisy = torch.from_numpy(noisy).float()
            recon, _, _, _ = model(x_noisy)
            loss = loss_fn(recon, x_clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        wall = time.time() - t0
        print(f"    epoch {ep:02d} mse={epoch_loss/max(n_batches,1):.5f} wall={wall:.1f}s", flush=True)
    return model


def extract_latent(model, X):
    model.eval()
    out = []
    bs = 4096
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            xt = torch.from_numpy(X[s:s+bs]).float()
            _, _, h2, h3 = model(xt)
            out.append(np.hstack([h2.numpy(), h3.numpy()]).astype(np.float32))
    return np.vstack(out)


def main():
    t_total = time.time()
    print("Loading data...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"  Train {train.shape}  Test {test.shape}", flush=True)

    X_train, X_test, names = encode_for_dae(train, test, y)
    print(f"  DAE input: train {X_train.shape}  test {X_test.shape}  d={X_train.shape[1]}", flush=True)

    X_all = np.vstack([X_train, X_test])
    t0 = time.time()
    model = train_dae(X_all, input_dim=X_all.shape[1])
    print(f"  DAE wall: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    latent_train = extract_latent(model, X_train)
    latent_test = extract_latent(model, X_test)
    print(f"  Latent: train {latent_train.shape} test {latent_test.shape}  "
          f"extract wall {time.time()-t0:.1f}s", flush=True)

    np.save(ART / "d15b_dae_X_train_latent.npy", latent_train)
    np.save(ART / "d15b_dae_X_test_latent.npy", latent_test)
    print(f"  Saved -> {ART}/d15b_dae_X_{{train,test}}_latent.npy", flush=True)
    print(f"  Total wall: {time.time()-t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
