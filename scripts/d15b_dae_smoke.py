"""Stage-1 smoke for d15b DAE recipe.

Spot-check on 100k train + 50k test subsample. Friction: per
`single-base-fe-additions-noise-wall`, FE-class candidates need a 10-min
spot-check before full 5-fold compute.

Pipeline:
    1. Subsample 100k train (stratified) + 50k test rows.
    2. Encode features for DAE input:
         - High-card cat: target-mean encode (Driver, Race) -- train rows
           use OOF-style fold-mean (cheap: fixed split). Test rows use
           full-train mean.
         - Low-card cat: one-hot (Compound, Year).
         - Numerics: standardize.
    3. Train 3-layer DAE on (sub_train + sub_test) concat:
         input_dim -> 256 -> 512 -> 256 -> input_dim, ReLU, Adam lr=1e-3.
       Swap-noise: per batch, replace 15% of cells per row with values
       resampled from same column (in-batch).
       3 epochs, batch=512.
    4. Extract activations from middle (512) and last hidden (256), concat
       -> 768-d latent.
    5. Train LightGBM 200 rounds on (raw_subset + latent) for 1-fold
       (StratifiedKFold split, train 80k, val 20k).
    6. Compute val AUC and rho vs subsample PRIMARY OOF.

KILL: val AUC < 0.92 OR rho >= 0.998 -> STOP.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED = 42
PRIMARY_OOF_PATH = ART / "oof_d13e_compound_stint_tau20000_strat.npy"

DAE_HIDDEN = (256, 512, 256)
SWAP_NOISE_FRAC = 0.15
DAE_EPOCHS = 3
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
    """Smoothed target-mean encode (single-pass; for DAE input only)."""
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
    """Build the matrix that goes into the DAE.

    Returns:
        X_dae_train, X_dae_test  (float32 arrays)
        feature_names: list[str]
    """
    parts_train, parts_test, names = [], [], []

    # numerics: standardize on combined
    num_train = train_df[NUMERICS].astype(np.float32).values
    num_test = test_df[NUMERICS].astype(np.float32).values
    combined = np.vstack([num_train, num_test])
    mu = combined.mean(axis=0)
    sd = combined.std(axis=0) + 1e-6
    parts_train.append((num_train - mu) / sd)
    parts_test.append((num_test - mu) / sd)
    names += NUMERICS

    # high-card: target-mean (mean encoded once on full train; for DAE)
    for c in HIGH_CARD:
        et, ev = target_mean_encode(
            train_df[c].astype(str).values, y,
            test_df[c].astype(str).values,
        )
        # standardize the encoded col too
        all_e = np.concatenate([et, ev])
        mu_e, sd_e = all_e.mean(), all_e.std() + 1e-6
        parts_train.append(((et - mu_e) / sd_e).reshape(-1, 1))
        parts_test.append(((ev - mu_e) / sd_e).reshape(-1, 1))
        names.append(f"{c}_te")

    # low-card: one-hot
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
    """In-batch swap-noise: each cell with prob=frac replaced by another
    row's value in same column."""
    if rng is None:
        rng = np.random.default_rng()
    n, d = batch.shape
    mask = (rng.random((n, d)) < frac)
    # for each (i,j) where mask, swap with random row in same column
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
    X_t = torch.from_numpy(X_all).float()
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
        print(f"    epoch {ep} mse={epoch_loss/max(n_batches,1):.4f} wall={wall:.1f}s")
    return model


def extract_latent(model, X):
    """Extract concatenated h2 (512) + h3 (256) -> 768-d latent."""
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
    rng = np.random.default_rng(SEED)

    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y_full = train[TARGET].astype(int).values
    print(f"  Train {train.shape}  Test {test.shape}")

    # Stratified 100k subsample of train, 50k of test (random)
    n_sub_train = 100_000
    n_sub_test = 50_000
    skf = StratifiedKFold(n_splits=int(np.ceil(len(train) / n_sub_train)),
                          shuffle=True, random_state=SEED)
    _, sub_idx = next(skf.split(np.zeros(len(train)), y_full))
    sub_idx = np.sort(sub_idx[:n_sub_train])
    test_idx = rng.choice(len(test), size=n_sub_test, replace=False)
    test_idx.sort()

    sub_train = train.iloc[sub_idx].reset_index(drop=True)
    sub_test = test.iloc[test_idx].reset_index(drop=True)
    y_sub = y_full[sub_idx]
    print(f"  Subsample: train {sub_train.shape}, test {sub_test.shape}, prior {y_sub.mean():.4f}")

    # Encode for DAE
    X_dae_train, X_dae_test, dae_names = encode_for_dae(sub_train, sub_test, y_sub)
    print(f"  DAE input shape: train {X_dae_train.shape}, test {X_dae_test.shape}, "
          f"d={X_dae_train.shape[1]}")

    # Train DAE on concat
    X_all = np.vstack([X_dae_train, X_dae_test])
    t0 = time.time()
    model = train_dae(X_all, input_dim=X_all.shape[1])
    print(f"  DAE wall: {time.time()-t0:.1f}s")

    # Extract latent
    t0 = time.time()
    latent_train = extract_latent(model, X_dae_train)
    latent_test = extract_latent(model, X_dae_test)
    print(f"  Latent shape: train {latent_train.shape} test {latent_test.shape}, "
          f"extract wall {time.time()-t0:.1f}s")

    # Build LGBM input: raw_subset (numeric + label-encoded categoricals) + latent
    raw_train = sub_train[NUMERICS].astype(np.float32).copy()
    raw_test = sub_test[NUMERICS].astype(np.float32).copy()
    for c in HIGH_CARD + LOW_CARD:
        all_vals = pd.concat([sub_train[c], sub_test[c]], ignore_index=True
                             ).astype(str).unique()
        mp = {v: i for i, v in enumerate(sorted(all_vals))}
        raw_train[c] = sub_train[c].astype(str).map(mp).astype(np.int32).values
        raw_test[c] = sub_test[c].astype(str).map(mp).astype(np.int32).values

    cat_cols = HIGH_CARD + LOW_CARD
    cat_pos = [list(raw_train.columns).index(c) for c in cat_cols]
    raw_train_arr = raw_train.values.astype(np.float32)
    raw_test_arr = raw_test.values.astype(np.float32)
    Xfull_train = np.hstack([raw_train_arr, latent_train])
    Xfull_test = np.hstack([raw_test_arr, latent_test])
    print(f"  LGBM input: train {Xfull_train.shape} test {Xfull_test.shape}")

    # 1-fold split: 80k train, 20k val
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf2.split(np.zeros(len(y_sub)), y_sub))
    print(f"  Fold split: tr {len(tr_idx)} va {len(va_idx)}")

    lgb_train = lgb.Dataset(
        Xfull_train[tr_idx], y_sub[tr_idx],
        categorical_feature=cat_pos,
        free_raw_data=False,
    )
    lgb_val = lgb.Dataset(
        Xfull_train[va_idx], y_sub[va_idx],
        categorical_feature=cat_pos,
        reference=lgb_train,
        free_raw_data=False,
    )
    params = dict(
        objective="binary", metric="auc",
        num_leaves=63, learning_rate=0.05,
        min_child_samples=200, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1,
        verbose=-1, seed=SEED,
    )
    t0 = time.time()
    bst = lgb.train(
        params, lgb_train, num_boost_round=200,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    print(f"  LGBM wall: {time.time()-t0:.1f}s  best_iter={bst.best_iteration}")
    val_pred = bst.predict(Xfull_train[va_idx], num_iteration=bst.best_iteration)
    val_auc = float(roc_auc_score(y_sub[va_idx], val_pred))

    # rho vs subsample PRIMARY OOF
    primary_oof_full = np.load(PRIMARY_OOF_PATH)[:, 1]
    primary_oof_sub = primary_oof_full[sub_idx][va_idx]
    rho, _ = spearmanr(val_pred, primary_oof_sub)
    rho = float(rho)

    print(f"\n=== STAGE-1 RESULT ===")
    print(f"  val AUC: {val_auc:.5f}")
    print(f"  rho vs PRIMARY (subsample val): {rho:.6f}")
    print(f"  total wall: {time.time()-t_total:.1f}s")

    kill_auc = val_auc < 0.92
    kill_rho = rho >= 0.998
    if kill_auc or kill_rho:
        decision = "KILL"
        why = []
        if kill_auc:
            why.append(f"val AUC {val_auc:.4f} < 0.92")
        if kill_rho:
            why.append(f"rho {rho:.4f} >= 0.998")
        print(f"  DECISION: KILL ({'; '.join(why)})")
    else:
        decision = "PASS"
        print(f"  DECISION: PASS")

    return {
        "val_auc": val_auc, "rho": rho, "decision": decision,
        "wall_s": time.time() - t_total,
    }


if __name__ == "__main__":
    main()
