"""Day-15 Branch B — Swap-noise DAE + LGBM-on-latent (Kaggle GPU T4x2).

Jahrer Porto-Seguro recipe ported to F1 PitNextLap. Train a 3-hidden-layer
swap-noise denoising autoencoder unsupervised on (train+test) combined,
extract concatenated middle+last hidden activations as latent feature
matrix, then train LightGBM 5-fold on (raw + latent) and (latent only).
Both feed K=21 stack as new candidate bases.

CPU-side smoke (scripts/d15b_dae_smoke.py) ran 30 min on 70k rows × 782
features without finishing — full 627k × 20 ep + 439k 5-fold LGBM is
many hours on CPU. GPU shrinks DAE to ~7-10 min, LGBM still CPU-bound
~30-45 min. Total kernel wall ~45-75 min. Comfortable under Kaggle's 9h cap.

Why fine for GroupKF probe deferral:
  Strat is the LB proxy (R1, U3 i.i.d. test). Public LB outcome is what
  matters; GKF probe is R5 final-window concern. This kernel is Strat-only.

Outputs (under /kaggle/working/):
  oof_d15b_lgbm_dae_full_strat.npy   (n_train, 2)
  test_d15b_lgbm_dae_full_strat.npy  (n_test, 2)
  oof_d15b_lgbm_dae_only_strat.npy   (n_train, 2)
  test_d15b_lgbm_dae_only_strat.npy  (n_test, 2)
  d15b_dae_lgbm_gpu_results.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

# DAE config
DAE_HIDDEN = (256, 512, 256)
SWAP_NOISE_FRAC = 0.15
DAE_EPOCHS = 20
DAE_BATCH = 4096
LR = 1e-3

# LGBM config (CPU)
LGBM_PARAMS = dict(
    objective="binary", metric="auc",
    num_leaves=63, learning_rate=0.05,
    min_child_samples=200, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1,
    verbose=-1, seed=SEED,
)
LGBM_BOOST = 2000
LGBM_EARLY_STOP = 100

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
HIGH_CARD = ["Driver", "Race"]
LOW_CARD = ["Compound", "Year"]


def boot_env():
    """Verify GPU + install lightgbm if needed (Kaggle base image lacks it
    sometimes, depending on the GPU image version)."""
    print(f"[boot] Python {sys.version.split()[0]}, torch {torch.__version__}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"[boot] CUDA: {torch.version.cuda}, "
              f"device 0: {torch.cuda.get_device_name(0)}, capability: {cap}")
        if cap[0] < 7:
            print("[boot] WARNING: device capability < sm_70; PyTorch sm_60 (P100) "
                  "may not support current torch ops. T4 (sm_75) expected.")
    else:
        print("[boot] WARNING: CUDA not available; will run on CPU (slow)")
    try:
        import lightgbm  # noqa: F401
        print(f"[boot] lightgbm {lightgbm.__version__} present")
    except ImportError:
        print("[boot] installing lightgbm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])


def find_data_dir() -> Path:
    """rglob train.csv anywhere under /kaggle/input. Pattern from cb-slow-wide-gpu."""
    base = Path("/kaggle/input")
    if not base.exists():
        # local fallback for sanity testing
        local = Path("data")
        if (local / "train.csv").exists():
            print(f"[data] local fallback: {local}")
            return local
        raise RuntimeError(f"/kaggle/input missing; ls /kaggle: {os.listdir('/kaggle')}")
    matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv anywhere under {base}")
    train_path = matches[0]
    print(f"[data] found train.csv at {train_path}")
    return train_path.parent


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
    """Build the matrix that goes into the DAE. Returns float32 arrays + names."""
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
            train_df[c].astype(str).values, y, test_df[c].astype(str).values,
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


def swap_noise_torch(batch: torch.Tensor, frac: float = SWAP_NOISE_FRAC) -> torch.Tensor:
    """In-batch swap-noise on GPU. For each cell with prob=frac, replace with
    a random row's value in same column."""
    n, d = batch.shape
    mask = torch.rand_like(batch) < frac
    rand_rows = torch.randint(0, n, (n, d), device=batch.device)
    col_idx = torch.arange(d, device=batch.device).unsqueeze(0).expand(n, -1)
    swapped = batch[rand_rows, col_idx]
    return torch.where(mask, swapped, batch)


def train_dae(X_all: np.ndarray, input_dim: int, device, epochs=DAE_EPOCHS,
              batch=DAE_BATCH, hidden=DAE_HIDDEN, lr=LR, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DAE(input_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    n = X_all.shape[0]
    X_t = torch.from_numpy(X_all).float().to(device)
    print(f"[dae] train: n={n} d={input_dim} hidden={hidden} epochs={epochs} "
          f"batch={batch} device={device}")
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(n)
        idx_t = torch.from_numpy(idx).to(device)
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for s in range(0, n, batch):
            b = idx_t[s:s + batch]
            clean = X_t[b]
            noisy = swap_noise_torch(clean)
            recon, _, _, _ = model(noisy)
            loss = loss_fn(recon, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        wall = time.time() - t0
        print(f"  epoch {ep:2d} mse={epoch_loss/max(n_batches,1):.4f} wall={wall:.1f}s")
    return model


def extract_latent(model: DAE, X: np.ndarray, device) -> np.ndarray:
    """Concatenated h2 (512) + h3 (256) -> 768-d latent."""
    model.eval()
    out = []
    bs = 8192
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        for s in range(0, X.shape[0], bs):
            xt = X_t[s:s+bs]
            _, _, h2, h3 = model(xt)
            out.append(torch.cat([h2, h3], dim=1).cpu().numpy().astype(np.float32))
    return np.vstack(out)


def lgbm_5fold(X_train: np.ndarray, y: np.ndarray, X_test: np.ndarray,
               cat_pos: list[int], name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """5-fold StratifiedKFold OOF + averaged test preds."""
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test_avg = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs, fold_walls, fold_iters = [], [], []
    print(f"[lgbm] {name}: X_train {X_train.shape} X_test {X_test.shape}")
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        ds_tr = lgb.Dataset(X_train[tr], y[tr], categorical_feature=cat_pos,
                            free_raw_data=False)
        ds_va = lgb.Dataset(X_train[va], y[va], categorical_feature=cat_pos,
                            reference=ds_tr, free_raw_data=False)
        bst = lgb.train(
            LGBM_PARAMS, ds_tr, num_boost_round=LGBM_BOOST,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va] = bst.predict(X_train[va], num_iteration=bst.best_iteration)
        test_avg += bst.predict(X_test, num_iteration=bst.best_iteration) / N_FOLDS
        auc_fold = float(roc_auc_score(y[va], oof[va]))
        wall = time.time() - t0
        fold_aucs.append(auc_fold)
        fold_walls.append(wall)
        fold_iters.append(int(bst.best_iteration))
        print(f"  fold {k}: AUC={auc_fold:.5f} iters={bst.best_iteration} "
              f"wall={wall:.1f}s")
    auc_oof = float(roc_auc_score(y, oof))
    print(f"[lgbm] {name}: OOF AUC = {auc_oof:.5f}")
    return oof, test_avg, dict(
        oof_auc=auc_oof, fold_aucs=fold_aucs,
        fold_walls=fold_walls, fold_iters=fold_iters,
    )


def main():
    t_total = time.time()
    boot_env()
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    print(f"[main] writing outputs under {out_dir}")

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train[TARGET].astype(int).values
    print(f"[data] train {train.shape}, test {test.shape}, prior {y.mean():.4f}")

    X_dae_train, X_dae_test, dae_names = encode_for_dae(train, test, y)
    print(f"[encode] DAE input train {X_dae_train.shape} test {X_dae_test.shape} "
          f"d={X_dae_train.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train DAE on concatenated train + test (unsupervised)
    X_all = np.vstack([X_dae_train, X_dae_test])
    t0 = time.time()
    model = train_dae(X_all, input_dim=X_all.shape[1], device=device)
    dae_wall = time.time() - t0
    print(f"[dae] total wall: {dae_wall:.1f}s")

    # Extract latent for full train + test
    t0 = time.time()
    latent_train = extract_latent(model, X_dae_train, device)
    latent_test = extract_latent(model, X_dae_test, device)
    extract_wall = time.time() - t0
    print(f"[dae] latent train {latent_train.shape} test {latent_test.shape} "
          f"extract wall {extract_wall:.1f}s")

    # Save raw latents too (debuggable; small)
    np.save(out_dir / "d15b_dae_X_train_latent.npy", latent_train)
    np.save(out_dir / "d15b_dae_X_test_latent.npy", latent_test)

    # Build raw+cat feature matrix for LGBM (label-encode categoricals)
    raw_train = train[NUMERICS].astype(np.float32).copy()
    raw_test = test[NUMERICS].astype(np.float32).copy()
    cat_cols = HIGH_CARD + LOW_CARD
    for c in cat_cols:
        all_vals = pd.concat([train[c], test[c]], ignore_index=True
                              ).astype(str).unique()
        mp = {v: i for i, v in enumerate(sorted(all_vals))}
        raw_train[c] = train[c].astype(str).map(mp).astype(np.int32).values
        raw_test[c] = test[c].astype(str).map(mp).astype(np.int32).values

    raw_train_arr = raw_train.values.astype(np.float32)
    raw_test_arr = raw_test.values.astype(np.float32)
    cat_pos_full = [list(raw_train.columns).index(c) for c in cat_cols]
    Xfull_train = np.hstack([raw_train_arr, latent_train])
    Xfull_test = np.hstack([raw_test_arr, latent_test])

    # Variant A: raw + latent
    print("\n========= Variant A: raw + latent (d15b_lgbm_dae_full) =========")
    oof_full, test_full, res_full = lgbm_5fold(
        Xfull_train, y, Xfull_test, cat_pos=cat_pos_full,
        name="d15b_lgbm_dae_full",
    )

    # Variant B: latent only
    print("\n========= Variant B: latent only (d15b_lgbm_dae_only) =========")
    oof_only, test_only, res_only = lgbm_5fold(
        latent_train, y, latent_test, cat_pos=[],
        name="d15b_lgbm_dae_only",
    )

    # Save 2-col canonical artifacts
    for name, oof, test_pred in [
        ("d15b_lgbm_dae_full", oof_full, test_full),
        ("d15b_lgbm_dae_only", oof_only, test_only),
    ]:
        np.save(out_dir / f"oof_{name}_strat.npy",
                np.column_stack([1 - oof, oof]))
        np.save(out_dir / f"test_{name}_strat.npy",
                np.column_stack([1 - test_pred, test_pred]))
        print(f"[save] oof_{name}_strat.npy + test_{name}_strat.npy")

    results = dict(
        n_train=int(len(train)), n_test=int(len(test)),
        dae_input_dim=int(X_all.shape[1]),
        latent_dim=int(latent_train.shape[1]),
        dae_epochs=DAE_EPOCHS, dae_batch=DAE_BATCH,
        dae_hidden=list(DAE_HIDDEN), swap_noise_frac=SWAP_NOISE_FRAC,
        dae_wall_s=float(dae_wall), latent_extract_wall_s=float(extract_wall),
        variants=dict(d15b_lgbm_dae_full=res_full,
                      d15b_lgbm_dae_only=res_only),
        total_wall_s=float(time.time() - t_total),
    )
    (out_dir / "d15b_dae_lgbm_gpu_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n[done] total wall {results['total_wall_s']:.0f}s")
    print(f"[done] OOF full: {res_full['oof_auc']:.5f}  "
          f"only: {res_only['oof_auc']:.5f}")
    print(f"[done] results JSON: d15b_dae_lgbm_gpu_results.json")


if __name__ == "__main__":
    main()
