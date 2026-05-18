"""Round-7 Phase A — swap-noise denoising autoencoder.

Porto Seguro 1st-place precedent (Michael Jahrer 2017). Untried
embedding-class diversity orthogonal to the row-FE + sequence-HMM +
attention-transformer classes already tested. Predicted EV +0.3-0.5
bp at K=14+Path-B if encoder embeddings capture latent feature
structure that LightGBM splits can't directly access.

Architecture:
- Input: 14 raw numerics + 9 R4 segment-interaction features = 23 dim
  standardized.
- Encoder: 3-layer MLP [23 → 256 → 256 → 128 bottleneck] ReLU + dropout.
- Decoder: mirror [128 → 256 → 256 → 23].
- Swap-noise augmentation: 15% feature values per row replaced with
  values from OTHER rows in the same column (training only).
- Pretraining: MSE reconstruction loss on train + test combined
  (AV-AUC 0.502 → safe transductive).
- Downstream: LightGBM 5-fold StratifiedKFold OOF on
  [raw 14 + encoder 128 = 142 dim].

Outputs:
- oof_r7_swapnoise_dae_strat.npy
- test_r7_swapnoise_dae_strat.npy
- r7_swapnoise_dae_results.json
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import time
import os
from pathlib import Path

# Force-reinstall torch 2.4 for sm_60 (P100) support. Same pattern
# as r6-transformer-v2-gpu. Requires enable_internet: true in
# kernel-metadata.json.
print("[boot] force-reinstall torch 2.4 (sm_60 P100 support) ...", flush=True)
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
except subprocess.CalledProcessError as e:
    print(f"[boot] torch 2.4 reinstall failed (continuing with default): {e}",
          flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

print(f"[boot] Python {sys.version.split()[0]}, torch {torch.__version__}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"[boot] CUDA: {torch.version.cuda}, "
          f"device 0: {torch.cuda.get_device_name(0)}, capability: {cap}")

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

# DAE hyperparams
BOTTLENECK = 128
HIDDEN = 256
N_LAYERS_ENC = 3
DROPOUT = 0.2
SWAP_FRAC = 0.15
DAE_EPOCHS = 50
DAE_LR = 1e-3
DAE_WD = 1e-5
DAE_BATCH = 1024

# Downstream LGBM
LGB_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    lambda_l2=1.0, max_depth=-1, n_jobs=-1, verbose=-1,
    random_state=SEED,
)
MAX_ROUNDS = 2000

RAW_NUMERICS = ["TyreLife", "RaceProgress", "Stint", "Position_Change",
                "Cumulative_Degradation", "LapTime (s)", "LapTime_Delta",
                "LapNumber", "Position"]
COMPOUND_VOCAB = {"HARD": 0, "MEDIUM": 1, "SOFT": 2, "INTERMEDIATE": 3, "WET": 4}


def _is_named(driver_series: pd.Series) -> pd.Series:
    return ~driver_series.astype(str).str.match(r"^D\d{3}$").fillna(False)


def add_seg_features(df: pd.DataFrame) -> pd.DataFrame:
    """R4 segment-interaction features. Pure functions of non-label cols."""
    df = df.copy()
    is_inter = (df["Compound"] == "INTERMEDIATE").astype(np.float32)
    is_wet = (df["Compound"] == "WET").astype(np.float32)
    is_hard = (df["Compound"] == "HARD").astype(np.float32)
    is_med = (df["Compound"] == "MEDIUM").astype(np.float32)
    is_soft = (df["Compound"] == "SOFT").astype(np.float32)
    named = _is_named(df["Driver"]).astype(np.float32)
    stint1 = (df["Stint"] == 1).astype(np.float32)
    stint2 = (df["Stint"] == 2).astype(np.float32)
    df["cumdeg_inter"] = df["Cumulative_Degradation"] * is_inter
    df["cumdeg_wet"] = df["Cumulative_Degradation"] * is_wet
    df["cumdeg_hard"] = df["Cumulative_Degradation"] * is_hard
    df["cumdeg_medium"] = df["Cumulative_Degradation"] * is_med
    df["cumdeg_soft"] = df["Cumulative_Degradation"] * is_soft
    df["poschg_named"] = df["Position_Change"] * named
    df["is_named"] = named
    df["is_wet_s1"] = is_wet * stint1
    df["is_inter_s2"] = is_inter * stint2
    return df


class DAE(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # Encoder
        layers, in_d = [], n_features
        for _ in range(N_LAYERS_ENC - 1):
            layers += [nn.Linear(in_d, HIDDEN), nn.ReLU(),
                       nn.Dropout(DROPOUT)]
            in_d = HIDDEN
        layers += [nn.Linear(in_d, BOTTLENECK), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        # Decoder
        layers, in_d = [], BOTTLENECK
        for _ in range(N_LAYERS_ENC - 1):
            layers += [nn.Linear(in_d, HIDDEN), nn.ReLU(),
                       nn.Dropout(DROPOUT)]
            in_d = HIDDEN
        layers += [nn.Linear(in_d, n_features)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def swap_noise(batch, frac=SWAP_FRAC):
    """For each (row, col), with probability `frac` replace value with
    value from another randomly-chosen row in same column."""
    n, d = batch.shape
    mask = (torch.rand(n, d, device=batch.device) < frac)
    src = torch.randint(0, n, (n, d), device=batch.device)
    swapped = batch.clone()
    for c in range(d):
        m_col = mask[:, c]
        if m_col.any():
            swapped[m_col, c] = batch[src[m_col, c], c]
    return swapped


def main():
    t0 = time.time()
    print(f"=== R7 Phase A: swap-noise DAE ===", flush=True)

    def find_data(name):
        base = Path("/kaggle/input")
        if base.exists():
            matches = list(base.rglob(name))
            if matches:
                return matches[0].parent
        return Path("data")
    DATA_DIR = find_data("train.csv")
    print(f"  DATA_DIR: {DATA_DIR}", flush=True)
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    y_all = train[TARGET].astype(int).values
    print(f"  train {train.shape}  test {test.shape}", flush=True)

    train = add_seg_features(train)
    test = add_seg_features(test)
    # Add compound int + categorical int encodings
    train["Compound_int"] = train["Compound"].map(COMPOUND_VOCAB).astype(np.float32)
    test["Compound_int"] = test["Compound"].map(COMPOUND_VOCAB).astype(np.float32)

    SEG_FEATS = ["cumdeg_inter", "cumdeg_wet", "cumdeg_hard", "cumdeg_medium",
                 "cumdeg_soft", "poschg_named", "is_named", "is_wet_s1",
                 "is_inter_s2"]
    DAE_INPUTS = RAW_NUMERICS + ["Compound_int"] + SEG_FEATS
    print(f"  DAE inputs ({len(DAE_INPUTS)} dim): {DAE_INPUTS}", flush=True)

    # Standardize
    means = train[DAE_INPUTS].mean()
    stds = train[DAE_INPUTS].std() + 1e-6
    X_tr = ((train[DAE_INPUTS] - means) / stds).fillna(0).values.astype(np.float32)
    X_te = ((test[DAE_INPUTS] - means) / stds).fillna(0).values.astype(np.float32)
    X_all = np.concatenate([X_tr, X_te], axis=0)
    print(f"  DAE pretraining input: train+test = {X_all.shape}", flush=True)

    # DAE training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}", flush=True)
    n_feat = X_all.shape[1]
    model = DAE(n_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=DAE_LR, weight_decay=DAE_WD)
    n_steps = (len(X_all) + DAE_BATCH - 1) // DAE_BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DAE_EPOCHS * n_steps)
    X_all_t = torch.tensor(X_all, device=device)
    print(f"  DAE pretraining {DAE_EPOCHS} epochs × {n_steps} steps ...", flush=True)
    for epoch in range(DAE_EPOCHS):
        t_ep = time.time()
        perm = torch.randperm(len(X_all_t))
        ep_loss = 0.0
        for i in range(0, len(X_all_t), DAE_BATCH):
            batch = X_all_t[perm[i:i + DAE_BATCH]]
            noisy = swap_noise(batch, SWAP_FRAC)
            recon, _ = model(noisy)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item() * len(batch)
        avg = ep_loss / len(X_all_t)
        if epoch % 5 == 0 or epoch == DAE_EPOCHS - 1:
            print(f"    epoch {epoch+1}/{DAE_EPOCHS}  loss={avg:.5f}  "
                  f"wall={time.time()-t_ep:.1f}s", flush=True)

    # Extract embeddings
    print(f"  extracting embeddings...", flush=True)
    model.eval()
    embs_tr = np.zeros((len(X_tr), BOTTLENECK), dtype=np.float32)
    embs_te = np.zeros((len(X_te), BOTTLENECK), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_tr), DAE_BATCH):
            b = torch.tensor(X_tr[i:i + DAE_BATCH], device=device)
            _, z = model(b)
            embs_tr[i:i + DAE_BATCH] = z.cpu().numpy()
        for i in range(0, len(X_te), DAE_BATCH):
            b = torch.tensor(X_te[i:i + DAE_BATCH], device=device)
            _, z = model(b)
            embs_te[i:i + DAE_BATCH] = z.cpu().numpy()

    # Downstream LGBM: raw 14 + segment-interaction 9 + DAE 128 = 151 dim
    raw14 = ["Driver", "Compound", "Race", "Year", "PitStop", "LapNumber",
             "Stint", "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
             "Cumulative_Degradation", "RaceProgress", "Position_Change"]
    # Label-encode cats
    cat_cols = ["Driver", "Compound", "Race"]
    for c in cat_cols:
        u = pd.concat([train[c], test[c]], ignore_index=True).unique()
        m = {v: i for i, v in enumerate(u)}
        train[c] = train[c].map(m).astype(np.int32)
        test[c] = test[c].map(m).astype(np.int32)

    emb_cols = [f"dae_{i}" for i in range(BOTTLENECK)]
    for i, c in enumerate(emb_cols):
        train[c] = embs_tr[:, i]
        test[c] = embs_te[:, i]
    feat_cols = raw14 + SEG_FEATS + emb_cols
    print(f"  downstream LGBM features: {len(feat_cols)}", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []
    for fold, (ti, vi) in enumerate(skf.split(np.zeros(len(y_all)), y_all), 1):
        t_f = time.time()
        X_tr_f = train.iloc[ti][feat_cols].fillna(0).values
        X_va_f = train.iloc[vi][feat_cols].fillna(0).values
        X_te_f = test[feat_cols].fillna(0).values
        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=MAX_ROUNDS)
        m.fit(X_tr_f, y_all[ti],
              eval_set=[(X_va_f, y_all[vi])],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])
        oof_va = m.predict_proba(X_va_f)[:, 1]
        oof[vi] = oof_va
        test_pred += m.predict_proba(X_te_f)[:, 1] / N_FOLDS
        fold_aucs.append(float(roc_auc_score(y_all[vi], oof_va)))
        print(f"  fold {fold}: AUC={fold_aucs[-1]:.5f}  "
              f"iters={m.best_iteration_}  wall={time.time()-t_f:.1f}s",
              flush=True)

    auc_full = float(roc_auc_score(y_all, oof))
    print(f"\n  Full OOF AUC: {auc_full:.5f}  "
          f"fold-std={np.std(fold_aucs):.5f}  "
          f"total wall={time.time()-t0:.1f}s", flush=True)

    # Save
    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "scripts/artifacts"
    np.save(f"{out_dir}/oof_r7_swapnoise_dae_strat.npy",
            np.column_stack([1 - oof, oof]).astype(np.float64))
    np.save(f"{out_dir}/test_r7_swapnoise_dae_strat.npy",
            np.column_stack([1 - test_pred, test_pred]).astype(np.float64))
    with open(f"{out_dir}/r7_swapnoise_dae_results.json", "w") as f:
        json.dump(dict(name="r7_swapnoise_dae", oof_auc=auc_full,
                       fold_aucs=fold_aucs,
                       dae_inputs=DAE_INPUTS, bottleneck=BOTTLENECK,
                       dae_epochs=DAE_EPOCHS, swap_frac=SWAP_FRAC), f, indent=2)
    print(f"  -> {out_dir}/oof+test_r7_swapnoise_dae_strat.npy", flush=True)


if __name__ == "__main__":
    main()
