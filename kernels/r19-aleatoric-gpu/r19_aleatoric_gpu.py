"""R19 inventor track — aleatoric NN with mean + log-variance heads.

Probe 3 of inventor track. R17 listwise lifted (LB +0.10 bp); R18
multi-task underfit (-194 bp direct, -0.09 bp K=19 vs K=18). R19
tests UNCERTAINTY as the missing lever: same shared-trunk arch as R18
but instead of side tasks, predict (logit_mean, log_variance) per row
via Kendall & Gal 2017-style aleatoric heteroscedastic loss.

Architecture: shared trunk (3 hidden x 256 GELU+LN+drop), 2 heads:
  - head_mean    -> logit mu          (1 unit)
  - head_logvar  -> log sigma^2       (1 unit)

Training:
  sigma = exp(0.5 * logvar)
  logit_sample = mu + sigma * eps,  eps ~ N(0,1)        [reparam]
  L = BCEWithLogitsLoss(logit_sample, y)
        + 1e-3 * (logvar.mean())                        [soft var reg]

Inference outputs (per-row):
  prob_mean = sigmoid(mu)              -> oof_R19_aleatoric_mean_strat.npy
  sigma     = exp(0.5 * logvar)        -> oof_R19_aleatoric_sigma_strat.npy

Both used as orthogonal Path-B base columns
(K=20 = K=17 + R17 + R19_mean + R19_sigma).

Strat-only 5-fold; fold-safe TE rebuild per fold (Rule 24).
Per-fold checkpointing to /kaggle/working.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def gpu_boot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=10).strip()
        print(f"[boot] GPU info: {out}", flush=True)
    except Exception as e:
        print(f"[boot] nvidia-smi failed: {e}", flush=True)


def install_torch():
    print("[setup] force-reinstall torch 2.4 (T4 sm_75 support) ...",
          flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
    import importlib
    for m in ("torch", "torchvision"):
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    import torch
    print(f"[setup] torch version: {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"[setup] CUDA: {torch.version.cuda}, "
              f"dev: {torch.cuda.get_device_name(0)}, "
              f"cap: {torch.cuda.get_device_capability(0)}", flush=True)


def find_data_dir():
    base = Path("/kaggle/input")
    matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv under {base}")
    return matches[0].parent


def find_scripts_dir():
    base = Path("/kaggle/input")
    for cand in base.rglob("p1_features.py"):
        return cand.parent
    raise RuntimeError(f"no p1_features.py under {base}; "
                       f"ls /kaggle/input: {os.listdir(base)}")


def main():
    t0 = time.time()
    print("== R19 aleatoric NN (mean + logvar heads) ==", flush=True)
    gpu_boot()
    install_torch()

    scripts_dir = find_scripts_dir()
    sys.path.insert(0, str(scripts_dir))
    print(f"[setup] scripts at {scripts_dir}", flush=True)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] torch {torch.__version__} device={device}", flush=True)
    if device.type == "cuda":
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    from p1_features import (
        TE_CONFIGS, apply_fs_a, feature_columns_for_lgbm,
        fit_fs_a, make_features_static,
    )
    from p1_single_cb import fold_safe_te_for_fold

    class AleatoricMLP(nn.Module):
        def __init__(self, n_in, hidden=256, dropout=0.1):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_in, hidden), nn.GELU(),
                nn.LayerNorm(hidden), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.LayerNorm(hidden), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.LayerNorm(hidden), nn.Dropout(dropout),
            )
            self.head_mean = nn.Linear(hidden, 1)
            self.head_logvar = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.trunk(x)
            mu = self.head_mean(h).squeeze(-1)
            logvar = self.head_logvar(h).squeeze(-1)
            logvar = torch.clamp(logvar, min=-10.0, max=4.0)
            return mu, logvar

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"[data] train {train.shape}  test {test.shape}", flush=True)

    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True).values

    sorted_ids = train_S[ID_COL].values
    orig_train_ids = train[ID_COL].values
    id_to_sorted_pos = {tid: i for i, tid in enumerate(sorted_ids)}
    test_sorted_ids = test_S[ID_COL].values
    test_orig_ids = test[ID_COL].values
    test_id_to_sorted_pos = {tid: i for i, tid in enumerate(test_sorted_ids)}

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    sample_ti = fold_list[0][0]
    sample_fs_a = fit_fs_a(train_S.iloc[sample_ti])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in feats and c not in cat_cols:
            cat_cols.append(c)
    feats = feats + [n for _, _, n in TE_CONFIGS]
    print(f"[feat] {len(feats)} features, {len(cat_cols)} cat", flush=True)

    EPOCHS, LR, WD, BATCH = 20, 3e-4, 1e-4, 8192
    LOGVAR_REG = 1e-3
    N_MC_SAMPLES = 1

    oof_mean = np.zeros(len(y), dtype=np.float64)
    oof_sigma = np.zeros(len(y), dtype=np.float64)
    test_mean = np.zeros(len(test_S), dtype=np.float64)
    test_sigma = np.zeros(len(test_S), dtype=np.float64)
    fold_aucs, fold_walls = [], []

    for fold, (ti, vi) in enumerate(fold_list, 1):
        t_f = time.time()
        print(f"\n--- Fold {fold}/{N_FOLDS} | ti={len(ti)} va={len(vi)} ---",
              flush=True)

        fs_a = fit_fs_a(train_S.iloc[ti])
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)

        y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
        fold_safe_te_for_fold(train_ti, train_va, test_fold, y_ti,
                              fold, N_FOLDS)

        X_tr = train_ti.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        X_va = train_va.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        X_te = test_fold.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        mu_X = X_tr.mean(axis=0)
        sd_X = X_tr.std(axis=0) + 1e-6
        X_tr = (X_tr - mu_X) / sd_X
        X_va = (X_va - mu_X) / sd_X
        X_te = (X_te - mu_X) / sd_X

        y_tr = train_ti[TARGET].astype(np.float32).values
        y_va = train_va[TARGET].astype(int).values

        n_in = X_tr.shape[1]
        model = AleatoricMLP(n_in).to(device)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        bce = nn.BCEWithLogitsLoss()

        Xt = torch.tensor(X_tr, dtype=torch.float32)
        yt = torch.tensor(y_tr, dtype=torch.float32)
        ds = TensorDataset(Xt, yt)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=False)

        for ep in range(EPOCHS):
            model.train()
            tot = 0.0
            n = 0
            for batch in loader:
                xb, yb = batch
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                mu, logvar = model(xb)
                sigma = (0.5 * logvar).exp()
                losses = []
                for _ in range(N_MC_SAMPLES):
                    eps = torch.randn_like(mu)
                    logit_s = mu + sigma * eps
                    losses.append(bce(logit_s, yb))
                L_bce = torch.stack(losses).mean()
                L_reg = LOGVAR_REG * logvar.mean()
                loss = L_bce + L_reg
                loss.backward()
                opt.step()
                tot += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            sched.step()
            if (ep + 1) % 5 == 0 or ep == 0:
                model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
                    mu_v, lv_v = model(Xv)
                    prob_v = torch.sigmoid(mu_v).cpu().numpy()
                auc = float(roc_auc_score(y_va, prob_v))
                print(f"  fold {fold}/{N_FOLDS} ep {ep+1}/{EPOCHS} "
                      f"loss {tot/n:.5f} val AUC {auc:.5f} "
                      f"logvar[mean]={float(lv_v.mean()):+.3f}", flush=True)

        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
            mu_v, lv_v = model(Xv)
            prob_v = torch.sigmoid(mu_v).cpu().numpy()
            sigma_v = (0.5 * lv_v).exp().cpu().numpy()
            Xte_t = torch.tensor(X_te, dtype=torch.float32).to(device)
            mu_te, lv_te = model(Xte_t)
            prob_te = torch.sigmoid(mu_te).cpu().numpy()
            sigma_te = (0.5 * lv_te).exp().cpu().numpy()

        oof_mean[vi] = prob_v
        oof_sigma[vi] = sigma_v
        test_mean += prob_te / N_FOLDS
        test_sigma += sigma_te / N_FOLDS

        auc_va = float(roc_auc_score(y_va, prob_v))
        fold_aucs.append(auc_va)
        wall = time.time() - t_f
        fold_walls.append(wall)
        print(f"  fold {fold} wall {wall:.0f}s AUC={auc_va:.5f} "
              f"sigma[mean]={float(sigma_v.mean()):.3f}", flush=True)

        order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
        order_back_test = np.array([test_id_to_sorted_pos[t] for t in test_orig_ids])
        np.save(WORK / "oof_R19_aleatoric_mean_strat.npy",
                oof_mean[order_back_train].astype(np.float32))
        np.save(WORK / "oof_R19_aleatoric_sigma_strat.npy",
                oof_sigma[order_back_train].astype(np.float32))
        np.save(WORK / "test_R19_aleatoric_mean_strat.npy",
                test_mean[order_back_test].astype(np.float32))
        np.save(WORK / "test_R19_aleatoric_sigma_strat.npy",
                test_sigma[order_back_test].astype(np.float32))

        del model, opt, sched, loader, ds, Xt, yt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    auc_full = float(roc_auc_score(y, oof_mean))
    sigma_mean_overall = float(oof_sigma.mean())
    sigma_std_overall = float(oof_sigma.std())
    print(f"\nR19 mean-head OOF AUC: {auc_full:.6f}", flush=True)
    print(f"R19 sigma stats: mean={sigma_mean_overall:.4f} "
          f"std={sigma_std_overall:.4f}", flush=True)

    summary = dict(
        round="R19_aleatoric_NN_gpu",
        oof_auc_mean_head=auc_full,
        sigma_oof_mean=sigma_mean_overall,
        sigma_oof_std=sigma_std_overall,
        fold_aucs=fold_aucs,
        fold_walls_s=fold_walls,
        wall_total_s=time.time() - t0,
        epochs=EPOCHS, batch=BATCH, lr=LR, weight_decay=WD,
        logvar_reg=LOGVAR_REG, n_mc_samples=N_MC_SAMPLES,
        n_feats=len(feats),
    )
    (WORK / "r19_aleatoric_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTotal wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
