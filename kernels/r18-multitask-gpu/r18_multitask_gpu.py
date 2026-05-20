"""R18 inventor track — multi-task joint NN, Kaggle GPU port.

Same hypothesis + architecture as scripts/probe_r18_multitask_joint.py
(local CPU run kept dying mid-fold from container preempts; ported to
Kaggle T4x2 for deterministic completion).

Architecture: PyTorch MLP, shared trunk (3 hidden × 256 GELU+LN+drop),
4 task heads:
  - PitNextLap (1u, BCE, weight 1.0)         <- primary
  - LapsUntilPit (1u, MSE, weight 0.3)
  - StintCompletion (1u, MSE, weight 0.3)
  - NextCompound (5u, CE masked, weight 0.3)

Strat-only (5-fold). Fold-safe TE rebuild per fold (Rule 24).

Outputs (/kaggle/working/):
  oof_R18_multitask_pit_head_strat.npy   (n_train,) in original train.csv order
  test_R18_multitask_pit_head_strat.npy  (n_test,)  in original test.csv order
  r18_multitask_results.json
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
N_COMPOUND = 5

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
    print("== R18 multi-task NN (Kaggle GPU port) ==", flush=True)
    gpu_boot()

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
    from b_laps_until_pit import build_laps_until_pit
    from probe_r13_cb_stint_completion import build_stint_completion
    from probe_r14_cb_next_compound import build_next_compound_target

    class MultiTaskMLP(nn.Module):
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
            self.head_pit = nn.Linear(hidden, 1)
            self.head_laps = nn.Linear(hidden, 1)
            self.head_stint = nn.Linear(hidden, 1)
            self.head_compound = nn.Linear(hidden, N_COMPOUND)

        def forward(self, x):
            h = self.trunk(x)
            return (self.head_pit(h).squeeze(-1),
                    self.head_laps(h).squeeze(-1),
                    self.head_stint(h).squeeze(-1),
                    self.head_compound(h))

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

    oof_pit = np.zeros(len(y), dtype=np.float64)
    test_pit = np.zeros(len(test_S), dtype=np.float64)
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

        ti_df = train_S.iloc[ti].reset_index(drop=True)
        y_laps_ti = build_laps_until_pit(ti_df).astype(np.float32)
        y_stint_ti = build_stint_completion(ti_df).astype(np.float32)
        y_comp_ti_raw = build_next_compound_target(ti_df)
        comp_mask_ti = (y_comp_ti_raw >= 0).astype(np.float32)
        y_comp_ti = np.where(y_comp_ti_raw >= 0, y_comp_ti_raw,
                             0).astype(np.int64)
        laps_mu, laps_sd = float(y_laps_ti.mean()), float(y_laps_ti.std() + 1e-6)
        y_laps_ti = (y_laps_ti - laps_mu) / laps_sd
        stint_mu, stint_sd = float(y_stint_ti.mean()), float(y_stint_ti.std() + 1e-6)
        y_stint_ti = (y_stint_ti - stint_mu) / stint_sd

        X_tr = train_ti.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        X_va = train_va.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        X_te = test_fold.reindex(columns=feats, fill_value=0).fillna(0).astype(np.float32).values
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-6
        X_tr = (X_tr - mu) / sd
        X_va = (X_va - mu) / sd
        X_te = (X_te - mu) / sd

        y_pit_ti = train_ti[TARGET].astype(np.float32).values
        y_pit_va = train_va[TARGET].astype(int).values

        n_in = X_tr.shape[1]
        model = MultiTaskMLP(n_in).to(device)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss(reduction="none")

        Xt = torch.tensor(X_tr, dtype=torch.float32)
        y_pit_t = torch.tensor(y_pit_ti, dtype=torch.float32)
        y_laps_t = torch.tensor(y_laps_ti, dtype=torch.float32)
        y_stint_t = torch.tensor(y_stint_ti, dtype=torch.float32)
        y_comp_t = torch.tensor(y_comp_ti, dtype=torch.long)
        mask_t = torch.tensor(comp_mask_ti, dtype=torch.float32)
        ds = TensorDataset(Xt, y_pit_t, y_laps_t, y_stint_t, y_comp_t, mask_t)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=False)

        for ep in range(EPOCHS):
            model.train()
            tot = 0.0
            n = 0
            for batch in loader:
                xb, yp, yl, ys, yc, m = batch
                xb = xb.to(device, non_blocking=True)
                yp = yp.to(device, non_blocking=True)
                yl = yl.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)
                yc = yc.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                opt.zero_grad()
                p_pit, p_laps, p_stint, p_comp = model(xb)
                L_pit = bce(p_pit, yp)
                L_laps = mse(p_laps, yl)
                L_stint = mse(p_stint, ys)
                ce_per = ce(p_comp, yc)
                L_comp = (ce_per * m).sum() / (m.sum() + 1e-6)
                loss = L_pit + 0.3 * (L_laps + L_stint + L_comp)
                loss.backward()
                opt.step()
                tot += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            sched.step()
            if (ep + 1) % 5 == 0 or ep == 0:
                model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
                    p_pit_va_t, _, _, _ = model(Xv)
                    p_pit_va_pr = torch.sigmoid(p_pit_va_t).cpu().numpy()
                auc = float(roc_auc_score(y_pit_va, p_pit_va_pr))
                print(f"  fold {fold}/{N_FOLDS} ep {ep+1}/{EPOCHS} "
                      f"loss {tot/n:.5f} val AUC {auc:.5f}", flush=True)

        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
            p_pit_va_t, _, _, _ = model(Xv)
            p_pit_va_pr = torch.sigmoid(p_pit_va_t).cpu().numpy()
            Xte_t = torch.tensor(X_te, dtype=torch.float32).to(device)
            p_pit_te_t, _, _, _ = model(Xte_t)
            p_pit_te_pr = torch.sigmoid(p_pit_te_t).cpu().numpy()

        oof_pit[vi] = p_pit_va_pr
        test_pit += p_pit_te_pr / N_FOLDS

        auc_va = float(roc_auc_score(y_pit_va, p_pit_va_pr))
        fold_aucs.append(auc_va)
        wall = time.time() - t_f
        fold_walls.append(wall)
        print(f"  fold {fold} wall {wall:.0f}s AUC={auc_va:.5f}",
              flush=True)

        # Checkpoint after each fold (defensive against any preempt)
        order_back_train = np.array([id_to_sorted_pos[t] for t in orig_train_ids])
        order_back_test = np.array([test_id_to_sorted_pos[t] for t in test_orig_ids])
        np.save(WORK / "oof_R18_multitask_pit_head_strat.npy",
                oof_pit[order_back_train].astype(np.float32))
        np.save(WORK / "test_R18_multitask_pit_head_strat.npy",
                test_pit[order_back_test].astype(np.float32))

        del model, opt, sched, loader, ds, Xt, y_pit_t, y_laps_t, y_stint_t
        del y_comp_t, mask_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    auc_full = float(roc_auc_score(y, oof_pit))
    print(f"\nR18 OOF AUC: {auc_full:.6f}", flush=True)

    summary = dict(
        round="R18_multitask_joint_NN_gpu",
        oof_auc=auc_full,
        fold_aucs=fold_aucs,
        fold_walls_s=fold_walls,
        wall_total_s=time.time() - t0,
        epochs=EPOCHS, batch=BATCH, lr=LR, weight_decay=WD,
        n_feats=len(feats),
    )
    (WORK / "r18_multitask_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTotal wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
