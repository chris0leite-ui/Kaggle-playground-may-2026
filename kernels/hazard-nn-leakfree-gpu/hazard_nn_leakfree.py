"""T1.4 LEAK-FREE — hazard-NN with stint-leakage-free Strat folds.

Day-9 K=19 hazard-NN stack submitted at OOF 0.95446, LB 0.94711 (gap
-73.5bp). Diagnosis: hazard-target bfill propagates val-row PitNextLap
backward through every earlier row in the same (Race,Driver,Year,Stint)
group. Per P6, 80% of consecutive-lap pairs cross fold boundaries under
Strat — so 80% of the tr rows' hazard buckets encode val-row labels.

This kernel keeps the StratifiedKFold(5, random_state=42) outer split
(so OOF aligns with the M5q pool for downstream K=N+1 stacking) but
inside each fold:
  1. Identify val_stints = unique (Race,Driver,Year,Stint) of val rows.
  2. tr_clean = tr indices whose stint NOT IN val_stints.
  3. Train only on tr_clean. Predict val + test as before.

This breaks the multi-row label leak: no training row shares a stint
with any val row, so val rows' PitNextLap cannot propagate into
training hazard buckets.

Cost: ~60% of stints in train have ≥1 val row under Strat (P6's
implication). We lose ~36% of tr rows per fold. Trains ~225k rows
instead of ~351k. Wall: similar to leaky bag (~36s/fold).

Smoke first (1 seed, no bag) to measure leak magnitude:
  - leaky bag's seed-42 5-fold OOF was 0.94319 (Day-9 v2)
  - leak-free seed-42 5-fold OOF = X (this kernel)
  - leak magnitude = (0.94319 - X) bp

If X is materially below 0.940, the hazard NN is structurally
redundant with the binary pool (sole signal was the leak). If X is
in 0.940-0.945, the architecture has real signal AND we can build a
legitimate K=19 stack. If X >= 0.945, we may even resubmit.

Outputs (under /kaggle/working/):
  oof_d10_hazard_nn_leakfree_strat.npy
  test_d10_hazard_nn_leakfree_strat.npy
  hazard_nn_leakfree_results.json
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET, ID_COL = "PitNextLap", "id"
SEED = 42
N_FOLDS = 5
N_BUCKETS = 20
BASE_S = 0.94075
LEAKY_BAG_OOF_REF = 0.94319  # seed-42 5-fold OOF from leaky bag v2

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def gpu_boot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=10).strip()
        print(f"[boot] GPU info: {out}")
    except Exception as e:
        print(f"[boot] nvidia-smi failed: {e}")


def install_torch():
    print("[setup] force-reinstall torch 2.4 (sm_60 P100 support) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
    import importlib
    if "torch" in sys.modules:
        importlib.reload(sys.modules["torch"])
    import torch
    print(f"[setup] torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[setup] CUDA: {torch.version.cuda}, "
              f"device 0: {torch.cuda.get_device_name(0)}, "
              f"capability: {torch.cuda.get_device_capability(0)}")


def find_data_dir():
    base = Path("/kaggle/input")
    matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv anywhere under {base}; "
                           f"ls /kaggle: {os.listdir('/kaggle')}")
    return matches[0].parent


def compute_hazard_targets(df, n_buckets=N_BUCKETS):
    n = len(df)
    df_s = (df.reset_index(drop=False)
              .rename(columns={"index": "_orig"})
              .sort_values(["Race", "Driver", "Year", "Stint", "LapNumber"])
              .reset_index(drop=True))
    g = df_s.groupby(["Race", "Driver", "Year", "Stint"], sort=False)
    next_pit_lap = np.where(df_s[TARGET].values == 1,
                            df_s["LapNumber"].values, np.nan)
    df_s["_next_pit"] = next_pit_lap
    df_s["_next_pit"] = g["_next_pit"].bfill()
    df_s["_stint_end"] = g["LapNumber"].transform("max")

    gap = (df_s["_next_pit"] - df_s["LapNumber"]).values
    eb = np.where(np.isnan(gap), -1, gap).astype(np.int64)
    eb = np.where(eb >= n_buckets, -1, eb).astype(np.int64)
    cap = (df_s["_stint_end"].values - df_s["LapNumber"].values).clip(
        0, n_buckets - 1).astype(np.int64)

    orig = df_s["_orig"].values
    event_bucket = np.zeros(n, dtype=np.int64)
    censor_cap = np.zeros(n, dtype=np.int64)
    event_bucket[orig] = eb
    censor_cap[orig] = cap
    return event_bucket, censor_cap


def hazard_nll(haz_logits, event_bucket, censor_cap):
    import torch
    import torch.nn.functional as F
    K = haz_logits.shape[1]
    log_surv = -F.softplus(haz_logits)
    log_haz = -F.softplus(-haz_logits)
    arange = torch.arange(K, device=haz_logits.device).unsqueeze(0)

    is_event = (event_bucket >= 0)
    eb = event_bucket.clamp(min=0).unsqueeze(1)
    pre_event = (log_surv * (arange < eb).float()).sum(dim=1)
    eb_safe = event_bucket.clamp(min=0)
    event_term = log_haz.gather(1, eb_safe.unsqueeze(1)).squeeze(1)
    nll_event = -(pre_event + event_term)
    cc = censor_cap.unsqueeze(1)
    nll_cens = -(log_surv * (arange <= cc).float()).sum(dim=1)
    nll = torch.where(is_event, nll_event, nll_cens)
    return nll.mean()


def prepare_features(train, test):
    cat_cols = ["Driver", "Compound", "Race"]
    num_cols = [c for c in train.columns
                if c not in [TARGET, ID_COL] + cat_cols]
    cat_dims = []
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[c].astype(str), test[c].astype(str)])
        le.fit(combined.values)
        train[c] = le.transform(train[c].astype(str).values)
        test[c] = le.transform(test[c].astype(str).values)
        cat_dims.append(int(len(le.classes_)))
    for c in num_cols:
        med = train[c].median()
        train[c] = train[c].fillna(med).astype(np.float32)
        test[c] = test[c].fillna(med).astype(np.float32)
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols]).astype(np.float32)
    test[num_cols] = scaler.transform(test[num_cols]).astype(np.float32)
    X_num = train[num_cols].values.astype(np.float32)
    X_num_t = test[num_cols].values.astype(np.float32)
    X_cat = train[cat_cols].values.astype(np.int64)
    X_cat_t = test[cat_cols].values.astype(np.int64)
    y = train[TARGET].astype(int).values
    return X_num, X_cat, X_num_t, X_cat_t, y, cat_dims


def make_hazard_net(num_dim, cat_dims, n_buckets=N_BUCKETS,
                    hidden=(384, 256, 128), dropout=0.15):
    import torch
    import torch.nn as nn
    emb_dims = [min(50, max(4, (c + 1) // 2)) for c in cat_dims]

    class HazardNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([
                nn.Embedding(c, e) for c, e in zip(cat_dims, emb_dims)
            ])
            in_dim = num_dim + sum(emb_dims)
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers += [nn.Linear(prev, n_buckets)]
            self.mlp = nn.Sequential(*layers)

        def forward(self, x_num, x_cat):
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
            x = torch.cat([x_num] + embs, dim=1)
            return self.mlp(x)
    return HazardNet()


def train_one_fold(seed, X_num, X_cat, X_num_t, X_cat_t, y, eb_full, cc_full,
                   tr_clean, va, cat_dims, device, epochs=250, lr=2e-3,
                   patience=25, batch=8192):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = make_hazard_net(X_num.shape[1], cat_dims).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    def to_t(idx):
        return (torch.from_numpy(X_num[idx]).to(device),
                torch.from_numpy(X_cat[idx]).to(device),
                torch.from_numpy(eb_full[idx]).to(device),
                torch.from_numpy(cc_full[idx]).to(device))

    Xn_tr, Xc_tr, eb_tr, cc_tr = to_t(tr_clean)
    Xn_va, Xc_va, eb_va, cc_va = to_t(va)
    Xn_te = torch.from_numpy(X_num_t).to(device)
    Xc_te = torch.from_numpy(X_cat_t).to(device)

    best_auc = -np.inf
    best_state = None
    bad = 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr_clean), device=device)
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            optimizer.zero_grad()
            logits = net(Xn_tr[idx], Xc_tr[idx])
            loss = hazard_nll(logits, eb_tr[idx], cc_tr[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            logits_va = []
            for s in range(0, len(va), batch):
                e = min(s + batch, len(va))
                logits_va.append(net(Xn_va[s:e], Xc_va[s:e]).cpu().numpy())
            logits_va = np.concatenate(logits_va, axis=0)
            p_va = 1.0 / (1.0 + np.exp(-logits_va[:, 0]))
            auc = float(roc_auc_score(y[va], p_va))
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p_va = []
        for s in range(0, len(va), batch):
            e = min(s + batch, len(va))
            l = net(Xn_va[s:e], Xc_va[s:e]).cpu().numpy()
            p_va.append(1.0 / (1.0 + np.exp(-l[:, 0])))
        p_va = np.concatenate(p_va)
        p_te = []
        for s in range(0, len(X_num_t), batch):
            e = min(s + batch, len(X_num_t))
            l = net(Xn_te[s:e], Xc_te[s:e]).cpu().numpy()
            p_te.append(1.0 / (1.0 + np.exp(-l[:, 0])))
        p_te = np.concatenate(p_te)
    return p_va, p_te, best_auc


def build_stint_id(df):
    """Encode (Race,Driver,Year,Stint) into a single int64 hash for fast
    set-membership testing of which tr rows live in val stints."""
    arr = (df["Race"].astype(str) + "|" + df["Driver"].astype(str) + "|" +
           df["Year"].astype(str) + "|" + df["Stint"].astype(str)).values
    le = LabelEncoder()
    return le.fit_transform(arr).astype(np.int64)


def main():
    t0 = time.time()
    gpu_boot()
    install_torch()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    print("[prep] computing hazard event buckets ...")
    eb_full, cc_full = compute_hazard_targets(train)
    print(f"[prep] events={(eb_full>=0).sum()} censored={(eb_full<0).sum()}")

    print("[prep] building stint_id ...")
    stint_id = build_stint_id(train)
    n_stints = len(np.unique(stint_id))
    print(f"[prep] stints: {n_stints}")

    X_num, X_cat, X_num_t, X_cat_t, y, cat_dims = prepare_features(train, test)
    print(f"cat_dims={cat_dims}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof_seed = np.zeros(len(y), dtype=np.float64)
    test_seed = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []
    fold_walls = []
    drop_stats = []

    for k, (tr, va) in enumerate(splits):
        t_fold = time.time()
        # Stints touched by val
        val_stint_ids = set(stint_id[va].tolist())
        tr_keep_mask = ~np.isin(stint_id[tr], list(val_stint_ids))
        tr_clean = tr[tr_keep_mask]
        n_dropped = len(tr) - len(tr_clean)
        n_val_stints = len(val_stint_ids)
        drop_stats.append(dict(
            fold=k, tr_orig=int(len(tr)), tr_clean=int(len(tr_clean)),
            tr_dropped=int(n_dropped), drop_pct=float(n_dropped / len(tr) * 100),
            n_val_stints=int(n_val_stints),
            n_val_rows=int(len(va)),
        ))
        print(f"  f{k}: tr_orig={len(tr)} tr_clean={len(tr_clean)} "
              f"dropped={n_dropped} ({n_dropped/len(tr)*100:.1f}%) "
              f"val_stints={n_val_stints}")

        p_va, p_te, fauc = train_one_fold(
            SEED, X_num, X_cat, X_num_t, X_cat_t, y, eb_full, cc_full,
            tr_clean, va, cat_dims, device)
        oof_seed[va] = p_va
        test_seed += p_te / N_FOLDS
        fold_aucs.append(fauc)
        wall = time.time() - t_fold
        fold_walls.append(wall)
        print(f"    AUC={fauc:.5f}  wall={wall:.0f}s")

        np.save(WORK / "_partial_oof_seed42.npy", oof_seed)
        np.save(WORK / "_partial_test_seed42.npy", test_seed)

    full_auc = float(roc_auc_score(y, oof_seed))
    delta_baseline = (full_auc - BASE_S) * 1e4
    delta_leaky = (full_auc - LEAKY_BAG_OOF_REF) * 1e4
    print(f"\n=== LEAK-FREE 5-fold OOF AUC: {full_auc:.5f} ===")
    print(f"  Δ baseline:                {delta_baseline:+.1f}bp")
    print(f"  Δ leaky bag seed-42 OOF:   {delta_leaky:+.1f}bp "
          f"(= leak magnitude with sign flipped)")
    print(f"  fold AUCs: {[round(a, 5) for a in fold_aucs]}")
    print(f"  fold walls: {[round(w, 1) for w in fold_walls]}")

    np.save(WORK / "oof_d10_hazard_nn_leakfree_strat.npy",
            np.column_stack([1 - oof_seed, oof_seed]))
    np.save(WORK / "test_d10_hazard_nn_leakfree_strat.npy",
            np.column_stack([1 - test_seed, test_seed]))
    sub = sample_sub.copy(); sub[TARGET] = test_seed
    sub.to_csv(WORK / "submission_d10_hazard_nn_leakfree.csv", index=False)

    # Cleanup partials
    for f in [WORK / "_partial_oof_seed42.npy", WORK / "_partial_test_seed42.npy"]:
        if f.exists():
            f.unlink()

    res = dict(
        seed=SEED,
        leakfree_oof_auc=full_auc,
        leaky_bag_oof_ref=LEAKY_BAG_OOF_REF,
        leak_magnitude_bp=-delta_leaky,
        delta_vs_baseline_bp=delta_baseline,
        fold_aucs=fold_aucs,
        fold_walls=fold_walls,
        drop_stats=drop_stats,
        cv="StratifiedKFold(5)+stint_drop",
        metric="roc_auc",
        n_buckets=N_BUCKETS,
        total_wall_s=time.time() - t0,
        notes=("Leak-free hazard NN diagnostic: stint-drop within-fold, "
               "outer split unchanged at random_state=42. OOF aligned with "
               "M5q pool for downstream K=N+1 stacking."),
    )
    (WORK / "hazard_nn_leakfree_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
