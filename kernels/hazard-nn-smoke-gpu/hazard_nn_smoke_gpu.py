"""T1.4 — Hazard-rate NN 1-fold SMOKE on Kaggle T4 (HANDOVER Day-9 G3).

Rule 2: 1-fold smoke before any GPU 5-fold. Rule 16 pre-flight: distinct
problem class (K=20 discrete hazard + nnet-survival NLL) — NOT in
mechanism_families_explored; NOT vulnerable to rank-lock-on-K=18 by
construction.

Why hazard-NN (vs RealMLP / TabM, our prior NN base attempts):
  RealMLP-TD: binary PitNextLap classifier with embeddings (in M5q).
  TabM-D:     binary classifier with k=32 BatchEnsemble heads (smoke FAIL Day-9).
  Hazard-NN:  predicts the ENTIRE laps-until-pit survival curve via K=20
    discrete hazards h_k = P(pit at bucket k | survived to bucket k).
    P(PitNextLap=1) marginalizes from h_0 directly, but the network is
    REGULARIZED by the multi-bucket NLL across the stint — different
    inductive bias from binary heads. Gensheimer & Narasimhan 2019
    (nnet-survival).

Bucket scheme: bucket k = "pit happens (k+1) laps from now" for k=0..19.
  - row r in same stint as pit at lap L: event bucket = (L - r_lap - 1)
    so bucket 0 = PitNextLap=1, bucket 1 = pit-in-2-laps, ...
  - if no pit visible in same stint: censored at last visible bucket.

P(PitNextLap=1) = h_0 (pulled directly from network output).

Loss (nnet-survival): for event at bucket k*:
    log(1 - h_0) + log(1 - h_1) + ... + log(1 - h_{k*-1}) + log(h_{k*})
For censored at bucket c*:
    log(1 - h_0) + ... + log(1 - h_{c*})

Pinned: SEED=42, N_FOLDS=5, fold 0 only.

Smoke gate (HANDOVER-aligned):
  - fold-0 AUC >= 0.945 → PROMOTE to 5-fold
  - 5-fold projection < 60 min → safe under 1h GPU cap
  - either failing → HOLD; pivot to TabM HPO or G4 SCARF

Outputs (under /kaggle/working/):
  hazard_nn_smoke_results.json
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
SEED, N_FOLDS = 42, 5
SMOKE_FOLD = 0
N_BUCKETS = 20
BASE_S = 0.94075
REALMLP_FOLD0_REF = 0.94722
SMOKE_GATE_AUC = 0.945

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
    """Force-reinstall torch 2.4 (sm_60 P100 support); same lesson as
    realmlp_gpu.py — Kaggle silently routes GpuT4x2 → P100 sometimes."""
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
    if not base.exists():
        raise RuntimeError(f"/kaggle/input missing; ls /kaggle: {os.listdir('/kaggle')}")
    matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv anywhere under {base}")
    train_path = matches[0]
    print(f"Found train.csv at {train_path}")
    return train_path.parent


def compute_hazard_targets(df: pd.DataFrame, n_buckets: int = N_BUCKETS):
    """For each row, compute event-bucket (or -1 if censored) and
    censor_cap (last bucket index for the censored-NLL term).

    Bucket k = pit happens (k+1) laps from current row. PitNextLap=1 → k=0.

    NOTE on the synthetic dataset: the playground-series-s6e5 data has
    ~21k (Race,Driver,Year,Stint) groups with MULTIPLE PitNextLap=1 rows
    (45k extra events vs one-per-stint assumption). We use bfill within
    each group to find the NEXT PitNextLap=1 lap from each row's
    perspective — the correct hazard interpretation regardless of how
    many events the synthetic data emits per stint.

    Vectorised; ~2-3s on the full 439k-row train set.
    """
    n = len(df)
    df_s = (df.reset_index(drop=False)
              .rename(columns={"index": "_orig"})
              .sort_values(["Race", "Driver", "Year", "Stint", "LapNumber"])
              .reset_index(drop=True))
    g = df_s.groupby(["Race", "Driver", "Year", "Stint"], sort=False)

    # Within group, within sorted-lap order, propagate the next pit-lap
    # backwards (so each row sees the lap of its nearest future PitNextLap=1).
    next_pit_lap = np.where(df_s[TARGET].values == 1,
                            df_s["LapNumber"].values, np.nan)
    df_s["_next_pit"] = next_pit_lap
    df_s["_next_pit"] = g["_next_pit"].bfill()
    df_s["_stint_end"] = g["LapNumber"].transform("max")

    gap = (df_s["_next_pit"] - df_s["LapNumber"]).values  # NaN if censored
    event_bucket_sorted = np.where(np.isnan(gap), -1, gap).astype(np.int64)
    # gap >= n_buckets → censored at last bucket
    event_bucket_sorted = np.where(event_bucket_sorted >= n_buckets, -1,
                                    event_bucket_sorted).astype(np.int64)
    cap_full = (df_s["_stint_end"].values - df_s["LapNumber"].values).clip(
        0, n_buckets - 1).astype(np.int64)

    # Reorder back to original index
    orig = df_s["_orig"].values
    event_bucket = np.zeros(n, dtype=np.int64)
    censor_cap = np.zeros(n, dtype=np.int64)
    event_bucket[orig] = event_bucket_sorted
    censor_cap[orig] = cap_full
    return event_bucket, censor_cap


def hazard_nll(haz_logits, event_bucket, censor_cap):
    """nnet-survival NLL.

    haz_logits : [B, K] real-valued logits (sigmoid → hazard h_k).
    event_bucket : [B] int (-1 if censored).
    censor_cap : [B] int (last bucket index for censored rows, ≥0).
    """
    import torch
    import torch.nn.functional as F
    B, K = haz_logits.shape
    # log(1 - h_k) = -softplus(logit_k)
    # log(h_k)     = logit_k - softplus(logit_k) = -softplus(-logit_k)
    log_surv = -F.softplus(haz_logits)               # [B, K]
    log_haz = -F.softplus(-haz_logits)               # [B, K]
    arange = torch.arange(K, device=haz_logits.device).unsqueeze(0)  # [1,K]

    is_event = (event_bucket >= 0)
    # mask_pre[b, k] = 1 if k < event_bucket[b] (sum log(1-h_k))
    eb = event_bucket.clamp(min=0).unsqueeze(1)      # [B, 1]
    mask_pre_event = (arange < eb).float()            # [B, K]
    pre_event = (log_surv * mask_pre_event).sum(dim=1)  # [B]
    # event term: log(h_event_bucket)
    eb_safe = event_bucket.clamp(min=0)
    event_term = log_haz.gather(1, eb_safe.unsqueeze(1)).squeeze(1)
    nll_event = -(pre_event + event_term)

    # censored: sum log(1 - h_k) for k <= censor_cap
    cc = censor_cap.unsqueeze(1)                      # [B, 1]
    mask_cens = (arange <= cc).float()
    nll_cens = -(log_surv * mask_cens).sum(dim=1)

    nll = torch.where(is_event, nll_event, nll_cens)
    return nll.mean()


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    """Pull features out, label-encode cats, standard-scale numerics.

    Returns: X_num, X_cat (LongTensor-friendly int), cat_dims, y
    Numerics: PitStop, LapNumber, Stint, TyreLife, Year, Position,
      LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress,
      Position_Change.
    Cats: Driver, Compound, Race.
    """
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

    X_num_train = train[num_cols].values.astype(np.float32)
    X_num_test = test[num_cols].values.astype(np.float32)
    X_cat_train = train[cat_cols].values.astype(np.int64)
    X_cat_test = test[cat_cols].values.astype(np.int64)
    y = train[TARGET].astype(int).values
    return X_num_train, X_cat_train, X_num_test, X_cat_test, y, cat_dims, num_cols, cat_cols


def make_hazard_net(num_dim, cat_dims, emb_dims=None,
                    hidden=(384, 256, 128), dropout=0.15, n_buckets=N_BUCKETS):
    import torch
    import torch.nn as nn
    if emb_dims is None:
        # Roughly min(50, (cardinality+1)//2)
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


def main():
    t0 = time.time()
    gpu_boot()
    install_torch()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"loaded train={train.shape} test={test.shape}  t={time.time()-t0:.1f}s")

    # Hazard targets BEFORE standardisation (uses raw LapNumber/Stint/etc.)
    print("[prep] computing hazard event buckets ...")
    t1 = time.time()
    event_bucket, censor_cap = compute_hazard_targets(train, N_BUCKETS)
    n_event = (event_bucket >= 0).sum()
    print(f"[prep] events={n_event} ({n_event/len(train)*100:.1f}%) "
          f"censored={(event_bucket<0).sum()}  wall={time.time()-t1:.1f}s")
    print(f"[prep] event-bucket histogram (top 5): "
          f"{pd.Series(event_bucket[event_bucket>=0]).value_counts().sort_index().head().to_dict()}")

    X_num, X_cat, X_num_t, X_cat_t, y, cat_dims, num_cols, cat_cols = \
        prepare_features(train, test)
    print(f"num_cols ({len(num_cols)}): {num_cols}")
    print(f"cat_cols: {cat_cols}  cat_dims: {cat_dims}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[SMOKE_FOLD]
    print(f"=== SMOKE fold {SMOKE_FOLD} (train={len(tr)} val={len(va)}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = make_hazard_net(num_dim=X_num.shape[1], cat_dims=cat_dims).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"net params: {n_params:,}")

    LR = 2e-3
    EPOCHS = 250
    PATIENCE = 25
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    def to_tensor(idx):
        return (torch.from_numpy(X_num[idx]).to(device),
                torch.from_numpy(X_cat[idx]).to(device),
                torch.from_numpy(event_bucket[idx]).to(device),
                torch.from_numpy(censor_cap[idx]).to(device),
                torch.from_numpy(y[idx]).to(device))

    Xn_tr, Xc_tr, eb_tr, cc_tr, y_tr = to_tensor(tr)
    Xn_va, Xc_va, eb_va, cc_va, y_va = to_tensor(va)

    BATCH = 8192  # T4 has plenty of memory for our small MLP
    best_val_auc = -np.inf
    best_state = None
    bad = 0
    t_fold = time.time()

    for ep in range(EPOCHS):
        # --- train ---
        net.train()
        perm = torch.randperm(len(tr), device=device)
        running = 0.0; nb = 0
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            optimizer.zero_grad()
            logits = net(Xn_tr[idx], Xc_tr[idx])
            loss = hazard_nll(logits, eb_tr[idx], cc_tr[idx])
            loss.backward()
            optimizer.step()
            running += loss.item(); nb += 1
        train_nll = running / max(nb, 1)
        scheduler.step()

        # --- val ---
        net.eval()
        with torch.no_grad():
            haz_logits_va = []
            nll_running = 0.0; nb = 0
            for s in range(0, len(va), BATCH):
                e = min(s + BATCH, len(va))
                logits = net(Xn_va[s:e], Xc_va[s:e])
                nll = hazard_nll(logits, eb_va[s:e], cc_va[s:e])
                nll_running += nll.item(); nb += 1
                haz_logits_va.append(logits.cpu().numpy())
            val_nll = nll_running / max(nb, 1)
            haz_logits_va = np.concatenate(haz_logits_va, axis=0)
            # P(PitNextLap=1) = h_0 = sigmoid(logit_0)
            p_pit = 1.0 / (1.0 + np.exp(-haz_logits_va[:, 0]))
            val_auc = float(roc_auc_score(y[va], p_pit))

        improved = val_auc > best_val_auc + 1e-5
        cur_lr = optimizer.param_groups[0]["lr"]
        msg = (f"ep{ep:03d}: lr={cur_lr:.1e}  train_nll={train_nll:.4f}  "
               f"val_nll={val_nll:.4f}  val_AUC={val_auc:.5f}  "
               f"{'★' if improved else ''}")
        if ep % 5 == 0 or improved:
            print(msg)
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early-stop at ep{ep} (no AUC improve for {PATIENCE} epochs)")
                break

    fold_wall = time.time() - t_fold
    fold_auc = best_val_auc
    delta_bp_baseline = (fold_auc - BASE_S) * 1e4
    delta_bp_realmlp = (fold_auc - REALMLP_FOLD0_REF) * 1e4
    five_fold_proj_min = (fold_wall * N_FOLDS) / 60.0
    auc_gate = "PASS" if fold_auc >= SMOKE_GATE_AUC else "FAIL"
    wall_gate = "PASS" if five_fold_proj_min < 60 else "FAIL"
    overall = "PROMOTE" if (auc_gate == "PASS" and wall_gate == "PASS") else "HOLD"

    print(f"\nFold-0 best val AUC={fold_auc:.5f}  wall={fold_wall:.0f}s "
          f"({fold_wall/60:.1f}min)")
    print(f"  Δ baseline(BASE_S={BASE_S:.5f})={delta_bp_baseline:+.1f}bp")
    print(f"  Δ realmlp_e4_fold0({REALMLP_FOLD0_REF:.5f})={delta_bp_realmlp:+.1f}bp")
    print(f"  smoke_gate AUC>={SMOKE_GATE_AUC}: {auc_gate}")
    print(f"5-fold wall projection: {five_fold_proj_min:.1f}min  gate<60min: {wall_gate}")
    print(f"OVERALL: {overall}")

    res = dict(
        smoke_fold=SMOKE_FOLD,
        n_buckets=N_BUCKETS,
        fold_auc=fold_auc,
        fold_wall_s=fold_wall,
        delta_vs_baseline_bp=delta_bp_baseline,
        delta_vs_realmlp_e4_fold0_bp=delta_bp_realmlp,
        five_fold_projection_min=five_fold_proj_min,
        smoke_gate_auc=SMOKE_GATE_AUC,
        auc_gate=auc_gate,
        wall_gate=wall_gate,
        overall=overall,
        n_params=int(n_params),
        cv="StratifiedKFold(5)",
        metric="roc_auc",
        seed=SEED,
        n_folds=N_FOLDS,
        n_event=int(n_event),
        n_censored=int((event_bucket < 0).sum()),
        total_wall_s=time.time() - t0,
        notes=("Hazard-rate NN 1-fold smoke (T1.4). PROMOTE -> 5-fold + "
               "3-seed bag; HOLD -> pivot to TabM-HPO or G4 SCARF."),
    )
    (WORK / "hazard_nn_smoke_results.json").write_text(json.dumps(res, indent=2))
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
