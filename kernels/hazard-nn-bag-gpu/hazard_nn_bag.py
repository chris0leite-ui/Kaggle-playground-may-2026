"""T1.4 — Hazard-rate NN 5-fold × 3-seed BAG (Day-9 PROMOTE follow-up).

Smoke v2 cleared the gate: fold-0 AUC 0.94513, 5-fold projection 3.0min.
This kernel runs the full 5-fold for 3 seeds (42, 123, 456) and
rank-averages OOF + test predictions. Total wall: ~10min on T4.

Architecture matches smoke v2:
  hidden=(384, 256, 128), dropout=0.15
  K=20 hazard buckets, nnet-survival NLL
  Adam lr=2e-3 → cosine to 1e-5 over 250 epochs, patience 25
  batch=8192, embeddings (Driver, Compound, Race) auto-sized

Strat-only (R1).
P(PitNextLap=1) = h_0 (sigmoid of bucket-0 logit), per row.

Outputs (under /kaggle/working/):
  oof_d9_hazard_nn_strat.npy   shape (n_train, 2) -- rank-averaged
  test_d9_hazard_nn_strat.npy  shape (n_test, 2)  -- rank-averaged
  hazard_nn_bag_results.json
  submission_d9_hazard_nn.csv  (held; do not submit until K=19 PASS)
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
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET, ID_COL = "PitNextLap", "id"
SEEDS = [42, 123, 456]
N_FOLDS = 5
N_BUCKETS = 20
BASE_S = 0.94075
REALMLP_OOF_REF = 0.94722  # fold-0 reference

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
    """Vectorised event-bucket / censor-cap compute. Same as smoke v2.

    NOTE: synthetic data has multi-PitNextLap=1-per-stint; bfill within
    (Race,Driver,Year,Stint) finds the next pit lap from each row.
    """
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
                   tr, va, cat_dims, device, epochs=250, lr=2e-3,
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
                torch.from_numpy(cc_full[idx]).to(device),
                torch.from_numpy(y[idx]).to(device))

    Xn_tr, Xc_tr, eb_tr, cc_tr, _ = to_t(tr)
    Xn_va, Xc_va, eb_va, cc_va, _ = to_t(va)
    Xn_te = torch.from_numpy(X_num_t).to(device)
    Xc_te = torch.from_numpy(X_cat_t).to(device)

    best_auc = -np.inf
    best_state = None
    bad = 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr), device=device)
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

    # Predict val + test from best checkpoint
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

    X_num, X_cat, X_num_t, X_cat_t, y, cat_dims = prepare_features(train, test)
    print(f"cat_dims={cat_dims}")

    # CRITICAL: outer fold split is FIXED at random_state=42 (matches M5q
    # pool's fold structure). Only the torch model seed varies per seed.
    # If we let random_state=seed, each seed's val rows would intersect
    # other seeds' train rows → bag OOF has in-sample contamination
    # against M5q's fold partition, and any K=N+1 stack lift would be fake.
    OUTER_SEED = 42
    skf_fixed = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=OUTER_SEED)
    splits_fixed = list(skf_fixed.split(np.zeros(len(y)), y))

    seed_oofs = []   # list of (n_train,) arrays per seed (probabilities)
    seed_tests = []  # list of (n_test,) arrays per seed
    seed_results = []
    for seed in SEEDS:
        print(f"\n=== SEED {seed} (model-seed; outer fold split=SEED={OUTER_SEED}) ===")
        oof_seed = np.zeros(len(y), dtype=np.float64)
        test_seed = np.zeros(len(test), dtype=np.float64)
        fold_aucs = []
        for k, (tr, va) in enumerate(splits_fixed):
            t_fold = time.time()
            p_va, p_te, fauc = train_one_fold(
                seed, X_num, X_cat, X_num_t, X_cat_t, y, eb_full, cc_full,
                tr, va, cat_dims, device)
            oof_seed[va] = p_va
            test_seed += p_te / N_FOLDS
            fold_aucs.append(fauc)
            print(f"  seed{seed} f{k}: AUC={fauc:.5f}  wall={time.time()-t_fold:.0f}s")
            # Save partial state every fold (resilience against kernel kills)
            np.save(WORK / f"_partial_oof_seed{seed}.npy", oof_seed)
            np.save(WORK / f"_partial_test_seed{seed}.npy", test_seed)
        full_auc = float(roc_auc_score(y, oof_seed))
        print(f"  → seed{seed} 5-fold OOF AUC: {full_auc:.5f}  "
              f"(fold std={np.std(fold_aucs):.5f})")
        seed_oofs.append(oof_seed)
        seed_tests.append(test_seed)
        seed_results.append(dict(seed=seed, oof_auc=full_auc,
                                  fold_aucs=fold_aucs))
        # Persist per-seed OOFs so we can stack them back independently
        # if the bag turns out to be tie-locked.
        np.save(WORK / f"oof_d9_hazard_nn_seed{seed}_strat.npy",
                np.column_stack([1 - oof_seed, oof_seed]))
        np.save(WORK / f"test_d9_hazard_nn_seed{seed}_strat.npy",
                np.column_stack([1 - test_seed, test_seed]))

    # Rank-average across seeds (more robust than mean for AUC)
    n_train = len(y); n_test = len(test)
    oof_ranks = np.column_stack([rankdata(o) / n_train for o in seed_oofs])
    test_ranks = np.column_stack([rankdata(t) / n_test for t in seed_tests])
    oof_bag = oof_ranks.mean(axis=1)
    test_bag = test_ranks.mean(axis=1)
    bag_auc = float(roc_auc_score(y, oof_bag))
    delta_baseline = (bag_auc - BASE_S) * 1e4
    delta_realmlp = (bag_auc - REALMLP_OOF_REF) * 1e4
    print(f"\n=== BAG (rank-averaged across {len(SEEDS)} seeds) ===")
    print(f"OOF AUC: {bag_auc:.5f}  Δ baseline {delta_baseline:+.1f}bp  "
          f"Δ realmlp_e4 {delta_realmlp:+.1f}bp")

    # Save in the (1-p, p) two-column convention used by the M5q pool
    np.save(WORK / "oof_d9_hazard_nn_strat.npy",
            np.column_stack([1 - oof_bag, oof_bag]))
    np.save(WORK / "test_d9_hazard_nn_strat.npy",
            np.column_stack([1 - test_bag, test_bag]))
    sub = sample_sub.copy(); sub[TARGET] = test_bag
    sub.to_csv(WORK / "submission_d9_hazard_nn.csv", index=False)

    res = dict(
        seeds=SEEDS,
        seed_results=seed_results,
        bag_oof_auc=bag_auc,
        delta_vs_baseline_bp=delta_baseline,
        delta_vs_realmlp_e4_fold0_bp=delta_realmlp,
        cv="StratifiedKFold(5)",
        metric="roc_auc",
        n_buckets=N_BUCKETS,
        total_wall_s=time.time() - t0,
        notes=("Hazard-NN 5-fold x 3-seed bag, rank-averaged. Save OOF/test "
               "and submission. K=19 stack vs K=18 PRIMARY (LB 0.95026) "
               "evaluated locally; submission HELD until ρ-vs-K=18 < 0.999 "
               "and stack OOF >= K=18 + 0.5bp."),
    )
    (WORK / "hazard_nn_bag_results.json").write_text(json.dumps(res, indent=2))
    # Cleanup partial files
    for s in SEEDS:
        for prefix in ("_partial_oof", "_partial_test"):
            f = WORK / f"{prefix}_seed{s}.npy"
            if f.exists():
                f.unlink()
    print("DONE", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
