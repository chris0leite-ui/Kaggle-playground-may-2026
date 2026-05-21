"""R35 — FT-Transformer (Feature-Tokenizer Transformer) on raw F1 + K=17 logits.

Architecture (Gorishniy et al. 2021, arXiv:2106.11189):
- Each input feature becomes a token of dim D after tokenisation.
- Numerical features: token = W_i * x_i + b_i (per-feature linear).
- Categorical features: token = lookup embedding for category index.
- Add [CLS] token; transformer encoder over (n_features+1, D); take [CLS]
  output -> MLP -> logit.

Why this is structurally novel vs r5/r6 transformer-on-sequence: those did
self-attention ACROSS LAPS within a (Driver, Race, Year) sequence. R35
does self-attention ACROSS FEATURES within a single row. Orthogonal axis.

Inputs:
- Raw F1 numericals: TyreLife, RaceProgress, Stint, LapNumber, Position,
  Position_Change, Cumulative_Degradation, LapTime_s, LapTime_Delta (9)
- Raw F1 categoricals: Year, Race, Driver, Compound (4)
- K=17 base logits (17) — re-use R21's pool as auxiliary feature tokens.

Total: 9 num + 4 cat + 17 num (K17 logit features) = 30 tokens + [CLS] = 31.

Training: 5-fold StratifiedKFold on rows. Per fold: train BCEWithLogits on
train_fold rows; predict on valid_fold. Test predictions averaged across 5
folds.

Output: oof_R35_ft_transformer_strat.npy + test_R35_ft_transformer_strat.npy
"""
from __future__ import annotations
import json
import math
import sys
import subprocess
import time
import os
from pathlib import Path

print("[boot] force-reinstall torch 2.4 (sm_60 P100 support) ...", flush=True)
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
except subprocess.CalledProcessError as e:
    print(f"[boot] torch 2.4 reinstall failed: {e}", flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

print(f"[boot] Python {sys.version.split()[0]}, torch {torch.__version__}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"[boot] CUDA device 0: {torch.cuda.get_device_name(0)}, cap: {cap}")

TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

NUM_FEATS = ["TyreLife", "RaceProgress", "Stint", "LapNumber", "Position",
             "Position_Change", "Cumulative_Degradation", "LapTime (s)",
             "LapTime_Delta"]
CAT_FEATS = ["Year", "Race", "Driver", "Compound"]

K17_FILES = [
    ("yekenot",   "oof_d17_h1d_yekenot_full_strat.npy",     "test_d17_h1d_yekenot_full_strat.npy"),
    ("cb_v4",     "oof_p1_single_cb_v4_gpu_strat.npy",      "test_p1_single_cb_v4_gpu_strat.npy"),
    ("hgbc_deep", "oof_f1_hgbc_deep_strat.npy",             "test_f1_hgbc_deep_strat.npy"),
    ("d16_orig",  "oof_d16_orig_continuous_only_strat.npy", "test_d16_orig_continuous_only_strat.npy"),
    ("qAT",       "dgp_v3_qAT_K1_oof.npy",                  "dgp_v3_qAT_K1_test.npy"),
    ("qAV",       "dgp_v3_qAV_K1_7feat_oof.npy",            "dgp_v3_qAV_K1_7feat_test.npy"),
    ("qAO",       "dgp_v3_qAO_knn_multi_oof.npy",           "dgp_v3_qAO_knn_multi_test.npy"),
    ("qAA",       "dgp_v3_qAA_stint_imputed_oof.npy",       "dgp_v3_qAA_stint_imputed_test.npy"),
    ("qAF",       "dgp_v3_qAF_d16plus_oof.npy",             "dgp_v3_qAF_d16plus_test.npy"),
    ("qAK",       "dgp_v3_qAK_knn3_oof.npy",                "dgp_v3_qAK_knn3_test.npy"),
    ("K27_100k",  "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy",
                  "test_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"),
    ("seg_fe",    "oof_r4_segment_fe_strat.npy",            "test_r4_segment_fe_strat.npy"),
    ("HMM",       "oof_r4_hmm_seq_strat.npy",               "test_r4_hmm_seq_strat.npy"),
    ("R12_cb_horizon",          "oof_R12_cb_horizon_strat.npy",         "test_R12_cb_horizon_strat.npy"),
    ("R13_cb_stint_completion", "oof_R13_cb_stint_completion_strat.npy","test_R13_cb_stint_completion_strat.npy"),
    ("R14_tabm",                "oof_R14_tabm_strat.npy",               "test_R14_tabm_strat.npy"),
    ("R15_xendcg_per_seg",      "oof_R15_xendcg_per_seg_strat.npy",     "test_R15_xendcg_per_seg_strat.npy"),
]

D_MODEL = 96
N_HEADS = 8
N_LAYERS = 3
DROPOUT = 0.15
LR = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2048
EPOCHS = 8

torch.manual_seed(SEED)
np.random.seed(SEED)


def find_data_dir():
    for p in Path("/kaggle/input").rglob("train.csv"):
        return p.parent
    raise RuntimeError("train.csv not found")


def find_artifacts_dir():
    for p in Path("/kaggle/input").rglob("oof_R15_xendcg_per_seg_strat.npy"):
        return p.parent
    raise RuntimeError("artifacts dir not found")


def _pos(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[:, -1] if a.shape[1] >= 2 else a.ravel()
    return a.astype(np.float32)


class FTDataset(Dataset):
    def __init__(self, num, cat, k17, y=None):
        self.num = num      # (N, n_num) float32
        self.cat = cat      # (N, n_cat) int64
        self.k17 = k17      # (N, 17) float32
        self.y = y          # (N,) float32 or None

    def __len__(self):
        return len(self.num)

    def __getitem__(self, i):
        out = {"num": self.num[i], "cat": self.cat[i], "k17": self.k17[i]}
        if self.y is not None:
            out["y"] = self.y[i]
        return out


class FeatureTokenizer(nn.Module):
    """Each numerical -> (D,) via per-feature W*x+b; each cat -> (D,) via embedding."""
    def __init__(self, n_num, cat_cardinalities, d_model):
        super().__init__()
        self.num_w = nn.Parameter(torch.randn(n_num, d_model) * 0.05)
        self.num_b = nn.Parameter(torch.zeros(n_num, d_model))
        self.cat_emb = nn.ModuleList([
            nn.Embedding(c, d_model) for c in cat_cardinalities
        ])

    def forward(self, num, cat):
        # num: (B, n_num), cat: (B, n_cat)
        num_tok = num.unsqueeze(-1) * self.num_w + self.num_b   # (B, n_num, D)
        cat_tok = torch.stack([self.cat_emb[i](cat[:, i])
                               for i in range(cat.shape[1])], dim=1)  # (B, n_cat, D)
        return torch.cat([num_tok, cat_tok], dim=1)             # (B, n_num+n_cat, D)


class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cardinalities, n_k17, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_model)
        # K17 features get their own tokenizer (treat as num)
        self.k17_w = nn.Parameter(torch.randn(n_k17, d_model) * 0.05)
        self.k17_b = nn.Parameter(torch.zeros(n_k17, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, num, cat, k17):
        feat_tok = self.tokenizer(num, cat)
        k17_tok = k17.unsqueeze(-1) * self.k17_w + self.k17_b
        cls = self.cls.expand(num.shape[0], 1, -1)
        x = torch.cat([cls, feat_tok, k17_tok], dim=1)
        x = self.encoder(x)
        logit = self.head(x[:, 0]).squeeze(-1)
        return logit


def main():
    t0 = time.time()
    print("== R35 inventor: FT-Transformer on raw F1 + K17 logits ==", flush=True)

    data_dir = find_data_dir()
    art_dir = find_artifacts_dir()
    print(f"[setup] data at {data_dir}", flush=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y_train = train[TARGET].astype(np.float32).values
    print(f"[data] train {train.shape}  test {test.shape}", flush=True)

    # Categorical encoding (shared train+test)
    cat_cardinalities = []
    cat_train = np.zeros((len(train), len(CAT_FEATS)), dtype=np.int64)
    cat_test = np.zeros((len(test), len(CAT_FEATS)), dtype=np.int64)
    for i, c in enumerate(CAT_FEATS):
        combined = pd.concat([train[c].astype(str), test[c].astype(str)], ignore_index=True)
        codes, _ = pd.factorize(combined)
        cat_cardinalities.append(int(codes.max() + 1))
        cat_train[:, i] = codes[:len(train)]
        cat_test[:, i] = codes[len(train):]
        print(f"[cat] {c}: {cat_cardinalities[-1]} levels", flush=True)

    # Numerical: scale by train stats
    num_train_raw = train[NUM_FEATS].astype(np.float32).fillna(0).values
    num_test_raw = test[NUM_FEATS].astype(np.float32).fillna(0).values
    scaler = StandardScaler()
    num_train = scaler.fit_transform(num_train_raw).astype(np.float32)
    num_test = scaler.transform(num_test_raw).astype(np.float32)

    # K=17 features (already in [0,1] probability space)
    oof_cols, test_cols = [], []
    for name, of, tf in K17_FILES:
        oof_cols.append(_pos(np.load(art_dir / of)))
        test_cols.append(_pos(np.load(art_dir / tf)))
    k17_train = np.column_stack(oof_cols).astype(np.float32)
    k17_test = np.column_stack(test_cols).astype(np.float32)
    print(f"[data] K17 train {k17_train.shape}  test {k17_test.shape}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device: {device}", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y_train)), y_train))

    oof_meta = np.zeros(len(y_train), dtype=np.float64)
    test_meta = np.zeros(len(test), dtype=np.float64)
    fold_aucs, walls = [], []

    for k, (ti, vi) in enumerate(fold_list, 1):
        t_f = time.time()
        print(f"\n--- Fold {k}/{N_FOLDS} | ti={len(ti)} vi={len(vi)} ---",
              flush=True)
        ds_tr = FTDataset(num_train[ti], cat_train[ti], k17_train[ti], y_train[ti])
        ds_va = FTDataset(num_train[vi], cat_train[vi], k17_train[vi], y_train[vi])
        ds_te = FTDataset(num_test,      cat_test,      k17_test)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=4096,       shuffle=False, num_workers=0, pin_memory=True)
        dl_te = DataLoader(ds_te, batch_size=4096,       shuffle=False, num_workers=0, pin_memory=True)

        model = FTTransformer(
            n_num=len(NUM_FEATS),
            cat_cardinalities=cat_cardinalities,
            n_k17=k17_train.shape[1],
            d_model=D_MODEL, n_heads=N_HEADS,
            n_layers=N_LAYERS, dropout=DROPOUT,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*len(dl_tr))
        loss_fn = nn.BCEWithLogitsLoss()
        scaler_amp = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        best_auc = 0
        best_state = None
        for ep in range(1, EPOCHS+1):
            model.train()
            ep_loss, n = 0.0, 0
            for batch in dl_tr:
                num = batch["num"].to(device, non_blocking=True)
                cat = batch["cat"].to(device, non_blocking=True)
                k17 = batch["k17"].to(device, non_blocking=True)
                y   = batch["y"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if scaler_amp is not None:
                    with torch.amp.autocast("cuda"):
                        logit = model(num, cat, k17)
                        loss = loss_fn(logit, y)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(opt)
                    scaler_amp.update()
                else:
                    logit = model(num, cat, k17)
                    loss = loss_fn(logit, y)
                    loss.backward()
                    opt.step()
                sched.step()
                ep_loss += loss.item() * len(y)
                n += len(y)
            ep_loss /= max(n, 1)

            # Validate
            model.eval()
            preds_va = []
            with torch.no_grad():
                for batch in dl_va:
                    num = batch["num"].to(device)
                    cat = batch["cat"].to(device)
                    k17 = batch["k17"].to(device)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        logit = model(num, cat, k17)
                    preds_va.append(torch.sigmoid(logit).float().cpu().numpy())
            preds_va = np.concatenate(preds_va)
            auc_va = float(roc_auc_score(y_train[vi], preds_va))
            print(f"  ep {ep}: loss={ep_loss:.4f}  val_AUC={auc_va:.5f}",
                  flush=True)
            if auc_va > best_auc:
                best_auc = auc_va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Restore best
        model.load_state_dict(best_state)
        model.eval()

        # Final OOF + test predictions
        with torch.no_grad():
            preds_va = []
            for batch in dl_va:
                num = batch["num"].to(device); cat = batch["cat"].to(device); k17 = batch["k17"].to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logit = model(num, cat, k17)
                preds_va.append(torch.sigmoid(logit).float().cpu().numpy())
            preds_va = np.concatenate(preds_va)
            oof_meta[vi] = preds_va

            preds_te = []
            for batch in dl_te:
                num = batch["num"].to(device); cat = batch["cat"].to(device); k17 = batch["k17"].to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logit = model(num, cat, k17)
                preds_te.append(torch.sigmoid(logit).float().cpu().numpy())
            preds_te = np.concatenate(preds_te)
            test_meta += preds_te / N_FOLDS

        fold_auc = float(roc_auc_score(y_train[vi], preds_va))
        wall = time.time() - t_f
        fold_aucs.append(fold_auc)
        walls.append(wall)
        print(f"  fold {k} final AUC={fold_auc:.5f}  wall={wall:.0f}s", flush=True)

    auc_full = float(roc_auc_score(y_train, oof_meta))
    print(f"\n[result] R35 FT-Transformer OOF AUC: {auc_full:.6f}", flush=True)

    np.save(Path("/kaggle/working") / "oof_R35_ft_transformer_strat.npy",
            oof_meta.astype(np.float32))
    np.save(Path("/kaggle/working") / "test_R35_ft_transformer_strat.npy",
            test_meta.astype(np.float32))
    print(f"[save] wrote oof_R35_ft_transformer_strat.npy + test_..._strat.npy",
          flush=True)

    summary = dict(
        round="R35_ft_transformer_gpu",
        oof_auc=auc_full, fold_aucs=fold_aucs, fold_walls_s=walls,
        wall_total_s=time.time() - t0,
        n_num=len(NUM_FEATS), n_cat=len(CAT_FEATS), n_k17=k17_train.shape[1],
        cat_cardinalities=cat_cardinalities,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
    )
    (Path("/kaggle/working") / "r35_ft_transformer_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"[save] wrote summary  total wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
