"""Round-5 Phase F — gap-aware transformer on per-(Year, Race, Driver) sequences.

Predicted +1.25 bp midpoint per Round-3 headroom math (highest-EV mechanism
in the remaining queue). The Round-4 plateau-break (mechanism-orthogonal
stacking of row-FE + sequence-HMM) suggests cross-class combinations work
where single-class refinement fails — the transformer adds attention-class
diversity orthogonal to both row-FE and HMM-state-class.

Architecture:
- Input: per-(Year, Race, Driver) sequence, lap-ordered.
- Per-row features (9 numerics): Compound_int (5-way one-hot via embedding),
  TyreLife, RaceProgress, Stint, Position_Change, Cumulative_Degradation,
  LapTime_s, LapTime_Delta, LapNumber_norm.
- Sinusoidal positional encoding indexed by LapNumber (gap-aware: VSC/SC
  laps skip values, so PE captures the actual lap index, not row index).
- Transformer encoder: 4 layers × 128 dim × 8 heads, dropout 0.1.
- Output: per-row sigmoid → P(PitNextLap).
- Loss: BCEWithLogits per row; masked over padding positions.

5-fold Stratified OOF on rows (not sequences). At training: forward over
FULL sequence; loss only on rows in train-fold. At eval: predict on val-fold
rows.

Outputs:
- oof_r5_transformer_strat.npy  (2-col [1-p, p] convention)
- test_r5_transformer_strat.npy
- r5_transformer_results.json
"""
from __future__ import annotations
import json
import math
import sys
import subprocess
import time
import os
from pathlib import Path

# Kaggle's default torch (>=2.6) drops sm_60 (P100) support. The
# kernel can be assigned P100 even when GpuT4x2 is requested; force
# torch==2.4 to keep P100 working. Same pattern as kernels/d15b-dae-gpu.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

print(f"[boot] Python {sys.version.split()[0]}, torch {torch.__version__}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"[boot] CUDA: {torch.version.cuda}, "
          f"device 0: {torch.cuda.get_device_name(0)}, capability: {cap}")

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_SEQ_LEN = 40  # max in train is 38; pad to 40 for safety
SEQ_FEATURES = ["TyreLife", "RaceProgress", "Stint", "Position_Change",
                "Cumulative_Degradation", "LapTime (s)", "LapTime_Delta",
                "LapNumber"]
COMPOUND_VOCAB = {"HARD": 0, "MEDIUM": 1, "SOFT": 2, "INTERMEDIATE": 3, "WET": 4}

D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
LR = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
EPOCHS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)


class SeqDataset(Dataset):
    def __init__(self, sequences, labels, label_mask, lap_indices):
        self.sequences = sequences      # (n_seq, MAX_SEQ_LEN, n_feat)
        self.labels = labels            # (n_seq, MAX_SEQ_LEN)
        self.label_mask = label_mask    # (n_seq, MAX_SEQ_LEN) — 1 if loss applies
        self.lap_indices = lap_indices  # (n_seq, MAX_SEQ_LEN) — int lap number (-1 if pad)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        return (self.sequences[i], self.labels[i],
                self.label_mask[i], self.lap_indices[i])


class GapAwareTransformer(nn.Module):
    def __init__(self, n_features, n_compounds=5):
        super().__init__()
        self.compound_emb = nn.Embedding(n_compounds + 1, 16)  # +1 for pad
        # Linear projection from (8 numerics + 16-dim compound emb) → D_MODEL
        self.input_proj = nn.Linear(len(SEQ_FEATURES) + 16, D_MODEL)
        # Sinusoidal PE indexed by LapNumber (max ~70)
        max_lap = 80
        pe = torch.zeros(max_lap, D_MODEL)
        position = torch.arange(0, max_lap).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, D_MODEL, 2).float()
                              * -(math.log(10000.0) / D_MODEL))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x_seq, x_compound, lap_idx, pad_mask):
        # x_seq:  (B, T, n_feat) numerics
        # x_compound: (B, T) int compound (with pad idx = 5)
        # lap_idx: (B, T) int LapNumber (pad = 0)
        # pad_mask: (B, T) bool — True at padding positions
        ce = self.compound_emb(x_compound)  # (B, T, 16)
        h = torch.cat([x_seq, ce], dim=-1)  # (B, T, n_feat+16)
        h = self.input_proj(h)              # (B, T, D)
        # Add positional encoding by lap index (gap-aware)
        lap_clamped = lap_idx.clamp(min=0, max=self.pe.size(0) - 1)
        pe_lookup = self.pe[lap_clamped]    # (B, T, D)
        h = h + pe_lookup
        h = self.dropout(h)
        # Transformer expects key_padding_mask: True at PAD positions
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        logits = self.head(h).squeeze(-1)   # (B, T)
        return logits


def build_seqs(df, has_target):
    """Group by (Year, Race, Driver), sort by LapNumber, pad to MAX_SEQ_LEN.
    Returns: sequences (n,T,n_feat), compound_idx (n,T), labels (n,T),
    label_mask (n,T), lap_idx (n,T), row_to_seq_pos (df_index → (seq, pos)).
    """
    df = df.copy()
    df["Compound_int"] = df["Compound"].map(COMPOUND_VOCAB).astype(np.int64)
    # standardize numerics with train stats — these will be passed in
    sort_cols = ["Year", "Race", "Driver", "LapNumber"]
    df_sorted = df.sort_values(sort_cols)
    sort_idx = df_sorted.index.values  # original df index → sorted position
    n_feat = len(SEQ_FEATURES)
    # Group iteration to build dense tensors
    groups = list(df_sorted.groupby(["Year", "Race", "Driver"], sort=False))
    n_seq = len(groups)
    sequences = np.zeros((n_seq, MAX_SEQ_LEN, n_feat), dtype=np.float32)
    compound_idx = np.full((n_seq, MAX_SEQ_LEN), 5, dtype=np.int64)  # 5 = pad
    labels = np.zeros((n_seq, MAX_SEQ_LEN), dtype=np.float32)
    label_mask = np.zeros((n_seq, MAX_SEQ_LEN), dtype=np.float32)
    lap_idx_arr = np.zeros((n_seq, MAX_SEQ_LEN), dtype=np.int64)
    pad_mask = np.ones((n_seq, MAX_SEQ_LEN), dtype=bool)  # True = pad
    df_idx_to_pos = {}
    for si, (_, g) in enumerate(groups):
        L = min(len(g), MAX_SEQ_LEN)
        sequences[si, :L] = g[SEQ_FEATURES].values[:L].astype(np.float32)
        compound_idx[si, :L] = g["Compound_int"].values[:L]
        lap_idx_arr[si, :L] = g["LapNumber"].values[:L].astype(np.int64)
        pad_mask[si, :L] = False
        if has_target:
            labels[si, :L] = g[TARGET].values[:L].astype(np.float32)
            label_mask[si, :L] = 1.0
        else:
            label_mask[si, :L] = 1.0  # all rows have predictions, no loss
        for pi, orig in enumerate(g.index.values[:L]):
            df_idx_to_pos[int(orig)] = (si, pi)
    return (sequences, compound_idx, labels, label_mask,
            lap_idx_arr, pad_mask, df_idx_to_pos)


def main():
    print(f"=== R5 Phase F: gap-aware transformer ===")
    t0 = time.time()

    def find_data_dir(name="train.csv"):
        base = Path("/kaggle/input")
        if base.exists():
            matches = list(base.rglob(name))
            if matches:
                return matches[0].parent
        return Path("data")
    DATA_DIR = find_data_dir("train.csv")
    print(f"  DATA_DIR resolved to: {DATA_DIR}")
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    print(f"  train {train.shape}  test {test.shape}")

    y_all = train[TARGET].astype(int).values

    # Standardize numerics using train stats
    means = train[SEQ_FEATURES].mean()
    stds = train[SEQ_FEATURES].std() + 1e-6
    train_s = train.copy()
    test_s = test.copy()
    for c in SEQ_FEATURES:
        train_s[c] = (train_s[c] - means[c]) / stds[c]
        test_s[c] = (test_s[c] - means[c]) / stds[c]

    print(f"  Building train sequences...")
    (tr_seq, tr_comp, tr_lab, tr_mask, tr_lap, tr_pad, tr_pos) = build_seqs(train_s, has_target=True)
    print(f"  Train: {len(tr_seq):,} sequences, total rows in seqs: "
          f"{(~tr_pad).sum():,}")

    print(f"  Building test sequences...")
    (te_seq, te_comp, te_lab, te_mask, te_lap, te_pad, te_pos) = build_seqs(test_s, has_target=False)
    print(f"  Test:  {len(te_seq):,} sequences, total rows in seqs: "
          f"{(~te_pad).sum():,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # 5-fold Stratified OOF on rows
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_iter = list(skf.split(np.zeros(len(y_all)), y_all))
    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    # Per-fold mask: which positions are training (loss applies) vs val
    # We build a row-fold map then convert to (seq, pos) masks.
    row_fold = np.full(len(y_all), -1, dtype=np.int64)
    for fold, (_, vi) in enumerate(fold_iter):
        row_fold[vi] = fold

    fold_aucs = []

    for fold in range(N_FOLDS):
        t_fold = time.time()
        print(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
        # Build train-fold mask on tr_lab positions
        is_train_row = (row_fold != fold)
        # Mask per (seq, pos): True if position is non-pad AND row_fold != fold
        train_mask_2d = np.zeros_like(tr_pad, dtype=bool)
        for orig_idx, (si, pi) in tr_pos.items():
            if is_train_row[orig_idx]:
                train_mask_2d[si, pi] = True
        # Val mask: non-pad AND row_fold == fold
        val_mask_2d = (~tr_pad) & (~train_mask_2d)

        train_loss_mask = torch.tensor(train_mask_2d, dtype=torch.float32)
        val_loss_mask   = torch.tensor(val_mask_2d, dtype=torch.float32)

        ds_train = SeqDataset(torch.tensor(tr_seq), torch.tensor(tr_comp),
                              torch.tensor(tr_lap), torch.tensor(tr_pad))
        dl_train = DataLoader(range(len(ds_train.sequences)),
                              batch_size=BATCH_SIZE, shuffle=True)

        model = GapAwareTransformer(n_features=len(SEQ_FEATURES)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS * len(dl_train))

        x_seq_t = torch.tensor(tr_seq).to(device)
        x_comp_t = torch.tensor(tr_comp).to(device)
        x_lap_t = torch.tensor(tr_lap).to(device)
        x_pad_t = torch.tensor(tr_pad).to(device)
        y_t = torch.tensor(tr_lab).to(device)
        train_mask_t = train_loss_mask.to(device)

        for epoch in range(EPOCHS):
            model.train()
            t_epoch = time.time()
            ep_loss = 0.0; ep_n = 0
            # Shuffle sequence order each epoch
            perm = torch.randperm(len(tr_seq))
            for i in range(0, len(tr_seq), BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE].to(device)
                xs = x_seq_t[idx]
                xc = x_comp_t[idx]
                xl = x_lap_t[idx]
                xp = x_pad_t[idx]
                yy = y_t[idx]
                mm = train_mask_t[idx]
                logits = model(xs, xc, xl, xp)
                loss_per_pos = F.binary_cross_entropy_with_logits(
                    logits, yy, reduction="none")
                loss = (loss_per_pos * mm).sum() / (mm.sum() + 1e-6)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                ep_loss += loss.item() * mm.sum().item()
                ep_n += mm.sum().item()
            ep_avg = ep_loss / max(ep_n, 1)
            print(f"    Epoch {epoch+1}/{EPOCHS}  loss={ep_avg:.5f}  "
                  f"wall={time.time()-t_epoch:.1f}s")

        # Val predictions on this fold
        model.eval()
        with torch.no_grad():
            val_logits = torch.zeros((len(tr_seq), MAX_SEQ_LEN), device=device)
            for i in range(0, len(tr_seq), BATCH_SIZE):
                idx = torch.arange(i, min(i + BATCH_SIZE, len(tr_seq))).to(device)
                xs = x_seq_t[idx]
                xc = x_comp_t[idx]
                xl = x_lap_t[idx]
                xp = x_pad_t[idx]
                logits = model(xs, xc, xl, xp)
                val_logits[idx] = logits
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        # Map back from (seq, pos) → original df row index for VAL rows
        for orig_idx, (si, pi) in tr_pos.items():
            if not is_train_row[orig_idx]:
                oof[orig_idx] = float(val_probs[si, pi])

        # Test predictions
        x_te_seq_t = torch.tensor(te_seq).to(device)
        x_te_comp_t = torch.tensor(te_comp).to(device)
        x_te_lap_t = torch.tensor(te_lap).to(device)
        x_te_pad_t = torch.tensor(te_pad).to(device)
        with torch.no_grad():
            test_logits = torch.zeros((len(te_seq), MAX_SEQ_LEN), device=device)
            for i in range(0, len(te_seq), BATCH_SIZE):
                idx = torch.arange(i, min(i + BATCH_SIZE, len(te_seq))).to(device)
                xs = x_te_seq_t[idx]
                xc = x_te_comp_t[idx]
                xl = x_te_lap_t[idx]
                xp = x_te_pad_t[idx]
                logits = model(xs, xc, xl, xp)
                test_logits[idx] = logits
            test_probs_fold = torch.sigmoid(test_logits).cpu().numpy()
        # Aggregate test predictions per-row over folds
        test_row_pred = np.zeros(len(test), dtype=np.float64)
        for orig_idx, (si, pi) in te_pos.items():
            test_row_pred[orig_idx] = float(test_probs_fold[si, pi])
        test_pred += test_row_pred / N_FOLDS

        # Fold AUC
        val_y = y_all[row_fold == fold]
        val_p = oof[row_fold == fold]
        auc_va = roc_auc_score(val_y, val_p)
        fold_aucs.append(float(auc_va))
        print(f"    Fold {fold+1} VAL AUC: {auc_va:.5f}  "
              f"wall={time.time()-t_fold:.1f}s")
        del model, opt, sched
        if device == "cuda":
            torch.cuda.empty_cache()

    auc_full = float(roc_auc_score(y_all, oof))
    print(f"\n  Full OOF AUC: {auc_full:.5f}  "
          f"fold-std={np.std(fold_aucs):.5f}  total wall={time.time()-t0:.1f}s")

    # Save outputs (2-col convention)
    out_oof = np.column_stack([1 - oof, oof]).astype(np.float64)
    out_test = np.column_stack([1 - test_pred, test_pred]).astype(np.float64)
    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "scripts/artifacts"
    np.save(f"{out_dir}/oof_r5_transformer_strat.npy", out_oof)
    np.save(f"{out_dir}/test_r5_transformer_strat.npy", out_test)
    with open(f"{out_dir}/r5_transformer_results.json", "w") as f:
        json.dump(dict(name="r5_transformer", oof_auc=auc_full,
                       fold_aucs=fold_aucs, epochs=EPOCHS,
                       d_model=D_MODEL, n_layers=N_LAYERS,
                       n_heads=N_HEADS, max_seq_len=MAX_SEQ_LEN), f, indent=2)
    print(f"  -> saved oof + test arrays to {out_dir}/")


if __name__ == "__main__":
    main()
