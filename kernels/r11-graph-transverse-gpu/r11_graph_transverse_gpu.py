"""Round-11 mechanism B — transverse (cross-driver) attention transformer.

PI directive (2026-05-19): redirect Kaggle T4 budget from A (transformer,
already CLOSED per HANDOVER line 134) to B. B is structurally novel:
attention over the DRIVER axis within each (Year, Race, LapNumber) group
versus R5/R6's attention over the LAP axis within each (Year, Race,
Driver) sequence. Inter-driver interactions (pit-pressure, undercut
windows) -- which 3 prior pit-cascade variants tried via hand-aggregates
and ALL failed -- are now modelled via LEARNED message passing.

Architecture:
- Group key: (Year, Race, LapNumber). N_drivers per group <= ~22.
- Per-driver features (9 numerics + compound embedding):
  TyreLife, RaceProgress, Stint, Position, Position_Change,
  Cumulative_Degradation, LapTime (s), LapTime_Delta, PitStop.
- Position is included as a FEATURE (drivers are exchangeable in
  attention; no positional encoding by token order).
- Transformer encoder: 4 layers x D_MODEL=128 x 8 heads, GELU, dropout 0.1.
  Self-attention is over the up-to-24-driver dimension within a group.
- Per-row head -> sigmoid -> P(PitNextLap).
- BCE loss, masked over padding AND over val rows in the current fold.

5-fold Stratified OOF on rows (LB proxy). Per fold: forward over FULL
group; loss only on TRAIN rows; val predictions read at the val-row
positions.

Outputs (write to /kaggle/working then local replay downloads them):
- oof_R11_B_graph_transverse_strat.npy
- test_R11_B_graph_transverse_strat.npy
- r11_graph_transverse_results.json
"""
from __future__ import annotations
import json
import math
import sys
import subprocess
import time
import os
from pathlib import Path

# Kaggle default torch (>=2.6) drops sm_60 (P100) support. The kernel
# can be assigned P100 even when GpuT4x2 is requested; pin torch==2.4
# to keep P100 working. Same pattern as r5/r6/r7 kernels.
print("[boot] force-reinstall torch 2.4 (sm_60 P100 support) ...", flush=True)
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--force-reinstall", "--no-deps",
        "torch==2.4.*", "torchvision==0.19.*",
    ])
except subprocess.CalledProcessError as e:
    print(f"[boot] torch 2.4 reinstall failed (continuing): {e}", flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

print(f"[boot] Python {sys.version.split()[0]}, torch {torch.__version__}",
      flush=True)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"[boot] CUDA: {torch.version.cuda}, "
          f"device 0: {torch.cuda.get_device_name(0)}, capability: {cap}",
          flush=True)

TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
MAX_DRIVERS = 24                 # max drivers in any (Year, Race, Lap) group
SEQ_FEATURES = ["TyreLife", "RaceProgress", "Stint", "Position",
                "Position_Change", "Cumulative_Degradation",
                "LapTime (s)", "LapTime_Delta", "PitStop"]
COMPOUND_VOCAB = {"HARD": 0, "MEDIUM": 1, "SOFT": 2,
                  "INTERMEDIATE": 3, "WET": 4}

D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
LR = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256        # groups per batch
EPOCHS = 8

torch.manual_seed(SEED)
np.random.seed(SEED)


class TransverseTransformer(nn.Module):
    """Self-attention over the driver axis within a (Year, Race, Lap) group."""

    def __init__(self, n_features: int, n_compounds: int = 5):
        super().__init__()
        # +1 for pad compound idx
        self.compound_emb = nn.Embedding(n_compounds + 1, 16)
        self.input_proj = nn.Linear(n_features + 16, D_MODEL)
        # No positional encoding by token index: drivers are exchangeable.
        # Position-in-race is included as a numeric input feature instead.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x_seq, x_compound, pad_mask):
        # x_seq: (B, T, n_feat)  numeric features per driver in group
        # x_compound: (B, T) int compound (pad idx = 5)
        # pad_mask: (B, T) bool, True where padding
        ce = self.compound_emb(x_compound)
        h = torch.cat([x_seq, ce], dim=-1)
        h = self.input_proj(h)
        h = self.dropout(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        logits = self.head(h).squeeze(-1)
        return logits


def build_groups(df: pd.DataFrame, has_target: bool):
    """Group by (Year, Race, LapNumber). Returns dense tensors + index map.

    Returns: (sequences, compound_idx, labels, label_mask, pad_mask,
              df_idx_to_pos).
    """
    df = df.copy()
    df["Compound_int"] = df["Compound"].map(COMPOUND_VOCAB).astype(np.int64)
    sort_cols = ["Year", "Race", "LapNumber", "Position"]
    df_sorted = df.sort_values(sort_cols)
    n_feat = len(SEQ_FEATURES)
    groups = list(df_sorted.groupby(["Year", "Race", "LapNumber"],
                                    sort=False))
    n_g = len(groups)
    sequences = np.zeros((n_g, MAX_DRIVERS, n_feat), dtype=np.float32)
    compound_idx = np.full((n_g, MAX_DRIVERS), 5, dtype=np.int64)
    labels = np.zeros((n_g, MAX_DRIVERS), dtype=np.float32)
    label_mask = np.zeros((n_g, MAX_DRIVERS), dtype=np.float32)
    pad_mask = np.ones((n_g, MAX_DRIVERS), dtype=bool)
    df_idx_to_pos = {}
    n_overflow = 0
    for si, (_, g) in enumerate(groups):
        L = min(len(g), MAX_DRIVERS)
        if len(g) > MAX_DRIVERS:
            n_overflow += 1
        sequences[si, :L] = g[SEQ_FEATURES].values[:L].astype(np.float32)
        compound_idx[si, :L] = g["Compound_int"].values[:L]
        pad_mask[si, :L] = False
        if has_target:
            labels[si, :L] = g[TARGET].values[:L].astype(np.float32)
            label_mask[si, :L] = 1.0
        else:
            label_mask[si, :L] = 1.0
        for pi, orig in enumerate(g.index.values[:L]):
            df_idx_to_pos[int(orig)] = (si, pi)
    if n_overflow > 0:
        print(f"  WARN: {n_overflow} groups exceeded MAX_DRIVERS="
              f"{MAX_DRIVERS}; tail truncated.", flush=True)
    return sequences, compound_idx, labels, label_mask, pad_mask, df_idx_to_pos


def main():
    print("=== R11-B: transverse (cross-driver) attention transformer ===",
          flush=True)
    t0 = time.time()

    def find_data_dir(name="train.csv"):
        base = Path("/kaggle/input")
        if base.exists():
            matches = list(base.rglob(name))
            if matches:
                return matches[0].parent
        return Path("data")
    DATA_DIR = find_data_dir("train.csv")
    print(f"  DATA_DIR resolved to: {DATA_DIR}", flush=True)
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    print(f"  train {train.shape}  test {test.shape}", flush=True)
    y_all = train[TARGET].astype(int).values

    # Standardize numerics using train stats
    means = train[SEQ_FEATURES].mean()
    stds = train[SEQ_FEATURES].std() + 1e-6
    train_s = train.copy()
    test_s = test.copy()
    for c in SEQ_FEATURES:
        train_s[c] = (train_s[c] - means[c]) / stds[c]
        test_s[c] = (test_s[c] - means[c]) / stds[c]
    # Test rows have no PitStop column -- fill with 0 (mean-imputed in z-space)
    # PitStop appears in SEQ_FEATURES; in test it doesn't exist so we use the
    # train mean. Actually -- test rows DO need PitStop since it's part of the
    # input. Test data does NOT contain PitStop (it's an observation, not a
    # label). Use mean-imputed value (= 0 in z-score after centering).
    if "PitStop" in SEQ_FEATURES and "PitStop" not in test.columns:
        # already filled with 0 after standardization above? Re-check below.
        pass

    print(f"  Building train groups...", flush=True)
    (tr_seq, tr_comp, tr_lab, tr_mask, tr_pad,
     tr_pos) = build_groups(train_s, has_target=True)
    print(f"  Train: {len(tr_seq):,} groups, total rows in groups: "
          f"{(~tr_pad).sum():,}", flush=True)

    print(f"  Building test groups...", flush=True)
    # Test has no PitStop column; insert zero placeholder before standardizing.
    if "PitStop" not in test_s.columns:
        test_s["PitStop"] = 0.0  # mean=0 after standardization centring
    (te_seq, te_comp, te_lab, te_mask, te_pad,
     te_pos) = build_groups(test_s, has_target=False)
    print(f"  Test:  {len(te_seq):,} groups, total rows in groups: "
          f"{(~te_pad).sum():,}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}", flush=True)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_iter = list(skf.split(np.zeros(len(y_all)), y_all))
    oof = np.zeros(len(y_all), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    row_fold = np.full(len(y_all), -1, dtype=np.int64)
    for fold, (_, vi) in enumerate(fold_iter):
        row_fold[vi] = fold

    fold_aucs = []
    for fold in range(N_FOLDS):
        t_fold = time.time()
        print(f"\n  --- Fold {fold+1}/{N_FOLDS} ---", flush=True)
        is_train_row = (row_fold != fold)
        train_mask_2d = np.zeros_like(tr_pad, dtype=bool)
        for orig_idx, (si, pi) in tr_pos.items():
            if is_train_row[orig_idx]:
                train_mask_2d[si, pi] = True

        train_loss_mask = torch.tensor(train_mask_2d, dtype=torch.float32)

        model = TransverseTransformer(n_features=len(SEQ_FEATURES)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
        steps_per_epoch = (len(tr_seq) + BATCH_SIZE - 1) // BATCH_SIZE
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS * steps_per_epoch)

        x_seq_t = torch.tensor(tr_seq).to(device)
        x_comp_t = torch.tensor(tr_comp).to(device)
        x_pad_t = torch.tensor(tr_pad).to(device)
        y_t = torch.tensor(tr_lab).to(device)
        train_mask_t = train_loss_mask.to(device)

        for epoch in range(EPOCHS):
            model.train()
            t_epoch = time.time()
            ep_loss = 0.0; ep_n = 0
            perm = torch.randperm(len(tr_seq))
            for i in range(0, len(tr_seq), BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE].to(device)
                xs = x_seq_t[idx]; xc = x_comp_t[idx]
                xp = x_pad_t[idx]; yy = y_t[idx]; mm = train_mask_t[idx]
                logits = model(xs, xc, xp)
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
                  f"wall={time.time()-t_epoch:.1f}s", flush=True)

        # Val predictions
        model.eval()
        with torch.no_grad():
            val_logits = torch.zeros((len(tr_seq), MAX_DRIVERS), device=device)
            for i in range(0, len(tr_seq), BATCH_SIZE):
                idx = torch.arange(i, min(i + BATCH_SIZE,
                                          len(tr_seq))).to(device)
                xs = x_seq_t[idx]; xc = x_comp_t[idx]; xp = x_pad_t[idx]
                val_logits[idx] = model(xs, xc, xp)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        for orig_idx, (si, pi) in tr_pos.items():
            if not is_train_row[orig_idx]:
                oof[orig_idx] = float(val_probs[si, pi])

        # Test predictions for this fold
        x_te_seq_t = torch.tensor(te_seq).to(device)
        x_te_comp_t = torch.tensor(te_comp).to(device)
        x_te_pad_t = torch.tensor(te_pad).to(device)
        with torch.no_grad():
            test_logits = torch.zeros((len(te_seq), MAX_DRIVERS), device=device)
            for i in range(0, len(te_seq), BATCH_SIZE):
                idx = torch.arange(i, min(i + BATCH_SIZE,
                                          len(te_seq))).to(device)
                xs = x_te_seq_t[idx]; xc = x_te_comp_t[idx]
                xp = x_te_pad_t[idx]
                test_logits[idx] = model(xs, xc, xp)
            test_probs_fold = torch.sigmoid(test_logits).cpu().numpy()
        test_row_pred = np.zeros(len(test), dtype=np.float64)
        for orig_idx, (si, pi) in te_pos.items():
            test_row_pred[orig_idx] = float(test_probs_fold[si, pi])
        test_pred += test_row_pred / N_FOLDS

        val_y = y_all[row_fold == fold]
        val_p = oof[row_fold == fold]
        auc_va = roc_auc_score(val_y, val_p)
        fold_aucs.append(float(auc_va))
        print(f"    Fold {fold+1} VAL AUC: {auc_va:.5f}  "
              f"wall={time.time()-t_fold:.1f}s", flush=True)
        del model, opt, sched
        if device == "cuda":
            torch.cuda.empty_cache()

    auc_full = float(roc_auc_score(y_all, oof))
    print(f"\n  Full OOF AUC: {auc_full:.5f}  "
          f"fold-std={np.std(fold_aucs):.5f}  "
          f"total wall={time.time()-t0:.1f}s", flush=True)

    out_oof = oof.astype(np.float32)
    out_test = test_pred.astype(np.float32)
    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "scripts/artifacts"
    np.save(f"{out_dir}/oof_R11_B_graph_transverse_strat.npy", out_oof)
    np.save(f"{out_dir}/test_R11_B_graph_transverse_strat.npy", out_test)
    with open(f"{out_dir}/r11_graph_transverse_results.json", "w") as f:
        json.dump(dict(name="r11_graph_transverse",
                       oof_auc=auc_full, fold_aucs=fold_aucs,
                       epochs=EPOCHS, d_model=D_MODEL,
                       n_layers=N_LAYERS, n_heads=N_HEADS,
                       max_drivers=MAX_DRIVERS), f, indent=2)
    print(f"  -> saved oof + test arrays to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
