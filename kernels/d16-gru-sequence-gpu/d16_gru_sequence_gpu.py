"""Day-16 H1 — Causal-GRU per (Driver, Race, Year) lap sequence (Kaggle T4x2).

α4 prediction-unit reframe (d13 problem-decomposition tree). The K=21+
Path-B pool sees ONLY per-row independent features. No base in the pool
consumes the (Driver, Race, Year) lap-temporal sequence. Hazard-NN
(d9 -315 bp; d10 leakfree OOF 0.92013) failed because of bucket-loss
implementation; this is a clean causal GRU with row-level BCE.

Architecture
------------
Per row r in sequence S(driver, race, year) sorted by LapNumber:
  * Numeric features (z-scored, train-fit): 10 cols
  * Categorical embeddings: Driver(16) + Compound(4) + Year(4) + Stint(4)
    Race is the same across the sequence -> 1-time embedding concat
  * Per-row input vector x_r ∈ R^(10 + 28 = 38)
Sequence pass:
  h_0 = 0
  h_r = GRUCell(x_r, h_{r-1})    # CAUSAL: only sees past+current
  logit_r = Linear(h_r) -> 1
  p_r = sigmoid(logit_r)

Loss: BCE per row, summed over train-fold rows only (5-fold Strat).
Sequence prefix is the same across folds; valid-fold rows compute
forward-pass under the SAME GRU weights but only train-fold losses
update parameters. Causal -> no future-row leakage.

Data layout
-----------
Group key = (Driver, Race, Year).
- ~21k train sequences (mean 21 laps), ~9k test sequences.
- Test sequences are completely separate (i.i.d. row split confirmed
  per U3, but groups don't cross). Test forward-pass uses model trained
  on full-train (5-fold mean of model weights -> used for test pred).

Outputs (under /kaggle/working/):
  oof_d16_gru_seq_strat.npy      (n_train, 2)
  test_d16_gru_seq_strat.npy     (n_test, 2)
  d16_gru_seq_results.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# P100 (sm_60) compat: Kaggle silently routes GpuT4x2 jobs to P100 nodes.
# Force-reinstall torch 2.4 BEFORE any `import torch`.
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
CATS = ["Driver", "Compound", "Race", "Year"]

EMBED_DIMS = {"Driver": 16, "Compound": 4, "Race": 4, "Year": 4}

GRU_HIDDEN = 96
DROPOUT = 0.10
LR = 2e-3
EPOCHS = 12
BATCH_SEQS = 128
GRAD_CLIP = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def boot_env():
    print(f"[boot] Python {sys.version.split()[0]}, torch {torch.__version__}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"[boot] CUDA: {torch.version.cuda}, "
              f"device 0: {torch.cuda.get_device_name(0)}, capability: {cap}")
    else:
        print("[boot] CUDA NOT available; running on CPU (slow)", flush=True)


def find_data_dir() -> Path:
    base = Path("/kaggle/input")
    matches = list(base.rglob("train.csv"))
    if not matches:
        raise RuntimeError(f"no train.csv under {base}")
    return matches[0].parent


def load_data():
    d = find_data_dir()
    print(f"[data] reading from {d}", flush=True)
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    print(f"[data] train={len(train)}  test={len(test)}", flush=True)
    return train, test


def encode_cats(train, test):
    """Label-encode categoricals (with shared vocab from train+test)."""
    encoders = {}
    full = pd.concat([train[CATS], test[CATS]], axis=0, ignore_index=True)
    for c in CATS:
        vals = full[c].astype(str).unique().tolist()
        enc = {v: i for i, v in enumerate(vals)}
        encoders[c] = enc
    for df in (train, test):
        for c in CATS:
            df[c + "_idx"] = df[c].astype(str).map(encoders[c]).astype(np.int64)
    return encoders


def build_sequences(df: pd.DataFrame, has_target: bool):
    """Group rows by (Driver, Race, Year), sort by LapNumber within group.
    Return:
      groups: list of (n_rows_in_seq, idx_into_df, idx_into_df_sorted)
      seq_lens: array of sequence lengths
    Each row keeps its original df index so we can scatter OOF preds back.
    """
    df = df.copy()
    df["__rowid"] = np.arange(len(df))
    grouped = (
        df.groupby(["Driver_idx", "Race_idx", "Year_idx"], sort=False)
          .apply(lambda g: g.sort_values("LapNumber"), include_groups=False)
          .reset_index(level=[0, 1, 2])
    )
    # Now grouped is sorted by group then LapNumber, with __rowid intact.
    keys = grouped[["Driver_idx", "Race_idx", "Year_idx"]].values
    # Find group boundaries.
    diffs = np.any(keys[1:] != keys[:-1], axis=1)
    boundaries = np.r_[0, np.flatnonzero(diffs) + 1, len(grouped)]
    seq_starts = boundaries[:-1]
    seq_ends = boundaries[1:]
    seq_lens = seq_ends - seq_starts
    return grouped, seq_starts, seq_ends, seq_lens


def standardize_numerics(train_df, test_df):
    arr = train_df[NUMERICS].astype(np.float32).values
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    train_df[NUMERICS] = (arr - mean) / std
    test_df[NUMERICS] = (test_df[NUMERICS].astype(np.float32).values - mean) / std
    return mean, std


class GRUSeqModel(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], num_dim: int):
        super().__init__()
        self.embeds = nn.ModuleDict({
            c: nn.Embedding(vocab_sizes[c], EMBED_DIMS[c])
            for c in CATS
        })
        in_dim = num_dim + sum(EMBED_DIMS.values())
        self.gru = nn.GRU(input_size=in_dim, hidden_size=GRU_HIDDEN,
                          num_layers=1, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(GRU_HIDDEN, 1)

    def forward(self, num: torch.Tensor, cat: dict, lengths: torch.Tensor):
        """
        num: (B, T_max, num_dim)
        cat: dict[c -> (B, T_max)]
        lengths: (B,)
        Returns logits: (B, T_max)
        """
        emb = torch.cat([self.embeds[c](cat[c]) for c in CATS], dim=-1)
        x = torch.cat([num, emb], dim=-1)
        # Pack-pad for GRU efficiency
        packed = pack_padded_sequence(x, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True,
                                     total_length=num.shape[1])
        out = self.dropout(out)
        logits = self.head(out).squeeze(-1)
        return logits


def collate_seqs(seq_idx_list, df_grouped, seq_starts, seq_ends,
                 num_cols, cat_cols):
    """Build a padded batch from a list of sequence indices."""
    nums = []
    cats = {c: [] for c in cat_cols}
    targets = []
    masks = []  # 1 where valid (not pad)
    rowids = []
    lens = []
    max_len = max(int(seq_ends[i] - seq_starts[i]) for i in seq_idx_list)
    for i in seq_idx_list:
        s, e = int(seq_starts[i]), int(seq_ends[i])
        ll = e - s
        rows = df_grouped.iloc[s:e]
        num = rows[num_cols].values.astype(np.float32)
        if ll < max_len:
            pad = np.zeros((max_len - ll, num.shape[1]), dtype=np.float32)
            num = np.concatenate([num, pad], axis=0)
        nums.append(num)
        for c in cat_cols:
            v = rows[c + "_idx"].values.astype(np.int64)
            if ll < max_len:
                v = np.concatenate([v, np.zeros(max_len - ll, dtype=np.int64)])
            cats[c].append(v)
        if "PitNextLap" in rows.columns:
            t = rows["PitNextLap"].values.astype(np.float32)
        else:
            t = np.zeros(ll, dtype=np.float32)
        if ll < max_len:
            t = np.concatenate([t, np.zeros(max_len - ll, dtype=np.float32)])
        targets.append(t)
        m = np.zeros(max_len, dtype=np.float32)
        m[:ll] = 1.0
        masks.append(m)
        rid = rows["__rowid"].values
        if ll < max_len:
            rid = np.concatenate([rid, -np.ones(max_len - ll, dtype=np.int64)])
        rowids.append(rid)
        lens.append(ll)
    num_tensor = torch.from_numpy(np.stack(nums))
    cat_tensors = {c: torch.from_numpy(np.stack(cats[c])) for c in cat_cols}
    targets = torch.from_numpy(np.stack(targets))
    masks = torch.from_numpy(np.stack(masks))
    rowids = np.stack(rowids)
    lens = torch.tensor(lens, dtype=torch.long)
    return num_tensor, cat_tensors, targets, masks, rowids, lens


def run_one_fold(fold, train_grouped, t_seq_starts, t_seq_ends, t_seq_lens,
                 vocab_sizes, num_dim, train_oof_idx, valid_oof_idx, n_train):
    """Train GRU on all sequences but only count loss on rows that belong
    to this fold's train subset. Compute OOF preds for this fold's valid
    rows. valid_oof_idx is a boolean mask over the original train df rows."""
    model = GRUSeqModel(vocab_sizes, num_dim).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    n_seqs = len(t_seq_starts)
    seq_perm_rng = np.random.RandomState(SEED + fold)

    fold_oof = np.zeros(n_train, dtype=np.float32)
    fold_oof_count = np.zeros(n_train, dtype=np.float32)
    best_auc = -1.0
    best_state = None
    history = []

    train_mask_full = np.zeros(n_train, dtype=bool)
    train_mask_full[train_oof_idx] = True
    valid_mask_full = np.zeros(n_train, dtype=bool)
    valid_mask_full[valid_oof_idx] = True

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = seq_perm_rng.permutation(n_seqs)
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()
        for batch_start in range(0, n_seqs, BATCH_SEQS):
            idx_batch = order[batch_start:batch_start + BATCH_SEQS].tolist()
            num, catd, tgt, mask, rowids, lens = collate_seqs(
                idx_batch, train_grouped, t_seq_starts, t_seq_ends,
                NUMERICS, CATS,
            )
            num = num.to(DEVICE, non_blocking=True)
            catd = {c: v.to(DEVICE, non_blocking=True) for c, v in catd.items()}
            tgt = tgt.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            # Loss mask: only TRAIN-fold rows; pad rows excluded by mask
            train_row_mask = torch.from_numpy(
                train_mask_full[np.where(rowids >= 0, rowids, 0)]
                & (rowids >= 0)
            ).to(DEVICE).float()
            train_row_mask = train_row_mask * mask  # belt+suspenders
            logits = model(num, catd, lens)
            loss_per = bce(logits, tgt)
            denom = train_row_mask.sum().clamp_min(1.0)
            loss = (loss_per * train_row_mask).sum() / denom
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            epoch_loss += float(loss.item()) * float(denom.item())
            epoch_n += float(denom.item())

        # End-of-epoch: full-pass eval on valid rows
        model.eval()
        valid_preds = np.zeros(n_train, dtype=np.float32)
        valid_count = np.zeros(n_train, dtype=np.float32)
        with torch.no_grad():
            for batch_start in range(0, n_seqs, BATCH_SEQS):
                idx_batch = list(range(batch_start,
                                       min(batch_start + BATCH_SEQS, n_seqs)))
                num, catd, tgt, mask, rowids, lens = collate_seqs(
                    idx_batch, train_grouped, t_seq_starts, t_seq_ends,
                    NUMERICS, CATS,
                )
                num = num.to(DEVICE, non_blocking=True)
                catd = {c: v.to(DEVICE, non_blocking=True) for c, v in catd.items()}
                logits = model(num, catd, lens)
                p = torch.sigmoid(logits).cpu().numpy()
                for b in range(rowids.shape[0]):
                    rids = rowids[b]
                    valid_idx = (rids >= 0)
                    rids_v = rids[valid_idx]
                    valid_preds[rids_v] += p[b, valid_idx]
                    valid_count[rids_v] += 1.0
        valid_preds = np.where(valid_count > 0, valid_preds / np.maximum(valid_count, 1), 0.0)
        valid_only = valid_preds[valid_mask_full]
        try:
            from numpy import asarray
            y_va = train_grouped.set_index("__rowid").loc[
                np.flatnonzero(valid_mask_full)
            ]["PitNextLap"].values.astype(int)
            auc_va = roc_auc_score(y_va, valid_preds[valid_mask_full])
        except Exception as e:
            auc_va = float("nan")
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(epoch_n, 1.0)
        print(f"[fold {fold} ep {epoch:2d}] loss={avg_loss:.5f} "
              f"valid_AUC={auc_va:.5f} ({elapsed:.0f}s)", flush=True)
        history.append({"epoch": epoch, "loss": avg_loss, "valid_auc": auc_va,
                         "wall_s": elapsed})
        if not np.isnan(auc_va) and auc_va > best_auc:
            best_auc = auc_va
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    # Final eval at best epoch (already computed valid_preds at last best epoch
    # implicitly — recompute to be safe).
    model.eval()
    valid_preds = np.zeros(n_train, dtype=np.float32)
    valid_count = np.zeros(n_train, dtype=np.float32)
    with torch.no_grad():
        for batch_start in range(0, n_seqs, BATCH_SEQS):
            idx_batch = list(range(batch_start,
                                   min(batch_start + BATCH_SEQS, n_seqs)))
            num, catd, tgt, mask, rowids, lens = collate_seqs(
                idx_batch, train_grouped, t_seq_starts, t_seq_ends,
                NUMERICS, CATS,
            )
            num = num.to(DEVICE, non_blocking=True)
            catd = {c: v.to(DEVICE, non_blocking=True) for c, v in catd.items()}
            logits = model(num, catd, lens)
            p = torch.sigmoid(logits).cpu().numpy()
            for b in range(rowids.shape[0]):
                rids = rowids[b]
                valid_idx = (rids >= 0)
                rids_v = rids[valid_idx]
                valid_preds[rids_v] += p[b, valid_idx]
                valid_count[rids_v] += 1.0
    valid_preds = np.where(valid_count > 0, valid_preds / np.maximum(valid_count, 1), 0.0)
    return model, valid_preds, best_auc, history


def predict_test(model, test_grouped, te_seq_starts, te_seq_ends, n_test):
    model.eval()
    preds = np.zeros(n_test, dtype=np.float32)
    counts = np.zeros(n_test, dtype=np.float32)
    n_seqs = len(te_seq_starts)
    with torch.no_grad():
        for batch_start in range(0, n_seqs, BATCH_SEQS):
            idx_batch = list(range(batch_start,
                                   min(batch_start + BATCH_SEQS, n_seqs)))
            num, catd, tgt, mask, rowids, lens = collate_seqs(
                idx_batch, test_grouped, te_seq_starts, te_seq_ends,
                NUMERICS, CATS,
            )
            num = num.to(DEVICE, non_blocking=True)
            catd = {c: v.to(DEVICE, non_blocking=True) for c, v in catd.items()}
            logits = model(num, catd, lens)
            p = torch.sigmoid(logits).cpu().numpy()
            for b in range(rowids.shape[0]):
                rids = rowids[b]
                valid_idx = (rids >= 0)
                rids_v = rids[valid_idx]
                preds[rids_v] += p[b, valid_idx]
                counts[rids_v] += 1.0
    preds = np.where(counts > 0, preds / np.maximum(counts, 1), 0.0)
    return preds


def main():
    boot_env()
    t_start = time.time()
    train, test = load_data()
    n_train, n_test = len(train), len(test)

    encoders = encode_cats(train, test)
    vocab_sizes = {c: max(train[c + "_idx"].max(),
                          test[c + "_idx"].max()) + 1
                   for c in CATS}
    print(f"[encode] vocab sizes: {vocab_sizes}", flush=True)

    standardize_numerics(train, test)

    # Build sorted-grouped views once.
    print(f"[seq] building train sequences ...", flush=True)
    tr_grouped, t_starts, t_ends, t_lens = build_sequences(train, has_target=True)
    print(f"[seq] train seqs: {len(t_starts)} (mean len {t_lens.mean():.1f})",
          flush=True)
    print(f"[seq] building test sequences ...", flush=True)
    te_grouped, te_starts, te_ends, te_lens = build_sequences(test, has_target=False)
    print(f"[seq] test seqs:  {len(te_starts)} (mean len {te_lens.mean():.1f})",
          flush=True)

    y = train["PitNextLap"].astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n_train, dtype=np.float32)
    test_pred = np.zeros(n_test, dtype=np.float32)
    fold_aucs = []
    fold_histories = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y)):
        print(f"\n=== FOLD {fold} (train {len(tr_idx)} / valid {len(va_idx)}) ===",
              flush=True)
        model, fold_oof_preds, best_auc, hist = run_one_fold(
            fold, tr_grouped, t_starts, t_ends, t_lens, vocab_sizes,
            num_dim=len(NUMERICS),
            train_oof_idx=tr_idx, valid_oof_idx=va_idx, n_train=n_train,
        )
        oof[va_idx] = fold_oof_preds[va_idx]
        # Predict test from this fold's model
        fold_test_pred = predict_test(model, te_grouped, te_starts, te_ends, n_test)
        test_pred += fold_test_pred / N_FOLDS
        fold_aucs.append(float(best_auc))
        fold_histories.append(hist)
        print(f"=== FOLD {fold} best valid AUC {best_auc:.5f} "
              f"(t={time.time()-t_start:.0f}s) ===", flush=True)

    full_auc = float(roc_auc_score(y, oof))
    print(f"\n[FINAL] OOF AUC = {full_auc:.5f}  per-fold {fold_aucs}", flush=True)

    # Save artifacts in (n, 2) format matching the rest of the harness.
    oof2 = np.column_stack([1.0 - oof, oof])
    test2 = np.column_stack([1.0 - test_pred, test_pred])
    np.save(WORK / "oof_d16_gru_seq_strat.npy", oof2)
    np.save(WORK / "test_d16_gru_seq_strat.npy", test2)

    results = dict(
        n_train=n_train, n_test=n_test,
        vocab_sizes=vocab_sizes,
        gru_hidden=GRU_HIDDEN, dropout=DROPOUT,
        embed_dims=EMBED_DIMS, epochs=EPOCHS, batch_seqs=BATCH_SEQS,
        lr=LR, seed=SEED, n_folds=N_FOLDS,
        full_oof_auc=full_auc,
        fold_aucs=fold_aucs,
        fold_histories=fold_histories,
        wall_total_s=time.time() - t_start,
    )
    (WORK / "d16_gru_seq_results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] artifacts written to {WORK}", flush=True)


if __name__ == "__main__":
    main()
