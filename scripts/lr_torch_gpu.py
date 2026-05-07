"""scripts/lr_torch_gpu.py — PyTorch LR replica of Chris's GPU LR-stacker.

Chris Deotte's 2nd-place s6e4 stacker was a multinomial LR fit in
PyTorch on GPU with: (i) class_weight='balanced', (ii) L2 penalty on
coefficients only (not bias), (iii) Adam optimizer, (iv) 5-fold CV
identical to ours, (v) forward selection over a 125-base bank.

For binary AUC the multinomial equivalent collapses to regular LR.
We replicate the architecture as a learning artifact (CPU here; works
unchanged on GPU — change DEVICE='cuda').

Default feature set: a "wide" stack — degree-2 polynomial of 11
numerics + KBins(20)-quantile-OHE + cat OHE. ~250 features per row,
sparse-ish; trains in seconds with batched SGD.

Saves oof + test artifacts and prints timing + AUC for comparison
with the sklearn equivalents in lr_bank.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    KBinsDiscretizer, OneHotEncoder, PolynomialFeatures, StandardScaler,
)

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5

NUM_COLS = [
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "Year",
]
CAT_COLS = ["Driver", "Compound", "Race"]


def build_wide_features(train, test):
    """Wide LR-friendly feature matrix: poly2 + KBins(20)-quantile-OHE + cat-OHE."""
    num_tr = train[NUM_COLS].fillna(0).values.astype(np.float32)
    num_te = test[NUM_COLS].fillna(0).values.astype(np.float32)

    pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    Xp_tr = pf.fit_transform(num_tr)
    Xp_te = pf.transform(num_te)
    sc = StandardScaler()
    Xp_tr = sc.fit_transform(Xp_tr).astype(np.float32)
    Xp_te = sc.transform(Xp_te).astype(np.float32)

    combined = np.vstack([num_tr, num_te])
    kb = KBinsDiscretizer(n_bins=20, encode="onehot", strategy="quantile", subsample=None)
    kb.fit(combined)
    Bk_tr = kb.transform(num_tr)
    Bk_te = kb.transform(num_te)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    Oc_tr = enc.transform(train[CAT_COLS])
    Oc_te = enc.transform(test[CAT_COLS])

    Xtr = sparse.hstack([sparse.csr_matrix(Xp_tr), Bk_tr, Oc_tr], format="csr")
    Xte = sparse.hstack([sparse.csr_matrix(Xp_te), Bk_te, Oc_te], format="csr")
    return Xtr.astype(np.float32), Xte.astype(np.float32)


def sparse_csr_to_torch(M: sparse.csr_matrix, device: str) -> torch.Tensor:
    """Convert scipy CSR to torch sparse_csr."""
    Mc = M.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(Mc.indptr.astype(np.int64)),
        torch.from_numpy(Mc.indices.astype(np.int64)),
        torch.from_numpy(Mc.data.astype(np.float32)),
        size=Mc.shape, device=device,
    )


def fit_lr_torch(Xtr_sp, y_tr, Xva_sp, Xte_sp, l2: float = 1.0, lr: float = 0.05,
                 epochs: int = 30, batch: int = 16384, balanced: bool = True,
                 device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Fit logistic regression in PyTorch with L2-coef-only, balanced class weight.

    Returns (val_p, test_p). L2 is applied to W only (Chris's recipe), not bias.
    """
    n, d = Xtr_sp.shape
    Xtr = sparse_csr_to_torch(Xtr_sp, device)
    Xva = sparse_csr_to_torch(Xva_sp, device)
    Xte = sparse_csr_to_torch(Xte_sp, device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)

    W = torch.zeros(d, requires_grad=True, device=device)
    b = torch.zeros(1, requires_grad=True, device=device)
    opt = torch.optim.Adam([W, b], lr=lr)
    pw = None
    if balanced:
        pw = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                          dtype=torch.float32, device=device)

    idx = torch.arange(n, device=device)
    for ep in range(epochs):
        perm = idx[torch.randperm(n, device=device)]
        for s in range(0, n, batch):
            sel = perm[s:s+batch]
            Xb = torch.index_select(Xtr, 0, sel).to_dense()
            yb = yt[sel]
            logits = Xb @ W + b
            if pw is not None:
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss = loss + 0.5 * l2 / n * (W * W).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        val_logits = (Xva.to_dense() @ W + b).cpu().numpy()
        test_logits = (Xte.to_dense() @ W + b).cpu().numpy()
    val_p = 1.0 / (1.0 + np.exp(-val_logits))
    test_p = 1.0 / (1.0 + np.exp(-test_logits))
    return val_p, test_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lr_torch_wide")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--balanced", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, threads: {torch.get_num_threads()}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values

    print("building wide feature matrix...")
    t0 = time.time()
    Xtr, Xte = build_wide_features(train, test)
    print(f"  Xtr {Xtr.shape}, density {Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1]):.4f} "
          f"({time.time()-t0:.1f}s)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(Xte.shape[0], dtype=np.float64)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        ts = time.time()
        val_p, test_p = fit_lr_torch(
            Xtr[tr], y[tr], Xtr[va], Xte,
            l2=args.l2, lr=args.lr, epochs=args.epochs,
            batch=args.batch, balanced=args.balanced, device=device)
        oof[va] = val_p
        test_pred += test_p / N_FOLDS
        auc_fold = float(roc_auc_score(y[va], val_p))
        print(f"  fold {fold}: AUC {auc_fold:.5f}  ({time.time()-ts:.1f}s)")

    elapsed = time.time() - t0
    auc = float(roc_auc_score(y, oof))
    print(f"\n[{args.name}] OOF AUC {auc:.5f}  total {elapsed:.1f}s")

    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / f"oof_{args.name}_strat.npy", oof2)
    np.save(ART / f"test_{args.name}_strat.npy", test2)

    res = dict(name=args.name, auc=auc, time_s=elapsed,
               n_features=int(Xtr.shape[1]), epochs=args.epochs, lr=args.lr,
               l2=args.l2, batch=args.batch, balanced=args.balanced, device=device)
    (ART / f"{args.name}_results.json").write_text(json.dumps(res, indent=2))
    print(f"  saved oof_{args.name}_strat.npy + json")


if __name__ == "__main__":
    main()
