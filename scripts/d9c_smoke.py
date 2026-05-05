"""50k 1-fold smoke for the FM model — verifies kernel + timing."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import d9c_fm as d9c

N_SAMPLE = 50_000
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
rng = np.random.default_rng(0)
samp = rng.choice(len(train), size=N_SAMPLE, replace=False)
train_s = train.iloc[samp].reset_index(drop=True)
y = train_s[d9c.TARGET].astype(int).values

idx = np.arange(len(y))
rng.shuffle(idx)
tr, va = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)):]

t0 = time.time()
Xtr_csr, Xte_csr = d9c.build_main_effect_hashes(train_s, test, tr)
idx_tr_full = d9c.csr_to_index_array(Xtr_csr)
idx_te = d9c.csr_to_index_array(Xte_csr)
print(f"feat-build wall {time.time()-t0:.1f}s; shapes "
      f"train_idx {idx_tr_full.shape} test_idx {idx_te.shape}")
p_va, p_te = d9c.fit_fm_one_fold(idx_tr_full, idx_te, y, tr, va, seed=42)
print(f"smoke total wall {time.time()-t0:.1f}s; val_AUC {roc_auc_score(y[va], p_va):.5f}")
