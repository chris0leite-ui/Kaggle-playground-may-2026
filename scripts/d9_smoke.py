"""Smoke-test the 10 rule kernels in d9_math_heuristics.py on a 50k-row
subsample, single fold. Per Rule 2 (smoke + 1-fold time-probe). Verifies
correctness + timing before the full 5-fold pass."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import d9_math_heuristics as d9

N_SAMPLE = 50_000

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y_full = train[d9.TARGET].astype(int).values
rng = np.random.default_rng(0)
samp = rng.choice(len(train), size=N_SAMPLE, replace=False)
train_s = train.iloc[samp].reset_index(drop=True)
y = train_s[d9.TARGET].astype(int).values

# Pre-compute neighbours on the FULL union (uses original train+test indices)
full = pd.concat([train_s.assign(__src="tr"), test.assign(__src="te")],
                 ignore_index=True)
nx, pv = d9._compute_neighbour_compounds(full)
next_train = nx[full["__src"].values == "tr"]
next_test = nx[full["__src"].values == "te"]
prev_train = pv[full["__src"].values == "tr"]
prev_test = pv[full["__src"].values == "te"]

X = train_s.drop(columns=[d9.TARGET, d9.ID_COL], errors="ignore").copy()
X_test = test.drop(columns=[d9.ID_COL], errors="ignore").copy()
X_enc, X_test_enc = d9.encode_features(X.copy(), X_test.copy())

idx = np.arange(len(y))
rng.shuffle(idx)
tr, va = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)):]

hash_train = d9._make_hash_features(train_s)
hash_test = d9._make_hash_features(test)

approaches = [
    ("R5_weibull_compound", d9.rule_weibull_compound, True, {}),
    ("R6_next_compound", d9.rule_next_compound, True,
     dict(next_train=next_train, next_test=next_test)),
    ("R7_prev_compound", d9.rule_prev_compound, True,
     dict(prev_train=prev_train, prev_test=prev_test)),
    ("R8_position_progress", d9.rule_position_progress, True, {}),
    ("R9_laptime_delta_z", d9.rule_laptime_delta_z, True, {}),
    ("R10_driver_eb", d9.rule_driver_eb, True, {}),
    ("R11_stint_overdue", d9.rule_stint_overdue, True, {}),
    ("R12_cumdeg_knee", d9.rule_cumdeg_knee, True, {}),
    ("R13_race_lapbin", d9.rule_race_lapbin, True, {}),
    ("R14_hash_lr_3way", d9.rule_hash_lr_3way, False,
     dict(hash_train=hash_train, hash_test=hash_test)),
]

print(f"smoke: N_train={len(y)}  N_test={len(test)}  tr={len(tr)} va={len(va)}")
totals = {}
for name, fn, _, kwargs in approaches:
    t0 = time.time()
    rp_tr, rp_va, rp_te = fn(train_s, test, tr, va, y[tr], **kwargs)
    wall = time.time() - t0
    rule_auc = roc_auc_score(y[va], rp_va)
    totals[name] = wall
    print(f"  {name:<24s} rule_AUC={rule_auc:.5f}  shape_tr={rp_tr.shape}  "
          f"shape_va={rp_va.shape}  shape_te={rp_te.shape}  wall={wall:.2f}s")
print(f"\nsmoke total wall: {sum(totals.values()):.1f}s")
