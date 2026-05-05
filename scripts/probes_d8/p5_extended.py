"""Extended P5: clarify the 97.4% claim about test-side successors."""
import json, os
import numpy as np, pandas as pd
ROOT = "/home/user/Kaggle-playground-may-2026"
test = pd.read_csv(os.path.join(ROOT, "data/test.csv"))
train = pd.read_csv(os.path.join(ROOT, "data/train.csv"))

# Variant A: successor within test only (already computed: 12.4%)
# Variant B: same (Race,Driver,Year), LapNumber+1 anywhere (train OR test)
all_df = pd.concat([train.assign(_src="train"), test.assign(_src="test")], ignore_index=True)
key_to_src = {}
for r,d,y,l,s in zip(all_df.Race, all_df.Driver, all_df.Year, all_df.LapNumber, all_df._src):
    key_to_src[(r,d,y,l)] = s

# For each test row, does (Race,Driver,Year, LapNumber+1) exist anywhere?
exists_anywhere = []
in_train = []
in_test = []
for r,d,y,l in zip(test.Race, test.Driver, test.Year, test.LapNumber):
    s = key_to_src.get((r,d,y,l+1))
    exists_anywhere.append(s is not None)
    in_train.append(s == "train")
    in_test.append(s == "test")

ea = np.array(exists_anywhere)
it = np.array(in_train)
ie = np.array(in_test)
print(f"test rows: {len(test)}")
print(f"frac with successor anywhere (train+test combined): {ea.mean():.4f}")
print(f"frac with successor IN TRAIN: {it.mean():.4f}")
print(f"frac with successor IN TEST:  {ie.mean():.4f}")

# Variant C: "same (Race, Driver) successor" without Year
key_rdl = {}
for r,d,l in zip(test.Race, test.Driver, test.LapNumber):
    key_rdl[(r,d,l)] = True
exists_rdl_test = []
for r,d,l in zip(test.Race, test.Driver, test.LapNumber):
    exists_rdl_test.append((r,d,l+1) in key_rdl)
print(f"frac with (Race,Driver,LapNumber+1) IN TEST: {np.mean(exists_rdl_test):.4f}")

# Variant D: predecessor in train (lookback)
key_train = {}
for r,d,y,l in zip(train.Race, train.Driver, train.Year, train.LapNumber):
    key_train[(r,d,y,l)] = True
prev_in_train = []
for r,d,y,l in zip(test.Race, test.Driver, test.Year, test.LapNumber):
    prev_in_train.append((r,d,y,l-1) in key_train)
print(f"frac with PREDECESSOR (LapNumber-1) IN TRAIN: {np.mean(prev_in_train):.4f}")

# write extended JSON
out = {
    "frac_succ_anywhere": float(ea.mean()),
    "frac_succ_in_train": float(it.mean()),
    "frac_succ_in_test":  float(ie.mean()),
    "frac_succ_rdl_in_test": float(np.mean(exists_rdl_test)),
    "frac_pred_in_train": float(np.mean(prev_in_train)),
}
with open(os.path.join(ROOT, "scripts/probes_d8/p5_extended.json"), "w") as fh:
    json.dump(out, fh, indent=2)
