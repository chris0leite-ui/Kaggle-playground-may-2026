"""Day-8 structural data probes (P1..P10).
CPU-only, fast (<2 min per probe). Reads train/test CSV + M5q OOF and
emits compact numerical summaries to JSON for the markdown report.
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

ROOT = "/home/user/Kaggle-playground-may-2026"
OUT = os.path.join(ROOT, "scripts/probes_d8/out.json")

t0 = time.time()
def tlog(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

tlog("loading train/test")
train = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
test = pd.read_csv(os.path.join(ROOT, "data/test.csv"))
tlog(f"train {train.shape}, test {test.shape}")

results = {}

# ---------------- P1 sequence reconstructability ----------------
tlog("P1 sequence reconstructability")
def grouped_lap_stats(df, name):
    g = df.groupby(["Race", "Driver", "Year", "Stint"], sort=False)
    sizes = g.size()
    # contiguity check: max - min + 1 == n
    laps_max = g["LapNumber"].max()
    laps_min = g["LapNumber"].min()
    contig = (laps_max - laps_min + 1 == sizes).mean()
    return {
        "name": name,
        "n_groups": int(sizes.shape[0]),
        "n_rows": int(sizes.sum()),
        "mean_group_len": float(sizes.mean()),
        "median_group_len": float(sizes.median()),
        "max_group_len": int(sizes.max()),
        "p95_group_len": float(sizes.quantile(0.95)),
        "frac_contiguous_groups": float(contig),
        "frac_groups_ge5": float((sizes >= 5).mean()),
        "frac_groups_ge10": float((sizes >= 10).mean()),
    }

p1 = {
    "train_groups": grouped_lap_stats(train, "train"),
    "test_groups": grouped_lap_stats(test, "test"),
}
# Also: across train+test, how many test (Race,Driver,Year,Stint) groups
# have ALL their successive laps in test (i.e. no overlap with train for
# the same group)?
keys_train = set(map(tuple, train[["Race","Driver","Year","Stint"]].values))
keys_test_unique = test[["Race","Driver","Year","Stint"]].drop_duplicates()
shared_groups = sum(tuple(r) in keys_train for r in keys_test_unique.values)
p1["test_group_count"] = int(len(keys_test_unique))
p1["test_groups_overlap_train"] = int(shared_groups)
p1["test_groups_unique_to_test"] = int(len(keys_test_unique) - shared_groups)
results["P1"] = p1
tlog(f"P1 done in {time.time()-t0:.1f}s")

# ---------------- P2 train/test row-level proximity ----------------
tlog("P2 row-level proximity (sampled, sklearn NN)")
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
num_cols = [
    "LapNumber","Stint","TyreLife","Position","LapTime (s)","LapTime_Delta",
    "Cumulative_Degradation","RaceProgress","Position_Change","Year","PitStop",
]
def encode(df, drv_map, cmp_map, race_map):
    X = df[num_cols].fillna(0).to_numpy(dtype="float32")
    extra = np.column_stack([
        df["Driver"].map(drv_map).fillna(-1).to_numpy("float32"),
        df["Compound"].map(cmp_map).fillna(-1).to_numpy("float32"),
        df["Race"].map(race_map).fillna(-1).to_numpy("float32"),
    ])
    return np.column_stack([X, extra]).astype("float32")

drv_map = {d:i for i,d in enumerate(sorted(set(train.Driver) | set(test.Driver)))}
cmp_map = {d:i for i,d in enumerate(sorted(set(train.Compound) | set(test.Compound)))}
race_map = {d:i for i,d in enumerate(sorted(set(train.Race) | set(test.Race)))}

# Subsample for speed (40k train, 20k test)
rng = np.random.default_rng(0)
tr_idx = rng.choice(len(train), size=min(40000, len(train)), replace=False)
te_idx = rng.choice(len(test), size=min(20000, len(test)), replace=False)
Xtr = encode(train.iloc[tr_idx], drv_map, cmp_map, race_map)
Xte = encode(test.iloc[te_idx], drv_map, cmp_map, race_map)
sc = StandardScaler().fit(np.vstack([Xtr, Xte]))
Xtr_s = sc.transform(Xtr).astype("float32")
Xte_s = sc.transform(Xte).astype("float32")
nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
nn.fit(Xtr_s)
d, _ = nn.kneighbors(Xte_s)
d = d.ravel()
p2 = {
    "n_train_sample": int(Xtr.shape[0]),
    "n_test_sample": int(Xte.shape[0]),
    "n_features": int(Xtr.shape[1]),
    "mean_nn_dist": float(d.mean()),
    "p5":  float(np.percentile(d, 5)),
    "p25": float(np.percentile(d, 25)),
    "p50": float(np.percentile(d, 50)),
    "p75": float(np.percentile(d, 75)),
    "p95": float(np.percentile(d, 95)),
    "max": float(d.max()),
    "frac_dist_lt_0p1": float((d < 0.1).mean()),
    "frac_dist_lt_0p5": float((d < 0.5).mean()),
    "frac_dist_lt_1p0": float((d < 1.0).mean()),
}
results["P2"] = p2
tlog(f"P2 done in {time.time()-t0:.1f}s")

# ---------------- P3 Year x Race anomaly ----------------
tlog("P3 year x race anomaly")
yr_pit = train.groupby("Year")["PitNextLap"].agg(["mean","count"]).reset_index()
yr_race_pit = train.groupby(["Year","Race"])["PitNextLap"].agg(["mean","count"]).reset_index()
# 2023 sub-rates
y23 = yr_race_pit[yr_race_pit.Year == 2023].sort_values("mean")
yr_train_count = train.groupby("Year").size()
yr_test_count = test.groupby("Year").size()
# train share per (Year, Race)
tr_yr_race = train.groupby(["Year","Race"]).size()
te_yr_race = test.groupby(["Year","Race"]).size()
years_all = sorted(set(train.Year) | set(test.Year))
share_compare = []
for y in years_all:
    tr_n = int(yr_train_count.get(y, 0))
    te_n = int(yr_test_count.get(y, 0))
    share_compare.append({
        "year": int(y),
        "train_n": tr_n, "test_n": te_n,
        "train_share": tr_n/len(train),
        "test_share":  te_n/len(test),
    })
# Identify anomalous low-pit-rate (Year,Race): pit_rate < 0.05 and count > 500
anomalies = yr_race_pit[(yr_race_pit["mean"] < 0.05) & (yr_race_pit["count"] > 500)]
results["P3"] = {
    "per_year": yr_pit.to_dict(orient="records"),
    "year_share_compare": share_compare,
    "n_anomalous_low_yr_race": int(len(anomalies)),
    "anomalous_low_examples": anomalies.head(15).to_dict(orient="records"),
    "y2023_min10": y23.head(10).to_dict(orient="records"),
    "y2023_max10": y23.tail(10).to_dict(orient="records"),
}
tlog(f"P3 done in {time.time()-t0:.1f}s")

# ---------------- P4 Stint-2 deep dive ----------------
tlog("P4 stint-2 deep dive")
s2 = train[train.Stint == 2].copy()
# pit rate per (Race, Compound)
rc = s2.groupby(["Race","Compound"])["PitNextLap"].agg(["mean","count"])
rc = rc[rc["count"] >= 100].reset_index()
# laps-into-stint
def laps_into_stint(df):
    # earliest lap of each (Race, Driver, Year, Stint)
    base = df.groupby(["Race","Driver","Year","Stint"])["LapNumber"].transform("min")
    return df["LapNumber"] - base
train["laps_into_stint"] = laps_into_stint(train)
s2 = train[train.Stint == 2].copy()
# bucket
bins = [-1,0,2,5,10,15,20,30,1000]
labels = ["0","1-2","3-5","6-10","11-15","16-20","21-30","30+"]
s2["lis_bucket"] = pd.cut(s2["laps_into_stint"], bins=bins, labels=labels)
lis_pit = s2.groupby("lis_bucket", observed=True)["PitNextLap"].agg(["mean","count"]).reset_index()
# Stint-2 starts: TyreLife=1
s2_starts = s2[(s2.TyreLife <= 1.0)]
# previous compound: compound of Stint 1 for same (Race,Driver,Year)
s1 = train[train.Stint == 1].groupby(["Race","Driver","Year"])["Compound"].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else None).rename("prev_compound").reset_index()
s2 = s2.merge(s1, on=["Race","Driver","Year"], how="left")
prev_compound_pit = s2.groupby(["prev_compound","Compound"])["PitNextLap"].agg(["mean","count"]).reset_index()
prev_compound_pit = prev_compound_pit[prev_compound_pit["count"] > 200]
results["P4"] = {
    "n_stint2_rows": int(len(s2)),
    "stint2_share": float(len(s2)/len(train)),
    "stint2_pit_rate_overall": float(s2["PitNextLap"].mean()),
    "rc_top_pit": rc.sort_values("mean", ascending=False).head(10).to_dict(orient="records"),
    "rc_bot_pit": rc.sort_values("mean", ascending=True).head(10).to_dict(orient="records"),
    "lis_pit": lis_pit.to_dict(orient="records"),
    "prev_compound_x_compound_pit": prev_compound_pit.sort_values("mean", ascending=False).head(15).to_dict(orient="records"),
}
tlog(f"P4 done in {time.time()-t0:.1f}s")

# ---------------- P5 test-only feature computability ----------------
tlog("P5 test-only feature computability")
# Build (id, Race, Driver, Year, Stint, LapNumber)
test_keyed = test[["id","Race","Driver","Year","Stint","LapNumber","Compound","PitStop"]].copy()
# successor in test: same (Race, Driver, Year), LapNumber+1
# Use a hash: (Race,Driver,Year,LapNumber) → row
test_lookup = {(r,d,y,l): i for i,(r,d,y,l) in enumerate(zip(test["Race"], test["Driver"], test["Year"], test["LapNumber"]))}
# successor flags
nxt_idx = np.array([test_lookup.get((r,d,y,l+1), -1) for r,d,y,l in zip(test["Race"], test["Driver"], test["Year"], test["LapNumber"])])
has_next = nxt_idx >= 0
nxt2_idx = np.where(has_next, np.array([test_lookup.get((r,d,y,l+2), -1) for r,d,y,l in zip(test["Race"], test["Driver"], test["Year"], test["LapNumber"])]), -1)
has_next2 = nxt2_idx >= 0
# lead PitStop in test
lead_pit = np.full(len(test), np.nan)
lead_pit[has_next] = test["PitStop"].iloc[nxt_idx[has_next]].to_numpy()
# next compound: scan forward until Stint changes
test_arr = test[["Race","Driver","Year","Stint","Compound","LapNumber"]].to_numpy()
race_a = test["Race"].to_numpy()
drv_a = test["Driver"].to_numpy()
yr_a = test["Year"].to_numpy()
st_a = test["Stint"].to_numpy()
cmp_a = test["Compound"].to_numpy()
lap_a = test["LapNumber"].to_numpy()
# build per-(Race,Driver,Year) sorted index by LapNumber
gkey = list(zip(race_a, drv_a, yr_a))
import collections
group_to_rows = collections.defaultdict(list)
for i, k in enumerate(gkey):
    group_to_rows[k].append(i)
for k in group_to_rows:
    group_to_rows[k].sort(key=lambda i: lap_a[i])
next_compound = np.full(len(test), None, dtype=object)
laps_until_eos = np.full(len(test), -1, dtype=np.int32)
for k, rows in group_to_rows.items():
    n = len(rows)
    sts = st_a[rows]
    cmps = cmp_a[rows]
    for j in range(n):
        # find next position whose Stint != current
        cur_st = sts[j]
        # laps until end-of-stint (max consecutive same-Stint)
        end = j
        while end+1 < n and sts[end+1] == cur_st:
            end += 1
        laps_until_eos[rows[j]] = end - j
        # next compound: first row at end+1 (if exists) Compound
        if end+1 < n:
            next_compound[rows[j]] = cmps[end+1]
n_test = len(test)
p5 = {
    "n_test": int(n_test),
    "frac_with_next_lap": float(has_next.mean()),
    "frac_with_next_next_lap": float(has_next2.mean()),
    "frac_with_lead_pitstop_observable": float(np.isfinite(lead_pit).mean()),
    "frac_with_next_compound_observable": float((next_compound != None).mean()),
    "frac_laps_until_eos_known_pos": float((laps_until_eos > 0).mean()),
    "median_laps_until_eos_when_known": float(np.median(laps_until_eos[laps_until_eos > 0])) if (laps_until_eos > 0).any() else None,
    "frac_next_compound_same":  float(((next_compound == cmp_a) & (next_compound != None)).mean()),
}
# also: for Stint-2 rows with TyreLife <=1 (Stint-2 starts) what frac of test rows have that?
results["P5"] = p5
tlog(f"P5 done in {time.time()-t0:.1f}s")

# ---------------- P6 lap-grouping leakage check ----------------
tlog("P6 lap-grouping leakage")
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
y = train["PitNextLap"].astype(int).to_numpy()
gk_groups = (train["Race"].astype(str) + "|" + train["Driver"].astype(str)).to_numpy()
gk_unique = np.unique(gk_groups)
# fold sizes for GroupKFold(5)
gk = GroupKFold(n_splits=5)
gk_sizes = []
for tr_i, va_i in gk.split(np.zeros(len(train)), y, groups=gk_groups):
    gk_sizes.append(int(len(va_i)))
# StratifiedKFold within-group leakage: for each (Race, Driver, Year, Stint),
# count how often consecutive laps land in different folds.
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_id = np.full(len(train), -1, dtype=np.int8)
for f, (_, va) in enumerate(sk.split(np.zeros(len(train)), y)):
    fold_id[va] = f
# group consecutive lap fold split fraction
df = train[["Race","Driver","Year","Stint","LapNumber"]].copy()
df["fold"] = fold_id
df = df.sort_values(["Race","Driver","Year","Stint","LapNumber"])
df["prev_fold"] = df.groupby(["Race","Driver","Year","Stint"])["fold"].shift(1)
df["prev_lap"]  = df.groupby(["Race","Driver","Year","Stint"])["LapNumber"].shift(1)
df["consec"]    = (df["LapNumber"] - df["prev_lap"] == 1)
mask = df["consec"].fillna(False)
diff_fold = (df["fold"] != df["prev_fold"]) & mask
results["P6"] = {
    "n_unique_race_driver": int(len(gk_unique)),
    "groupkfold5_fold_sizes": gk_sizes,
    "groupkfold5_fold_size_min": int(min(gk_sizes)),
    "groupkfold5_fold_size_max": int(max(gk_sizes)),
    "n_consecutive_lap_pairs": int(mask.sum()),
    "n_consec_diff_fold": int(diff_fold.sum()),
    "frac_consec_lap_pairs_diff_fold_strat5": float(diff_fold.sum()/max(1,mask.sum())),
}
tlog(f"P6 done in {time.time()-t0:.1f}s")

# ---------------- P7 compound transition ----------------
tlog("P7 compound transition")
# For each (Race, Driver, Year), sequence of Stint→Compound (use mode per stint)
seq_df = train.groupby(["Race","Driver","Year","Stint"])["Compound"].agg(lambda x: x.mode().iloc[0]).reset_index()
seq_df = seq_df.sort_values(["Race","Driver","Year","Stint"])
seq_per_race = seq_df.groupby(["Race","Driver","Year"])["Compound"].apply(tuple)
sequences = Counter(seq_per_race)
# Top sequences total
top_seq = sequences.most_common(15)
# 1-stop vs 2-stop counts
len_dist = Counter(len(s) for s in seq_per_race)
# Stint-2 conditional pit rate given (s1, s2)
# Use stint1 compound + stint2 compound merge
s1c = train[train.Stint==1].groupby(["Race","Driver","Year"])["Compound"].agg(lambda x: x.mode().iloc[0]).rename("s1_cmp")
s2c = train[train.Stint==2].groupby(["Race","Driver","Year"])["Compound"].agg(lambda x: x.mode().iloc[0]).rename("s2_cmp")
both = pd.concat([s1c, s2c], axis=1).dropna().reset_index()
s2t = train[train.Stint==2].merge(both, on=["Race","Driver","Year"], how="left")
s2_cond = s2t.groupby(["s1_cmp","s2_cmp"])["PitNextLap"].agg(["mean","count"]).reset_index()
s2_cond = s2_cond[s2_cond["count"] > 100]
results["P7"] = {
    "len_distribution": {str(k):int(v) for k,v in len_dist.items()},
    "top_sequences_overall": [
        {"sequence":"->".join(k),"count":int(v)} for k,v in top_seq
    ],
    "stint2_pit_given_s1_s2_cmp": s2_cond.sort_values("mean", ascending=False).to_dict(orient="records"),
}
tlog(f"P7 done in {time.time()-t0:.1f}s")

# ---------------- P8 lap-number distribution per race ----------------
tlog("P8 race length")
race_max_lap = train.groupby(["Race","Year"])["LapNumber"].max().reset_index()
# typical max-lap per Race
race_typical = race_max_lap.groupby("Race")["LapNumber"].agg(["min","max","median","mean","std","count"]).reset_index()
race_typical = race_typical.sort_values("median", ascending=False)
results["P8"] = {
    "race_typical_length": race_typical.to_dict(orient="records"),
}
tlog(f"P8 done in {time.time()-t0:.1f}s")

# ---------------- P9 driver embedding plausibility ----------------
tlog("P9 driver counts")
drv_counts = train["Driver"].value_counts()
n_drivers_train = int(drv_counts.shape[0])
n_drivers_test = int(test["Driver"].nunique())
# how many drivers in test not in train
in_test_only = set(test.Driver) - set(train.Driver)
n_lt10 = int((drv_counts < 10).sum())
n_lt100 = int((drv_counts < 100).sum())
n_lt1000 = int((drv_counts < 1000).sum())
top50 = drv_counts.head(50).sum()
top50_share = float(top50 / drv_counts.sum())
# pit rate variance for low-count drivers
drv_pit = train.groupby("Driver")["PitNextLap"].agg(["mean","count"])
low_count = drv_pit[drv_pit["count"] < 50]
high_count = drv_pit[drv_pit["count"] >= 1000]
results["P9"] = {
    "n_drivers_train": n_drivers_train,
    "n_drivers_test": n_drivers_test,
    "n_drivers_test_only": int(len(in_test_only)),
    "n_drivers_lt10": n_lt10,
    "n_drivers_lt100": n_lt100,
    "n_drivers_lt1000": n_lt1000,
    "top50_share_of_rows": top50_share,
    "low_count_pit_mean": float(low_count["mean"].mean()),
    "low_count_pit_std":  float(low_count["mean"].std()),
    "high_count_pit_mean":float(high_count["mean"].mean()),
    "high_count_pit_std": float(high_count["mean"].std()),
}
tlog(f"P9 done in {time.time()-t0:.1f}s")

# ---------------- P10 anti-corr feature search ----------------
tlog("P10 oof residual cohorts")
oof = np.load(os.path.join(ROOT, "scripts/artifacts/oof_m5q_strat.npy"))
oof_pos = oof[:, 1]
res = train["PitNextLap"].to_numpy(dtype="float32") - oof_pos.astype("float32")
train["_residual"] = res
# (Race, Stint)
rs = train.groupby(["Race","Stint"])["_residual"].agg(["mean","std","count"]).reset_index()
rs_big = rs[(rs["count"]>=200) & (rs["mean"].abs() >= 0.02)].sort_values("mean", key=lambda s: s.abs(), ascending=False)
# (Compound, TyreLife decile)
train["_tl_dec"] = pd.qcut(train["TyreLife"].rank(method="first"), 10, labels=False)
ct = train.groupby(["Compound","_tl_dec"])["_residual"].agg(["mean","std","count"]).reset_index()
ct_big = ct[(ct["count"]>=200) & (ct["mean"].abs()>=0.02)].sort_values("mean", key=lambda s: s.abs(), ascending=False)
# (Year, Position)
yp = train.groupby(["Year","Position"])["_residual"].agg(["mean","std","count"]).reset_index()
yp_big = yp[(yp["count"]>=200) & (yp["mean"].abs()>=0.02)].sort_values("mean", key=lambda s: s.abs(), ascending=False)

results["P10"] = {
    "global_residual_mean": float(res.mean()),
    "global_residual_std":  float(res.std()),
    "n_race_stint_cohorts_with_bias_ge_002": int(len(rs_big)),
    "race_stint_top": rs_big.head(15).to_dict(orient="records"),
    "n_cmp_tl_cohorts_with_bias_ge_002": int(len(ct_big)),
    "cmp_tl_top": ct_big.head(15).to_dict(orient="records"),
    "n_year_pos_cohorts_with_bias_ge_002": int(len(yp_big)),
    "year_pos_top": yp_big.head(15).to_dict(orient="records"),
}
tlog(f"P10 done in {time.time()-t0:.1f}s")

with open(OUT, "w") as fh:
    json.dump(results, fh, default=lambda o: int(o) if hasattr(o, "item") else str(o), indent=2)
tlog(f"WROTE {OUT}")
