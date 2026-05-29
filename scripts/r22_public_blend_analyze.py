"""R22 public-submission blend analysis.

Loads our PRIMARY (C5) + all scraped public submission CSVs, computes pairwise
Spearman correlations, ranks against PRIMARY, sketches candidate rank-mean blends.
PI-authorized 2026-05-29 (reverses Day-8 + Day-22 'original work only' directive).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = "/home/user/Kaggle-playground-may-2026"
PRIMARY = f"{ROOT}/submissions/submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv"
PUBLIC = f"{ROOT}/submissions/public_scrape"
OUT_DIR = f"{ROOT}/submissions/public_scrape/_blends"
os.makedirs(OUT_DIR, exist_ok=True)

PUB_FILES = {
    "raunakdey07_95454": "raunakdey07_f1-pit-stops-blender-0-95454/submission.csv",
    "usmankhan0":       "muhammadusmankhan0_predicting-f1-pitstops-with-blending/submission.csv",
    "arunklenin_solo":  "arunklenin_ps6e5-f1-pit-stops-prediction-fe-ensemble/submission.csv",
    "arunklenin_blend": "arunklenin_ps6e5-f1-pit-stops-prediction-fe-ensemble/submission_blended.csv",
    "sarvesh_lgbm":     "sarveshchhetri_f1-pit-stop-prediction-lightgbm-blending/submission.csv",
    "mikhail":          "mikhailnaumov_f1-pit-stops-ensemble/submission.csv",
    "anthony_resnet":   "anthonytherrien_predicting-f1-pit-stops-nn-residual-network/submission.csv",
    "nawfeel_95452":    "nawfeelrahman1124444_s6e5-0-95452/submission.csv",
    "safar1_95452":     "safar1_lb-score-0-95452/submission.csv",
    "safar1_95449":     "safar1_lb-score-0-95449/submission.csv",
    "kalyan_95450":     "kalyankkr_f1-pitstop-ensemble-blender-0-95450/submission.csv",
}

def read_sub(p):
    df = pd.read_csv(p)
    cols = [c for c in df.columns if c.lower() != "id"]
    assert len(cols) == 1, f"unexpected cols in {p}: {df.columns.tolist()}"
    return df.set_index("id" if "id" in df.columns else df.columns[0])[cols[0]].rename(os.path.basename(p))

# load PRIMARY
pri = read_sub(PRIMARY).rename("PRIMARY_C5")
print(f"PRIMARY shape={pri.shape}  mean={pri.mean():.4f}  std={pri.std():.4f}")
print(f"  floor rate (<=1e-3): {(pri <= 1e-3).mean()*100:.2f}%  ceil (>=0.999): {(pri >= 0.999).mean()*100:.4f}%")

# load all public, align to PRIMARY index, drop any that mismatch
records = {"PRIMARY_C5": pri}
for name, rel in PUB_FILES.items():
    p = f"{PUBLIC}/{rel}"
    if not os.path.exists(p):
        print(f"MISSING {p}")
        continue
    s = read_sub(p)
    if not s.index.equals(pri.index):
        # try reindex
        s2 = s.reindex(pri.index)
        miss = s2.isna().sum()
        if miss > 0:
            print(f"MISALIGN {name}: {miss} missing after reindex; SKIP")
            continue
        s = s2
    records[name] = s.rename(name)
    print(f"  {name:24s} n={len(s)} mean={s.mean():.4f} std={s.std():.4f} "
          f"floor%={ (s <= 1e-3).mean()*100:6.2f}  ceil%={ (s >= 0.999).mean()*100:.4f}")

df = pd.DataFrame(records)
print(f"\nFinal frame: {df.shape}")

# Pairwise Spearman on RANK-NORMALIZED values (per pre_submit_diff lesson)
ranks = pd.DataFrame({c: rankdata(df[c].values, method="average") for c in df.columns})
print("\nPairwise Spearman ρ (rank-normalized first):")
rho_mat = ranks.corr(method="pearson")  # Pearson on ranks == Spearman, but consistent ties
print(rho_mat.round(5).to_string())

# Sort public subs by ρ vs PRIMARY (most diverse first; closest last)
rho_pri = rho_mat["PRIMARY_C5"].drop("PRIMARY_C5").sort_values()
print("\nPublic ρ vs PRIMARY_C5 (most diverse first):")
print(rho_pri.round(5).to_string())

# Save the ρ matrix for inspection
rho_mat.to_csv(f"{OUT_DIR}/rho_matrix.csv")

# ---- Candidate blend ladder ----
# Anchor: raunakdey07_95454 (top public LB 0.95454, +5bp vs ours).
# Strategy A: pure-public mean (proves the "raunakdey blender" recipe survives our pipeline)
# Strategy B: raunakdey * w_high + (PRIMARY + one orthogonal public) * residual
# Strategy C: arunklenin-style 0.8 * raunakdey + 0.1 * own + 0.1 * other_blend

# rank-uniform predictions (so blends are scale-invariant)
N = len(df)
def rank_uniform(s):
    return (rankdata(s.values, method="average") - 0.5) / N

ru = {c: rank_uniform(df[c]) for c in df.columns}

def weighted_blend_rank(weights):
    """weights: dict {name: w}, sum=1."""
    tot = sum(weights.values())
    w = {k: v / tot for k, v in weights.items()}
    out = np.zeros(N)
    for name, wgt in w.items():
        out = out + wgt * ru[name]
    return out

def save_blend(weights, fname):
    pred = weighted_blend_rank(weights)
    sub = pd.DataFrame({"id": df.index, "PitNextLap": pred})
    p = f"{OUT_DIR}/{fname}"
    sub.to_csv(p, index=False)
    return p, pred

# All public submissions ordered by claimed LB (descending). Numbers are rough.
claimed_lb = {
    "raunakdey07_95454": 0.95454,
    "usmankhan0":        0.95453,   # unknown but high-vote recent
    "arunklenin_blend":  0.95453,
    "arunklenin_solo":   None,
    "sarvesh_lgbm":      None,
    "mikhail":           None,
    "anthony_resnet":    None,
    "nawfeel_95452":     0.95452,
    "safar1_95452":      0.95452,
    "safar1_95449":      0.95449,
    "kalyan_95450":      0.95450,
}

# B0: pure raunakdey only (sanity baseline; if our pipeline writes it through cleanly,
# the LB should reproduce the public 0.95454)
save_blend({"raunakdey07_95454": 1.0}, "B0_raunakdey_only.csv")

# B1: rank-mean of top-3 distinct-author high-LB blenders
save_blend({"raunakdey07_95454": 1.0,
            "usmankhan0":        1.0,
            "kalyan_95450":      1.0}, "B1_top3_mean.csv")

# B2: 0.6 raunakdey + 0.4 our PRIMARY  (hedge with originality)
save_blend({"raunakdey07_95454": 0.6,
            "PRIMARY_C5":        0.4}, "B2_raunakdey_60_ours_40.csv")

# B3: arunklenin recipe-style: 0.8 raunakdey + 0.1 arunklenin_solo + 0.1 PRIMARY
save_blend({"raunakdey07_95454": 0.80,
            "arunklenin_solo":   0.10,
            "PRIMARY_C5":        0.10}, "B3_raunakdey80_arun10_ours10.csv")

# B4: top-3 public + small slice of PRIMARY (anti-overfit hedge)
save_blend({"raunakdey07_95454": 0.45,
            "usmankhan0":        0.25,
            "kalyan_95450":      0.15,
            "PRIMARY_C5":        0.15}, "B4_top3_45_25_15_ours15.csv")

# B5: 5-public diversified rank-mean (no own)
save_blend({"raunakdey07_95454": 1.0,
            "usmankhan0":        1.0,
            "arunklenin_blend":  1.0,
            "kalyan_95450":      1.0,
            "nawfeel_95452":     1.0}, "B5_top5_equal_mean.csv")

# B6: 90% public-mean + 10% PRIMARY (light hedge)
save_blend({"raunakdey07_95454": 0.30,
            "usmankhan0":        0.30,
            "kalyan_95450":      0.30,
            "PRIMARY_C5":        0.10}, "B6_top3_30_30_30_ours10.csv")

# B7: most-diverse-public-anchor + raunakdey (assuming arunklenin_solo or anthony_resnet
# is most diverse from raunakdey; pick by ρ above)

# print summary
print("\nBlend candidates written to", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".csv"):
        p = f"{OUT_DIR}/{f}"
        size = os.path.getsize(p)
        print(f"  {f}: {size} bytes")
