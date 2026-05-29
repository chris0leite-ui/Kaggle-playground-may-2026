"""R22 public blend v2 — deduped sources, refined ladder, pre-submit ρ check.

Deduped distinct lineages:
  A = raunakdey07_95454  (canonical top-of-LB anchor; usmankhan0 == anthony_resnet == A)
  B = arunklenin_solo    (real FE; ρ=0.974 vs A — most diverse public)
  C = arunklenin_blend   (0.8A + own)  ρ=0.9987 vs A
  D = nawfeel_95452      (safar1_95452 identical)
  E = kalyan_95450       (slight lineage variation)
  F = sarvesh_lgbm       (recent; centered probs)
  G = mikhail            (29 votes)
  H = safar1_95449       (older anchor)
  P = PRIMARY_C5  (ours; ρ=0.99154 vs A — original work)

Per R27, ρ vs PRIMARY > 0.999 → REGRESSION_RISK / TIE_ZONE. Raunakdey is 0.99154
(comfortably under 0.999 abort), so submission is SAFE under R27.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from shutil import copyfile

ROOT = "/home/user/Kaggle-playground-may-2026"
PUB = f"{ROOT}/submissions/public_scrape"
OUT = f"{ROOT}/submissions/public_scrape/_blends"
os.makedirs(OUT, exist_ok=True)

PRIMARY = f"{ROOT}/submissions/submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv"

SRC = {
    "A_raunakdey":      f"{PUB}/raunakdey07_f1-pit-stops-blender-0-95454/submission.csv",
    "B_arunsolo":       f"{PUB}/arunklenin_ps6e5-f1-pit-stops-prediction-fe-ensemble/submission.csv",
    "C_arunblend":      f"{PUB}/arunklenin_ps6e5-f1-pit-stops-prediction-fe-ensemble/submission_blended.csv",
    "D_nawfeel":        f"{PUB}/nawfeelrahman1124444_s6e5-0-95452/submission.csv",
    "E_kalyan":         f"{PUB}/kalyankkr_f1-pitstop-ensemble-blender-0-95450/submission.csv",
    "F_sarvesh":        f"{PUB}/sarveshchhetri_f1-pit-stop-prediction-lightgbm-blending/submission.csv",
    "G_mikhail":        f"{PUB}/mikhailnaumov_f1-pit-stops-ensemble/submission.csv",
    "H_safar495449":    f"{PUB}/safar1_lb-score-0-95449/submission.csv",
    "P_PRIMARY":        PRIMARY,
}

CLAIMED_LB = {
    "A_raunakdey":   0.95454,
    "B_arunsolo":   None,
    "C_arunblend":  0.95453,
    "D_nawfeel":    0.95452,
    "E_kalyan":     0.95450,
    "F_sarvesh":    None,
    "G_mikhail":    None,
    "H_safar495449":0.95449,
    "P_PRIMARY":    0.95404,
}

def read_sub(p):
    df = pd.read_csv(p)
    pred_col = [c for c in df.columns if c.lower() != "id"][0]
    return df.set_index("id")[pred_col].rename(p)

# load
data = {}
for k, p in SRC.items():
    s = read_sub(p)
    data[k] = s
ids = data["A_raunakdey"].index

df = pd.DataFrame({k: data[k].reindex(ids).values for k in SRC}, index=ids)
N = len(df)
print(f"N={N}")

# rank-uniform each col
ru = {k: (rankdata(df[k].values, method="average") - 0.5) / N for k in df.columns}

def blend(weights, fname, meta=""):
    tot = sum(weights.values())
    w = {k: v / tot for k, v in weights.items()}
    out = np.zeros(N)
    for k, wgt in w.items():
        out = out + wgt * ru[k]
    sub = pd.DataFrame({"id": ids, "PitNextLap": out})
    sub.to_csv(f"{OUT}/{fname}", index=False)
    # ρ band vs PRIMARY (R27 check)
    rho_pri = np.corrcoef(rankdata(out), rankdata(df["P_PRIMARY"].values))[0,1]
    rho_anc = np.corrcoef(rankdata(out), rankdata(df["A_raunakdey"].values))[0,1]
    print(f"  {fname:50s}  weights={w}  ρ_PRI={rho_pri:.5f}  ρ_A={rho_anc:.5f}  {meta}")
    return out, rho_pri, rho_anc

# ------- Save the RAW raunakdey (canonical anchor) as B0 for pure-passthrough sanity submit -------
copyfile(SRC["A_raunakdey"], f"{OUT}/V0_raunakdey_RAW.csv")
print(f"V0_raunakdey_RAW.csv: raw copy of public top (claimed LB 0.95454)")

# ------- Ladder (rank-uniform-blended; all clearly under R27 0.999 abort vs PRIMARY) -------
# V1: raunakdey only (rank-uniform pipeline through; should reproduce LB ~0.95454 at AUC)
blend({"A_raunakdey": 1.0},
      "V1_A_only_RU.csv", "pipeline reproducibility check")

# V2: most-diverse-public-only — A + B (arunklenin_solo)
blend({"A_raunakdey": 0.7, "B_arunsolo": 0.3},
      "V2_A70_B30.csv", "A+arunsolo diversity")

# V3: 3-public diversified
blend({"A_raunakdey": 0.5, "B_arunsolo": 0.2, "E_kalyan": 0.15, "G_mikhail": 0.15},
      "V3_A50_B20_E15_G15.csv", "4-source diversified")

# V4: hedge-with-ours — A + B + P (the diamond of distinct lineages)
blend({"A_raunakdey": 0.70, "B_arunsolo": 0.15, "P_PRIMARY": 0.15},
      "V4_A70_B15_P15.csv", "A+arun+ours hedge")

# V5: heavy hedge with ours (anti-public-overfit)
blend({"A_raunakdey": 0.50, "B_arunsolo": 0.20, "P_PRIMARY": 0.30},
      "V5_A50_B20_P30.csv", "anti-overfit heavy hedge")

# V6: arunklenin recipe-style — 0.8 anchor + 0.1 own + 0.1 our PRIMARY
blend({"A_raunakdey": 0.80, "B_arunsolo": 0.10, "P_PRIMARY": 0.10},
      "V6_A80_B10_P10.csv", "arun-recipe + our hedge")

# V7: 5-lineage public mean (no own)
blend({"A_raunakdey": 1.0, "B_arunsolo": 1.0, "E_kalyan": 1.0, "G_mikhail": 1.0, "H_safar495449": 1.0},
      "V7_five_public_mean.csv", "5-source equal mean")

# V8: very heavy A + small ours (smallest deviation from public top)
blend({"A_raunakdey": 0.90, "P_PRIMARY": 0.10},
      "V8_A90_P10.csv", "near-anchor + 10% hedge")

# V9: most aggressive diversity — equal weights across 5 lineages including ours
blend({"A_raunakdey": 1.0, "B_arunsolo": 1.0, "E_kalyan": 1.0, "G_mikhail": 1.0, "P_PRIMARY": 1.0},
      "V9_five_equal_with_ours.csv", "5-equal incl ours")

print("\nFinal listing:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".csv") and not f.startswith("rho_"):
        size = os.path.getsize(f"{OUT}/{f}")
        print(f"  {f}  ({size} bytes)")
