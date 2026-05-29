"""R22 weight-sweep generator for Day-30.

After Day-29 results in (V0 / V4 / V8 / V10), this script generates a
parametric set of A/B/P weight combinations to refine the curve.

Usage: python scripts/r22_weight_sweep.py
Writes CSVs to submissions/public_scrape/_blends/sweep_*.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "/home/user/Kaggle-playground-may-2026"
PUB = f"{ROOT}/submissions/public_scrape"
OUT = f"{PUB}/_blends"
os.makedirs(OUT, exist_ok=True)

A_path = f"{PUB}/raunakdey07_f1-pit-stops-blender-0-95454/submission.csv"
B_path = f"{PUB}/arunklenin_ps6e5-f1-pit-stops-prediction-fe-ensemble/submission.csv"
P_path = f"{ROOT}/submissions/submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv"

A_df = pd.read_csv(A_path).set_index("id")
ids = A_df.index
N = len(ids)
def ru(p):
    s = pd.read_csv(p).set_index("id").reindex(ids).iloc[:, 0].values
    return (rankdata(s, method="average") - 0.5) / N
A = ru(A_path); B = ru(B_path); P = ru(P_path)

# Weight grid: choose to span the curve
# Total weight always sums to 1.0
GRID = [
    # (a, b, p, name)
    (0.95, 0.00, 0.05, "S0_A95_P05"),
    (0.90, 0.05, 0.05, "S1_A90_B05_P05"),
    (0.85, 0.10, 0.05, "S2_A85_B10_P05"),
    (0.85, 0.05, 0.10, "S3_A85_B05_P10"),
    (0.80, 0.10, 0.10, "S4_A80_B10_P10"),  # = V6 (rebuilt for clarity)
    (0.80, 0.15, 0.05, "S5_A80_B15_P05"),
    (0.80, 0.05, 0.15, "S6_A80_B05_P15"),
    (0.75, 0.15, 0.10, "S7_A75_B15_P10"),
    (0.75, 0.10, 0.15, "S8_A75_B10_P15"),
    (0.65, 0.20, 0.15, "S9_A65_B20_P15"),
    (0.70, 0.30, 0.00, "S10_A70_B30"),     # no-PRIMARY variant
    (0.60, 0.40, 0.00, "S11_A60_B40"),     # heavier arunsolo no-PRIMARY
]

for a, b, p, name in GRID:
    assert abs(a + b + p - 1.0) < 1e-9, (a, b, p, name)
    out = a * A + b * B + p * P
    sub = pd.DataFrame({"id": ids, "PitNextLap": out})
    fname = f"sweep_{name}.csv"
    sub.to_csv(f"{OUT}/{fname}", index=False)
    rho_v4 = np.corrcoef(rankdata(out), rankdata(pd.read_csv(f"{ROOT}/submissions/submission_d29_R22V4_raunakdey70_arunsolo15_C5_15.csv").set_index("id").reindex(ids).iloc[:, 0].values))[0, 1]
    print(f"{name:22s}  A={a:.2f} B={b:.2f} P={p:.2f}  ρ_vs_V4={rho_v4:.5f}")

print(f"\nWrote {len(GRID)} sweep candidates to {OUT}/sweep_*.csv")
