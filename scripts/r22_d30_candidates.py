"""Day-30 candidate generator: easy-sophistication blends + V8 weight sweep.

Anchor = raunakdey07 public blender (LB 0.95453).
PRIMARY (P) = C5 our K=20 stack (LB 0.95404).
V8 = 0.9 A + 0.1 P (LB 0.95456) is the reference.

Categories produced:
  W: V8 weight sweep (A=0.95..0.75, P matches)
  O: alternative "ours" at 0.10 weight (C7, R31, C1=K=19)
  M: multi-ours blends
  L: logit-space blend
  E: power-mean (Lp) blends
  C: confidence-conditional weights
"""

import os, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit

ROOT = "/home/user/Kaggle-playground-may-2026"
PUB = f"{ROOT}/submissions/public_scrape"
OUT = f"{PUB}/_blends_d30"
os.makedirs(OUT, exist_ok=True)

ANCHOR_PATH = f"{PUB}/raunakdey07_f1-pit-stops-blender-0-95454/submission.csv"
PRIMARY_C5 = f"{ROOT}/submissions/submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv"
OURS_C7    = f"{ROOT}/submissions/submission_d21_d22_C7_K20_R21_K27_R28_renorm.csv"
OURS_C1_K19= f"{ROOT}/submissions/submission_d21_C1_K19_R21_rankmean_89_11.csv"
OURS_R31   = f"{ROOT}/submissions/submission_R31_K18_R30_R72_rankmean_50_15_35.csv"
OURS_R25   = f"{ROOT}/submissions/submission_R25_K18_R21_rankmean_w89.csv"
V8_REF     = f"{ROOT}/submissions/submission_d29_R22V8_raunakdey90_C5_10.csv"

A_df = pd.read_csv(ANCHOR_PATH).set_index("id")
ids = A_df.index
N = len(ids)
def ru(path):
    s = pd.read_csv(path).set_index("id").reindex(ids).iloc[:, 0].values
    return (rankdata(s, method="average") - 0.5) / N

A = ru(ANCHOR_PATH)
P = ru(PRIMARY_C5)
O_C7   = ru(OURS_C7)
O_C1   = ru(OURS_C1_K19)
O_R31  = ru(OURS_R31)
O_R25  = ru(OURS_R25)
V8     = ru(V8_REF)

def save(name, vec, desc=""):
    fname = f"D30_{name}.csv"
    pd.DataFrame({"id": ids, "PitNextLap": vec}).to_csv(f"{OUT}/{fname}", index=False)
    rho = np.corrcoef(rankdata(vec), rankdata(V8))[0, 1]
    print(f"  {name:30s}  ρ_vs_V8={rho:.5f}  {desc}")
    return rho

print("=== W: V8 weight sweep ===")
for a, p, lbl in [(0.95, 0.05, "W1_A95_P05"), (0.85, 0.15, "W2_A85_P15"),
                  (0.80, 0.20, "W3_A80_P20"), (0.75, 0.25, "W4_A75_P25")]:
    save(lbl, a*A + p*P, f"A={a} P={p}")

print("\n=== O: alternative 'ours' at 10% ===")
save("O1_A90_C7_10",   0.9*A + 0.1*O_C7,  "C7 (LB-tied C5)")
save("O2_A90_R31_10",  0.9*A + 0.1*O_R31, "R31 (3-way K18+R30+R7.2)")
save("O3_A90_C1K19_10", 0.9*A + 0.1*O_C1,  "K=19 d21-only mem-inf")
save("O4_A90_R25_10",  0.9*A + 0.1*O_R25, "R25 prior PRIMARY K=18")

print("\n=== M: multi-ours blends ===")
save("M1_A85_C5_C7_R31",  0.85*A + 0.08*P + 0.04*O_C7  + 0.03*O_R31, "A85 + 3-ours sum=15")
save("M2_A85_C5_R31_C1",  0.85*A + 0.08*P + 0.04*O_R31 + 0.03*O_C1,  "A85 + 3 distinct mech")
save("M3_A90_C5_R31",     0.90*A + 0.07*P + 0.03*O_R31,             "A90 + 2 ours sum=10")

print("\n=== L: logit-space V8 ===")
A_logit = logit(np.clip(A, 1e-6, 1-1e-6))
P_logit = logit(np.clip(P, 1e-6, 1-1e-6))
V8_logit = expit(0.9*A_logit + 0.1*P_logit)
save("L1_LOGIT_A90_P10", V8_logit, "logit-mean 0.9A + 0.1P then expit")
V8_logit_p15 = expit(0.85*A_logit + 0.15*P_logit)
save("L2_LOGIT_A85_P15", V8_logit_p15, "logit-mean 0.85/0.15")

print("\n=== E: power-mean Lp blends (rank-based) ===")
def pmean(a, p, q, w_a, w_p):
    # power-mean of two non-neg sequences; q != 0
    # Lp_mean = (w_a * a^q + w_p * p^q) ^ (1/q)
    return (w_a * np.power(a, q) + w_p * np.power(p, q)) ** (1.0/q)
save("E1_PMEAN_q2_A90_P10", pmean(A, P, 2.0, 0.9, 0.1), "q=2 quadratic mean")
save("E2_PMEAN_q05_A90_P10", pmean(A, P, 0.5, 0.9, 0.1), "q=0.5 sqrt mean")
save("E3_PMEAN_q3_A90_P10", pmean(A, P, 3.0, 0.9, 0.1), "q=3 cubic mean")
# geometric mean (q→0 limit) = product^weights
geo = np.exp(0.9*np.log(np.clip(A,1e-6,None)) + 0.1*np.log(np.clip(P,1e-6,None)))
save("E4_GEO_A90_P10", geo, "geometric mean (q→0)")

print("\n=== C: confidence-conditional V8 ===")
# When anchor is in mid-range (|A - 0.5| < 0.25), give ours more weight (0.20)
# When anchor is extreme (|A - 0.5| > 0.25), give ours less weight (0.05)
dist_mid = np.abs(A - 0.5)
w_p = np.where(dist_mid < 0.25, 0.20, 0.05)
w_a = 1.0 - w_p
save("C1_CONFCOND_A_mid_P20_ext_P05", w_a*A + w_p*P, "ours weight 0.20 in mid, 0.05 at tails")
# Smooth version: w_p = 0.05 + 0.15 * (1 - 2*|A-0.5|)
w_p_smooth = 0.05 + 0.15 * (1.0 - 2.0*dist_mid)
w_a_smooth = 1.0 - w_p_smooth
save("C2_CONFCOND_SMOOTH", w_a_smooth*A + w_p_smooth*P, "smooth ours weight 0.05-0.20 by |A-0.5|")
# Trust-ours-only-on-disagreement: w_p large when |A - P| > threshold
disagree = np.abs(A - P)
w_p_dis = 0.05 + 0.15 * (disagree / disagree.max())
w_a_dis = 1.0 - w_p_dis
save("C3_DISAGREE_W", w_a_dis*A + w_p_dis*P, "ours weight grows with |A-P|")

print(f"\n=== Total {len(os.listdir(OUT))} candidates in {OUT} ===")
