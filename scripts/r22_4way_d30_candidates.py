"""Day-30 final-slot + Day-31 candidates: 4-way blends incorporating
brand-new (today) orthogonal public bases (leonchani_nn, leonchani_lgbm,
joypig_lgbm) into the V8 family.

R27 TIE-band check vs current PRIMARY W2 reported per candidate.
"""
import pandas as pd, numpy as np, os
from scipy.stats import rankdata, spearmanr

OUTDIR = "submissions/public_scrape/_blends_d30b"
os.makedirs(OUTDIR, exist_ok=True)

def load(p):
    s = pd.read_csv(p).set_index("id")["PitNextLap"]
    return s

raun = load("submissions/public_scrape/raunakdey07_f1-pit-stops-blender-0-95454/submission.csv")
c5 = load("submissions/submission_d21_d22_C5_K20_R21_K27_rankmean_82_08_10.csv")
w2 = load("submissions/public_scrape/_blends_d30/D30_W2_A85_P15.csv")

leon_nn = load("submissions/public_scrape/_day30_scrape/leonchani_02-nn-predicting-f1-pit-stops/submission.csv")
leon_lgbm = load("submissions/public_scrape/_day30_scrape/leonchani_02-lightgbm-optuna-predicting-f1-pit-stops/submission.csv")
joypig = load("submissions/public_scrape/_day30_scrape/joypig_lightgbm/submission.csv")
sarvesh3 = load("submissions/public_scrape/_day30_scrape/sarveshchhetri_triple-boost-f1-pit-prediction-blending/submission.csv")
parth = load("submissions/public_scrape/_day30_scrape/parthsarnobat_s6e5-f1-pit-stops-catboost-xgboost-blend/submission.csv")

# Align indexes
idx = raun.index
for nm, s in [("raun",raun),("c5",c5),("w2",w2),("leon_nn",leon_nn),("leon_lgbm",leon_lgbm),("joypig",joypig),("sarvesh3",sarvesh3),("parth",parth)]:
    assert (s.index == idx).all() or len(s) == len(idx), f"{nm} index mismatch"

# rank-mean blend in (0,1) space
def rm(*series_weights):
    """series_weights: list of (series, weight). Rank-mean: avg of normalized ranks."""
    N = len(series_weights[0][0])
    acc = np.zeros(N)
    wsum = 0.0
    for s, w in series_weights:
        r = (rankdata(s.values, method="average") - 1) / (N - 1)
        acc += w * r
        wsum += w
    return pd.Series(acc / wsum, index=series_weights[0][0].index, name="PitNextLap")

# Candidate panel
candidates = {
    # Z1-Z3: replace C5 in V8 with single new orthogonal base
    "Z1_A90_LEONLGBM10":  rm((raun,0.90),(leon_lgbm,0.10)),
    "Z2_A90_LEONNN10":    rm((raun,0.90),(leon_nn,0.10)),
    "Z3_A90_JOYPIG10":    rm((raun,0.90),(joypig,0.10)),
    # X1-X3: 4-way add new orthogonal alongside C5 (small)
    "X1_A85_C5_LEONLGBM": rm((raun,0.85),(c5,0.10),(leon_lgbm,0.05)),
    "X2_A85_C5_LEONNN":   rm((raun,0.85),(c5,0.10),(leon_nn,0.05)),
    "X3_A85_C5_JOYPIG":   rm((raun,0.85),(c5,0.10),(joypig,0.05)),
    # Y1-Y3: 4-way diversification at heavier external weight
    "Y1_A80_C5_LEONLGBM_LEONNN": rm((raun,0.80),(c5,0.10),(leon_lgbm,0.05),(leon_nn,0.05)),
    "Y2_A80_C5_LEONLGBM_JOYPIG": rm((raun,0.80),(c5,0.10),(leon_lgbm,0.05),(joypig,0.05)),
    "Y3_A80_LEONLGBM_LEONNN":    rm((raun,0.80),(leon_lgbm,0.10),(leon_nn,0.10)),  # control: no C5
    # T1-T2: triple-boost variants
    "T1_A85_C5_SARVESH3":         rm((raun,0.85),(c5,0.10),(sarvesh3,0.05)),
    "T2_A85_C5_PARTH":            rm((raun,0.85),(c5,0.10),(parth,0.05)),
    # Q1: aggressive multi-orth blend (4 external + C5)
    "Q1_A70_C5_LEON2_JOY":        rm((raun,0.70),(c5,0.10),(leon_lgbm,0.07),(leon_nn,0.07),(joypig,0.06)),
    # P1: even more extreme — minority raunakdey
    "P1_A60_C5_LEON2_JOY_PARTH":  rm((raun,0.60),(c5,0.10),(leon_lgbm,0.075),(leon_nn,0.075),(joypig,0.075),(parth,0.075)),
}

print(f"{'cand':35s} {'ρ_W2':>10s} {'ρ_raun':>10s} {'ρ_C5':>10s} {'mean':>7s}")
print("-"*80)
for nm, blend in candidates.items():
    rW2 = spearmanr(w2, blend)[0]
    rR = spearmanr(raun, blend)[0]
    rC5 = spearmanr(c5, blend)[0]
    print(f"{nm:35s} {rW2:>10.6f} {rR:>10.6f} {rC5:>10.6f} {blend.mean():>7.4f}")
    df = pd.DataFrame({"id": idx, "PitNextLap": blend.values})
    df.to_csv(f"{OUTDIR}/D30b_{nm}.csv", index=False)

print(f"\nWritten to {OUTDIR}/")
