"""d18 F6 — Information-theoretic ceiling map.

For each (Compound, Stint, Year) cell, compute KL(P_synth || P_orig)
on the 7 KS-low features. Diagnostic only — no base produced.

Method per cell:
  1. Subset orig and synth_train to that cell.
  2. If both have ≥200 rows: fit GMM(K=4) on orig cell, score synth cell.
     KL_approx = E_synth[log P_synth - log P_orig], where E_synth is
     approximated by GMM-fit on synth and orig has GMM-fit too.
  3. Otherwise: skip (low-data cells).

Per-cell KL identifies:
  - Low-KL cells: orig-transfer should work strongly → orig-LGBM-restricted
    transfer ceiling is higher.
  - High-KL cells: synthesizer corruption is severe → orig-transfer hits
    a ceiling.

This map informs F1 (replay forensics) cell focus and future Path-B
cohort-axis design.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
SEED = 42
N_COMP = 4
LAPTIME = "LapTime (s)"
TARGET = "PitNextLap"

KS_LOW_FEATS = ["TyreLife", "Position", LAPTIME, "Cumulative_Degradation",
                "RaceProgress", "LapTime_Delta", "LapNumber"]


def kl_gmm(g_orig, g_synth, X_synth_cell):
    """Estimate KL(P_synth || P_orig) ≈ mean(log P_synth(x) - log P_orig(x))
    over x ~ synth_cell (Monte-Carlo with synth samples)."""
    if len(X_synth_cell) < 50:
        return None
    # Up to 5000 samples
    if len(X_synth_cell) > 5000:
        idx = np.random.RandomState(SEED).choice(len(X_synth_cell), 5000,
                                                  replace=False)
        Xs = X_synth_cell[idx]
    else:
        Xs = X_synth_cell
    ll_o = g_orig.score_samples(Xs)
    ll_s = g_synth.score_samples(Xs)
    return float(np.mean(ll_s - ll_o))


def main():
    t0 = time.time()
    print("[F6 KL ceiling]  loading data")
    tr = pd.read_csv("data/train.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()

    # Standardize using orig stats
    Xo = orig[KS_LOW_FEATS].astype(float).values
    Xt = tr[KS_LOW_FEATS].astype(float).values
    np.nan_to_num(Xo, copy=False); np.nan_to_num(Xt, copy=False)
    mu = Xo.mean(axis=0); sd = Xo.std(axis=0) + 1e-8
    Xo = (Xo - mu) / sd; Xt = (Xt - mu) / sd

    cells = []
    for cmp in sorted(set(tr["Compound"].astype(str)) & set(orig["Compound"].astype(str))):
        for stint in sorted(set(tr["Stint"].astype(int)) & set(orig["Stint"].astype(int))):
            for year in sorted(set(tr["Year"].astype(int)) & set(orig["Year"].astype(int))):
                tr_mask = ((tr["Compound"] == cmp) &
                           (tr["Stint"] == stint) &
                           (tr["Year"] == year)).values
                or_mask = ((orig["Compound"] == cmp) &
                           (orig["Stint"] == stint) &
                           (orig["Year"] == year)).values
                n_t = int(tr_mask.sum()); n_o = int(or_mask.sum())
                if n_t < 200 or n_o < 200:
                    continue
                X_t = Xt[tr_mask]; X_o = Xo[or_mask]
                # Fit GMMs (small n_comp because cells are smaller)
                k = min(N_COMP, max(2, n_o // 200))
                try:
                    g_o = GaussianMixture(n_components=k, covariance_type="diag",
                                          max_iter=100, random_state=SEED,
                                          reg_covar=1e-2).fit(X_o)
                    g_t = GaussianMixture(n_components=k, covariance_type="diag",
                                          max_iter=100, random_state=SEED,
                                          reg_covar=1e-2).fit(X_t)
                except Exception:
                    continue
                kl_st_o = kl_gmm(g_o, g_t, X_t)        # KL(synth || orig)
                kl_o_st = kl_gmm(g_t, g_o, X_o)        # KL(orig || synth)
                # Cell target rate (in train)
                rate_t = float(tr.loc[tr_mask, TARGET].astype(int).mean())
                rate_o = float(orig.loc[or_mask, TARGET].astype(int).mean())
                cells.append(dict(
                    Compound=cmp, Stint=int(stint), Year=int(year),
                    n_train=n_t, n_orig=n_o, k=k,
                    kl_synth_to_orig=kl_st_o, kl_orig_to_synth=kl_o_st,
                    target_rate_train=rate_t, target_rate_orig=rate_o,
                ))

    df = pd.DataFrame(cells)
    df = df.sort_values("kl_synth_to_orig", ascending=False)
    print(f"\n[KL map]  {len(df)} (Compound, Stint, Year) cells with ≥200/200 rows")
    print(df.to_string(index=False, max_colwidth=20))
    df.to_csv(ART / "d18_f6_kl_ceiling.csv", index=False)

    summary = dict(
        n_cells=len(df),
        kl_synth_to_orig_quantiles={
            "q05": float(df["kl_synth_to_orig"].quantile(0.05)),
            "q25": float(df["kl_synth_to_orig"].quantile(0.25)),
            "q50": float(df["kl_synth_to_orig"].quantile(0.50)),
            "q75": float(df["kl_synth_to_orig"].quantile(0.75)),
            "q95": float(df["kl_synth_to_orig"].quantile(0.95)),
        },
        most_corrupted_cells=df.head(5).to_dict("records"),
        cleanest_cells=df.tail(5).to_dict("records"),
        wall_s=time.time() - t0,
    )
    (ART / "d18_f6_kl_ceiling_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done F6]  wall {time.time()-t0:.0f}s")
    print(f"  KL(synth||orig) median {df['kl_synth_to_orig'].median():.3f}")
    print(f"  most-corrupted cell: {df.iloc[0].to_dict()}")
    print(f"  cleanest cell:       {df.iloc[-1].to_dict()}")


if __name__ == "__main__":
    main()
