"""Phase C — Mechanism #5: laps_remaining_if_no_2nd_compound_yet.

FIA 2-compound rule: dry races require >=2 distinct compounds.
If at lap N a driver has used only 1 compound type so far, they MUST
pit before race end. Encode the pressure as:
    needs_2nd = 1 if unique_compounds_so_far == 1 else 0
    laps_remaining = race_total_laps - LapNumber
    pressure = laps_remaining * needs_2nd

Aggregate keys: (Year, Race, Driver, LapNumber). compounds_used_so_far
is the set of unique compounds in rows of the same (Year, Race, Driver)
group with LapNumber <= current.

Race total laps: max(LapNumber) over train+test for each (Year, Race)
— transductive but label-free, R25-safe.

Outputs:
- audit/2026-05-20-laps-remain-probe.{log,json}
- scripts/artifacts/oof_R16_lapsremain_strat.npy + test variant
  (rank-normalized to original train.csv id order)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    n_train = len(train)
    both = pd.concat([train, test], ignore_index=True, sort=False)
    # race total laps from train+test combined (label-free transductive)
    race_total = both.groupby(["Year", "Race"])["LapNumber"].max().rename(
        "race_total_laps").reset_index()
    both = both.merge(race_total, on=["Year", "Race"], how="left")
    both["laps_remaining"] = (both["race_total_laps"] - both["LapNumber"]).clip(lower=0)

    # compounds used so far per (Year, Race, Driver) up to LapNumber
    both = both.sort_values(["Year", "Race", "Driver", "LapNumber"]).reset_index(drop=True)
    both["compound_int"] = both["Compound"].astype("category").cat.codes
    # cumulative unique count: track per group using expanding-set logic
    grp = ["Year", "Race", "Driver"]
    # mark first-occurrence within group of each compound
    seen = both.groupby(grp + ["compound_int"]).cumcount() == 0
    both["new_compound_flag"] = seen.astype(int)
    both["unique_compounds_so_far"] = both.groupby(grp)["new_compound_flag"].cumsum()
    both["needs_2nd"] = (both["unique_compounds_so_far"] == 1).astype(int)
    both["pressure"] = both["laps_remaining"] * both["needs_2nd"]
    # ratio version (scale-invariant)
    both["pressure_ratio"] = both["pressure"] / both["race_total_laps"].clip(lower=1)

    # Wet/intermediate flag — FIA rule strict to dry only
    both["is_wet_or_inter"] = both["Compound"].isin(["WET", "INTERMEDIATE"]).astype(int)

    # restore original index order via id
    both = both.sort_values("id").reset_index(drop=True)
    train_out = both.iloc[:n_train].reset_index(drop=True)
    test_out = both.iloc[n_train:].reset_index(drop=True)
    return train_out, test_out


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"train {train.shape}  test {test.shape}")
    y = train[TARGET].astype(int).values
    p_primary = np.load(ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy")
    print(f"PRIMARY OOF AUC: {roc_auc_score(y, p_primary):.5f}")

    train_f, test_f = build_features(train, test)
    print("\n=== FEATURE STATS (train) ===")
    for c in ["needs_2nd", "laps_remaining", "pressure", "pressure_ratio",
              "unique_compounds_so_far", "is_wet_or_inter"]:
        s = train_f[c]
        print(f"  {c}: mean={s.mean():.4f} std={s.std():.4f} "
              f"min={s.min():.4g} max={s.max():.4g}")

    # G1 — standalone OOF AUC of each engineered scalar
    print("\n=== G1 STANDALONE AUCs ===")
    feats = ["needs_2nd", "pressure", "pressure_ratio", "laps_remaining",
             "unique_compounds_so_far"]
    for c in feats:
        a = float(roc_auc_score(y, train_f[c].values))
        print(f"  {c}: AUC {a:.5f}")

    # Fold layout
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_idx = np.empty(len(y), dtype=int)
    for k, (_, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_idx[vi] = k

    # G2 — LR-meta with K17 + new features (dry rows + flagged wet/inter)
    print("\n=== G2 LR-META vs K17 PRIMARY ===")
    p_clip = np.clip(p_primary, 1e-6, 1 - 1e-6)
    L = logit(p_clip)
    X_base = L.reshape(-1, 1).astype(np.float64)
    X_full = np.column_stack([L,
        train_f["pressure"].values,
        train_f["pressure_ratio"].values,
        train_f["needs_2nd"].values,
        train_f["unique_compounds_so_far"].values,
        train_f["is_wet_or_inter"].values,
    ]).astype(np.float64)

    def lr_oof(X):
        oof = np.zeros(len(y))
        for k in range(N_FOLDS):
            iv = fold_idx == k
            it = ~iv
            lr = LogisticRegression(C=1.0, max_iter=300)
            lr.fit(X[it], y[it])
            oof[iv] = lr.predict_proba(X[iv])[:, 1]
        return oof
    oof_base = lr_oof(X_base)
    oof_full = lr_oof(X_full)
    auc_base = roc_auc_score(y, oof_base)
    auc_full = roc_auc_score(y, oof_full)
    delta_bp = (auc_full - auc_base) * 1e4
    print(f"  base (K17-only LR-meta) AUC: {auc_base:.5f}")
    print(f"  + lapsremain feats   AUC:    {auc_full:.5f}")
    print(f"  Δ vs base: {delta_bp:+.3f} bp")
    print(f"  Δ vs PRIMARY R15 (.95449): "
          f"{(auc_full - roc_auc_score(y, p_primary)) * 1e4:+.3f} bp")

    # Stratum decomposition (real vs synth)
    drv = train_f["Driver"].astype(str)
    is_real = (~drv.str.match(r"^D\d{3}$")).values
    if is_real.sum() > 0:
        print(f"\n  Real-F1 Δ:  {(roc_auc_score(y[is_real], oof_full[is_real]) - roc_auc_score(y[is_real], oof_base[is_real]))*1e4:+.3f} bp")
        print(f"  Synth Δ:    {(roc_auc_score(y[~is_real], oof_full[~is_real]) - roc_auc_score(y[~is_real], oof_base[~is_real]))*1e4:+.3f} bp")

    # If G2 ≥ +0.02 bp, save as new base for Day-20 K=18 stack-add
    GATE_BP = 0.02
    passed = delta_bp >= GATE_BP
    print(f"\n  G2 gate (+{GATE_BP:.2f} bp floor): {'PASS' if passed else 'fail'}")

    if passed:
        # Build test predictions via LR-meta refit on full train
        oof_te = np.column_stack([
            logit(np.clip(np.zeros(len(test_f)) + 0.5, 1e-6, 1 - 1e-6)),  # placeholder
            test_f["pressure"].values,
            test_f["pressure_ratio"].values,
            test_f["needs_2nd"].values,
            test_f["unique_compounds_so_far"].values,
            test_f["is_wet_or_inter"].values,
        ]).astype(np.float64)
        # Replace placeholder with actual K17 test predictions
        p_test_K17 = np.load(ART / "test_K17_xendcg_pathb_dcs_tau100000.npy")
        oof_te[:, 0] = logit(np.clip(p_test_K17, 1e-6, 1 - 1e-6))
        lr = LogisticRegression(C=1.0, max_iter=300)
        lr.fit(X_full, y)
        test_pred = lr.predict_proba(oof_te)[:, 1]

        # Rank-normalize jointly for Path-B uniformity
        combined = np.concatenate([oof_full, test_pred])
        ranks = rankdata(combined)
        eps = 1.0 / (2 * len(ranks))
        uni = np.clip((ranks - 0.5) / len(ranks), eps, 1 - eps)
        oof_uni = uni[:len(oof_full)]
        test_uni = uni[len(oof_full):]
        np.save(ART / "oof_R16_lapsremain_strat.npy", oof_uni.astype(np.float32))
        np.save(ART / "test_R16_lapsremain_strat.npy", test_uni.astype(np.float32))
        print("  saved oof_R16_lapsremain_strat.npy + test_R16_lapsremain_strat.npy")

        # ρ vs PRIMARY for R27 band check
        rho = spearmanr(oof_uni, p_primary)[0]
        print(f"  ρ_OOF vs R15 PRIMARY: {rho:.6f}")

    out = dict(
        date="2026-05-20",
        primary_oof_auc=float(roc_auc_score(y, p_primary)),
        g1_aucs={c: float(roc_auc_score(y, train_f[c].values)) for c in feats},
        auc_lr_base=float(auc_base),
        auc_lr_full=float(auc_full),
        delta_bp=float(delta_bp),
        delta_vs_primary_bp=float((auc_full - roc_auc_score(y, p_primary)) * 1e4),
        g2_passed=bool(passed),
        wall_s=time.time() - t0,
    )
    Path("audit/2026-05-20-laps-remain-probe.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nWall: {out['wall_s']:.1f}s")
    print("Wrote audit/2026-05-20-laps-remain-probe.json")


if __name__ == "__main__":
    main()
