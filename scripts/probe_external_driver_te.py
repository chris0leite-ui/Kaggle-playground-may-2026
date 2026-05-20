"""Phase A — KILL-OR-CONFIRM probe (per plan, 2026-05-19/20).

Tests whether external Driver-aggregate TE (from Rohan Rao F1
1950-2024 dataset) adds anything beyond what internal Driver TE
absorbs. Mirrors probe_real_f1_axis.py fold layout exactly.

Verdict rule:
  ALIVE iff (auc_external - auc_internal) >= 0.005 standalone
           OR (auc_joint_lr - auc_internal_lr) >= 0.003.
  DEAD otherwise. Mechanisms #1/#3/#4 collapse if DEAD.

Writes audit/2026-05-20-external-driver-te-killprobe.{log,json}.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
EXT = Path("external/rohan_rao_f1")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def build_external_pit_rate(alpha: float) -> dict[str, float]:
    drivers = pd.read_csv(EXT / "drivers.csv")
    pits = pd.read_csv(EXT / "pit_stops.csv")
    races = pd.read_csv(EXT / "races.csv")
    # Year filter: 2011-2024 (pit_stops table starts at 2011 per dataset)
    races = races[races["year"] <= 2024]
    pits_with_y = pits.merge(races[["raceId", "year"]], on="raceId", how="inner")

    n_pits = pits_with_y.groupby("driverId").size()
    n_races = pits_with_y.groupby("driverId")["raceId"].nunique()
    df = pd.DataFrame({"n_pits": n_pits, "n_races": n_races}).fillna(0)
    df["pit_rate"] = df["n_pits"] / df["n_races"].clip(lower=1)
    df = df.merge(drivers[["driverId", "code"]], left_index=True,
                   right_on="driverId")
    df = df[df["code"].notna() & (df["code"] != r"\N")]
    global_mean = float(df["pit_rate"].mean())
    df["smoothed"] = (df["n_pits"] + alpha * global_mean) / (df["n_races"] + alpha)
    mapping = dict(zip(df["code"], df["smoothed"]))
    return mapping, global_mean


def fold_safe_internal_te(driver_arr, y, fold_idx, alpha=20.0):
    te = np.zeros(len(driver_arr), dtype=np.float32)
    for k in range(N_FOLDS):
        is_val = fold_idx == k
        d_tr = driver_arr[~is_val]
        y_tr = y[~is_val]
        gm = float(y_tr.mean())
        agg = pd.DataFrame({"d": d_tr, "y": y_tr}).groupby("d")["y"].agg(
            ["sum", "count"])
        smoothed = (agg["sum"] + alpha * gm) / (agg["count"] + alpha)
        m = smoothed.to_dict()
        te[is_val] = pd.Series(driver_arr[is_val]).map(m).fillna(gm).values
    return te


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].astype(int).values
    p_primary = np.load(ART / "oof_K17_xendcg_pathb_dcs_tau100000.npy")
    driver = train["Driver"].astype(str).values
    is_real = (~pd.Series(driver).str.match(r"^D\d{3}$")).values.astype(int)
    print(f"PRIMARY OOF AUC: {roc_auc_score(y, p_primary):.5f}")
    print(f"Real-F1 rows: {is_real.sum()} ({is_real.mean()*100:.1f}%)")

    # Build fold index
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_idx = np.empty(len(y), dtype=int)
    for k, (_, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_idx[vi] = k

    # ----- Internal TE (auc_internal) ------------------------------
    print("\n=== INTERNAL TE (alpha=20) ===")
    int_te = fold_safe_internal_te(driver, y, fold_idx, alpha=20.0)
    auc_internal = roc_auc_score(y, int_te)
    print(f"  auc_internal (standalone OOF): {auc_internal:.5f}")

    # ----- External TE alpha sweep (auc_external) -----------------
    print("\n=== EXTERNAL TE alpha sweep ===")
    best_alpha, best_auc_ext, best_ext = None, -1.0, None
    coverage = None
    ext_results = {}
    for alpha in (10.0, 25.0, 50.0, 100.0):
        m, gm = build_external_pit_rate(alpha=alpha)
        ext = pd.Series(driver).map(m)
        match = ext.notna()
        if coverage is None:
            real_match = match.values & (is_real == 1)
            real_total = (is_real == 1).sum()
            coverage = float(real_match.sum() / real_total)
            print(f"  External coverage on real-F1 rows: "
                  f"{coverage*100:.1f}% ({real_match.sum()}/{real_total})")
        # Synth/unmatched -> global mean of EXTERNAL pit_rate (not target)
        ext = ext.fillna(gm).astype(np.float32).values
        a = float(roc_auc_score(y, ext))
        ext_results[alpha] = a
        print(f"  alpha={alpha:>5.0f}  AUC={a:.5f}  ext_global_mean={gm:.3f}")
        if a > best_auc_ext:
            best_alpha, best_auc_ext, best_ext = alpha, a, ext
    auc_external = best_auc_ext
    print(f"  best alpha={best_alpha} AUC={auc_external:.5f}")

    # ----- LR-meta combinations -----------------------------------
    print("\n=== LR-META combinations ===")
    p_clip = np.clip(p_primary, 1e-6, 1 - 1e-6)
    L = logit(p_clip)
    te_is_external = (pd.Series(driver).map(
        build_external_pit_rate(best_alpha)[0]).notna().astype(int).values)

    def lr_oof(X):
        oof = np.zeros(len(y))
        for k in range(N_FOLDS):
            iv = fold_idx == k
            it = ~iv
            lr = LogisticRegression(C=1.0, max_iter=200)
            lr.fit(X[it], y[it])
            oof[iv] = lr.predict_proba(X[iv])[:, 1]
        return roc_auc_score(y, oof)

    # baseline: K17 + internal TE + is_real (same as today's probe)
    X_int = np.column_stack([L, L * is_real, int_te, is_real]).astype(np.float64)
    auc_internal_lr = lr_oof(X_int)
    print(f"  internal-only LR-meta:  {auc_internal_lr:.5f}")

    # joint: + external TE + te_is_external
    X_joint = np.column_stack([L, L * is_real, int_te, is_real,
                                best_ext, te_is_external]).astype(np.float64)
    auc_joint_lr = lr_oof(X_joint)
    print(f"  joint LR-meta:          {auc_joint_lr:.5f}")

    # Decompose
    real_m = is_real == 1
    synth_m = ~real_m

    # Verdict
    print("\n=== VERDICT ===")
    delta_standalone = auc_external - auc_internal
    delta_joint = auc_joint_lr - auc_internal_lr
    alive_standalone = delta_standalone >= 0.005
    alive_joint = delta_joint >= 0.003
    alive = alive_standalone or alive_joint
    verdict = "ALIVE" if alive else "DEAD"
    print(f"  Δ standalone (ext-int): {delta_standalone:+.5f}  "
          f"({'PASS' if alive_standalone else 'fail'}, thresh 0.005)")
    print(f"  Δ joint LR-meta:        {delta_joint:+.5f}  "
          f"({'PASS' if alive_joint else 'fail'}, thresh 0.003)")
    print(f"  VERDICT: {verdict}")

    out = dict(
        date="2026-05-20",
        primary_oof_auc=float(roc_auc_score(y, p_primary)),
        auc_internal=float(auc_internal),
        auc_external_alpha_sweep=ext_results,
        best_alpha=float(best_alpha),
        auc_external=float(auc_external),
        auc_internal_lr=float(auc_internal_lr),
        auc_joint_lr=float(auc_joint_lr),
        delta_standalone=float(delta_standalone),
        delta_joint=float(delta_joint),
        alive_standalone=bool(alive_standalone),
        alive_joint=bool(alive_joint),
        verdict=verdict,
        external_coverage_real_pct=float(coverage * 100),
        wall_s=time.time() - t0,
    )
    Path("audit/2026-05-20-external-driver-te-killprobe.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nWall: {out['wall_s']:.1f}s")
    print("Wrote audit/2026-05-20-external-driver-te-killprobe.json")


if __name__ == "__main__":
    main()
