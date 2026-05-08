"""scripts/probe_a3_7_uid_smoothing.py — EXP-A3-7 dry-run probe.

Origin: IEEE-CIS Fraud Detection 1st-place writeup ("UID synthesis +
post-hoc smoothing"); listed in `audit/2026-05-08-fe-research-extended.md`
as Tier-A3 entry #7.

Mechanism: synthesise a UID by concatenating `(Driver, Race, Year)`,
then replace each row's prediction with the UID-group mean.

Two variants tested:
  - **naive**: `groupby(uid).transform('mean')` — what IEEE applied
    at test time. For train-OOF this is mildly leaky in a benign sense
    (each row contributes to its own group mean) but matches the
    deployment recipe.
  - **loo**: leave-one-out group mean. Stricter for train-OOF AUC.

Verdict criteria:
  PASS:  K=4 PRIMARY OOF Δ ≥ +0.5 bp under EITHER variant.
  WEAK:  +0.1 to +0.5 bp.
  FAIL:  ≤ +0.1 bp or regression.

Pure post-process; ~2 min wall. No FE pipeline, no model retrain. This
is the cheapest funnel-design dry-run from the multi-model FE testing
campaign plan (`/root/.claude/plans/now-carefully-plan-how-polished-
dewdrop.md`).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
PRIMARY_OOF = ART / "oof_K4_fwd_pathb.npy"
PRIMARY_TEST = ART / "test_K4_fwd_pathb.npy"


def _pos(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def _build_uid(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "__" + df[c].astype(str)
    return s


def smooth_naive(pred: np.ndarray, uid: pd.Series) -> np.ndarray:
    """pred ← group_mean(uid)."""
    s = pd.Series(pred)
    return s.groupby(uid).transform("mean").values.astype(np.float64)


def smooth_loo(pred: np.ndarray, uid: pd.Series) -> np.ndarray:
    """Leave-one-out group mean: (sum_g - pred_i) / (count_g - 1).
    Falls back to single-row groups via the global mean.
    """
    s = pd.Series(pred)
    sum_g = s.groupby(uid).transform("sum").values
    cnt_g = s.groupby(uid).transform("count").values
    global_mean = float(s.mean())
    denom = cnt_g - 1
    out = np.where(denom > 0, (sum_g - pred) / np.maximum(denom, 1),
                   global_mean)
    return out.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid-cols", nargs="+",
                    default=["Driver", "Race", "Year"],
                    help="columns used to form the UID. Default: "
                         "Driver, Race, Year (the canonical IEEE/Otto choice).")
    ap.add_argument("--mix", type=float, default=0.5,
                    help="alpha in alpha*smoothed + (1-alpha)*original. "
                         "0 = no smoothing; 1 = pure group mean.")
    ap.add_argument("--save-prefix", default="probe_a3_7_uid_smoothing",
                    help="output prefix for OOF + test arrays.")
    args = ap.parse_args()

    print(f"=== EXP-A3-7 dry-run | UID = {args.uid_cols} | mix α = {args.mix} ===")
    t0 = time.time()

    # Load PRIMARY OOF + test
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    print(f"  primary OOF shape: {primary_oof.shape}")

    # Load train + test labels / UID columns
    train = pd.read_csv("data/train.csv",
                        usecols=args.uid_cols + [TARGET, "id"])
    test = pd.read_csv("data/test.csv", usecols=args.uid_cols + ["id"])
    y = train[TARGET].astype(int).values

    # Sanity: artifacts shape matches train length
    assert len(primary_oof) == len(train), \
        f"OOF length {len(primary_oof)} ≠ train length {len(train)}"
    assert len(primary_test) == len(test), \
        f"test pred length {len(primary_test)} ≠ test length {len(test)}"

    # Baseline
    auc_base = float(roc_auc_score(y, primary_oof))
    print(f"\n  baseline K=4 PRIMARY OOF AUC: {auc_base:.5f}")

    uid_train = _build_uid(train, args.uid_cols)
    uid_test = _build_uid(test, args.uid_cols)
    n_groups_train = uid_train.nunique()
    n_groups_test = uid_test.nunique()
    overlap = len(set(uid_train.unique()) & set(uid_test.unique()))
    print(f"  UID groups: train {n_groups_train}, test {n_groups_test}, "
          f"shared {overlap}")

    # Variant 1 — naive groupby mean
    smoothed_oof_naive = smooth_naive(primary_oof, uid_train)
    smoothed_test_naive = smooth_naive(primary_test, uid_test)
    blend_oof_naive = (args.mix * smoothed_oof_naive
                       + (1 - args.mix) * primary_oof)
    auc_naive = float(roc_auc_score(y, blend_oof_naive))
    auc_naive_pure = float(roc_auc_score(y, smoothed_oof_naive))
    delta_naive_pure = (auc_naive_pure - auc_base) * 1e4
    delta_naive_blend = (auc_naive - auc_base) * 1e4
    print(f"\n  [naive groupby-mean]")
    print(f"    pure smoothed OOF AUC: {auc_naive_pure:.5f}  Δ {delta_naive_pure:+.3f} bp")
    print(f"    α={args.mix} blend OOF AUC: {auc_naive:.5f}  Δ {delta_naive_blend:+.3f} bp")

    # Variant 2 — leave-one-out group mean
    smoothed_oof_loo = smooth_loo(primary_oof, uid_train)
    blend_oof_loo = (args.mix * smoothed_oof_loo
                     + (1 - args.mix) * primary_oof)
    auc_loo = float(roc_auc_score(y, blend_oof_loo))
    auc_loo_pure = float(roc_auc_score(y, smoothed_oof_loo))
    delta_loo_pure = (auc_loo_pure - auc_base) * 1e4
    delta_loo_blend = (auc_loo - auc_base) * 1e4
    print(f"\n  [leave-one-out groupby-mean]")
    print(f"    pure smoothed OOF AUC: {auc_loo_pure:.5f}  Δ {delta_loo_pure:+.3f} bp")
    print(f"    α={args.mix} blend OOF AUC: {auc_loo:.5f}  Δ {delta_loo_blend:+.3f} bp")

    # Verdict (best of variants × mix-pure)
    best_delta = max(delta_naive_pure, delta_naive_blend,
                     delta_loo_pure, delta_loo_blend)
    if best_delta >= 0.5:
        verdict = "PASS"
    elif best_delta >= 0.1:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    print(f"\n  best Δ across variants: {best_delta:+.3f} bp  →  verdict: {verdict}")

    # Save artifacts (variant the best Δ)
    candidates = [
        ("naive_pure", smoothed_oof_naive, smoothed_test_naive, delta_naive_pure),
        ("naive_blend", blend_oof_naive,
         args.mix * smoothed_test_naive + (1 - args.mix) * primary_test,
         delta_naive_blend),
        ("loo_pure", smoothed_oof_loo, smoothed_test_naive, delta_loo_pure),
        ("loo_blend", blend_oof_loo,
         args.mix * smoothed_test_naive + (1 - args.mix) * primary_test,
         delta_loo_blend),
    ]
    best = max(candidates, key=lambda c: c[3])
    print(f"  best variant: {best[0]} (Δ={best[3]:+.3f} bp)")

    np.save(ART / f"oof_{args.save_prefix}_strat.npy",
            np.column_stack([1 - best[1], best[1]]).astype(np.float64))
    np.save(ART / f"test_{args.save_prefix}_strat.npy",
            np.column_stack([1 - best[2], best[2]]).astype(np.float64))
    summary = dict(
        primary_oof_auc=auc_base,
        uid_cols=args.uid_cols,
        mix_alpha=args.mix,
        train_groups=int(n_groups_train),
        test_groups=int(n_groups_test),
        shared_groups=int(overlap),
        delta_bp=dict(
            naive_pure=delta_naive_pure,
            naive_blend=delta_naive_blend,
            loo_pure=delta_loo_pure,
            loo_blend=delta_loo_blend,
        ),
        best_variant=best[0],
        best_delta_bp=float(best[3]),
        verdict=verdict,
        wall_s=time.time() - t0,
    )
    (ART / f"{args.save_prefix}_results.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n  wall: {time.time()-t0:.1f}s")
    print(f"  → scripts/artifacts/oof_{args.save_prefix}_strat.npy")
    print(f"  → scripts/artifacts/test_{args.save_prefix}_strat.npy")
    print(f"  → scripts/artifacts/{args.save_prefix}_results.json")


if __name__ == "__main__":
    main()
