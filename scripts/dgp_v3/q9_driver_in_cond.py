"""Phase A5 probe — is Driver in the host's cond-vector, or post-hoc random?

Two contrasting predictions:
  H_in_cond:     each ghost driver concentrates in a specific (Year, Race)
                 cluster. Variance of (Year, Race) values per ghost is low.
  H_post_hoc:    ghost drivers are uniformly distributed across (Year, Race).
                 Variance high; per-driver year-distribution ~ marginal.

Output: scripts/artifacts/dgp_v3_q9_driver_cond.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def main() -> None:
    out: dict = {}
    ts = time.time()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
        test = test.rename(columns={"LapTime (s)": "LapTime"})
    synth = pd.concat(
        [train[[c for c in train.columns if c != "PitNextLap"]], test],
        ignore_index=True,
    )
    t(f"loaded synth {synth.shape}", ts)

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig_drivers = set(orig["Driver"].unique().tolist())

    synth["driver_class"] = synth["Driver"].apply(
        lambda d: "active" if d in orig_drivers
        else ("dprefix" if str(d).startswith("D") and len(str(d)) == 4 else "retired")
    )

    out["driver_class_counts"] = synth["driver_class"].value_counts().to_dict()
    t(f"driver classes: {out['driver_class_counts']}", ts)

    # For each ghost driver, compute the entropy of its (Year, Race) distribution.
    # Compare to the entropy of (Year, Race) marginal across all of synth.
    overall_yr = (
        synth.groupby(["Year", "Race"]).size() / len(synth)
    ).rename("p_marginal")
    overall_entropy = float(-(overall_yr * np.log(overall_yr.replace(0, np.nan))).dropna().sum())
    out["marginal_yr_entropy"] = overall_entropy

    # Sample 100 of each class
    rng = np.random.default_rng(0)
    out["per_class_yr_entropy"] = {}
    for cls in ["dprefix", "retired", "active"]:
        cls_drivers = synth.loc[synth["driver_class"] == cls, "Driver"].unique()
        if len(cls_drivers) == 0:
            continue
        sampled = list(rng.choice(cls_drivers, size=min(100, len(cls_drivers)), replace=False))
        ents = []
        per_driver_year_concentration = []
        per_driver_year_share_top = []
        for d in sampled:
            sub = synth[synth["Driver"] == d]
            if len(sub) < 10:
                continue
            p = sub.groupby(["Year", "Race"]).size() / len(sub)
            ent = float(-(p * np.log(p.replace(0, np.nan))).dropna().sum())
            ents.append(ent)
            year_p = sub.groupby("Year").size() / len(sub)
            top_share = float(year_p.max())
            per_driver_year_share_top.append(top_share)
            per_driver_year_concentration.append({
                "driver": d, "n_rows": int(len(sub)),
                "top_year_share": top_share,
                "yr_entropy": ent,
            })
        out["per_class_yr_entropy"][cls] = {
            "n_drivers_sampled": len(ents),
            "median_entropy": float(np.median(ents)) if ents else None,
            "p10_entropy": float(np.percentile(ents, 10)) if ents else None,
            "p90_entropy": float(np.percentile(ents, 90)) if ents else None,
            "median_top_year_share": float(np.median(per_driver_year_share_top))
                if per_driver_year_share_top else None,
            "examples_first_5": per_driver_year_concentration[:5],
        }

    t("per-class entropy done", ts)

    # Save
    fp = ART / "dgp_v3_q9_driver_cond.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    t(f"wrote {fp.name}", ts)

    # Print summary
    print("\n=== Driver class size ===")
    for k, v in out["driver_class_counts"].items():
        print(f"  {k:10s} {v:,} rows")
    print(f"\nMarginal (Year, Race) entropy = {out['marginal_yr_entropy']:.3f}")
    print(f"  (uniform 26 races x 4 years would give log(104) = {np.log(104):.3f})")

    print("\n=== Per-driver (Year, Race) distribution entropy ===")
    print("If driver in cond: entropy << marginal (driver concentrates in few cells)")
    print("If post-hoc random: entropy ≈ marginal")
    for cls, v in out["per_class_yr_entropy"].items():
        print(f"  {cls:10s} median_entropy={v['median_entropy']:.3f} "
              f"p10={v['p10_entropy']:.3f} p90={v['p90_entropy']:.3f} "
              f"median_top_year_share={v['median_top_year_share']:.3f}")


if __name__ == "__main__":
    main()
