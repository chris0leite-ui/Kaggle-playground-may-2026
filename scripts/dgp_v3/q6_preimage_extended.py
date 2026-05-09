"""Phase A2 parallel — extended preimage probe.

Use the now-known full set of literal-copy columns (LapTime, RaceProgress,
LapTime_Delta, Position, TyreLife, Position_Change) to find synth → orig
matches. Skip Cumulative_Degradation (only 71% literal-copy per Q2).

Three questions answered:
  Q6.1 — How many synth rows have a perfect 6-tuple match in orig?
  Q6.2 — When matched, what fraction inherit (Year, Compound, PitStop,
         PitNextLap) correctly?
  Q6.3 — Is each orig row used ~6× (627k/101k) in synth, or skewed?
         (Tests the upsampling mechanism.)

Output: scripts/artifacts/dgp_v3_q6_preimage.json
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

    orig = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    orig = orig.rename(columns={"LapTime (s)": "LapTime"})
    orig = orig.drop(columns=["Normalized_TyreLife"])
    orig = orig.dropna()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if "LapTime (s)" in train.columns:
        train = train.rename(columns={"LapTime (s)": "LapTime"})
        test = test.rename(columns={"LapTime (s)": "LapTime"})
    synth = pd.concat(
        [train[[c for c in train.columns if c != "PitNextLap"]], test],
        ignore_index=True,
    )
    t(f"loaded orig {orig.shape}, synth {synth.shape}", ts)

    # Build a per-row fingerprint key.
    # Literal-copy + preserved columns (KS < 0.06):
    #   LapTime, RaceProgress, LapTime_Delta, Position, TyreLife, Position_Change
    fp_cols = ["LapTime", "RaceProgress", "LapTime_Delta",
               "Position", "TyreLife", "Position_Change"]

    def keyfunc(df: pd.DataFrame) -> pd.Series:
        # Round floats to 6 decimal places to handle any FP wobble.
        parts = []
        for c in fp_cols:
            v = df[c]
            if v.dtype.kind == "f":
                v = v.round(6)
            parts.append(v.astype(str))
        return pd.Series(["|".join(t) for t in zip(*[p.values for p in parts])])

    synth["_key"] = keyfunc(synth)
    orig["_key"] = keyfunc(orig)
    t("fingerprint keys built", ts)

    # Q6.1 — match rate
    orig_keys = set(orig["_key"].tolist())
    synth_match_mask = synth["_key"].isin(orig_keys)
    out["Q6.1_match_rate_6tuple"] = float(synth_match_mask.mean())
    out["Q6.1_n_synth_matched"] = int(synth_match_mask.sum())
    out["Q6.1_n_orig_keys"] = len(orig_keys)
    t(f"6-tuple match rate: {out['Q6.1_match_rate_6tuple']:.4f}", ts)

    # Q6.2 — when matched, do (Year, Compound, PitStop) inherit?
    if synth_match_mask.sum() > 0:
        orig_lookup = orig.set_index("_key")
        # For ambiguous keys (multiple orig rows with same fingerprint), take first
        orig_lookup = orig_lookup[~orig_lookup.index.duplicated(keep="first")]
        sample_matched = synth[synth_match_mask].sample(
            min(50_000, int(synth_match_mask.sum())), random_state=0
        )
        joined = sample_matched.merge(
            orig_lookup[["Year", "Compound", "PitStop", "PitNextLap"]].rename(
                columns={
                    "Year": "Year_orig",
                    "Compound": "Compound_orig",
                    "PitStop": "PitStop_orig",
                    "PitNextLap": "PitNextLap_orig",
                }
            ),
            left_on="_key",
            right_index=True,
            how="left",
        )
        out["Q6.2_year_match"] = float((joined["Year"] == joined["Year_orig"]).mean())
        out["Q6.2_compound_match"] = float((joined["Compound"] == joined["Compound_orig"]).mean())
        out["Q6.2_pitstop_match"] = float((joined["PitStop"] == joined["PitStop_orig"]).mean())
        out["Q6.2_n_joined"] = int(len(joined))
        t(
            f"matched-row Y/C/PS inheritance: "
            f"{out['Q6.2_year_match']:.4f} / "
            f"{out['Q6.2_compound_match']:.4f} / "
            f"{out['Q6.2_pitstop_match']:.4f}",
            ts,
        )

    # Q6.3 — usage distribution per orig key
    synth_key_counts = synth["_key"].value_counts()
    matched_synth_keys = synth_key_counts[synth_key_counts.index.isin(orig_keys)]
    out["Q6.3_orig_key_usage"] = {
        "n_orig_keys": len(orig_keys),
        "n_orig_keys_used_at_least_once": int(matched_synth_keys[matched_synth_keys > 0].count()),
        "frac_orig_keys_unused": float(
            1 - (matched_synth_keys.count() / len(orig_keys))
        ),
        "median_uses_per_used_orig_key": float(matched_synth_keys.median()),
        "p90_uses_per_used_orig_key": float(matched_synth_keys.quantile(0.9)),
        "max_uses_per_orig_key": int(matched_synth_keys.max()) if len(matched_synth_keys) else 0,
    }
    t(f"orig usage: {out['Q6.3_orig_key_usage']}", ts)

    # Save
    fp = ART / "dgp_v3_q6_preimage.json"
    fp.write_text(json.dumps(out, indent=2))
    t(f"wrote {fp.name}", ts)

    # Print summary
    print("\n=== Q6 summary ===")
    print(f"  6-tuple match rate: {out['Q6.1_match_rate_6tuple']:.4f}")
    print(f"  matched rows: {out['Q6.1_n_synth_matched']:,} of {len(synth):,}")
    if "Q6.2_year_match" in out:
        print(f"  matched (Year, Compound, PitStop) inheritance: "
              f"{out['Q6.2_year_match']:.4f} / "
              f"{out['Q6.2_compound_match']:.4f} / "
              f"{out['Q6.2_pitstop_match']:.4f}")


if __name__ == "__main__":
    main()
