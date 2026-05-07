"""scripts/probe_target_structure.py — Probe #2 hiding-in-plain-sight.

Hypothesis: d13_data_probe.json shows max_pos_per_stint=30,
stint_with_2plus_pos=21,254, and frac_pos_aligned_with_next_pitstop=0.221.
The team has been stacking 23 bases against an undefined target. What IS
PitNextLap?

This script does pure EDA — no model fitting. ~30 min CPU expected wall.

Probes:
  T1.  Per-stint target distribution: where do positives sit within a stint
       relative to (lap-in-stint, normalized stint progress)?
  T2.  Conditional P(target=1 | NTL_bin × Compound).
  T3.  Conditional P(target=1 | (lap-in-stint, stint-size)). Is target=1
       concentrated in last-K laps of stint?
  T4.  Window-target hypothesis: if target=1 ⇔ "pit within next K laps" for
       some K, then total positives per stint should equal min(K, n_remaining).
  T5.  Position-in-Race hypothesis: target=1 driven by RaceProgress quantile.
  T6.  Multi-positive stint structure: when a stint has 2+ positives, are
       positives contiguous or scattered?

Output: scripts/artifacts/probe_target_structure.json + console summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
OUT = ART / "probe_target_structure.json"
TARGET = "PitNextLap"


def main() -> None:
    print("Loading train...")
    tr = pd.read_csv("data/train.csv")
    print(f"  train: {tr.shape}")

    out: dict = {}
    out["pos_rate_overall"] = float(tr[TARGET].mean())
    out["row_count"] = int(len(tr))

    # === T1 — positives within stint =====================================
    print("\n[T1] Stint-relative position of positives...")
    sort_cols = ["Race", "Driver", "Year", "Stint", "LapNumber"]
    tr2 = tr.sort_values(sort_cols).reset_index(drop=True)
    g_stint = tr2.groupby(["Race", "Driver", "Year", "Stint"], sort=False)
    tr2["lap_in_stint"] = g_stint.cumcount()  # 0-indexed
    tr2["stint_size"] = g_stint["Stint"].transform("size")
    tr2["lap_in_stint_norm"] = tr2["lap_in_stint"] / (tr2["stint_size"] - 1).clip(lower=1)
    tr2["lap_from_end"] = tr2["stint_size"] - 1 - tr2["lap_in_stint"]

    by_norm = (tr2.groupby(pd.cut(tr2["lap_in_stint_norm"], bins=10))
               .agg(rows=(TARGET, "size"), pos_rate=(TARGET, "mean"))
               .reset_index())
    by_norm["lap_in_stint_norm"] = by_norm["lap_in_stint_norm"].astype(str)
    out["T1_pos_by_stint_norm_decile"] = by_norm.to_dict(orient="records")

    by_lfe = (tr2.groupby(tr2["lap_from_end"].clip(upper=15))
                  .agg(rows=(TARGET, "size"), pos_rate=(TARGET, "mean"))
                  .reset_index().rename(columns={0: "lap_from_end"}))
    out["T1_pos_by_lap_from_stint_end"] = by_lfe.to_dict(orient="records")

    # === T2 — P(target=1 | NTL_bin × Compound) ===========================
    print("[T2] P(target | NTL × Compound)...")
    cml_p99 = tr2.groupby("Compound")["TyreLife"].quantile(0.99).to_dict()
    tr2["ntl"] = (tr2["TyreLife"] / tr2["Compound"].map(cml_p99)
                                                    .clip(lower=1)).clip(0, 1.5)
    tr2["ntl_bin"] = pd.cut(tr2["ntl"], bins=10).astype(str)
    by_ntl = (tr2.groupby(["Compound", "ntl_bin"])
                  .agg(rows=(TARGET, "size"), pos_rate=(TARGET, "mean"))
                  .reset_index())
    out["T2_pos_by_compound_ntl_bin"] = by_ntl.to_dict(orient="records")

    # === T3 — P(target=1 | lap-in-stint × stint-size) ====================
    print("[T3] P(target | lap-in-stint × stint-size)...")
    tr2["lis_clip"] = tr2["lap_in_stint"].clip(upper=20)
    tr2["ss_clip"] = tr2["stint_size"].clip(upper=30)
    by_lis = (tr2.groupby(["ss_clip", "lis_clip"])
                  .agg(rows=(TARGET, "size"), pos_rate=(TARGET, "mean"))
                  .reset_index())
    out["T3_pos_by_lis_x_ss"] = by_lis.to_dict(orient="records")

    # === T4 — windowed-target test =======================================
    print("[T4] Window-target hypothesis (positives per stint vs stint size)...")
    pos_per_stint = g_stint[TARGET].sum().reset_index(name="pos")
    sz_per_stint = g_stint.size().reset_index(name="sz")
    ps = pos_per_stint.merge(sz_per_stint, on=["Race", "Driver", "Year", "Stint"])
    ps_summary = (ps.groupby(ps["sz"].clip(upper=30))
                    .agg(stints=("pos", "size"),
                         pos_mean=("pos", "mean"),
                         pos_max=("pos", "max"),
                         pos_p25=("pos", lambda s: float(s.quantile(0.25))),
                         pos_p50=("pos", lambda s: float(s.quantile(0.50))),
                         pos_p75=("pos", lambda s: float(s.quantile(0.75))))
                    .reset_index())
    out["T4_pos_per_stint_by_size"] = ps_summary.to_dict(orient="records")
    # Frac pos / size by stint-size
    ps["pos_frac"] = ps["pos"] / ps["sz"].clip(lower=1)
    out["T4_pos_frac_overall_mean"] = float(ps["pos_frac"].mean())
    out["T4_pos_frac_by_size"] = (ps.groupby(ps["sz"].clip(upper=30))
                                    ["pos_frac"].mean().reset_index()
                                    .to_dict(orient="records"))

    # === T5 — RaceProgress hypothesis ====================================
    print("[T5] P(target | RaceProgress decile)...")
    tr2["rp_dec"] = pd.cut(tr2["RaceProgress"], bins=10).astype(str)
    by_rp = (tr2.groupby("rp_dec")
                .agg(rows=(TARGET, "size"), pos_rate=(TARGET, "mean"))
                .reset_index())
    out["T5_pos_by_RaceProgress_decile"] = by_rp.to_dict(orient="records")

    # === T6 — multi-positive stint structure =============================
    print("[T6] Multi-positive stint geometry...")
    # For each stint with >=2 positives, compute (a) is last positive at the
    # last lap?  (b) gap between first and last positive  (c) are positives
    # contiguous (gap == n_pos - 1)?
    multi = ps[ps["pos"] >= 2][["Race", "Driver", "Year", "Stint", "pos", "sz"]]
    multi_keys = multi.set_index(["Race", "Driver", "Year", "Stint"]).index
    tr_multi = tr2.set_index(["Race", "Driver", "Year", "Stint"])
    tr_multi = tr_multi.loc[tr_multi.index.isin(multi_keys)].reset_index()
    g_multi = tr_multi.groupby(["Race", "Driver", "Year", "Stint"])
    pos_only = tr_multi[tr_multi[TARGET] == 1]
    pos_g = pos_only.groupby(["Race", "Driver", "Year", "Stint"])
    first_pos = pos_g["lap_in_stint"].min()
    last_pos = pos_g["lap_in_stint"].max()
    n_pos = pos_g.size()
    sz_per_key = g_multi.size()
    # last-pos-is-last-lap fraction
    last_lap = sz_per_key - 1
    out["T6_n_multi_pos_stints"] = int(len(n_pos))
    out["T6_frac_last_pos_at_last_lap"] = float(
        (last_pos == last_lap.reindex(last_pos.index)).mean()
    )
    span = (last_pos - first_pos)
    contig = (span == (n_pos - 1))
    out["T6_frac_contiguous_positives"] = float(contig.mean())
    out["T6_pos_span_p25"] = float(span.quantile(0.25))
    out["T6_pos_span_p50"] = float(span.quantile(0.50))
    out["T6_pos_span_p75"] = float(span.quantile(0.75))
    out["T6_pos_span_max"] = int(span.max())

    # === T7 — within-Race scoring of overlap with PitStop_next ===========
    print("[T7] Lag/lead alignment of PitStop with target...")
    g_seq = tr2.groupby(["Race", "Driver", "Year"], sort=False)
    tr2["PitStop_next"] = g_seq["PitStop"].shift(-1)
    tr2["PitStop_lag1"] = g_seq["PitStop"].shift(1)
    tr2["PitStop_lag2"] = g_seq["PitStop"].shift(2)
    tr2["PitStop_next2"] = g_seq["PitStop"].shift(-2)
    tr2["PitStop_next3"] = g_seq["PitStop"].shift(-3)
    valid = tr2["PitStop_next"].notna()
    out["T7_frac_pos_with_next_PitStop_eq_1"] = float(
        ((tr2.loc[valid, TARGET] == 1) & (tr2.loc[valid, "PitStop_next"] == 1)).sum()
        / max(1, (tr2.loc[valid, TARGET] == 1).sum())
    )
    # multi-row windows
    for k in ["PitStop_lag2", "PitStop_lag1", "PitStop", "PitStop_next",
              "PitStop_next2", "PitStop_next3"]:
        v = tr2[k]
        msk = v.notna()
        if msk.sum() == 0:
            continue
        joint = ((tr2.loc[msk, TARGET] == 1) & (v[msk] == 1)).sum()
        denom_pos = (tr2.loc[msk, TARGET] == 1).sum()
        denom_v1 = (v[msk] == 1).sum()
        out[f"T7_match_{k}"] = {
            "P_pos_given_v1": float(joint / max(1, denom_v1)),
            "P_v1_given_pos": float(joint / max(1, denom_pos)),
        }

    # === T8 — Stint_next change vs target ================================
    print("[T8] Stint_next change vs target...")
    tr2["Stint_next"] = g_seq["Stint"].shift(-1)
    tr2["Compound_next"] = g_seq["Compound"].shift(-1)
    tr2["TyreLife_next"] = g_seq["TyreLife"].shift(-1)
    has_next = tr2["Stint_next"].notna()
    sub = tr2.loc[has_next].copy()
    sub["stint_change"] = (sub["Stint_next"] != sub["Stint"]).astype(int)
    sub["compound_change"] = (sub["Compound_next"] != sub["Compound"]).astype(int)
    sub["tyrelife_reset"] = (sub["TyreLife_next"] < sub["TyreLife"]).astype(int)
    out["T8_n_with_next"] = int(len(sub))
    for col in ["stint_change", "compound_change", "tyrelife_reset"]:
        joint_pos = ((sub[TARGET] == 1) & (sub[col] == 1)).sum()
        denom_pos = (sub[TARGET] == 1).sum()
        denom_v1 = (sub[col] == 1).sum()
        out[f"T8_{col}"] = {
            "P_v1_given_pos": float(joint_pos / max(1, denom_pos)),
            "P_pos_given_v1": float(joint_pos / max(1, denom_v1)),
            "v1_count": int(denom_v1),
            "pos_count": int(denom_pos),
        }
    # union of three
    sub["any_change"] = ((sub["stint_change"] + sub["compound_change"]
                          + sub["tyrelife_reset"]) > 0).astype(int)
    joint = ((sub[TARGET] == 1) & (sub["any_change"] == 1)).sum()
    out["T8_any_change"] = {
        "P_v1_given_pos": float(joint / max(1, (sub[TARGET] == 1).sum())),
        "P_pos_given_v1": float(joint / max(1, (sub["any_change"] == 1).sum())),
    }

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print(f"\nKey numbers:")
    print(f"  pos_rate_overall:                    {out['pos_rate_overall']:.4f}")
    print(f"  T4_pos_frac_overall_mean:            {out['T4_pos_frac_overall_mean']:.4f}")
    print(f"  T6_frac_contiguous_positives:        {out['T6_frac_contiguous_positives']:.4f}")
    print(f"  T6_frac_last_pos_at_last_lap:        {out['T6_frac_last_pos_at_last_lap']:.4f}")
    print(f"  T7_frac_pos_with_next_PitStop_eq_1:  {out['T7_frac_pos_with_next_PitStop_eq_1']:.4f}")
    if "T8_stint_change" in out:
        print(f"  T8 stint_change   P(target|stint_change)  = {out['T8_stint_change']['P_pos_given_v1']:.4f}")
        print(f"  T8 stint_change   P(stint_change|target)  = {out['T8_stint_change']['P_v1_given_pos']:.4f}")
        print(f"  T8 compound_change P(target|comp_change) = {out['T8_compound_change']['P_pos_given_v1']:.4f}")
        print(f"  T8 tyrelife_reset P(target|tl_reset)     = {out['T8_tyrelife_reset']['P_pos_given_v1']:.4f}")
        print(f"  T8 any_change     P(target|any)          = {out['T8_any_change']['P_pos_given_v1']:.4f}")
        print(f"  T8 any_change     P(any|target)          = {out['T8_any_change']['P_v1_given_pos']:.4f}")


if __name__ == "__main__":
    main()
