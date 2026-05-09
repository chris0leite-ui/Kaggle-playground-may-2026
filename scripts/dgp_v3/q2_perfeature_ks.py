"""Phase A1 probe — per-feature KS dissection (P17 from v1 plan).

For each numeric column in the schema, measure the Kolmogorov-Smirnov
distance between:
  - synth (host) and orig (aadigupta1601)
  - synth and orig sliced by year (to test 2023 anomaly)
  - synth and orig sliced by (Compound, Year)

Also measure literal-overlap (fraction of synth values in orig empirical
set), per column and per cell.

This is the cheap, no-GPU, evidence-rich probe that should run in <2 min.
It tells us:
  - Which columns the host literally copies vs synthesises.
  - Whether 2023's anomaly (F4) shows up as KS divergence on a specific
    sub-slice of orig.
  - Which (Compound, Year) cells the host fits well vs poorly.

Output: scripts/artifacts/dgp_v3_q2_ks.json + a printed summary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path("/home/user/Kaggle-playground-may-2026")
DATA = ROOT / "data"
EXT = ROOT / "external/aadigupta"
ART = ROOT / "scripts/artifacts"


def t(label: str, ts: float) -> None:
    print(f"  [{time.time()-ts:6.1f}s] {label}", flush=True)


def load_orig() -> pd.DataFrame:
    df = pd.read_csv(EXT / "f1_strategy_dataset_v4.csv")
    df = df.rename(columns={"LapTime (s)": "LapTime"})
    return df


def load_synth() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    cols = [c for c in train.columns if c != "PitNextLap"]
    df = pd.concat([train[cols], test[cols]], ignore_index=True)
    if "LapTime (s)" in df.columns:
        df = df.rename(columns={"LapTime (s)": "LapTime"})
    return df


def ks_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    s, p = ks_2samp(a, b)
    return float(s), float(p)


def literal_overlap(synth_vals: pd.Series, orig_vals: pd.Series) -> float:
    oset = set(orig_vals.dropna().unique().tolist())
    if not oset:
        return float("nan")
    return float(synth_vals.isin(oset).mean())


def main() -> None:
    out: dict = {}
    ts = time.time()

    orig = load_orig()
    synth = load_synth()
    t(f"loaded orig {orig.shape} synth {synth.shape}", ts)

    cont_cols = [
        "LapTime",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "RaceProgress",
        "Position_Change",
        "TyreLife",
    ]
    int_cols = ["LapNumber", "Stint", "Position", "Year", "PitStop"]
    all_cols = cont_cols + int_cols

    # 1. Global per-column KS + literal overlap
    out["global"] = {}
    for c in all_cols:
        if c not in synth.columns or c not in orig.columns:
            continue
        s = synth[c].dropna()
        o = orig[c].dropna()
        ks, p = ks_pair(s.values, o.values)
        ov = literal_overlap(s, o)
        out["global"][c] = {
            "ks": ks,
            "ks_p": p,
            "n_synth": int(len(s)),
            "n_orig": int(len(o)),
            "literal_overlap_frac": ov,
            "synth_unique": int(s.nunique()),
            "orig_unique": int(o.nunique()),
        }
    t("global KS done", ts)

    # 2. Per-year KS + literal overlap (the F4 2023 anomaly probe)
    out["per_year"] = {}
    for y in sorted(synth["Year"].unique().tolist()):
        s_y = synth[synth["Year"] == y]
        o_y = orig[orig["Year"] == y]
        out["per_year"][int(y)] = {
            "n_synth": int(len(s_y)),
            "n_orig": int(len(o_y)),
            "synth_pit_rate": (
                float(s_y["PitStop"].mean()) if "PitStop" in s_y.columns else None
            ),
            "orig_pit_rate": (
                float(o_y["PitStop"].mean()) if "PitStop" in o_y.columns else None
            ),
            "orig_pitnext_rate": (
                float(o_y["PitNextLap"].mean()) if "PitNextLap" in o_y.columns else None
            ),
            "ks": {},
        }
        for c in cont_cols:
            if c in s_y.columns and c in o_y.columns:
                ks, _ = ks_pair(s_y[c].dropna().values, o_y[c].dropna().values)
                ov = literal_overlap(s_y[c].dropna(), o_y[c].dropna())
                out["per_year"][int(y)]["ks"][c] = {"ks": ks, "literal": ov}
    t("per-year KS done", ts)

    # 3. Per-(Compound, Year) cell KS — only LapTime + Cumulative_Degradation,
    #    the two with most contrast in the global KS.
    out["per_cell"] = {}
    for c in ["LapTime", "Cumulative_Degradation"]:
        out["per_cell"][c] = {}
        for cmp_ in synth["Compound"].unique():
            for y in sorted(synth["Year"].unique().tolist()):
                s_cy = synth[(synth["Compound"] == cmp_) & (synth["Year"] == y)]
                o_cy = orig[(orig["Compound"] == cmp_) & (orig["Year"] == y)]
                if len(s_cy) < 50 or len(o_cy) < 50:
                    continue
                ks, _ = ks_pair(s_cy[c].dropna().values, o_cy[c].dropna().values)
                ov = literal_overlap(s_cy[c].dropna(), o_cy[c].dropna())
                key = f"{cmp_}_{int(y)}"
                out["per_cell"][c][key] = {
                    "n_s": int(len(s_cy)),
                    "n_o": int(len(o_cy)),
                    "ks": ks,
                    "literal": ov,
                }
    t("per-cell KS done", ts)

    # 4. PitStop-conditional KS (d18 f5 hypothesis)
    out["pitstop_cond"] = {}
    for c in cont_cols:
        if c not in synth.columns:
            continue
        ks_y0, _ = ks_pair(
            synth[synth["PitStop"] == 0][c].dropna().values,
            orig[orig["PitStop"] == 0][c].dropna().values,
        )
        ks_y1, _ = ks_pair(
            synth[synth["PitStop"] == 1][c].dropna().values,
            orig[orig["PitStop"] == 1][c].dropna().values,
        )
        out["pitstop_cond"][c] = {
            "ks_pitstop0": ks_y0,
            "ks_pitstop1": ks_y1,
            "asymmetry": ks_y1 - ks_y0,
        }
    t("PitStop-conditional KS done", ts)

    # 5. Driver-set comparison
    s_drivers = set(synth["Driver"].unique().tolist())
    o_drivers = set(orig["Driver"].unique().tolist())
    out["driver_vocab"] = {
        "n_synth_drivers": len(s_drivers),
        "n_orig_drivers": len(o_drivers),
        "common": len(s_drivers & o_drivers),
        "synth_only": len(s_drivers - o_drivers),
        "orig_only": len(o_drivers - s_drivers),
        "examples_synth_only": sorted(list(s_drivers - o_drivers))[:20],
        "examples_orig_only": sorted(list(o_drivers - s_drivers))[:20],
    }
    t("driver vocab done", ts)

    # Save
    fp = ART / "dgp_v3_q2_ks.json"
    fp.write_text(json.dumps(out, indent=2))
    t(f"wrote {fp.name}", ts)

    # Print summary
    print("\n=== Global per-column KS (synth vs orig) ===")
    rows = sorted(out["global"].items(), key=lambda kv: -kv[1]["ks"])
    for c, v in rows:
        print(
            f"  {c:30s} ks={v['ks']:.4f} literal_overlap={v['literal_overlap_frac']:.4f} "
            f"unique synth/orig={v['synth_unique']}/{v['orig_unique']}"
        )

    print("\n=== Per-year (n, pit rates, KS LapTime / CumDeg) ===")
    for y, v in sorted(out["per_year"].items()):
        ks_lt = v["ks"].get("LapTime", {}).get("ks", float("nan"))
        ks_cd = v["ks"].get("Cumulative_Degradation", {}).get("ks", float("nan"))
        lit_lt = v["ks"].get("LapTime", {}).get("literal", float("nan"))
        print(
            f"  Y={y} n_s={v['n_synth']:6} n_o={v['n_orig']:6} "
            f"synth_pit={v['synth_pit_rate']:.4f} orig_pit={v['orig_pit_rate']:.4f} "
            f"orig_pitnext={v['orig_pitnext_rate']:.4f} "
            f"ks(LT)={ks_lt:.4f} literal(LT)={lit_lt:.4f} ks(CD)={ks_cd:.4f}"
        )

    print("\n=== PitStop-conditional KS (d18 f5: large asymmetry = PitStop in cond) ===")
    for c, v in out["pitstop_cond"].items():
        print(
            f"  {c:30s} ks(PS=0)={v['ks_pitstop0']:.4f} "
            f"ks(PS=1)={v['ks_pitstop1']:.4f} asymm={v['asymmetry']:+.4f}"
        )

    print("\n=== Driver vocab ===")
    dv = out["driver_vocab"]
    print(
        f"  synth={dv['n_synth_drivers']} orig={dv['n_orig_drivers']} "
        f"common={dv['common']} synth_only={dv['synth_only']} orig_only={dv['orig_only']}"
    )
    print(f"  examples_synth_only: {dv['examples_synth_only'][:10]}")
    print(f"  examples_orig_only:  {dv['examples_orig_only'][:10]}")


if __name__ == "__main__":
    main()
