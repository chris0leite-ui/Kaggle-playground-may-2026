"""scripts/probe_understand_problem.py — combined diagnostic for the
"understand the problem better" question.

Three probes in one pass:

  Probe A — residual concentration map.
    Where does the current PRIMARY (K=27 + Path-B Compound x Stint, tau=100k,
    OOF 0.95432, LB 0.95368) lose AUC? Bin OOF residuals by (Compound,
    Stint, position-within-stint) and report per-bin AUC vs the global
    AUC. If residuals are concentrated in a small number of cells, a
    targeted feature is plausible. If diffuse, it is not.

  Probe B — OOF -> LB gap decomposition.
    The OOF -> LB gap has been -5 to -6 bp across very different stack
    compositions (Days 16, 17, 17 PM, 18 PM). Is that gap inside the
    sampling band of the public 80/20 split, or outside it? Bootstrap
    the public-LB simulation by drawing 1000 stratified 20%
    sub-samples from OOF and report the AUC distribution.

  Probe C — synth vs original stint coherence.
    AV-AUC = 0.502 at row level says synthetic and original are
    indistinguishable per row. Test the same question at sequence level:
    within (Race, Driver, Stint) groups, does the synthetic data violate
    physical constraints (Compound-constant, TyreLife monotone non-
    decreasing, LapNumber strictly increasing) at a different rate than
    the original?

Cost: <10 min CPU on the local sandbox.

Outputs:
  scripts/artifacts/probe_understand_problem.json  (machine-readable)
  + console summary printed at the end.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
OUT = ART / "probe_understand_problem.json"

PRIMARY_OOF = ART / "oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy"
TARGET = "PitNextLap"
SEED = 42
N_BOOTSTRAP = 1000
PUBLIC_FRAC = 0.20

POSITION_BUCKETS = [
    ("first", lambda lap_idx, n: lap_idx == 0),
    ("early", lambda lap_idx, n: 0 < lap_idx < max(1, int(0.5 * n))),
    ("mid", lambda lap_idx, n: int(0.5 * n) <= lap_idx < n - 1),
    ("last", lambda lap_idx, n: lap_idx == n - 1),
]


def load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    print("Loading train + PRIMARY OOF ...")
    tr = pd.read_csv("data/train.csv")
    oof = np.load(PRIMARY_OOF)
    if oof.ndim == 2:
        # Column 1 is the predicted positive probability (col 0 = neg class).
        oof = oof[:, 1]
    assert len(oof) == len(tr), f"OOF/train shape mismatch: {oof.shape} vs {tr.shape}"
    y = tr[TARGET].astype(int).values
    return tr, oof, y


def add_position_in_stint(tr: pd.DataFrame) -> pd.DataFrame:
    """Lap-position within (Race, Driver, Stint) ordered by LapNumber."""
    tr = tr.copy()
    tr = tr.sort_values(["Race", "Driver", "Stint", "LapNumber"], kind="stable")
    g = tr.groupby(["Race", "Driver", "Stint"], sort=False)
    tr["lap_idx_in_stint"] = g.cumcount()
    tr["stint_size"] = g["LapNumber"].transform("size")
    return tr.sort_index()


def position_bucket(lap_idx: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Vectorised assignment to one of the four POSITION_BUCKETS."""
    out = np.empty(len(lap_idx), dtype=object)
    out[:] = "mid"
    out[lap_idx == 0] = "first"
    out[(lap_idx > 0) & (lap_idx < (0.5 * n).astype(int).clip(min=1))] = "early"
    out[lap_idx == (n - 1)] = "last"
    return out


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) < 100 or len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def probe_a_residual_map(tr: pd.DataFrame, oof: np.ndarray,
                         y: np.ndarray) -> dict:
    """Per-bin AUC by (Compound, Stint-bin, lap-in-stint bucket)."""
    print("\n=== Probe A: residual concentration ===")
    tr = add_position_in_stint(tr)
    bucket = position_bucket(
        tr["lap_idx_in_stint"].values, tr["stint_size"].values
    )
    stint_bin = np.where(
        tr["Stint"].values <= 1, "S1",
        np.where(tr["Stint"].values == 2, "S2",
                 np.where(tr["Stint"].values == 3, "S3", "S4+")),
    )
    global_auc = roc_auc_score(y, oof)
    print(f"  global OOF AUC = {global_auc:.5f}")

    rows = []
    for compound in sorted(tr["Compound"].unique()):
        m_c = (tr["Compound"].values == compound)
        for sb in ["S1", "S2", "S3", "S4+"]:
            m_sb = m_c & (stint_bin == sb)
            for pb in ["first", "early", "mid", "last"]:
                m = m_sb & (bucket == pb)
                n = int(m.sum())
                pos = int(y[m].sum())
                auc = safe_auc(y[m], oof[m])
                rows.append({
                    "compound": compound, "stint_bin": sb, "position": pb,
                    "n": n, "pos": pos,
                    "pos_rate": pos / max(n, 1),
                    "auc": auc,
                    "delta_vs_global_bp": (auc - global_auc) * 1e4 if auc else None,
                })
    rows.sort(key=lambda r: (r["delta_vs_global_bp"] is None,
                             r["delta_vs_global_bp"] or 0))
    worst = [r for r in rows if r["auc"] is not None][:8]
    best = [r for r in rows if r["auc"] is not None][-5:]
    print("  worst 8 cells (auc - global, bp):")
    for r in worst:
        print(f"    {r['compound']:>12s} {r['stint_bin']:>3s} {r['position']:>5s}"
              f"  n={r['n']:>6d}  pos={r['pos']:>5d} ({r['pos_rate']*100:5.2f}%)"
              f"  auc={r['auc']:.4f}  d={r['delta_vs_global_bp']:+.1f} bp")

    # marginal collapse: AUC by Compound, Stint, position separately
    marg = {}
    for col_name, vals in [("compound", tr["Compound"].values),
                           ("stint_bin", stint_bin),
                           ("position", bucket)]:
        marg[col_name] = {}
        for v in sorted(set(vals)):
            m = (vals == v)
            marg[col_name][str(v)] = {
                "n": int(m.sum()),
                "auc": safe_auc(y[m], oof[m]),
            }

    return {
        "global_auc": global_auc,
        "cells": rows,
        "worst_cells": worst,
        "best_cells": best,
        "marginals": marg,
    }


def probe_b_oof_lb_gap(oof: np.ndarray, y: np.ndarray) -> dict:
    """Bootstrap public-LB simulation via repeated 20% draws."""
    print("\n=== Probe B: OOF -> LB gap bootstrap ===")
    rng = np.random.default_rng(SEED)
    n = len(y)
    full_auc = roc_auc_score(y, oof)
    sims = np.empty(N_BOOTSTRAP)
    pub_n = int(round(PUBLIC_FRAC * n))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pub_pos = int(round(PUBLIC_FRAC * len(pos_idx)))
    pub_neg = pub_n - pub_pos
    for i in range(N_BOOTSTRAP):
        s_pos = rng.choice(pos_idx, pub_pos, replace=False)
        s_neg = rng.choice(neg_idx, pub_neg, replace=False)
        s = np.concatenate([s_pos, s_neg])
        sims[i] = roc_auc_score(y[s], oof[s])
    print(f"  full OOF AUC                = {full_auc:.5f}")
    print(f"  bootstrapped 20% mean       = {sims.mean():.5f}")
    print(f"  bootstrapped 20% std        = {sims.std():.5f}")
    print(f"  bootstrapped 20% [2.5, 97.5] = "
          f"[{np.percentile(sims, 2.5):.5f}, {np.percentile(sims, 97.5):.5f}]")
    print(f"  observed PRIMARY LB         = 0.95368  (delta vs OOF "
          f"= {(0.95368 - full_auc) * 1e4:+.1f} bp)")
    return {
        "full_auc": full_auc,
        "boot_mean": float(sims.mean()),
        "boot_std": float(sims.std()),
        "boot_p2_5": float(np.percentile(sims, 2.5)),
        "boot_p97_5": float(np.percentile(sims, 97.5)),
        "observed_lb": 0.95368,
        "observed_gap_bp": (0.95368 - full_auc) * 1e4,
    }


def probe_c_stint_coherence() -> dict:
    """Compare stint-level constraint violation rates: synth vs orig."""
    print("\n=== Probe C: synth vs orig stint coherence ===")
    synth = pd.read_csv("data/train.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    print(f"  synth rows = {len(synth):,}    orig rows = {len(orig):,}")

    def metrics(df: pd.DataFrame, label: str) -> dict:
        df = df.copy()
        keys = [k for k in ["Year", "Race", "Driver", "Stint"] if k in df.columns]
        if "Stint" not in df.columns or "LapNumber" not in df.columns:
            return {"label": label, "error": "missing Stint/LapNumber"}
        df = df.sort_values(keys + ["LapNumber"], kind="stable").reset_index(drop=True)
        g = df.groupby(keys, sort=False)
        n_groups = g.ngroups
        # Compound constancy (should be 1.0)
        compound_const = (
            g["Compound"].nunique().eq(1).mean() if "Compound" in df.columns else None
        )
        # Within-stint diffs
        if "TyreLife" in df.columns:
            tl_d = g["TyreLife"].diff()
            tl_violations_per_group = (tl_d < 0).groupby(g.ngroup()).any()
            tl_mono = float(1.0 - tl_violations_per_group.mean())
        else:
            tl_mono = None
        ln_d = g["LapNumber"].diff()
        ln_violations_per_group = (ln_d <= 0).groupby(g.ngroup()).any()
        ln_mono = float(1.0 - ln_violations_per_group.mean())
        lap_gaps = ln_d.dropna()
        gap_eq_1 = float((lap_gaps == 1).mean()) if len(lap_gaps) else None
        gap_mean = float(lap_gaps.mean()) if len(lap_gaps) else None
        sl = g.size()
        return {
            "label": label,
            "n_rows": int(len(df)),
            "n_groups": int(n_groups),
            "compound_const_frac": float(compound_const) if compound_const is not None else None,
            "tyrelife_monotone_frac": tl_mono,
            "lapnumber_monotone_frac": ln_mono,
            "lap_gap_eq_1_frac": gap_eq_1,
            "lap_gap_mean": gap_mean,
            "stint_len_mean": float(sl.mean()),
            "stint_len_std": float(sl.std()),
            "stint_len_q": [int(sl.quantile(q)) for q in [0.05, 0.5, 0.95]],
        }

    out = {
        "synth": metrics(synth, "synth"),
        "orig": metrics(orig, "orig"),
    }
    for k in ["synth", "orig"]:
        m = out[k]
        if "error" in m:
            print(f"  {k}: ERROR {m['error']}")
            continue
        print(f"  {k:>5s}: groups={m['n_groups']:>7d}  "
              f"compound-const={m.get('compound_const_frac'):.4f}  "
              f"tyre-mono={m.get('tyrelife_monotone_frac'):.4f}  "
              f"lap-mono={m.get('lapnumber_monotone_frac'):.4f}  "
              f"gap=1 frac={m.get('lap_gap_eq_1_frac'):.4f}  "
              f"stint-len mean={m.get('stint_len_mean'):.2f}")
    return out


def main() -> None:
    t0 = time.time()
    tr, oof, y = load()
    a = probe_a_residual_map(tr, oof, y)
    b = probe_b_oof_lb_gap(oof, y)
    c = probe_c_stint_coherence()
    out = {"probe_a": a, "probe_b": b, "probe_c": c,
           "wall_s": time.time() - t0}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}.  Wall {out['wall_s']:.1f}s.")


if __name__ == "__main__":
    main()
