"""d16 Phase 1 — diagnostic measurement of orig↔synth divergence.

Five sub-probes, single script (all share data loading):
  P1.1  SDV QualityReport (column-shape + pair-trends)
  P1.2  Marginal KS / chi-sq per feature × {orig, synth_train, synth_test}
  P1.3  Pairwise binned-chi-sq grid (rank pairs by corruption)
  P1.4  Class-conditional divergence: KS of (X|y=1) vs (X|y=0) per dist
  P1.5  Per-stratum divergence (Compound, Year)

Outputs:
  scripts/artifacts/d16_phase1_results.json   — full numeric report
  scripts/artifacts/d16_phase1_summary.md     — narrative table
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

warnings.filterwarnings("ignore")
ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)

CAT = ["Driver", "Compound", "Race"]
CONT = ["LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
        "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
        "Position_Change"]
DISC = ["Year", "PitStop"]
TARGET = "PitNextLap"


def bin_quantile(s, k=10):
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()
    qs = np.linspace(0, 1, k + 1)
    edges = np.unique(s.quantile(qs).values)
    if len(edges) < 3:
        return None, None
    return edges, pd.cut(s, edges, include_lowest=True, duplicates="drop")


def chi2_2sample(a, b, k=10):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    edges, _ = bin_quantile(pd.concat([a, b]), k=k)
    if edges is None:
        return None
    ca = pd.cut(a, edges, include_lowest=True, duplicates="drop").value_counts().sort_index()
    cb = pd.cut(b, edges, include_lowest=True, duplicates="drop").value_counts().sort_index()
    df = pd.concat([ca, cb], axis=1).fillna(0).values + 0.5
    chi2, p, _, _ = chi2_contingency(df.T)
    return float(chi2), float(p), int(len(ca))


def cat_chi2(a, b):
    levels = sorted(set(a.dropna().astype(str)) | set(b.dropna().astype(str)))
    ca = a.astype(str).value_counts().reindex(levels, fill_value=0).values
    cb = b.astype(str).value_counts().reindex(levels, fill_value=0).values
    df = np.array([ca, cb]) + 0.5
    chi2, p, _, _ = chi2_contingency(df)
    return float(chi2), float(p), int(len(levels))


def main():
    t0 = time.time()
    log = []

    def step(msg):
        log.append(f"[{time.time() - t0:6.1f}s] {msg}")
        print(log[-1])

    step("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/f1_strategy_dataset_v4.csv")
    orig = orig[orig["Compound"].notna()].copy()
    orig = orig[~orig["Race"].isin(["Pre-Season Test", "Pre-Season Track Session"])].copy()
    step(f"shapes orig={orig.shape} tr={tr.shape} te={te.shape}")

    results: dict = {"meta": {"orig_n": int(len(orig)),
                              "tr_n": int(len(tr)),
                              "te_n": int(len(te))}}

    # ------------------------------------------------------------------
    # P1.2  marginal KS / chi-sq per feature
    # ------------------------------------------------------------------
    step("P1.2 marginal KS / chi-sq")
    p12 = {}
    for c in CONT + DISC:
        if c not in orig.columns:
            continue
        ks_tr = ks_2samp(orig[c].dropna(), tr[c].dropna())
        ks_te = ks_2samp(orig[c].dropna(), te[c].dropna())
        chi_tr = chi2_2sample(orig[c], tr[c], k=20)
        chi_te = chi2_2sample(orig[c], te[c], k=20)
        p12[c] = dict(
            ks_tr_stat=float(ks_tr.statistic), ks_tr_p=float(ks_tr.pvalue),
            ks_te_stat=float(ks_te.statistic), ks_te_p=float(ks_te.pvalue),
            chi2_tr_stat=chi_tr[0] if chi_tr else None,
            chi2_tr_p=chi_tr[1] if chi_tr else None,
            chi2_te_stat=chi_te[0] if chi_te else None,
            chi2_te_p=chi_te[1] if chi_te else None,
        )
    for c in CAT:
        chi_tr = cat_chi2(orig[c], tr[c])
        chi_te = cat_chi2(orig[c], te[c])
        p12[c] = dict(chi2_tr_stat=chi_tr[0], chi2_tr_p=chi_tr[1], n_levels=chi_tr[2],
                      chi2_te_stat=chi_te[0], chi2_te_p=chi_te[1])
    results["P1_2_marginals"] = p12

    # ------------------------------------------------------------------
    # P1.3  pairwise binned-chi-sq grid (orig vs synth_train)
    # ------------------------------------------------------------------
    step("P1.3 pairwise chi-sq grid")
    all_feats = [c for c in CONT + DISC if c in orig.columns]
    pair_results: list = []
    for i, fi in enumerate(all_feats):
        for fj in all_feats[i + 1:]:
            try:
                ai = orig[[fi, fj]].dropna()
                bi = tr[[fi, fj]].dropna().sample(min(50000, len(tr)), random_state=42)
                # bin both
                ei, _ = bin_quantile(pd.concat([ai[fi], bi[fi]]), k=5)
                ej, _ = bin_quantile(pd.concat([ai[fj], bi[fj]]), k=5)
                if ei is None or ej is None:
                    continue
                ai_b = pd.crosstab(
                    pd.cut(ai[fi], ei, include_lowest=True, duplicates="drop"),
                    pd.cut(ai[fj], ej, include_lowest=True, duplicates="drop"),
                )
                bi_b = pd.crosstab(
                    pd.cut(bi[fi], ei, include_lowest=True, duplicates="drop"),
                    pd.cut(bi[fj], ej, include_lowest=True, duplicates="drop"),
                )
                ai_n = ai_b.values.flatten() + 0.5
                bi_n = bi_b.reindex(index=ai_b.index, columns=ai_b.columns, fill_value=0).values.flatten() + 0.5
                chi2, p, _, _ = chi2_contingency(np.array([ai_n, bi_n]))
                pair_results.append(dict(f1=fi, f2=fj, chi2=float(chi2), p=float(p)))
            except Exception as e:
                pair_results.append(dict(f1=fi, f2=fj, error=str(e)))
    pair_results.sort(key=lambda d: -d.get("chi2", 0))
    results["P1_3_pairs_top20_corrupt"] = pair_results[:20]
    results["P1_3_pairs_bot10_clean"] = pair_results[-10:]

    # ------------------------------------------------------------------
    # P1.4  class-conditional divergence
    # ------------------------------------------------------------------
    step("P1.4 class-conditional divergence")
    p14 = {}
    for src_name, df in [("orig", orig), ("tr", tr)]:
        d0 = df[df[TARGET] == 0]
        d1 = df[df[TARGET] == 1]
        sub = {}
        for c in CONT:
            if c not in df.columns:
                continue
            ks = ks_2samp(d0[c].dropna(), d1[c].dropna())
            sub[c] = dict(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                          mean_y0=float(d0[c].mean()), mean_y1=float(d1[c].mean()))
        p14[src_name] = sub
    # similarity of (orig X|y=1) vs (synth_tr X|y=1)
    p14["orig_vs_tr_y1_KS"] = {}
    p14["orig_vs_tr_y0_KS"] = {}
    for c in CONT:
        if c not in orig.columns:
            continue
        a1 = orig[orig[TARGET] == 1][c].dropna()
        b1 = tr[tr[TARGET] == 1][c].dropna()
        a0 = orig[orig[TARGET] == 0][c].dropna()
        b0 = tr[tr[TARGET] == 0][c].dropna()
        p14["orig_vs_tr_y1_KS"][c] = float(ks_2samp(a1, b1).statistic)
        p14["orig_vs_tr_y0_KS"][c] = float(ks_2samp(a0, b0).statistic)
    results["P1_4_class_cond"] = p14

    # ------------------------------------------------------------------
    # P1.5  per-stratum divergence (Compound, Year)
    # ------------------------------------------------------------------
    step("P1.5 per-stratum divergence")
    p15: dict = {"by_compound": {}, "by_year": {}}
    for cmp in sorted(set(orig["Compound"].dropna()) & set(tr["Compound"].dropna())):
        sub = {}
        for c in CONT:
            if c not in orig.columns:
                continue
            a = orig[orig["Compound"] == cmp][c].dropna()
            b = tr[tr["Compound"] == cmp][c].dropna()
            if len(a) > 100 and len(b) > 100:
                sub[c] = float(ks_2samp(a, b).statistic)
        p15["by_compound"][str(cmp)] = sub
    for yr in sorted(set(orig["Year"]) & set(tr["Year"])):
        sub = {}
        for c in CONT:
            if c not in orig.columns:
                continue
            a = orig[orig["Year"] == yr][c].dropna()
            b = tr[tr["Year"] == yr][c].dropna()
            if len(a) > 100 and len(b) > 100:
                sub[c] = float(ks_2samp(a, b).statistic)
        p15["by_year"][str(yr)] = sub
    results["P1_5_per_stratum"] = p15

    # ------------------------------------------------------------------
    # P1.1  SDV QualityReport (last because slow)
    # ------------------------------------------------------------------
    step("P1.1 SDV QualityReport")
    try:
        from sdv.metadata import SingleTableMetadata
        from sdmetrics.reports.single_table import QualityReport

        common_cols = [c for c in tr.columns if c in orig.columns and c not in ["id"]]
        a = orig[common_cols].copy()
        b = tr[common_cols].sample(min(50000, len(tr)), random_state=42).copy()
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(a)
        report = QualityReport()
        report.generate(real_data=a, synthetic_data=b, metadata=meta.to_dict(), verbose=False)
        results["P1_1_sdv"] = dict(
            overall=float(report.get_score()),
            properties={k: float(v) for k, v in report.get_properties().set_index("Property")["Score"].items()},
        )
    except Exception as e:
        step(f"  SDV failed: {e}")
        results["P1_1_sdv"] = {"error": str(e)}

    # ------------------------------------------------------------------
    step("writing results")
    with open(ART / "d16_phase1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # markdown summary
    md = ["# d16 Phase 1 — diagnostic divergence orig↔synth", "",
          f"_runtime {time.time() - t0:.0f}s_", ""]

    md.append("## P1.1 SDV overall scores")
    sdv = results.get("P1_1_sdv", {})
    if "overall" in sdv:
        md.append(f"- overall: **{sdv['overall']:.4f}**")
        for k, v in sdv["properties"].items():
            md.append(f"- {k}: {v:.4f}")
    else:
        md.append(f"- error: {sdv.get('error')}")

    md.append("\n## P1.2 marginal divergence (orig vs synth-train)")
    md.append("| feature | KS-stat (orig vs tr) | KS-stat (orig vs te) | type |")
    md.append("|---|---:|---:|---|")
    for c, v in results["P1_2_marginals"].items():
        if "ks_tr_stat" in v:
            md.append(f"| {c} | {v['ks_tr_stat']:.4f} | {v['ks_te_stat']:.4f} | num |")
        else:
            md.append(f"| {c} | chi2 {v['chi2_tr_stat']:.0f} | chi2 {v['chi2_te_stat']:.0f} | cat ({v.get('n_levels','?')} lv) |")

    md.append("\n## P1.3 top-20 most-corrupted feature pairs (chi-sq)")
    md.append("| pair | chi-sq |")
    md.append("|---|---:|")
    for p in results["P1_3_pairs_top20_corrupt"]:
        if "chi2" in p:
            md.append(f"| {p['f1']} × {p['f2']} | {p['chi2']:.0f} |")

    md.append("\n## P1.4 class-conditional KS (X|y=1 vs X|y=0)")
    md.append("| feature | orig | synth_tr |")
    md.append("|---|---:|---:|")
    for c in CONT:
        o = results["P1_4_class_cond"]["orig"].get(c, {}).get("ks_stat")
        s = results["P1_4_class_cond"]["tr"].get(c, {}).get("ks_stat")
        if o is not None and s is not None:
            md.append(f"| {c} | {o:.4f} | {s:.4f} |")

    md.append("\n## P1.5 per-stratum divergence (compact summary)")
    md.append("Per-Year mean-of-KS over continuous features:")
    md.append("| Year | mean KS |")
    md.append("|---|---:|")
    for yr, sub in results["P1_5_per_stratum"]["by_year"].items():
        if sub:
            md.append(f"| {yr} | {np.mean(list(sub.values())):.4f} |")

    md.append("\nPer-Compound mean-of-KS over continuous features:")
    md.append("| Compound | mean KS |")
    md.append("|---|---:|")
    for cmp, sub in results["P1_5_per_stratum"]["by_compound"].items():
        if sub:
            md.append(f"| {cmp} | {np.mean(list(sub.values())):.4f} |")

    (ART / "d16_phase1_summary.md").write_text("\n".join(md))
    step("DONE")


if __name__ == "__main__":
    main()
