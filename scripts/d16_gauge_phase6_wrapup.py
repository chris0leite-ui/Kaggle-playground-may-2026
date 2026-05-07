"""d16 Phase 6 — wrap-up: synthesize all 5 phases into one audit note.

Reads:
  scripts/artifacts/d16_phase{1,2,3,4,5}_*  + summary.json files
  + cross-base ρ matrix between all new bases and PRIMARY

Writes:
  audit/2026-05-07-overnight-gauge-p-synth.md   — narrative + tables
  scripts/artifacts/d16_summary_xrho.json       — cross-rho matrix
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"


def safe_load(path):
    p = Path(path)
    if not p.exists():
        return None
    a = np.load(p)
    if a.ndim == 2:
        a = a[:, 1]
    return a


def safe_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    t0 = time.time()
    tr = pd.read_csv("data/train.csv")
    y = tr[TARGET].astype(int).values

    # ------------------------------------------------------------------
    # Collect all summaries
    # ------------------------------------------------------------------
    p1 = (ART / "d16_phase1_summary.md").read_text() if (ART / "d16_phase1_summary.md").exists() else "_phase 1 missing_"
    p1_json = safe_json(ART / "d16_phase1_results.json") or {}
    p2 = safe_json(ART / "d16_phase2_summary.json") or {}
    p3 = safe_json(ART / "d16_phase3_summary.json") or {}
    p4 = safe_json(ART / "d16_phase4_summary.json") or {}
    p5 = safe_json(ART / "d16_phase5_summary.json") or {}

    # ------------------------------------------------------------------
    # New OOF/test artifacts: gather AUC + ρ vs PRIMARY
    # ------------------------------------------------------------------
    primary_oof = safe_load(ART / "oof_PRIMARY_K22_strat.npy")
    primary_test = safe_load(ART / "test_PRIMARY_K22_strat.npy")
    primary_auc = float(roc_auc_score(y, primary_oof))

    new_bases = [
        ("d16_dr_rhat",            "P2.2  r̂(x) single-feature LGBM"),
        ("d16_dr_weighted_orig",   "P2.3  r̂-weighted orig + synth-pseudo"),
        ("d16_dr_split",           "P2.4  r̂-median segment-calibrated orig base"),
        ("d16_logp_gmm",           "P3.1  log p_orig single-feat (GMM)"),
        ("d16_logp_bgmm",          "P3.2  log p_orig single-feat (BGMM)"),
        ("d16_orig_no_laptime",    "P4.1  orig minus LapTime"),
        ("d16_orig_no_tyrelife_rp","P4.2  orig minus TyreLife+RP"),
        ("d16_orig_categorical_only","P4.3 orig categorical-only"),
        ("d16_orig_continuous_only","P4.4  orig continuous-only"),
        ("d16_path_b_rhat_q5_tau5000",  "P5.1  Path B r̂_q5 τ=5k"),
        ("d16_path_b_rhat_q5_tau20000", "P5.1  Path B r̂_q5 τ=20k"),
        ("d16_path_b_rhat_q5_tau100000","P5.1  Path B r̂_q5 τ=100k"),
        ("d16_path_b_logp_q5_tau5000",  "P5.2  Path B logp_q5 τ=5k"),
        ("d16_path_b_logp_q5_tau20000", "P5.2  Path B logp_q5 τ=20k"),
        ("d16_path_b_logp_q5_tau100000","P5.2  Path B logp_q5 τ=100k"),
        ("d16_path_b_compXrhat_q5_tau20000","P5.3  Path B Compound×r̂ τ=20k"),
    ]

    base_table = []
    for name, label in new_bases:
        oof = safe_load(ART / f"oof_{name}_strat.npy")
        test = safe_load(ART / f"test_{name}_strat.npy")
        if oof is None or test is None:
            base_table.append(dict(name=name, label=label, status="MISSING"))
            continue
        auc = float(roc_auc_score(y, oof))
        rho = float(np.corrcoef(test, primary_test)[0, 1])
        delta_bp = (auc - primary_auc) * 1e4
        base_table.append(dict(name=name, label=label,
                                std_oof_auc=auc, delta_bp_vs_primary=delta_bp,
                                rho_test=rho, status="OK"))

    # min-meta gates: only run for new single-base candidates that improved
    # (all single-feat probes are tiny weight so K=21+1 absorbs them; quick check via
    # K=2 LR-meta logit-stack of [PRIMARY, candidate])
    gate_table = []
    primary_logit = np.log(np.clip(primary_oof, 1e-6, 1 - 1e-6) /
                           (1 - np.clip(primary_oof, 1e-6, 1 - 1e-6)))
    primary_logit_te = np.log(np.clip(primary_test, 1e-6, 1 - 1e-6) /
                              (1 - np.clip(primary_test, 1e-6, 1 - 1e-6)))
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for entry in base_table:
        if entry["status"] != "OK":
            continue
        name = entry["name"]
        if "path_b" in name:  # Path B is its own meta; skip K=2 gate
            continue
        oof = safe_load(ART / f"oof_{name}_strat.npy")
        test = safe_load(ART / f"test_{name}_strat.npy")
        oof_logit = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6)))
        test_logit = np.log(np.clip(test, 1e-6, 1 - 1e-6) / (1 - np.clip(test, 1e-6, 1 - 1e-6)))
        Xtr = np.column_stack([primary_logit, oof_logit])
        Xte = np.column_stack([primary_logit_te, test_logit])
        oof_meta = np.zeros(len(y))
        for tri, vai in skf.split(Xtr, y):
            lr = LogisticRegression(max_iter=500)
            lr.fit(Xtr[tri], y[tri])
            oof_meta[vai] = lr.predict_proba(Xtr[vai])[:, 1]
        meta_auc = roc_auc_score(y, oof_meta)
        gate_table.append(dict(name=name,
                                k2_meta_auc=float(meta_auc),
                                k2_lift_bp=(meta_auc - primary_auc) * 1e4))

    # ------------------------------------------------------------------
    # Cross-rho matrix among new bases (test-side)
    # ------------------------------------------------------------------
    ok_names = [b["name"] for b in base_table if b["status"] == "OK" and "path_b" not in b["name"]]
    test_arr = []
    for n in ok_names:
        t = safe_load(ART / f"test_{n}_strat.npy")
        test_arr.append(t)
    test_arr.append(primary_test)
    if len(test_arr) >= 2:
        rho_mat = np.corrcoef(np.array(test_arr))
        xrho = {n1: {n2: float(rho_mat[i, j]) for j, n2 in enumerate(ok_names + ["PRIMARY"])}
                 for i, n1 in enumerate(ok_names + ["PRIMARY"])}
    else:
        xrho = {}
    with open(ART / "d16_summary_xrho.json", "w") as f:
        json.dump(xrho, f, indent=2)

    # ------------------------------------------------------------------
    # Write narrative audit
    # ------------------------------------------------------------------
    md = []
    md.append("# Overnight 2026-05-06/07 — gauge p_synth research sweep")
    md.append("")
    md.append("`branch: claude/autoencoder-synthetic-data-pEMB6`")
    md.append("`tag: gauge-p-synth-overnight`")
    md.append("")
    md.append("> Umbrella: translate \"what is the synthesizer's learned p(X,y)\" into prediction signal.")
    md.append("> 5 phases × 19 probes. CPU-only. 0 submits (Rule 1).")
    md.append("")
    md.append("## TL;DR")
    md.append("")
    md.append(f"- PRIMARY OOF AUC reference: **{primary_auc:.5f}**")
    if p2:
        md.append(f"- **AV-AUC orig vs synth** (P2.1): **{p2.get('P21_orig_vs_synth_auc', 'n/a')}**")
        md.append(f"  - top tells: {p2.get('P21_top5_features', [])}")
    if p1_json.get("P1_1_sdv", {}).get("overall"):
        sdv = p1_json["P1_1_sdv"]
        md.append(f"- **SDV overall**: {sdv['overall']:.4f} "
                  f"(column-shape {sdv['properties'].get('Column Shapes', 0):.4f}, "
                  f"pair-trends {sdv['properties'].get('Column Pair Trends', 0):.4f})")
    md.append("")
    # Best base
    ok = [b for b in base_table if b["status"] == "OK"]
    if ok:
        ok.sort(key=lambda b: b.get("delta_bp_vs_primary", -999), reverse=True)
        best = ok[0]
        md.append(f"- **Best new base**: `{best['name']}` ({best['label']}) "
                   f"OOF Δ {best['delta_bp_vs_primary']:+.2f} bp, ρ vs PRIMARY {best['rho_test']:.4f}")
    md.append("")

    # Phase tables
    md.append("## Phase 1 — divergence diagnostics")
    md.append(p1)
    md.append("")

    md.append("## Phase 2 — density ratio r̂(x)")
    md.append(f"- AV-AUC orig vs synth: **{p2.get('P21_orig_vs_synth_auc', 'n/a')}**")
    md.append(f"- top tell features (importance gain): {p2.get('P21_top5_features', [])}")
    md.append(f"- r̂ stats on synth_train: {p2.get('P21_rhat_stats', {})}")
    md.append(f"- P2.2 r̂ single-feat OOF AUC: {p2.get('P22_standalone_auc', 'n/a')}")
    md.append(f"- P2.3 r̂-weighted orig+pseudo OOF AUC: {p2.get('P23_standalone_auc', 'n/a')}")
    md.append(f"- P2.4 r̂-segmented orig OOF AUC: uncal {p2.get('P24_uncal_auc', 'n/a')}, calibrated {p2.get('P24_calibrated_auc', 'n/a')}")
    md.append("")

    md.append("## Phase 3 — log p_orig(x_synth)")
    md.append(f"- GMM(16, full) BIC: {p3.get('P3_1_gmm', {}).get('bic', 'n/a')}")
    md.append(f"- GMM single-feat AUC: {p3.get('P3_1_gmm', {}).get('single_feat_auc', 'n/a')}")
    md.append(f"- BGMM effective components: {p3.get('P3_2_bgmm', {}).get('effective_components', 'n/a')}")
    md.append(f"- BGMM single-feat AUC: {p3.get('P3_2_bgmm', {}).get('single_feat_auc', 'n/a')}")
    md.append(f"- ρ(GMM logp, BGMM logp) on synth_train: {p3.get('P3_gmm_bgmm_logp_correlation', 'n/a')}")
    md.append("")

    md.append("## Phase 4 — orig-transfer feature-subset diversification")
    if p4:
        md.append("| variant | n_feats | orig-held AUC | synth-train AUC |")
        md.append("|---|---:|---:|---:|")
        for k, v in p4.items():
            if not isinstance(v, dict):
                continue
            if "synth_train_auc" in v:
                md.append(f"| {k} | {v['n_features']} | {v['orig_held_auc']:.4f} | {v['synth_train_auc']:.4f} |")
        md.append("")
        md.append("Cross-ρ matrix (test side): see `scripts/artifacts/d16_phase4_summary.json`")
    md.append("")

    md.append("## Phase 5 — Path B on r̂ / log p_orig cohort")
    if p5 and p5.get("results"):
        md.append("| variant | OOF AUC | Δ vs PRIMARY (bp) | ρ test |")
        md.append("|---|---:|---:|---:|")
        for r in p5["results"]:
            md.append(f"| {r['name']} | {r['oof_auc']:.5f} | {r['delta_bp']:+.2f} | {r['rho_test']:.5f} |")
    md.append("")

    md.append("## All new bases — gate table (sorted by Δ vs PRIMARY)")
    md.append("| name | label | std OOF AUC | Δ bp | ρ test |")
    md.append("|---|---|---:|---:|---:|")
    for b in sorted(ok, key=lambda x: -x.get("delta_bp_vs_primary", -999)):
        md.append(f"| {b['name']} | {b['label']} | {b['std_oof_auc']:.5f} | "
                   f"{b['delta_bp_vs_primary']:+.2f} | {b['rho_test']:.5f} |")
    md.append("")

    md.append("## K=2 min-meta gate (PRIMARY + 1 candidate, LR-stack)")
    md.append("| name | K=2 AUC | K=2 lift bp |")
    md.append("|---|---:|---:|")
    for g in sorted(gate_table, key=lambda x: -x["k2_lift_bp"]):
        md.append(f"| {g['name']} | {g['k2_meta_auc']:.5f} | {g['k2_lift_bp']:+.3f} |")
    md.append("")

    md.append("## Synthesis")
    md.append("_to be added by hand based on the tables above._")
    md.append("")

    # Save
    out = Path("audit/2026-05-07-overnight-gauge-p-synth.md")
    out.write_text("\n".join(md))
    print(f"wrote {out}")

    # Also dump consolidated JSON for downstream consumers
    consolidated = dict(
        primary_oof_auc=primary_auc,
        phase1=p1_json,
        phase2=p2,
        phase3=p3,
        phase4=p4,
        phase5=p5,
        new_bases=base_table,
        k2_gate=gate_table,
        runtime_s=time.time() - t0,
    )
    with open(ART / "d16_overnight_consolidated.json", "w") as f:
        json.dump(consolidated, f, indent=2, default=str)


if __name__ == "__main__":
    main()
