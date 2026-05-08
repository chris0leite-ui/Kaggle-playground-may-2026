"""Phase F — Leakage asymmetry per base × per feature.

Quantifies the Strat-OOF vs GroupKF-OOF gap *per base* using the OOFs already
on disk, and decomposes the gap into within-Race vs cross-Race components.
Also probes per-Year and per-Stint calibration for the strongest meta to
identify post-hoc isotonic headroom.

Output: plots/eda_deep/F_leakage/*.png + F_summary.md
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

OUT = Path("plots/eda_deep/F_leakage")
OUT.mkdir(parents=True, exist_ok=True)
ART = Path("scripts/artifacts")


def load_oof(name: str, kind: str) -> np.ndarray | None:
    p = ART / f"oof_{name}_{kind}.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr[:, 1]
    return arr.astype(np.float32)


def main() -> None:
    train = pd.read_csv("data/train.csv")
    y = train["PitNextLap"].astype(int).to_numpy()
    race = train["Race"].to_numpy()

    findings: list[str] = ["# Phase F — Leakage asymmetry\n"]

    # ---- per-base ΔAUC table ----
    bases = [
        ("baseline_two_anchor", "GBDT"),
        ("e3_hgbc", "GBDT"),
        ("e5_optuna_lgbm", "GBDT"),
        ("m2_xgb", "GBDT"),
        ("cb_lossguide", "CatBoost"),
        ("cb_year-cat", "CatBoost"),
        ("cb_slow-wide-bag", "CatBoost"),
        ("a_horizon", "GBDT-formulation"),
        ("b_lapsuntilpit", "GBDT-formulation"),
        ("d2a_te", "TargetEnc"),
        ("d3a_te_unified", "TargetEnc"),
        ("d3b_seqfe", "SeqFE-GBDT"),
        ("d6_rule_residual", "RuleResid"),
        ("d6_rule_driver_compound", "RuleResid"),
        ("d6_rule_year_race", "RuleResid"),
        ("d9c_fm", "FM"),
        ("d9f_FM_A", "FM"),
        ("d9f_FM_B", "FM"),
        ("d9b_R14_L4", "SparseLR"),
        ("e1_catboost_sub", "CatBoost"),
        ("f1_hgbc_deep", "GBDT"),
        ("f2_hgbc_shallow", "GBDT"),
    ]

    rows = []
    for name, fam in bases:
        s = load_oof(name, "strat")
        g = load_oof(name, "groupkf")
        if s is None or g is None:
            continue
        auc_s = roc_auc_score(y, s)
        auc_g = roc_auc_score(y, g)
        # within-race AUC: average across races (only those with both classes)
        wr = []
        for r in pd.unique(race):
            m = race == r
            if y[m].sum() == 0 or y[m].sum() == m.sum():
                continue
            wr.append(roc_auc_score(y[m], s[m]))
        auc_wr = float(np.mean(wr)) if wr else np.nan
        rows.append({
            "base": name, "family": fam,
            "auc_strat": auc_s,
            "auc_groupkf": auc_g,
            "auc_within_race_strat": auc_wr,
            "delta_strat_minus_groupkf_bp": (auc_s - auc_g) * 1e4,
            "delta_strat_minus_withinrace_bp": (auc_s - auc_wr) * 1e4,
        })
    df = pd.DataFrame(rows).sort_values("delta_strat_minus_groupkf_bp", ascending=False)
    findings.append("\n## Per-base ΔAUC table (Strat − GroupKF, in bp)\n")
    findings.append("```\n" + df.round(4).to_string(index=False) + "\n```\n")
    df.to_csv(OUT / "per_base_delta_auc.csv", index=False)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"FM": "#2ca02c", "GBDT": "#d62728", "CatBoost": "#ff7f0e",
              "GBDT-formulation": "#9467bd", "TargetEnc": "#8c564b",
              "SeqFE-GBDT": "#e377c2", "RuleResid": "#1f77b4",
              "SparseLR": "#17becf"}
    for fam in df["family"].unique():
        sub = df[df["family"] == fam]
        ax.barh(sub["base"], sub["delta_strat_minus_groupkf_bp"],
                color=colors.get(fam, "gray"), label=fam)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Strat AUC − GroupKF AUC (bp)")
    ax.set_title("Per-base leakage gap (large bar = leakage-eater; near-zero = leakage-robust)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "per_base_delta_auc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- family aggregate ----
    fam_agg = (df.groupby("family")["delta_strat_minus_groupkf_bp"]
                  .agg(["count", "mean", "min", "max"])
                  .round(2)
                  .sort_values("mean", ascending=False))
    findings.append("\n## Family aggregate ΔAUC\n")
    findings.append("```\n" + fam_agg.to_string() + "\n```\n")

    # ---- calibration: PRIMARY meta vs cohort ----
    # Use d9h FM as part of best meta; load m5h primary meta
    m5q = load_oof("m5q", "strat")
    if m5q is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, cohort in [(axes[0], "Year"), (axes[1], "Stint"), (axes[2], "Compound")]:
            for v in sorted(train[cohort].unique()):
                m = train[cohort].to_numpy() == v
                if m.sum() < 1000:
                    continue
                # bin by predicted prob
                p = m5q[m]
                ybin = y[m]
                bins = np.quantile(p, np.linspace(0, 1, 11))
                bins = np.unique(bins)
                if len(bins) < 3:
                    continue
                cuts = np.digitize(p, bins[1:-1])
                centers = []
                rates = []
                for k in range(len(bins) - 1):
                    sel = cuts == k
                    if sel.sum() < 30:
                        continue
                    centers.append(p[sel].mean())
                    rates.append(ybin[sel].mean())
                ax.plot(centers, rates, marker="o", label=str(v), alpha=0.7)
            ax.plot([0, 1], [0, 1], color="black", lw=0.5, ls="--")
            ax.set_xlabel("predicted")
            ax.set_ylabel("observed")
            ax.set_title(f"M5q calibration by {cohort}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "calibration_per_cohort.png", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)

        # isotonic per cohort: how much does it lift?
        iso_results = []
        for cohort in ["Year", "Stint", "Compound"]:
            base_auc = roc_auc_score(y, m5q)
            adj = m5q.copy().astype(float)
            for v in sorted(train[cohort].unique()):
                m = train[cohort].to_numpy() == v
                if m.sum() < 500:
                    continue
                ir = IsotonicRegression(out_of_bounds="clip").fit(m5q[m], y[m])
                adj[m] = ir.predict(m5q[m])
            new_auc = roc_auc_score(y, adj)
            iso_results.append((cohort, base_auc, new_auc,
                                (new_auc - base_auc) * 1e4))
        findings.append("\n## Per-cohort isotonic calibration headroom (in-sample upper bound)\n")
        findings.append("| Cohort | Base AUC | Calibrated AUC | Δ bp (in-sample) |")
        findings.append("|---|---:|---:|---:|")
        for c, ba, na, d in iso_results:
            findings.append(f"| {c} | {ba:.5f} | {na:.5f} | {d:+.2f} |")
        findings.append("\n_(In-sample upper bound; real OOF lift ~half. Useful for "
                        "ranking which cohort split has most slack.)_\n")

    # ---- Per-feature leakage probe via simple LR-by-fold ----
    # Train a 1-feature LR using StratifiedKFold OOF and GroupKFold OOF; compare AUC.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    feats = ["TyreLife", "RaceProgress", "Stint", "LapNumber",
             "Cumulative_Degradation", "Position", "LapTime_Delta", "Year"]
    feat_rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    gkf = GroupKFold(5)
    for f in feats:
        x = train[f].fillna(-999).to_numpy().reshape(-1, 1)
        # Strat
        oof_s = np.zeros(len(y))
        for tr, va in skf.split(x, y):
            lr = LogisticRegression(max_iter=200).fit(x[tr], y[tr])
            oof_s[va] = lr.predict_proba(x[va])[:, 1]
        auc_s = roc_auc_score(y, oof_s)
        # GroupKF
        oof_g = np.zeros(len(y))
        for tr, va in gkf.split(x, y, groups=race):
            lr = LogisticRegression(max_iter=200).fit(x[tr], y[tr])
            oof_g[va] = lr.predict_proba(x[va])[:, 1]
        auc_g = roc_auc_score(y, oof_g)
        feat_rows.append((f, auc_s, auc_g, (auc_s - auc_g) * 1e4))
    feat_df = pd.DataFrame(feat_rows, columns=["feature", "auc_strat",
                                                "auc_groupkf", "delta_bp"])
    feat_df = feat_df.sort_values("delta_bp", ascending=False)
    findings.append("\n## Per-feature leakage AUC (single-feature LR)\n")
    findings.append("```\n" + feat_df.round(4).to_string(index=False) + "\n```\n")
    findings.append(
        "Features with high single-feature AUC and small Δbp are the leakage-robust "
        "signal — they generalize across Race. Features with Δbp > 50 carry "
        "Race-specific information and don't cleanly transfer to held-out Races.\n"
    )

    # ---- TL;DR ----
    summary = ["\n## TL;DR\n",
               f"- {len(df)} bases analyzed; FM family mean ΔAUC "
               f"{fam_agg.loc['FM','mean']:.0f}bp vs GBDT "
               f"{fam_agg.loc['GBDT','mean'] if 'GBDT' in fam_agg.index else float('nan'):.0f}bp",
               "- Calibration analysis on M5q shows per-cohort isotonic delta in `F_summary.md`",
               "- Single-feature LR table reveals which raw features carry leakage-eaten signal",
               ]
    findings = summary + findings

    md = "\n".join(findings) + "\n"
    Path("plots/eda_deep/F_summary.md").write_text(md)
    print(md[:3500])
    print("...")
    print(f"saved: {OUT}/")


if __name__ == "__main__":
    main()
