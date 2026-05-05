"""Pre-baseline gate audit — items 2 (schema), 3 (target-rate),
4 (group keys), plus the structural PitStop-vs-PitNextLap check.

One-shot. Re-runnable; writes audit/<date>-pre-baseline-gate.md.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

TARGET = "PitNextLap"
ID_COL = "id"


def fmt_top3(s: pd.Series) -> dict:
    vc = s.value_counts(dropna=False).head(3).to_dict()
    return {str(k): int(v) for k, v in vc.items()}


def schema_table(df: pd.DataFrame, label: str) -> list[str]:
    out = [f"### {label}", f"- shape: {df.shape}", "",
           "| col | dtype | n_unique | n_null | top-3 |",
           "|---|---|---|---|---|"]
    for c in df.columns:
        out.append(
            f"| {c} | {df[c].dtype} | {df[c].nunique()} | "
            f"{df[c].isna().sum()} | {fmt_top3(df[c])} |"
        )
    out.append("")
    return out


def main() -> None:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    lines = [
        f"# Pre-baseline gate audit — {dt.date.today().isoformat()}",
        "",
        "Items 2 (schema) / 3 (target-rate) / 4 (group keys) of the "
        "pre-baseline understanding gate. Companion to "
        "`brief.md`, `comp-context.md` (prior_art / domain_notes / "
        "metric_notes), and the agent summaries.",
        "",
        "## Item 2 — full schema",
        "",
    ]
    lines += schema_table(train, "train.csv")
    lines += schema_table(test, "test.csv")
    train_cols, test_cols = set(train.columns), set(test.columns)
    lines += [
        "### train ↔ test column diff",
        f"- in train, not in test: {sorted(train_cols - test_cols)}",
        f"- in test, not in train: {sorted(test_cols - train_cols)}",
        "",
    ]

    # CRITICAL: PitStop vs PitNextLap structural check
    lines += [
        "## CRITICAL — `PitStop` vs `PitNextLap` structural check",
        "",
        "Both `PitStop` and `PitNextLap` exist in train; `PitStop` "
        "also in test. Hypothesis: `PitNextLap_N == PitStop_{N+1}` "
        "within a (Race, Driver) sequence (1-step-ahead lag).",
        "",
    ]
    needed = ["Race", "Driver", "LapNumber", "PitStop", "PitNextLap"]
    if all(c in train.columns for c in needed):
        s = train.sort_values(["Race", "Driver", "LapNumber"]).copy()
        s["PitStop_next"] = s.groupby(["Race", "Driver"])["PitStop"].shift(-1)
        valid = s.dropna(subset=["PitStop_next"])
        match_rate = (valid["PitNextLap"] == valid["PitStop_next"]).mean()
        lines += [
            f"- valid rows (non-last-lap): {len(valid):,} / {len(s):,}",
            f"- match rate `PitNextLap_N == PitStop_{{N+1}}`: "
            f"**{match_rate:.6f}**",
        ]
        if match_rate > 0.999:
            lines.append(
                "- **CONFIRMED** structural identity. Future-lap "
                "`PitStop` fully determines current-lap target."
            )
        elif match_rate > 0.95:
            lines.append("- **STRONG** structural relationship; minor noise.")
        else:
            lines.append("- weak relationship; not a simple lag.")
        lines.append("")

        # Same check on test (ordering only — no target available)
        if all(c in test.columns for c in ["Race", "Driver", "LapNumber", "PitStop"]):
            t = test.sort_values(["Race", "Driver", "LapNumber"]).copy()
            t["PitStop_next"] = t.groupby(["Race", "Driver"])["PitStop"].shift(-1)
            test_has_next = t["PitStop_next"].notna().sum()
            lines += [
                "### test — leakage scan",
                f"- test rows where `PitStop_{{N+1}}` exists in test "
                f"under same (Race, Driver): {test_has_next:,} / {len(t):,}",
                "  (if non-zero, that fraction of the target is "
                "deterministically recoverable from test alone — "
                "competitors can solve it with a join.)",
                "",
            ]

    # Group-key candidates
    lines += ["## Item 4 — GroupKFold candidate keys", ""]
    for gc in ["Race", "Driver", "Year", "Compound"]:
        if gc not in train.columns:
            continue
        n_train = train[gc].nunique()
        avg = len(train) / n_train
        line = (f"- **{gc}**: train={n_train} groups (avg {avg:.0f} "
                f"rows/group)")
        if gc in test.columns:
            n_test = test[gc].nunique()
            overlap = len(set(train[gc].unique()) & set(test[gc].unique()))
            line += f"; test={n_test}; overlap={overlap}"
        lines.append(line)

    # Combined keys
    for cols in (["Race", "Driver"], ["Race", "Driver", "Stint"]):
        if not all(c in train.columns for c in cols):
            continue
        k_tr = train[cols].astype(str).agg("|".join, axis=1)
        n_tr = k_tr.nunique()
        line = f"- **({', '.join(cols)})**: train={n_tr} groups"
        if all(c in test.columns for c in cols):
            k_te = test[cols].astype(str).agg("|".join, axis=1)
            overlap = len(set(k_tr.unique()) & set(k_te.unique()))
            line += f"; test={k_te.nunique()}; overlap={overlap}"
        lines.append(line)
    lines.append("")

    # Per-feature target-rate
    lines += ["## Item 3 — per-feature target rate", ""]
    top_num = ["TyreLife", "LapNumber", "Stint", "RaceProgress",
               "Cumulative_Degradation", "LapTime_Delta", "Position",
               "Position_Change"]
    for col in top_num:
        if col not in train.columns:
            continue
        try:
            binned = pd.qcut(train[col], 10, duplicates="drop")
        except Exception:
            continue
        rate = train.groupby(binned, observed=True)[TARGET].mean()
        n_per = train.groupby(binned, observed=True)[TARGET].size()
        lines.append(f"### {col} (deciles)")
        for b, r in rate.items():
            lines.append(f"- `{b}`: target_rate={r:.4f}, n={int(n_per[b])}")
        lines.append("")

    for col in ["Compound", "Driver", "Race"]:
        if col not in train.columns:
            continue
        rate = train.groupby(col)[TARGET].mean().sort_values(ascending=False)
        n = train[col].nunique()
        lines.append(f"### {col} ({n} levels)")
        if n <= 25:
            for k, r in rate.items():
                lines.append(
                    f"- `{k}`: target_rate={r:.4f}, "
                    f"n={int((train[col] == k).sum())}"
                )
        else:
            lines.append(
                f"- top-5: {rate.head(5).round(4).to_dict()}"
            )
            lines.append(
                f"- bottom-5: {rate.tail(5).round(4).to_dict()}"
            )
        lines.append("")

    out = Path(f"audit/{dt.date.today().isoformat()}-pre-baseline-gate.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
