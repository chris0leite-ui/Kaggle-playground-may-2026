"""scripts/hypothesis_view.py — mechanism-families graph view.

Parses `CLAUDE.md` `mechanism_families_explored:` block, classifies each
entry by AXIS (model_class / meta_arch / target_reform / cohort_split /
fe_addition / external_data / pool_surgery / pseudo_aug / hyperparam),
and renders a markdown table grouped by axis (open coverage gaps surface
visually).

Inspired by RD-Agent's persistent-hypothesis-graph pattern (arXiv
2505.14738) — but kept lightweight: single source of truth is CLAUDE.md;
this script is a *view*, not a separate database.

Usage:
    python scripts/hypothesis_view.py                  # markdown to stdout
    python scripts/hypothesis_view.py --json data/hypothesis_graph.json
    python scripts/hypothesis_view.py --axis target_reform   # filter
    python scripts/hypothesis_view.py --status alive          # alive/dead/held

Heuristic axis-tagging via slug substrings + override dict. PI ratifies
the tagging by skimming the markdown view; corrections go to AXIS_OVERRIDE.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


CLAUDE_MD = Path("CLAUDE.md")


# Axis tagging: substring → axis. First match wins; order matters.
# Keep ordered by specificity (most specific first).
AXIS_RULES: list[tuple[str, str]] = [
    ("path_b", "meta_arch"),
    ("hier", "meta_arch"),
    ("groupkf", "meta_arch"),
    ("leak_corrected", "meta_arch"),
    ("lr_meta", "meta_arch"),
    ("gbdt_meta", "meta_arch"),
    ("lambdarank", "meta_arch"),
    ("two_level_stacking", "meta_arch"),
    ("kd_distillation", "meta_arch"),
    ("aux_feature_gbdt", "meta_arch"),
    ("orig_transfer", "external_data"),
    ("orig_multi_arch_bag", "external_data"),
    ("decode_normalized_tyrelife", "external_data"),
    ("physics_residual", "external_data"),
    ("leak_lookup", "external_data"),
    ("target_reformulation", "target_reform"),
    ("invlaps", "target_reform"),
    ("multi_target_nn", "target_reform"),
    ("t12_", "target_reform"),
    ("censored", "target_reform"),
    ("ratio_target", "target_reform"),
    ("stintlevel_survival", "target_reform"),
    ("stintgrouped_lambdamart", "target_reform"),
    ("cohort", "cohort_split"),
    ("year_segmented", "cohort_split"),
    ("compound_stint", "cohort_split"),
    ("driver_cluster", "cohort_split"),
    ("path_b_cohort", "cohort_split"),
    ("dae", "model_class"),
    ("factorization_machine", "model_class"),
    ("hash_lr", "model_class"),
    ("nn_with_embedding", "model_class"),
    ("tabnet", "model_class"),
    ("tabpfn", "model_class"),
    ("extra_trees", "model_class"),
    ("knn_distance", "model_class"),
    ("realmlp", "model_class"),
    ("gaussian_naive_bayes", "model_class"),
    ("aucpairwise_xgb", "model_class"),
    ("recursive_gbdt", "model_class"),
    ("baseline_lgbm", "model_class"),
    ("xgb_native", "model_class"),
    ("catboost_native", "model_class"),
    ("catboost_year-cat", "model_class"),
    ("catboost_yetirank", "model_class"),
    ("catboost_lossguide", "model_class"),
    ("catboost_gpu_multi", "model_class"),
    ("hgbc_label", "model_class"),
    ("hgbc_beta_variants", "model_class"),
    ("row_subsample_catboost", "model_class"),
    ("single_bag", "ensemble"),
    ("blend_aggregators", "ensemble"),
    ("2base_recursive_blend", "ensemble"),
    ("rule_residual", "fe_addition"),
    ("simple_math_rule", "fe_addition"),
    ("multi_rule_residual", "fe_addition"),
    ("within_stint", "fe_addition"),
    ("within_race", "fe_addition"),
    ("cross_driver", "fe_addition"),
    ("lap_mod", "fe_addition"),
    ("id_order_synth", "fe_addition"),
    ("masked_column", "fe_addition"),
    ("year_stint_sparse_lr", "fe_addition"),
    ("oof_target_encoding", "fe_addition"),
    ("unified_te_2way", "fe_addition"),
    ("sequence_fe", "fe_addition"),
    ("relative_state_fe", "fe_addition"),
    ("fm_new_input_features", "fe_addition"),
    ("fm_aug", "fe_addition"),
    ("fm_partition", "fe_addition"),
    ("gkf_full_22_stack", "pool_surgery"),
    ("move_c", "pool_surgery"),
    ("corr_pool_prune", "pool_surgery"),
    ("l1coef_pool_prune", "pool_surgery"),
    ("tier_break_l1_prune", "pool_surgery"),
    ("groupkf_full_pool_meta", "pool_surgery"),
    ("groupkf_stack_rebuild", "pool_surgery"),
    ("pseudo_label", "pseudo_aug"),
    ("partial_pseudo", "pseudo_aug"),
    ("adversarial_validation", "pseudo_aug"),
    ("alpha_calibrated", "hyperparam"),
    ("dirichlet", "hyperparam"),
    ("l1_meta_sweep", "hyperparam"),
    ("lr_meta_stacker", "meta_arch"),
    ("reformulation_lgbm", "target_reform"),
]

# Manual overrides (slug → axis) for entries the rules misclassify.
AXIS_OVERRIDE: dict[str, str] = {}

VERDICT_RULES: list[tuple[str, str]] = [
    ("PRIMARY", "alive"),
    ("HELD", "held"),
    ("HEDGE", "held"),
    ("FALSIFIED", "dead"),
    ("FAIL", "dead"),
    ("DEAD", "dead"),
    ("REGRESS", "dead"),
    ("REGRESSED", "dead"),
    ("NULL", "dead"),
    ("PASS", "alive"),
    ("TIE", "alive"),
]


def classify_axis(slug: str, body: str) -> str:
    if slug in AXIS_OVERRIDE:
        return AXIS_OVERRIDE[slug]
    text = f"{slug} {body}".lower()
    for substr, axis in AXIS_RULES:
        if substr in text:
            return axis
    return "uncategorized"


def classify_verdict(body: str) -> str:
    upper = body.upper()
    for tag, status in VERDICT_RULES:
        if tag in upper:
            return status
    return "untagged"


def parse_claude_md(path: Path = CLAUDE_MD) -> list[dict]:
    text = path.read_text()
    m = re.search(r"mechanism_families_explored:.*?\n(.*?)\n(?=\S|\Z)", text,
                  re.DOTALL)
    if not m:
        raise SystemExit("mechanism_families_explored block not found in CLAUDE.md")
    block = m.group(1)
    entries: list[dict] = []
    for line in block.splitlines():
        s = line.strip()
        if not s or not s.startswith("- "):
            continue
        rest = s[2:]
        if "#" in rest:
            slug, body = rest.split("#", 1)
            slug = slug.strip()
            body = body.strip()
        else:
            slug = rest.strip()
            body = ""
        if not slug or "," in slug:
            continue
        slug = slug.split()[0]
        entries.append(dict(
            slug=slug,
            note=body,
            axis=classify_axis(slug, body),
            status=classify_verdict(body),
        ))
    return entries


def render_markdown(entries: list[dict], axis_filter: str | None,
                    status_filter: str | None) -> str:
    by_axis: dict[str, list[dict]] = {}
    for e in entries:
        if axis_filter and e["axis"] != axis_filter:
            continue
        if status_filter and e["status"] != status_filter:
            continue
        by_axis.setdefault(e["axis"], []).append(e)

    lines = ["# Hypothesis graph — by axis", ""]
    counts = {a: len(v) for a, v in by_axis.items()}
    total = sum(counts.values())
    lines.append(f"Total entries: **{total}** "
                 f"(across {len(counts)} axes)")
    lines.append("")
    lines.append("Coverage:  " + ", ".join(
        f"`{a}`={n}" for a, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ))
    lines.append("")

    AXIS_ORDER = [
        "model_class", "meta_arch", "target_reform", "cohort_split",
        "fe_addition", "external_data", "pool_surgery", "pseudo_aug",
        "ensemble", "hyperparam", "uncategorized",
    ]
    for axis in AXIS_ORDER:
        if axis not in by_axis:
            continue
        rows = by_axis[axis]
        lines.append(f"## {axis} ({len(rows)} entries)")
        lines.append("")
        lines.append("| slug | status | note |")
        lines.append("|---|---|---|")
        for e in rows:
            note = e["note"][:120].replace("|", "/")
            lines.append(f"| `{e['slug']}` | {e['status']} | {note} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default=None,
                    help="filter to one axis (e.g. target_reform)")
    ap.add_argument("--status", default=None,
                    help="filter to one status (alive/held/dead/untagged)")
    ap.add_argument("--json", default=None,
                    help="write JSON dump of all entries to this path")
    ap.add_argument("--no-md", action="store_true",
                    help="suppress markdown output")
    args = ap.parse_args()

    entries = parse_claude_md()
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(entries, indent=2))
        print(f"→ {args.json}  ({len(entries)} entries)", file=sys.stderr)
    if not args.no_md:
        print(render_markdown(entries, args.axis, args.status))


if __name__ == "__main__":
    main()
