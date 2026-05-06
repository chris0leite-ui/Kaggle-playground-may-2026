"""scripts/research_seed.py — family-kickoff web-retrieval stub.

Embeds MLE-STAR's web-retrieval lever (Google Research blog 2025; arXiv
2506.15692) into our loop. MLE-STAR's +47bp over AIDE on MLE-bench-Lite
came mostly from grounding the seed solution in prior-comp recipes
fetched from the web. We do this on plateau (Rule 7); this script makes
it cheap to do at FAMILY KICKOFF too.

Usage:
    python scripts/research_seed.py FAMILY_NAME [--note ...]

Generates a structured markdown stub at:
    audit/research-seed-<slug>-YYYY-MM-DD.md

The stub contains:
    - family name + BOTE family prior (read from probe.py FAMILY_PRIORS
      if recognised; else placeholder)
    - prompt template the agent uses with WebSearch to fill citations
    - citation table (top finisher writeups, papers, GM blog posts)
    - "candidate recipes derived" section with predicted OOF/LB band

After running this, the agent is expected to invoke WebSearch from a
follow-up turn to populate the stub, then BOTE individual recipes.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

AUDIT = Path("audit")

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from probe import FAMILY_PRIORS  # type: ignore
except Exception:
    FAMILY_PRIORS = {}


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip().lower())
    return s.strip("-")[:60] or "family"


TEMPLATE = """# Research seed — {family}

> Generated {today} by `scripts/research_seed.py`. Stub for the agent
> to fill via WebSearch (MLE-STAR-style family-kickoff retrieval).
> Origin: `audit/2026-05-06-agentic-kaggle-research.md` tip #3.

## Context

- Family: `{family}`
- BOTE prior: {prior}
- PI note: {note}

## Search queries to run

The agent should run these (or adaptively similar) WebSearch queries
and capture findings in the table below. **Required:** at least 5
sources, ≥2 from prior comps, ≥1 from a paper, ≥1 from a Grandmaster
blog/Kaggle writeup. Cite URLs verbatim.

1. `{family} Kaggle tabular winning solution`
2. `{family} site:kaggle.com writeup`
3. `{family} prior comp playground series`
4. `{family} arxiv 2024..2026`
5. `{family} grandmaster blog post`

## Citations (fill via WebSearch)

| # | Source (URL) | Type | Headline finding | Recipe extractable? |
|---|---|---|---|---|
| 1 |   | comp/paper/blog |   | yes/no |
| 2 |   |   |   |   |
| 3 |   |   |   |   |
| 4 |   |   |   |   |
| 5 |   |   |   |   |

## Candidate recipes derived

For each recipe extracted from the citations above, predict:
- standalone OOF Δ vs PRIMARY (bp)
- ρ vs PRIMARY (band)
- expected LB Δ at family BOTE prior (bp)
- cost (CPU min)

| Recipe | std OOF Δ | ρ band | LB Δ | cost | 5-Q clear? |
|---|---|---|---|---|---|
|   |   |   |   |   |   |

## Pre-flight 5-question check (Rule 16) — answer per recipe

For the top-ranked recipe, answer:

1. Family in `mechanism_families_explored`?
2. In rank-lock-vulnerable bucket {{meta-only, rule_residual-on-raw,
   GBDT-on-binary-target, formulation-already-in-pool}}?
3. Predicted standalone OOF (cite precedent): _____
4. Predicted ρ vs PRIMARY (cite closest base): _____
5. At that ρ, closest gate-PASS/FAIL precedent: _____
6. **(NEW)** Training objective matches row-AUC metric? Yes/No + why.

If 1–6 don't return a coherent answer, downgrade BOTE EV midpoint by
0.3× before ranking.

## Next action

After filling the citations + recipes:
- Run `python scripts/probe.py bote NAME --family {family} --cost_min N
  --metric-aligned true/false --pi-predicted-lb-bp X` for the top recipe.
- If verdict is PURSUE, claim an ISSUES.md leaf and proceed.
- If verdict is SKIP, append the rule-out reasoning here and close.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", help="mechanism family / candidate-class name")
    ap.add_argument("--note", default="", help="PI free-text context")
    ap.add_argument("--out", default=None,
                    help="override output path (default audit/research-seed-<slug>-YYYY-MM-DD.md)")
    args = ap.parse_args()

    if args.family in FAMILY_PRIORS:
        p, (pess, med, opt) = FAMILY_PRIORS[args.family]
        prior = f"P(useful)={p:.2f}, bp band P/M/O = {pess:.1f}/{med:.1f}/{opt:.1f}"
    else:
        prior = "(family not in FAMILY_PRIORS — agent should propose one and amend probe.py)"

    today = date.today().isoformat()
    body = TEMPLATE.format(
        family=args.family,
        today=today,
        prior=prior,
        note=args.note or "(none)",
    )

    if args.out:
        out = Path(args.out)
    else:
        AUDIT.mkdir(parents=True, exist_ok=True)
        out = AUDIT / f"research-seed-{_slugify(args.family)}-{today}.md"

    if out.exists():
        print(f"⚠ {out} already exists; not overwriting. Pass --out to override.")
        return
    out.write_text(body)
    print(f"→ {out}")
    print("\nNext: open the stub and fill citations via WebSearch, then BOTE recipes.")


if __name__ == "__main__":
    main()
