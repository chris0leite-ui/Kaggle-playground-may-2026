# audit/ index

For mutable session state, see `state/` (sibling of `audit/`).
For rules + pointers, see `CLAUDE.md`.

## Top-level (load-bearing)

| File | Why it stays at top |
|------|---------------------|
| `friction.md` | Concise weekly summaries; first read each session. |
| `friction-archive.md` | Full 1,450-line historical friction; do not read by default. |
| `decisions.jsonl` | BOTE / outcome calibration log. |
| `2026-05-06-target-reform-leakage-audit.md` | Origin of Rule 24; mandatory pre-submit reading. |
| `2026-05-07-d19-overnight-research.md` | Day-19 closure notes. |
| `2026-05-15-d15-4branch-results.md` | Day-15 PRIMARY-trail context. |
| `2026-05-16-d16-virgin-axes-results.md` | Day-16 closure (11 of 11 null). |
| `archive-*-handover-*-sections.md` | Archived HANDOVER per-branch sessions. |

## Subdirectories

| Path | Contents |
|------|----------|
| `postmortems/` | Session-end postmortems (Rule 17 step 4b output). |
| `research/` | Research synthesis, strategy critiques, idea boards, prior-art notes. |
| `2026-05-06/` | Day-14/15-era per-probe audits (DGP, single-model, blend/rho probes). |
| `2026-05-07/` | Day-17/18/19 per-probe audits (CB recipe transfer, DGP decomp, LR diagnostics, Path-B variants, field-state). |
| `archive_pre_d13/` | Day-3 to Day-12 audits (kickoff through M-tier sweeps). |

## Quick lookups

- "Where is the day's CatBoost-recipe-transfer audit?" → `2026-05-07/2026-05-07-d17-cb-v4-yekenot-transfer.md`
- "Where is the leakage origin?" → `2026-05-06-target-reform-leakage-audit.md` (top-level).
- "What postmortems exist?" → `postmortems/`.
- "What does the LR-leverage finding say?" → `2026-05-07/2026-05-07-lr-leverage-six-probes.md`.
- "Where is the strategy-critique loop output?" → `research/2026-05-07-d17-strategy-critique.md`.
- "What's the full historical friction?" → `friction-archive.md`.
