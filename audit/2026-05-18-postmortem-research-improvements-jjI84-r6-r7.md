# Postmortem — 2026-05-18 research-improvements-jjI84 (Rounds 6-7)

Continues from `audit/2026-05-18-postmortem-research-improvements-jjI84-r4-r5.md`
(Rounds 4-5). This artifact covers Rounds 6 and 7 of the same
session day.

**Headline outcome**: new PRIMARY LB **0.95389** (R7.1 K=13 +
Path-B DriverClass × Stint τ=100k), +0.02 bp over R5.2 PRIMARY
(LB 0.95387). Top-5% gap closed 1.8 → 1.6 bp. 4 submissions across
R6+R7 (R6.1 0.95387 tied, R7.1 0.95389 new best, R7.2 0.95389
tied hedge). Total daily 7/10 used.

## What went wrong

Decision-quality view (priors at decision-time):

1. **R7 Phase A: 4 separate K=14+Path-B variants tested after
   first 2 were null.** Tested Compound×Stint, DriverClass×Stint,
   plain LR-meta, and K=15 super-combo with DAE. ~4 min CPU spent
   on closure. *Marginal*: thoroughness was defensible (wanted to
   close embedding-class as a K=11-pool adder) but stopping at 2
   would have saved CPU.
2. **R7.2 submitted at TIE_ZONE (ρ=0.99997).** Defensible: +0.264
   bp OOF lift may register on private LB; PI directive was
   "iterate"; slot would otherwise be idle. Outcome: public LB tied
   R7.1 at 0.95389 (as predicted). Decision was OK given priors,
   even if the public-LB outcome added no information beyond what
   we already knew.

PI-overrides: **None this round.** PI said "go" and ratified Round 7
plan via the approve-plan flow.

Rule-bypass failures: None visible.

## Frictions logged this session (R6 + R7 additions)

- `audit/friction.md` `two-axis-operator-sweep-missed` — Path-B
  has TWO hyperparameter axes (τ + segmentation); only τ was
  swept in R5. R7 found DriverClass × Stint beats default by +0.02
  bp LB.
- `audit/friction.md` `fold-bag-quantize-on-public-LB` — Fold-fit
  bag of LB-confirmed candidate at OOF +0.1-0.3 bp ties public LB
  at TIE_ZONE. Observed twice (R6.1, R7.2) in 2 days.

Both logged as frictions, **not promoted to improvements.md** per
PI directive. Awaiting more data points (3rd observation or
final-LB confirmation) before rule-rewrite.

## Promotion candidates (NOT promoted; held for further data)

Two patterns are visible but lack the corroboration needed for
rule promotion:

### G1 — Path-B-style operators need segmentation sweep too

**Pattern**: Round 5 introduced Path-B with hard-coded Compound ×
Stint. Round 7 found DriverClass × Stint beats it. 6 weeks of
default-segmentation as silent prior.

**Why not promoted**: Single new-segmentation win (DriverClass ×
Stint). Need to see whether more segmentation variants also lift
(deferred to next session) before generalising to "all per-segment
operators need 2-axis sweep at introduction."

### G2 — Fold-fit bag of LB-confirmed candidate → public-LB tie

**Pattern**: R6.1 fold-bag of R5.2 (OOF +0.212 bp → LB tied at
0.95387). R7.2 fold-bag of R7.1 (OOF +0.264 bp → LB tied at
0.95389). Two consecutive days, same mechanism (5-seed fold-fit
averaging), same outcome (TIE_ZONE ρ ≥ 0.99997, LB-quantized away).

**Why not promoted**: 2 data points across one comp. Need 3rd
observation OR final-LB confirmation that the +0.2 bp OOF showed
up on private LB. Until then, the rule "fold-bag is private-LB-only"
might be a coincidence of OOF→LB transfer band quirk.

## PI additions

PI directive at postmortem step 4: "Do not promote anything yet.
Just note, it inflictions or improvements. and prepare everything
so that they can start the next session."

→ Both candidates logged as friction entries; nothing promoted to
`improvements.md`. Wrap-up artifact (this file + next-session
prompt) prepared.

## Framework version at session-end

- Commit SHA at postmortem-write: `4b25269`
- Active rules: 0 + 1..36 + R1d/R2d/R5d/R7d/R8d defaults (per
  `CLAUDE.md`).
- Loaded skills this session: `kaggle-comp`, `postmortem`.

## Next-session prompt

See `audit/2026-05-19-next-session-prompt.md` (paste-ready PI block).
