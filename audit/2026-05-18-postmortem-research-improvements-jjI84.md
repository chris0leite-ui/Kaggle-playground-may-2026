# Postmortem — 2026-05-18 research-improvements-jjI84

## What went wrong

Decision-quality assessment (priors at decision-time, not hindsight):

- **D1 — Ran a2_2 + a3_1 full 5-fold without grep-ing the ledger.**
  `state/hypothesis-board.md` already recorded "K=4+1 +0.302 bp
  WEAK" for a2_2 mandatory_compound_rule and a similar predicted-
  null for a3_1. Cost: 115 min CPU on Round 1 for zero new
  information. Bad decision; the prior-result was retrievable
  pre-decision via a 30-second grep.

- **D2 — Skipped strategy-critic Section 5 (headroom math) until
  Round 3.** Plateau condition was clear from the prior handover
  (5 null mechanism classes overnight 2026-05-14 → Rule 14 + Rule
  7 triggers). The headroom math (~5 min) would have shown queue
  midpoint 1.4 bp discounted vs gap 1.9 bp — pivoting Round 1
  compute from mechanism probes to infrastructure (kNN-base
  rebuild). Bad decision given Rule 14 explicitly auto-fires at
  plateau.

- **D3 — Persona rotation order suboptimal.** Used 10 Wild Options
  + Junior ML in Round 2 (low-pressure brainstorm) then Senior ML
  in Round 3 (high-pressure review). The Senior ML's
  proxy-substitution surfaced in 5 minutes and yielded the
  Pearson ρ=0.998 K=4↔K=27 residual-correlation datum. Running
  Senior ML first would have re-prioritised Round 2's compute.

- **D4 — Did not check artifact-snapshot freshness at session
  start.** 30 seconds with `ls scripts/artifacts/ | grep
  dgp_v3_qA` would have surfaced the 6 missing slim-kNN bases.
  Every subsequent K=4+1 / K=21+1 / K=4+K27super+1 gate was 1.8-3.5
  bp behind the actual K=11+K=9 PRIMARY. The Round-3 Senior ML
  pressure-test eventually flagged this, but a session-start audit
  is the right place.

- **D5 — Lost ~10 min on KGAT_ token auth.** Set
  KAGGLE_USERNAME + KAGGLE_KEY + KAGGLE_API_TOKEN simultaneously;
  the CLI tried basic-auth and 403'd on private datasets.
  bootstrap.sh's docstring mentioned the conflict but the
  precedence wasn't documented in script-resolvable terms.

## PI overrides (calibration data)

- "Explain the options simple and concise" — Rule 0 reinforcement;
  I used `Tier-A2/A3 picks`, `ρ_test`, `K=11+1` jargon when PI
  wanted plain English.
- "Bootstrap and check the data" — I had been asking permission
  rather than running bootstrap.sh. Just-do-it preferred.
- "Artfacts are on kaggle private datasets" — I had falsely
  concluded artifacts were unavailable based on 403; PI corrected.
  The 403 was the KGAT_ token auth issue (D5), not a missing
  dataset.

## Rule-bypass failures

- **Rule 7** (research-before-saturation, mandatory after 3 nulls
  / 5 sat / 2 days no lift) was 10 days overdue at session start.
  Already noted in today's friction.md as `research-loop-overdue`.
- **Rule 14** (strategy-critic-loop at plateau) didn't fire on
  session start despite plateau condition being clear.
- **Rule 18** (claim leaf before probe) didn't bind on Round-1
  picks; ledger grep wasn't done.
- **Rule 22** (public-notebook scan) blocked by reCAPTCHA today;
  the agent failed to pivot to the authenticated `kaggle kernels
  list` path. Already noted in friction.md.

## Frictions logged this session

- `2026-05-18 research-loop-overdue` — 10-day gap from prior
  Research-loop despite Rule 7 thresholds met on 2026-05-14.
- `2026-05-18 kaggle-pages-recaptcha-gated` — WebFetch on Kaggle
  /code, /discussion returns only page title; switch R22 scan to
  authenticated `kaggle kernels list`.
- `2026-05-18 tier-a3-menu-stale` — 10 of 13 Tier-A2/A3 picks
  still pending 10 days after the original Research-loop synthesis.

Plus 3 frictions added by today's three Round-3 wrap commits:
`kggt-token-needs-isolated-auth`, `9-of-9-nulls-confirm-ceiling`,
`artifact-snapshot-blocks-k11-gating`,
`k4-k27-residual-correlation-0.998`,
`caruana-degenerates-without-diversity`.

## Promotion candidates — PI ratified all 5

1. **bootstrap.sh — auto-isolate KAGGLE_API_TOKEN when KGAT_
   prefix.** APPLIED to `bootstrap.sh:35-44`. Closes friction
   `kggt-token-needs-isolated-auth`.

2. **experiment-loop.md — step 0 mandatory ledger-grep gate.**
   APPLIED to `experiment-loop.md` as Step 0 before Heuristic
   baseline. Closes friction `tier-a3-menu-stale`.

3. **strategy-critic.md — Section 5 (headroom math) FIRST at
   plateau.** APPLIED to `strategy-critic.md` as an ordering note
   above the 5-question template. Closes friction
   `noise-ceiling-5-null-probes` (we should have done the headroom
   math before the probes, not after).

4. **personas.md — Senior ML Engineer FIRST on initial structural-
   ceiling claim.** APPLIED to `personas.md` as a new top row in
   the rotation table + a "Rotation order at plateau" note. Closes
   the implicit gap that Round 2 ran 9 probes before any
   methodological pressure-test.

5. **day-loop.md — snapshot-freshness audit at session start.**
   APPLIED to `day-loop.md` Auto-triggers section as the 6th
   bullet. Closes friction `artifact-snapshot-blocks-k11-gating`.

All 5 applied to skill files and logged in
`.claude/skills/kaggle-comp/improvements.md` under
`## Applied 2026-05-18 (round-1+2+3 wrap; PI-ratified)`.

## PI additions (from step 4 of the postmortem skill)

PI message verbatim: "Promoter suggested and think carefully what
to merge to main."

Interpretation: ratify all 5 candidates AND advise on which of
today's branch commits should merge to main. The merge advisory
is delivered in chat (separate from this postmortem artifact);
see the "What to merge to main" section the chat reply.

## Framework version at session-end

- Commit SHA (pre-postmortem): `871ca45`
- Active rules: CLAUDE.md lists ~37 rules (R0-R36 + dN defaults).
  No rules changed today; 5 skill-file edits applied (above).
- Loaded skills this session: `kaggle-comp` (default), `postmortem`
  (invoked at session end).

## What I'd do differently — 7-step framework (PI's framing)

Compressed walkthrough; full detail lives in the chat reply.

1. **Define** — session-start: write a 2026-05-18 problem
   statement to `audit/2026-05-18-problem-statement.md` per
   `problem-solving.md` Q3.5. The L1 should be "validate or break
   the row-feature ceiling claim with 13 days remaining" — not
   "iterate on mechanisms."
2. **Disaggregate** — draw MECE tree of plateau causes
   (A. row-features exhausted / B. ceiling claim wrong /
   C. public-private gap). I never drew this.
3. **Prioritise** — 2×2 (impact × effort). Verify-the-gate +
   pre-probe ledger grep + headroom math are HIGH-impact +
   LOW-effort cells; I treated them as steps 5-6 instead of step 1.
4. **Workplan** — Day 1: Senior ML pressure-test + headroom math
   in first hour. Decide on infrastructure investment (kNN
   rebuild ~3-6 hr CPU, ungated). C1 OpenF1 in parallel.
5. **Analyse** — with proper steps 1-4, save 3-4 hours of CPU
   and 1-2 hours of cognitive cycles in Round 1.
6. **Synthesise** — net result might still be 15/15 null, but the
   evidence would be valid (K=11+1 anchor, not K=4 proxy).
7. **Communicate** — PI gets pyramid earlier; decision is "rebuild
   kNN or accept K=27 fallback?" by hour 2 of session, not hour 8.
