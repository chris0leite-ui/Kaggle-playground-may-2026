# Postmortem — 2026-05-08 research-model-extensions-Ibwvn

Branch session: senior-ML-researcher framing of how to extend the K=4
PRIMARY base; designed and executed 14 probes across 4 hypothesis
lanes; documented closure.

## What went wrong

**Rule-bypass failures**

- **Rule 19 (probe.py BOTE seal) bypassed on all 14 probes.** Each
  probe was at the cheap-probe threshold (10–60 min CPU), but the
  harness's BOTE seal protects the calibration log. Running 14 probes
  without sealing means 14 missing rows in `audit/decisions.jsonl`,
  which biases the calibration table toward "things we ran with
  formal seals." The cheap-probe exception is NOT a calibration-log
  exception — see promotion candidate #3.

**Rule-gap failures**

- **No rule said "verify domain heuristics against synth labelling
  before encoding them."** Lane 2 P2.2 deterministic rule clamps
  regressed −9.48 bp because `P(pit | is_last_lap) = 0.38` in this
  synth, vs ~0 in real F1. A 1k-row sanity check would have flagged
  this in 30 seconds. See promotion candidate #2.
- **No rule said "verify target structure before reformulating."**
  Designed P1.2 (discrete-time hazard) for ~10 minutes before
  empirically verifying that PitNextLap is only 81% deterministic
  from PitStop[L+1] (and L+1 is observed for only 29% of train rows).
  Construction abandoned. Verification belonged at the start.

**Bad decisions (would-not-retake-given-same-priors)**

- None. The lane choices, probe designs, and abandonment of P1.2
  after the structural finding were appropriate given the priors at
  decision-time.

**Hindsight refinements (not retroactive blame)**

- Lane 1 P1.3 (per-gap isotonic) regressed −2.18 bp. Predictable from
  Day-16 friction `primary-hier-meta-globally-calibrated`. Lane 3
  P3.2 (per-Compound isotonic) regressed −1.78 bp — same pattern,
  second confirmation. See promotion candidate #1.

**PI override this session**

- **Lottery-vs-mechanism framing.** PI noted that top-LB has few
  submissions, so the 12.5-bp gap to leader is most likely a real
  mechanism, not lottery. Sharpened the search; redirected effort
  into actually testing the four lanes rather than wrapping early.
  Calibration data-point: agent's framing was wrong, PI corrected,
  agent updated immediately. Result: even though the four lanes
  closed null, the closure is now **decisively** documented as
  empirical, not "we wrapped early because it might be lottery."

## Frictions logged this session

Six new entries in `audit/friction.md ## Week of 2026-05-08`:

- `pool-collapse-K4-effective-rank-1.33`
- `non-LR-meta-on-K4-regresses` (A30 falsified)
- `gap-feature-absorbed-by-tyrelife-stint-lap-compound`
- `synth-divergence-from-F1-realism-on-last-lap`
- `pitnextlap-not-deterministic-from-observed-row-structure`
- `isotonic-overfits-when-base-already-calibrated` (2nd confirmation)

Cross-links: `audit/2026-05-08-four-lane-research-extension.md`,
probe artefacts in `scripts/artifacts/probe_lane{1,2,3,4}_*.json`
and `oof_lane*_strat.npy` (11 OOF arrays).

## Promotion candidates (PI ratified: ALL REJECTED 2026-05-08)

PI response: "no additions, no promotion." All three candidates remain
as friction-log entries only; no edits applied to
`.claude/skills/kaggle-comp/improvements.md` or `CLAUDE.md`.

### [REJECTED] Candidate 1 — extend Rule 33 with stratum-ECE precondition

**Tag:** `isotonic-per-stratum-needs-stratum-ECE-threshold`
(2 confirmations this session; possibly more in archive.)

**Where to insert:** Rule 33 in `CLAUDE.md ## Operating rules — concise`.

**What to add:** "Per-stratum isotonic / Platt rescaling requires the
stratum's pre-rescaling ECE > 1% to be worth attempting. When the
input is already well-calibrated per stratum, fold-restricted
isotonic overfits noise."

**Why:** P1.3 per-gap iso −2.18 bp; P3.2 per-Compound iso −1.78 bp;
both regressed despite ECE diagnostics showing near-zero
miscalibration per stratum (0.0001–0.0112 range across 12 strata
total).

**PI ratification (2026-05-08):** REJECTED — no promotion.

### [REJECTED] Candidate 2 — add Q7 to pre-flight 6-question check

**Tag:** `verify-domain-heuristic-against-synth-labelling`
(Lane 2 P2.2 lost 9.48 bp.)

**Where to insert:** Rule 16 in `CLAUDE.md ## Operating rules — concise`,
appended to the 6-question list.

**What to add:** "Q7 — for any heuristic / rule clamp imported from
domain knowledge: verify the rule empirically against this comp's
labelling on a 1k-sample probe BEFORE deploying. Synthesised data
may diverge from real-world rules in ways that invalidate the prior.
Q7 unanswered = forced SKIP."

**Why:** `P(pit | is_last_lap) = 0.38` in this synth, contra F1
realism (~0). Deterministic clamp regressed −9.48 bp at OOF; would
have lost a submission slot if deployed.

**PI ratification (2026-05-08):** REJECTED — no promotion.

### [REJECTED] Candidate 3 — tighten Rule 19 BOTE-seal threshold

**Tag:** `bote-seal-required-even-for-cheap-probes`
(14 probes this session, 0 BOTE seals.)

**Where to insert:** Rule 19 in `CLAUDE.md ## Operating rules — concise`.

**What to add:** "BOTE seal via `scripts/probe.py bote` is required
for any candidate ≥10 min CPU/GPU AND any candidate that produces an
OOF artefact suitable for stack-add. The cheap-probe exception is
NOT a calibration-log exception — even a 5-min probe without BOTE
is a missing row in `audit/decisions.jsonl`."

**Why:** Without sealed predictions per probe, the calibration table
becomes biased toward "things we ran with formal seals" and we lose
agent-vs-actual rows for cheap probes — exactly the regime where
priors are weakest.

**PI ratification (2026-05-08):** REJECTED — no promotion.

## PI additions (from step 4)

PI response 2026-05-08: "no additions, no promotion." None.

## Framework version at session-end

- Commit SHA: `36ced460bfeb55340e3bc2a4c86a720b92ea3283`
- Branch: `claude/research-model-extensions-Ibwvn`
- Active rules: 1..36 + Defaults R1..R8 (per
  `CLAUDE.md ## Operating rules — concise` and `## Defaults from
  prior-comp postmortem`)
- Loaded skills this session: `kaggle-comp`, `postmortem`
- Probes executed: 14 (4 lanes × 3-4 probes each + 1 base-level
  gap variant); 0 PASS, 7 AMBIG, 7 NULL/negative
- Calibration snapshot ran at wrap-up (Step 5b); per-probe BOTE
  seals were NOT logged this session (see promotion candidate #3)
