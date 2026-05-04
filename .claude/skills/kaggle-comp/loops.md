# Loops — agent-instruction form

Five loops. The agent runs Day-loop → Experiment-loop nested. The
human triggers Calibration / Research / Weekly when conditions hit.

## Day-loop

**Trigger**: every session start.

**Day boundary** (load-bearing):
- **A "day" = a Kaggle UTC submission-quota day** (5 submits/day,
  resets at UTC midnight). NOT a work-session boundary.
- **Day ends when EITHER** (a) `submissions_used_today == 5`, OR
  (b) PI explicitly declares EOD. Default = drive toward (a).
  Compute may continue past EOD; LB submits cannot.
- **A day is NOT done because experiments are conducted, even if
  the queue is empty.** If slots remain and PI hasn't called EOD,
  pick a new hypothesis and keep going.

**Stop**: 5/5 submissions used, OR PI declares EOD.

**Auto-trigger recognition (load-bearing — added 2026-05-04):** the
agent MUST recognize day-end from CONTEXT and execute steps 5-7
WITHOUT being prompted, asked, or invoked via slash command. The
PI does not run commands; the agent listens and acts. Day-end cues
the agent must catch:

- The most recent LB submit pushed `submissions_used_today` to 5
  (parse `kaggle competitions submissions` output or CLAUDE.md
  state block).
- PI's natural-language EOD signals: "the day is done", "let's
  wrap up", "stop submitting today", "no more LB submits", "EOD",
  "we're done for today", "let's call it", or any close paraphrase.
- Kaggle UTC midnight passes during an active session (slots reset).

When ANY cue fires, immediately execute the EOD workflow (steps 5-7)
in one batch without asking permission. The wrap is non-LB-touching
and Rule 1 does not apply. After the wrap, give the PI a 1-sentence
"day-N closed; wrap committed" notice and stop. Don't sit on
artifacts. Don't ask "should I write the wrap?" — just write it.

```
1. Load state (Haiku): comp-context.md, last 3 audit/, lb_status.py
2. Pick experiment (Sonnet): from queue or new hypothesis.
   *RE-RANK THE QUEUE BY EXPECTED LEARNING-PER-SLOT* at every
   replan, not by speculative lift. Best slot is the one that
   most reduces uncertainty about (a) OOF→LB calibration per
   mechanism family, (b) a pool member's behaviour, or (c) a
   structural-overfit signature. Heuristic-first if novel.
3. Execute Experiment-loop (see below). If gate-passing candidate
   ready and slot remains, propose to PI for single-shot submit.
4. After each LB result lands: update calibration ladder; if
   slots still remain and PI hasn't called EOD, return to step 2.
5. End-of-day audit (auto-trigger; no prompting): write
   audit/YYYY-MM-DD-day-N-wrap.md with FOUR REQUIRED SECTIONS:
   (a) 3-bullet PI summary,
   (b) Calibration ladder snapshot (today's submits + OOF→LB gaps),
   (c) Problems to address (load-bearing constraints surfaced today),
   (d) Hypotheses ranked by predicted-lift × CPU-feasibility +
       next-steps sequence (compute window + slot plan).
   Update CLAUDE.md state block (day, our_lb_best, headroom).
6. Append friction one-liners distilled from the day to
   audit/friction.md (NOT CLAUDE.md — see self-improvement.md).
7. Queue next session's first 3 experiments in CLAUDE.md
   hypothesis board. Then commit + push to feature branch AND
   merge to main (PI authorized 2026-05-04).
```

## Experiment-loop

**Trigger**: an experiment is selected.

**1-hour cap (revised 2026-05-04):** the cap applies to a
**SINGLE FOLD's actual wall time on full data**, not to the
extrapolated 5-fold-both-anchor projection. If one fold completes
within 1h on the production hardware, run it, see the result,
then decide whether to pursue the full 5-fold or shrink. This
unblocks heavyweight mechanisms (NN architectures, deep CatBoost
on CPU) for at least an exploratory single-fold probe.

```
1. Heuristic baseline (skip if N/A):
   - closed-form rule / threshold / hand-coded
   - bound the lift available
   - if heuristic already clears the candidate's predicted lift,
     stop here

2. Smoke (5-min cap):
   - 1 fold / 1 trial / 50k subsample
   - verify: COMPLETE status, expected artifacts emitted, no shape
     mismatch, no permission errors
   - if smoke fails, fix and re-smoke; do NOT push to production

3. 1-fold time-probe:
   - full feature set, full data, fold 0
   - measure wall time on same hardware as production
   - if 5-fold projection ≥ 1h, shrink config (folds, n_ens, epochs,
     subsample)

4. 5-fold production:
   - emit OOF + test .npy
   - compute candidate OOF on metric

5. 4-gate filter:
   - G1 standalone OOF clears anchor at recipe-bias
   - G2 blend with anchor lifts at α* > 0
   - G3 net rare-class-flip ratio ≥ 0.5
   - G4 direction asymmetry: more correct flips than incorrect

6. Minimal-input meta sanity check (if stacking candidate):
   - train candidate meta with ONLY 2 components (anchor + new)
   - if 2-comp OOF < anchor at recipe-bias, STOP. Do not LB-probe.

7. Reviewer audit:
   - subagent invocation, no parent-context anchoring
   - emit gate-result audit entry

8. Ask PI to submit:
   - present: candidate name, OOF, gate results, predicted LB lift,
     remaining slots today
   - PI confirms → single-shot kaggle competitions submit
   - log result, update calibration ladder
```

**Stop condition**: any gate fails, or PI declines, or LB result
lands.

## Calibration-loop

**Trigger**: every 5 LB submissions, OR after any negative-gap
entry > 5bp (LB above OOF), OR after any leakage incident.

```
1. Refresh calibration ladder (Haiku): parse all (OOF, LB) pairs,
   compute per-mechanism-family gap
2. Drift check: if any family's gap moves > 5bp from its trailing
   average, flag in CLAUDE.md and pause new submissions in that
   family
3. Refit blend weights if applicable (Sonnet) + re-run minimal-input
   meta on the refit
4. Commit calibration_ladder.md
```

## Research-loop

**Trigger** (mandatory): 3 consecutive nulls, OR 5 saturation events
at the same LB, OR 2 days without LB lift.

```
1. Web search (Opus): top public notebooks for the comp slug, top
   discussion threads in last 30 days
2. Read 2 prior-comp writeups in same domain (same metric, similar
   class imbalance, similar data type)
3. Persona rotation (Opus): invoke ML Researcher, return 5 untried
   mechanisms with citations
4. Rank by predicted EV × cost-to-test → top 3 to experiment queue,
   emit audit/YYYY-MM-DD-research.md with citations
```

The agent will want to skip this. The human PI should enforce it.

## Weekly-loop

**Trigger**: every 7 days, plus the start of the final 3-day window.

```
1. Re-read CLAUDE.md ⚠️ rules in full
2. Audit ceiling thesis: if any session claimed "structural ceiling",
   trigger Research-loop now
3. Persona rotation: rotate at least one on a stuck problem
4. Update REPORT.md with the week's results
5. Submission-budget audit: did we use 10/day? if not, why?
6. Friction distillation: scan audit/friction.md, find tags with
   ≥3 entries, edit guardrails / personas / examples in the skill
   itself. Reset friction.md (archive prior week).
   See self-improvement.md.
7. Commit a 5-line weekly summary to audit/
```

## Loop interaction

Day-loop wraps Experiment-loop. Calibration-loop is triggered by
submission count or leakage drift. Research-loop is triggered by
plateau detection (mandatory). Weekly-loop runs across day boundaries.

The agent's default failure mode is to stay in Experiment-loop,
ignoring Research-loop triggers. This is what the PI watches for.
