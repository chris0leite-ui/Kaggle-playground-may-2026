# Postmortem — 2026-05-21 research-improvements-jjI84

Branch: `claude/research-improvements-jjI84`. Session arc: Day-22 PM,
comp-day 21/31, 10 days remaining. Session began with R31 3-way submit
(LB-TIE), pivoted through pseudo-labels + FT-Transformer (both NULL),
then R22 IDEA-scan + R38/R39 (both NULL for blend lift). LB still 0.95402.

## What went wrong

**Decision-quality assessment** (decisions evaluated on priors at decision-time,
not outcomes — per `knowledge-base/concepts/decision-quality-vs-outcome-quality.md`):

1. **R38 (CB YetiRank + domain features on listwise cohort) was a BAD DECISION
   given priors.** The pilkwang domain features were designed for pointwise LGBM
   (which got 0.94812). The listwise cohort ranker R21 succeeded because of
   within-cohort discriminative power, NOT cohort-level features. Compound,
   EstimatedRaceLaps, TyreLifePct are CONSTANT or near-constant within each
   (Y,R,L) cohort, so they can't help a within-cohort ranking loss. **Should
   have inferred this before pushing R38** — the IDEA-scan finding was about
   FE for pointwise models, not for listwise rankers. Cost: 1 Kaggle GPU run
   (~5 min), -14 bp standalone OOF result.

2. **Stale HANDOVER top-5% gap was a propagation failure**, not a session-level
   decision flaw — but it cost the first half of the session aiming at the
   wrong target. Decision quality: NOT-BAD given the data we trusted, but
   the propagation failure is a process bug worth promoting.

3. **R31 (Driver,Y,R) cohort pushed in parallel with R32/R33 without checking
   group-size limit FIRST**. R32 (max=5621) and R33 (max=7908) both fail at
   CB GPU YetiRank's 1023 limit. Should have verified group sizes ≤ 1023 BEFORE
   pushing 3 kernels at once. Cost: 1 wasted kernel slot (R32) and 1 deferred
   kernel that would have errored anyway (R33).

4. **R34 (pseudo-label cohort YetiRank)** was a BORDERLINE BAD DECISION given
   priors. R21's value came from within-cohort listwise loss; injecting
   pseudo-labeled test rows into cohorts breaks within-cohort label structure.
   Should have anticipated this. But the experiment WAS the diagnostic that
   produced the friction note, so the cost (1 GPU run) was repaid in learning.

**PI-overrides this session:**
- PI said "no" to public-kernel CSV blending after I started exploring
  flexonafft/safar1 blends. Correctly redirected to own-bases. R22 IDEA-scan
  was authorized later, distinct from CSV-blending.
- PI asked "think hard" when I presented R39 enumeration. Should have made
  a recommendation instead of presenting a menu. Calibration data: PI
  prefers decisive recommendations with reasoning over multiple-choice
  questions when the path is clear.

**Rule-bypass failures:** None this session. Rule 27 (ρ-band check), R33
(inner-CV for post-hoc transforms), R3 (4-gate filter), R6 (heuristic before
heavy compute) all applied.

**Rule-gap failures:**
- No rule for "validate group-size limits before pushing parallel kernels."
  Promotion candidate.
- No rule for "hedge ladder analysis must use test-prediction .npy ρ, not
  mechanism-family difference assumption." Promotion candidate.

## Frictions logged this session

Cross-linked to `audit/friction.md` 2026-05-21 block (10 entries):
- `cb-gpu-yetirank-max-query-1023`
- `public-LB-gap-stale-in-handover`
- `per-cohort-isotonic-needs-explicit-block`
- `monitor-until-loop-empty-status-false-positive`
- `rank-blend-ceiling-at-LB`
- `pseudo-label-breaks-cohort-listwise-signal`
- `listwise-cohort-cant-use-cohort-constant-features`
- `domain-features-no-blend-orthogonality`
- `r22-ideascan-public-ceiling-is-itself-blender`
- `hedge-ladder-redundant-LB-tied-not-diversified`

## Promotion candidates (pending PI ratification)

### [ ] kickoff-runbook.md — refresh LB-percentile gaps at session start

**Tag:** `public-LB-gap-stale-in-handover` (HANDOVER had stale top-5% gap; session
aimed at wrong target for first half)

**Where to insert:** session-start sequence, BEFORE any compute planning

**What to add:**
```
At session start (after `git fetch origin`), run:
  kaggle competitions leaderboard playground-series-s6e5 --download -p /tmp/lb
  python -c "import pandas as pd; lb=pd.read_csv('/tmp/lb/*.csv'); ..."
Recompute and print top-5%, top-10% boundaries and OUR rank against the
current LB. If the printed gap differs from HANDOVER's quoted gap by >1 bp,
flag it before any experiment plan.
```

**Why:** 2026-05-21 friction `public-LB-gap-stale-in-handover` — burned ~2hr
aiming at top-5% when gap was actually -4.7 bp (top-5% mathematically
unreachable on own bases per Strategy-critic Section 5). HANDOVER's
"-0.7 bp" was stale by ~10 days of LB growth.

---

### [ ] guardrails.md OR agent-ops.md — pre-flight cohort-size validation

**Tag:** `cb-gpu-yetirank-max-query-1023` (R32/R33 wasted slot when group
size > 1023)

**Where to insert:** after Rule 16 (6-question pre-flight) — add Q7:

**What to add:**
```
Q7 (CB GPU listwise variants only): max cohort group size <= 1023?
   Compute via:
     sizes = train.groupby(cohort_cols).size()
     assert sizes.max() <= 1023, f"max={sizes.max()} exceeds CB GPU YetiRank limit"
   If > 1023: either sub-bin the cohort or run on CPU. Q7 unanswered = SKIP.
```

**Why:** 2026-05-21 friction `cb-gpu-yetirank-max-query-1023`. The friction
catalog already had `cb-gpu-query-cross-entropy-size-limit` (256 limit for QCE)
but YetiRank's 1023 limit was undocumented. R32 (Y,R,Stint max=5621) wasted
a kernel slot; R33 (Y,R max=7908) was deferred and would have errored
similarly.

---

### [ ] guardrails.md — hedge ladder must use test .npy ρ analysis

**Tag:** `hedge-ladder-redundant-LB-tied-not-diversified` (R25/R27/R31 are
pairwise ρ > 0.9999, no diversification despite different mechanism families)

**Where to insert:** new rule near R2d / R7d hedge rules

**What to add:**
```
Rule R2d (extension): HEDGE candidates must satisfy ρ_test(HEDGE, PRIMARY)
< 0.999 on the .npy test predictions. "Different mechanism family" or "different
blend recipe" is NOT sufficient — three rank-mean blends of overlapping bases
can have pairwise ρ > 0.9999 and provide ZERO private-LB diversification.
Compute via `scipy.stats.spearmanr` on raw test predictions BEFORE selecting
final pair.
```

**Why:** 2026-05-21 friction `hedge-ladder-redundant-LB-tied-not-diversified`.
R25 (K18+R21 rank-mean), R27 (4-way rank-mean), R31 (3-way rank-mean) all
LB-tied 0.95402 and pairwise ρ > 0.9999. If I'd selected R27 or R31 as the
HEDGE, private-LB diversification = 0. Only K=18 (pure Path-B, ρ=0.99774)
provides meaningful HEDGE diversification.

---

### [ ] day-loop.md — R22 public-notebook scan must trace blender provenance

**Tag:** `r22-ideascan-public-ceiling-is-itself-blender` (safar1's 0.95449
was a blend of nina2025 + mikhailnaumov, not a novel single-model approach)

**Where to insert:** R22 public-notebook scan procedure

**What to add:**
```
When scanning top LB public kernels: if a high-LB kernel reads other
submission CSVs as its inputs (look for `kaggle.com/notebooks/...` or
`kaggle.com/datasets/.../submission.csv` in `pd.read_csv` calls), it is
a BLENDER not a novel mechanism. Recurse to its INPUT kernels for actual
mechanism ideas. The blender's hyperparameters are tuning, not novelty.
```

**Why:** 2026-05-21 friction. The whole R22 scan effort on safar1 yielded no
novel mechanism because safar1 is itself a blender. Should have skipped to
mikhailnaumov / nina2025 single-model kernels directly.

## PI additions (from step 4)

PI replied "No additions" to the additions question. Postmortem stands as
drafted.

PI replied "No preference" on promotion ratification. Per skill ("apply
edits to improvements.md only after explicit yes"), the 4 candidates above
are **NOT promoted this session**. They remain documented here as draft
candidates for future PI ratification.

## Framework version at session-end

- Commit SHA: `205db87228f63472cf87f567c9d0c58f8a1b0375`
- Active rules: 36 (R0..R36 + R1d..R8d defaults; see CLAUDE.md)
- Loaded skills this session: `kaggle-comp`, `postmortem`
- Branch: `claude/research-improvements-jjI84`
