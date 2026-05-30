# Postmortem — 2026-05-30 research-improvements-jjI84

## What went wrong

Nothing flagged as a bad decision given decision-time priors. Two items
worth examining though — both informational, not retroactive blame.

1. **I2/I3 raw-NN probe (slots 1, 3) cost 149 bp combined.** Decision
   to fire d22/d21 raw at 10% concentration was justified by the
   plan-level "most-orthogonal axis sitting under-used" thesis
   (ρ_d22_vs_anchor=0.682 vs C5's 0.991, 25× more concentration vs V8's
   ~0.4% effective). Counter-prior was visible in `state/current.md`:
   d22 standalone OOF=0.835 (vs K=18 ~0.954) signals raw axis is NOISY
   and only useful as a chassis-filtered signal. I did not connect
   "low standalone OOF" with "low noise-tolerance budget at high blend
   concentration." Decision was defensible (information-bearing
   calibration probe; PI plan pre-approval); the missed inference is
   the learning. See friction `raw-NN-axis-needs-chassis-filter`.

2. **Slots 7-9 fired after plateau was already clear at slot 6.** W2/W3/W4
   and M1 all tied 0.95457 by slot 6. Sophistication slate (L1/E3/C1)
   could have stopped at 1 probe instead of 3. L1 tied; E3 and C1
   regressed 1 bp. Defensible under R12 (spend full quota) and PI's
   "be aggressive" directive — sophistications ARE a different mechanism
   class. But marginal EV after slot 7 was near-zero. Not bad enough to
   change behavior; documenting for calibration.

## PI overrides this session

None received during Day-30 execution. PI plan ratification (this
morning) carried multi-slot R27 override authorization for all 9 fires.
Calibration: 0 overrides / 9 candidates this session. Cumulative trend
toward `pi-stamp-risk` watch.

## Frictions logged this session

Four entries appended to `audit/friction.md ## 2026-05-30`:

- `raw-NN-axis-needs-chassis-filter` — d22/d21 raw at 10% direct blend
  regress 70+ bp; same axis through K=20 PathB chassis lifts +1-3 bp.
- `curvature-blends-no-lift-on-AUC-plateau` — E3 cubic + C1 conf-cond
  tied V8 0.95456 while linear blends hit plateau 0.95457.
- `V8-family-plateau-at-+1bp` — flat plateau across P=[0.15, 0.25]
  and multi-ours variants; +1 bp is the structural ceiling.
- `R27-batch-override-justified-by-prior-day-band-break` — all 9
  candidates ρ_vs_V8 above 0.9974; plan-level pre-approval carried
  per-slot override.

## Promotion candidates (PI ratified: **YES — both promoted**)

### [x] `.claude/skills/kaggle-comp/improvements.md` — raw orthogonal-axis blend ceiling

**Tag:** `raw-NN-axis-needs-chassis-filter`
(Day-30 ablation: d22/d21 raw at 10% blend → -72 to -77 bp; same axis
routed through K=20 PathB chassis → +1-3 bp.)

**Where to insert:** under a new "## Mechanism-ledger closures" section
or under existing "## Day-30+ entries".

**What to add:**
```
- raw-orthogonal-axis-direct-blend → FALSIFIED.
  When a candidate axis has ρ_vs_anchor < 0.9 AND standalone
  OOF < anchor − 50 bp, the axis is NOISY. Do not blend raw at
  >5% weight; route through a chassis (PathB/LR-meta) first.
  Direct 10% blend will regress 50-80 bp. Evidence: d22 raw and
  d21 raw at 10% on Day-30 cost 149 bp combined; same signals
  via K=20 PathB chassis (C5, V8, W2) lifted +1 to +3 bp.
```

**Why:** 2 slots burned (149 bp combined). Same pattern likely to
re-emerge whenever a new high-orthogonality / low-standalone-AUC
mechanism is candidated as a blend addition. Closes the mechanism
class for direct-blend treatment.

### [x] `CLAUDE.md` (R27 augmentation) — batch-authorized override band

**Tag:** `R27-batch-override-justified-by-prior-day-band-break`
(All 9 Day-30 candidates ρ_vs_V8 above 0.9974; per-slot override was
pre-paid by plan-level PI plan approval.)

**Where to insert:** R27 description + `state/current.md` ρ-band table.

**What to add:**
```
R27 4th band: BATCH_AUTHORIZED. When a multi-slot plan with
≥3 R27-trippers is PI-ratified in advance, individual per-slot
overrides do not need re-authorization. Plan ratification grants
batch override; agent fires per-slot and logs outcome.
```

**Why:** Day-30 fired 9 R27 trippers across one PI-approved plan;
asking per-slot would have been ritualistic. The structural
authorization scope is the plan, not each candidate. Documents
existing practice rather than introducing new behavior.

## PI additions

"Nothing to add — wrap as-is" (PI 2026-05-30 wrap-up).

## Framework version at session-end

- Commit SHA at session-end: `ed3355c` (Day-30 wrap commit will follow)
- Active rules: R0, R1-R36 (+ R1d/R2d/R5d/R7d/R8d defaults)
- Loaded skills this session: kaggle-comp, postmortem

## Calibration snapshot (R26)

Probe.py calibration table (last 15 rows) attached for
postmortem-step-2 PI-override count. Trend: PI override rate at 0/9
this session; cumulative 2-session trend toward `pi-stamp-risk` flag
on HANDOVER.md if pattern continues.
