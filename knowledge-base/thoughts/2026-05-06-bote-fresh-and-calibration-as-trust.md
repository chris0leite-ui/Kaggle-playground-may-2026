# 2026-05-06 — BOTE was fresh, and calibration is the trust mechanism

> Synthesis of PI's [F1.1.1 + F1.4 answers](../questions/2026-05-06-grilling-round-4.md).
> Two factual corrections + one structural pushback.

## Factual corrections (verified via git log today)

### 1. BOTE genuinely didn't exist at TabPFN time — but Rule 2 did

| Time (2026-05-06 UTC) | Event |
|---|---|
| 06:24-06:42 | TabPFN kernel v6→v9 iteration (`N_FOLDS=1`, `SMOKE_FOLD0_ONLY`, `SMOKE_N_ROWS=50k`) |
| 08:11 | TabPFN fold-0 smoke result, AUC 0.94439 → DEAD |
| 09:57 | `scripts/probe.py` (BOTE harness) created |
| 10:26 | CLAUDE.md Rule 19 (BOTE codified) merged |

So PI's claim "BOTE not established yet" is correct.

But the TabPFN kill was **not BOTE-driven**. It was **Rule 2 + PI
override**. The kernel-iteration sequence (v7-v9) shows the agent
trying to comply with Rule 2 (smoke / 1-fold time-probe). PI's
"test first" instruction is exactly Rule 2 enforcement.

→ F1.2 originally read as evidence BOTE works. Actual reading: Rule 2
plus PI override worked. Logged as [flag F-1](../flags/2026-05-06.md#f-1).

### 2. BOTE itself was agent-drafted

`e48820d` was merged from `claude/ml-handover-alignment-xvUN0` — an
agent feature branch. The most-trusted focus-setting rule in the
whole framework was authored by an agent and ratified by PI through
a merge. Direct empirical confirmation of the F1.3 finding (PI as
editor, not author). Logged as [flag F-2](../flags/2026-05-06.md#f-2).

## PI's F1.4 answer: calibration is the trust mechanism

PI: build trust at scale by tracking **calibration of BOTE estimates,
predictions, and decisions** over time. Accepts it'll be slow.

This is the right shape. Three problems underneath.

### Problem 1 — calibration is two-stage

BOTE outputs (per `probe.py`) include both a predicted OOF Δ and a
predicted LB Δ. Currently the calibration ladder shows massive gaps
on the second stage:

| Probe | Pred OOF Δ | Realized LB Δ | Amplification |
|---|---|---|---|
| d9c FM K=20 swap | +0.53 bp | +3 bp | 5.7× |
| d9f K=21 swap | +0.32 bp | +2 bp | 6.25× |
| d9h K=22 add aug12 | +0.01 bp | +3 bp | **300×** |
| d9i K=21 swap aug2way | **−0.19 bp** | +3 bp | sign-flipped |
| Path B Compound τ=100k | +0.30 bp | +2 bp | 6.7× |
| Path B Stint τ=100k | +0.86 bp | +7 bp | 11.6× |
| Path B Compound×Stint τ=20k | +1.00 bp | +8 bp | 8× |

→ For FM-class, BOTE/OOF are systematically *under*-predicting LB
lift, and at least once *direction-flipped*. Calibration must
disentangle:
- Stage 1: BOTE → OOF Δ (the agent's prediction quality).
- Stage 2: OOF Δ → LB Δ (the leakage / generalization gap, here
  exacerbated by StratifiedKFold within-group leakage per P6).

PI's answer doesn't yet pick which stage. Drilled in
[F1.6](../questions/2026-05-06-grilling-round-5.md#f16).

### Problem 2 — PURSUE-precision is measurable, SKIP-recall isn't

If the agent SKIPs a probe, you don't get ground truth. So:

- **PURSUE precision**: of probes BOTE said PURSUE, how many delivered
  predicted lift? Easy to measure (you ran them).
- **SKIP recall**: of probes BOTE said SKIP, how many would have
  delivered? Impossible to measure without running them — which
  defeats the purpose of skipping.

Without addressing this, "decision calibration" only watches one side
of the confusion matrix. Drilled in
[F1.7](../questions/2026-05-06-grilling-round-5.md#f17).

### Problem 3 — sample size and cross-comp aggregation

With ≤10 submissions/day and a ~30-day comp, even an aggressive
PURSUE rate yields O(100-300) ground-truthed datapoints. That's tight
for two-stage calibration with multiple base populations (GBDT vs FM
vs hier-meta — currently calibrated very differently).

Implication: this is **per-comp calibration too noisy alone**. Trust
mechanism must aggregate **across comps**. The KB then needs to be
the cross-comp calibration store, not just per-comp notes. Out of
scope for today; flag for later.

## What PI's answer commits them to

- The trust mechanism is **empirical / aggregate**, not per-instance.
- Building it is **slow and explicit** — PI accepted this.
- The KB (or some adjacent store) will need to hold a **calibration
  log** that survives across comps.
- Per-instance overrides (TabPFN-style) need to be **distinguishable
  from rule-driven kills** in the log, or aggregate calibration will
  silently overcount the framework's wins.

## Where to push next

- F1.6 — pick the calibration stage(s).
- F1.7 — SKIP-recall sampling strategy.
- F1.5 (carried over from round 4) — bipartite CLAUDE.md ownership.
