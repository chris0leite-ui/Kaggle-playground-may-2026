# EXPERIMENTS-NEXT.md — durable experiment ledger

This file is the **across-session memory** of what experiments are
worth running, why, and what we'd learn. Anything in `state/`,
`HANDOVER.md`, or `audit/` is volatile or session-specific; this file
is the permanent menu.

Read order on session start:
1. This file (anything tagged `pending` is fair game, `running` is
   already in flight elsewhere).
2. `ASSUMPTIONS.md` for the load-bearing facts each EXP rests on.
3. `WEAKNESSES.md` for the gaps each EXP is targeting.
4. `state/calibration-ladder.md` for the precedent each EXP cites.

Edits to this file are **append-only for the menu**; status updates
are in-place. When an EXP completes, its row gets a verdict and a
link to the audit note.

---

## Operating rules for this file

1. Every EXP entry must cite (a) the WEAKNESSES.md item it targets,
   (b) the ASSUMPTIONS.md row that justifies the prior, (c) cost in
   minutes, (d) what we learn if it passes AND what we learn if it
   fails.
2. EXPs are tiered by cost: A=cheap (≤30 min CPU), B=medium (1-3 hr),
   C=expensive (GPU kernel, >3 hr).
3. If an EXP is **falsified** in a session, mark `verdict: FALSIFIED`
   with the audit-note path. Don't delete — the next session needs to
   know this was tried.
4. The "what we learn if it fails" column is the load-bearing one.
   An EXP whose null result tells us nothing is not worth running.

---

## Strategic context (Day-19 baseline)

We are at LB 0.95368 (PRIMARY = K=27 + Path-B Compound × Stint, τ=100k).
Gap to top-5%: −3.7 bp. Gap to leader: −10.8 bp. Submissions used:
40 / 270.

**Two findings make the strategic frame:**
- **A26 (MEASURED, Day-19):** the 17 "extra" bases in K=27 buy us
  exactly 1.2 bp on LB (K=10 forward-greedy at LB 0.95356).
- **A25 (MEASURED, Day-19):** the K=27 logit pool sits in a 3.23-D
  subspace. All 27 bases share the same task framing (i.i.d. binary
  AUC on 14 columns). They share a **3-D logit subspace** because
  the task — not the model class — defines the directions.

The leader's +10.8 bp requires a **4th independent direction** in
the embedding. Pool surgery alone won't produce it; we need a base
with a structurally different inductive bias.

---

## Tier A — cheap, learning-rich, runnable today

### EXP-1 — K=10 + 1-base meta-add gate for previously-null bases
**Status:** in progress (Day-19 PM)
**Targets:** WEAKNESSES W5 (rank-lock vs pool-redundancy distinction);
ASSUMPTIONS A9 (rank-lock at K=22 inferred but not verified at K=10).
**Cost:** ~30 sec per candidate if OOF is cached; ~30 min per candidate
if it must be re-run.

What we test: bases that were NULL at K=22+ / K=24+ / K=27+ may have
been hidden inside the dense pool's redundancy. At K=10 (logit eff-rank
~3), the absorption capacity is smaller. If any candidate passes
K=10+1 plain LR-meta gate by ≥ +0.5 bp, it's a stack-add candidate
AND we have evidence the dense pool was concealing signal.

| Candidate | OOF persisted? | Original verdict |
|---|---|---|
| d16 GRU sequence | YES (kernels/d16-gru-sequence-gpu/output/) | NULL Δ −0.043 bp at K=22+1 |
| Field-state cross-driver aggregates | NO (regenerate, ~30 min) | NULL Δ −0.015 bp at K=24+1 |
| H9 transductive pseudo | NO (regenerate, ~30 min) | passed plain LR-meta +0.63 bp; absorbed by hier-meta |
| Combined-frame lead/lag | NO (regenerate, ~15 min) | NULL Δ −0.36 bp combined-frame premium |

**What we learn if any candidate passes:** the dense pool was hiding
a 4th-direction signal. We get a stack-add candidate AND a strong
prior to retest other formerly-null candidates.

**What we learn if all candidates fail:** the rank-lock is real and
pool-size-independent; the 4th direction requires a fundamentally
different *task framing*, not a different *model class*.

### EXP-2 — Per-stint LambdaRank base
**Status:** pending
**Targets:** A25 (3-D pool because all bases share i.i.d. binary AUC
framing); WEAKNESSES W5 (untried task framings).
**Cost:** ~25 min CPU.

LightGBM with `objective=lambdarank`, group-id =
`(Race, Driver, Year, Stint)`, target = PitNextLap. Each stint is a
small query (mean 3.87 rows). Within-stint, the model learns *which
row* is the pit row. Predicts a soft per-row rank.

The d13 LambdaMART work was killed for sparsity (63% all-zero stints)
on a different framing. Synth stints are short enough that a within-
stint ranker has a small but dense target space — qualitatively
different from the task tested.

**What we learn if it passes K=10+1:** a non-binary task framing
produces a 4th direction at the meta. Major reframe — opens hazard
models, two-stage cascades, position-aware models.

**What we learn if it fails:** rank-lock holds across task framings,
not just model classes. Suggests the binding constraint is at the
*feature-content* level, not the *learning-objective* level.

### EXP-3 — Inter-stint feature engineering
**Status:** pending
**Targets:** WEAKNESSES W5 (cross-stint memory); strategic angle that
F1 strategy is sequential across stints, not just within a stint.
**Cost:** ~30 min CPU.

Combined-frame, AV-safe at row level (per A3). Features per row:
- `prev_stint_length` — observed length of (Race, Driver, Year)
  previous stint
- `prev_compound` — what compound was used last
- `prev_pit_lap_in_race` — absolute LapNumber of last pit
- `stints_completed_so_far` — count of completed stints in this race
- `race_pit_count_so_far` — total pit count by this driver in this race

Single-LGBM 5-fold OOF on these features alone. Then K=10+1 gate
against the sparse pool.

**What we learn if it passes:** strategic context (memory across
stints) is missing from the current pool. Suggests a hierarchical-
sequence model (transformer with stint-completion tokens) would help.

**What we learn if it fails:** within-stint signal is sufficient;
cross-stint strategic memory is either noise or already extracted via
Stint number + Compound.

### EXP-4 — Stint-completion dual-head
**Status:** pending
**Targets:** A25 (task framing is the bottleneck); W3 (synth temporal
downsampling distorts sequences).
**Cost:** ~15 min CPU.

Decompose PitNextLap into two binary heads:
- Head A: `is_last_observed_row_in_this_stint`
- Head B: `pit_happened_given_last_row`

Train two separate LightGBMs; combine via `P(A=1) · P(B=1|A=1)` (with
a calibration adjustment). Single OOF.

The composition recovers PitNextLap with a different decomposition
than direct binary prediction. The downsampling-aware Head A is
explicitly modeling the synth's structure.

**What we learn if it passes K=10+1:** the synth's downsampling
masks a structural signal that explicit decomposition can recover.
This is a route to a *4th direction* without needing a fancy model.

**What we learn if it fails:** the team's previous "i.i.d. binary
is the right framing" stance is justified.

### EXP-5 — Minimal-pool sweep, K = {2..10} forward-greedy
**Status:** in progress (Day-19 PM)
**Targets:** A26 (sparse pool calibration); user's question "can we
reduce the pool further without LB regression."
**Cost:** ~10 min CPU.

For each k ∈ {2, 3, 5, 7, 9, 10}, fit plain LR-meta and Path-B C×S
τ=100k on the forward-greedy E9 pick order. Identify smallest k where
OOF is within 2 bp of K=10. Submit that pool to LB as a calibration
probe.

**What we learn if it passes:** confirms the bank is even more
redundant than K=10 indicates; we can run with k=5 or k=7 with no
LB cost. Operational simplification.

**What we learn if it fails:** K=10 is the right operating size;
sparser pools fall off quickly.

---

## Tier B — medium cost, requires structural compute

### EXP-6 — Loss-function diversity sweep
**Status:** pending
**Targets:** A25, W5.
**Cost:** ~1 hr CPU. Train three LightGBMs with the same features but
{`objective=binary`, focal-loss, pairwise-rank}. Compute pairwise ρ
across the three; if any pair is < 0.95, candidate base.

### EXP-7 — Hazard / survival model with parametric baseline
**Status:** pending
**Targets:** A25 (different task framing).
**Cost:** ~1.5 hr CPU. Cox-PH style with parametric baseline hazard
on (Compound, position-in-stint). Predicts P(pit | features) via the
hazard; maps back to row-level binary at meta gate.

### EXP-8 — Cross-driver field-state at meta-level (not as base)
**Status:** pending
**Targets:** the unfortunate framing of A24.
**Cost:** ~30 min CPU. Field-state aggregates were tested as a *base*
and absorbed by K=24+1 hier-meta. Test as **meta-feature columns
alongside the base predictions** at K=10 plain LR-meta. Different
mechanism: the meta sees field-state directly, not via routing through
a single LGBM base.

---

## Tier C — GPU, expensive

### EXP-9 — Sequence transformer with explicit gap modelling
**Status:** pending
**Targets:** W3 (synth downsampling), A25.
**Cost:** ~4-6 hr Kaggle T4×2.

A small attention-only model over (Race, Driver, Year) sequences with:
- Absolute `LapNumber` positional encoding (not just within-stint).
- Learned `gap_embedding` for the lap-gap to previous observed row.
- (Compound, Stint) embeddings as auxiliary inputs.
- Output: PitNextLap probability per row.

The d16 GRU was NULL because it ingested sequences without gap-mask.
A transformer with explicit gap modelling would learn from both
present and absent laps.

**What we learn if it passes:** the synth's downsampling masks a
sequence signal that gap-aware attention recovers. Strong evidence
for the leader's likely mechanism.

### EXP-10 — Driver-stable embedding NN
**Status:** pending
**Targets:** W5 (untried task framings); the observation that drivers
have stable behavioural patterns across years.
**Cost:** ~2 hr Kaggle GPU.

Small NN with a per-driver learned embedding (vector dim ~16). Trained
end-to-end on PitNextLap. The Driver column is currently encoded as
count/freq in the GBDT bases — not as a learned dense embedding from
sequence patterns.

---

## Research, not compute

### RES-1 — Public LB leader-pattern scrape
**Status:** pending
**Cost:** 30 min reading.

Pull public LB top-50 with submission counts. Look for cluster
structure (single high outlier = unique trick; tight cluster = shared
mechanism). Check if any top-team profiles have a public discussion
post or earlier-comp pattern that hints at mechanism class.

### RES-2 — Postmortem of the "Path-B amp myth" (Day-19 finding)
**Status:** pending
**Cost:** 1 hr writing.

Synthesise audit/2026-05-08 finding A9c (Path-B amp +0.0–0.4 bp at
all pool sizes; the Day-13 +18 bp claim was pool-specific noise) into
a postmortem and cross-reference all 11+ Path-B variants tested.
Promote `path-b-amp-is-statistical-noise` to a CLAUDE.md rule
candidate.

---

## Verdicts — closed EXPs

| EXP | Verdict | Audit |
|---|---|---|
| EXP-1 (GRU at K=10+1) | NULL Δ −0.045 bp (matches K=22+1 result of −0.043 bp) — rank-lock is pool-size-independent. Strong prior NOT to rerun field-state / H9 / lead-lag — they will show the same. | `scripts/probe_exp1_gru_retest.py` + `scripts/artifacts/probe_exp1_gru_retest.json` (Day-19 PM) |
| EXP-5 (minimal-pool sweep) | K=4 forward-greedy is the smallest pool within 2 bp OOF of K=10. K=4 LB **0.95351** vs PRIMARY 0.95368 (LB Δ −1.7 bp; *better* than OOF prediction of −2.93 bp by ~1.2 bp). Operational simplification: **K=4 captures 99% of the bank's LB value with 15% of the bases.** | `scripts/probe_minimal_pool_sweep.py` + `scripts/artifacts/probe_minimal_pool_sweep.json` (Day-19 PM) |
