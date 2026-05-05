# Day-3 learnings + observations on weaknesses + new hypotheses

Captured at ~22:30 UTC, mid-night-batch. Three runs in flight (RealMLP
fold 3/5; H1 fold 2/5; LR-FE just launched). EBM completed (NULL standalone).

## What we learned today

### 1. M5h is at the GBDT-pool OOF ceiling — and the LB ceiling

OOF transfer is collapsing:
  M5b 7-base → 0.94926 / 0.94891 LB / 92% transfer
  M5d 12-base → 0.95023 / 0.94963 / 74%
  M5h 13-base → 0.95043 / 0.94991 / 24%

We confirmed the LB ceiling at 0.94991 by submitting THREE structurally
different M5h variants — all tied at 0.94991:
  M5h        K=13          (LB 0.94991, gap −5.2bp)
  M5h2 v1    K=12 (drop a_horizon)  (LB 0.94991)
  M5j swap   K=13 (d3a/d2a swap)    (LB 0.94991)

Post-hoc Spearman analysis showed all three pairs ρ ≥ 0.9997 vs M5h —
near-identical RANKING, hence identical LB. Friction logged
(`tag: pre-submit-rank-diff-check`) and pre-submit diff helper added
(`scripts/pre_submit_diff.py`).

### 2. The pool's "diversity" is a 3-source illusion

Pool disagreement diagnostic (`audit/2026-05-04-d3-pool-disagreement.md`):

| Base | Spearman ρ vs mean of others |
|---|---:|
| a_horizon | 0.729 (most diverse) |
| b_lapsuntilpit | 0.734 |
| cb_slow-wide-bag | 0.840 |
| (10 others) | 0.88 – 0.92 |

Mean |p − consensus| on top-decile-disagreement test rows:
  a_horizon       0.530   ← huge dissent
  b_lapsuntilpit  0.349
  cb_slow-wide-bag 0.344
  (others)        ≤ 0.04   ← consensus clones

**The 13-base pool is effectively 3 sources of diversity + 10 GBDT
consensus clones.** This is why marginal in-pool tweaks (M5h2, M5j)
can't move LB — the rank is locked by the consensus.

### 3. Stint 2 is a SHARED BLIND SPOT

Per-segment OOF showed Stint 2 (post-first-pit, 30% of data) is the
worst large segment (M5h AUC 0.91631, −341bp from aggregate). The
disagreement diagnostic showed Stint 2 has the LOWEST mean pool std
(0.112). Combined: the entire GBDT pool is uniformly wrong on Stint 2
— a feature/interaction blind spot, not a model-variance issue.

### 4. Calibration is fine globally; per-group is overfit

  - Reliability bins: M5h is well-calibrated (gap ≤0.003 across deciles)
  - Per-(Year, Race) isotonic: in-sample +24.6bp, **inner-CV −10.9bp**
  - Per-Race isotonic: in-sample +11.8bp, inner-CV −5.3bp
  - Friction logged: `tag: posthoc-isotonic-overfits-OOF`.

### 5. Specialists fail; meta is optimal

Tested today as null:
  - Stint-2 specialist (single LGBM): −124bp on its own segment vs M5h.
    A 13-base stacker is strictly better on segment than a focused
    single-model.
  - Stint-2 LR-meta (re-stack): +0.6bp (within noise).
  - Hill-climb / LGBM-meta / L1-LR / geomean / mean-rank: ALL within
    fold-noise of M5h. **LR with [raw, rank, logit] is genuinely the
    right stacker for this pool.**

### 6. Tier-break L1 prune doesn't move LB

M5h2 v1 (drop a_horizon, lowest L1=0.154) tied M5h at LB 0.94991.
The "smaller pool → tighter LB transfer" hypothesis is falsified for
the 13→12 step.

### 7. Rule 2 violation cost ~50% of compute today

Skipped the 1-fold smoke probe before launching the full RealMLP
5-fold on Kaggle. Cost: ~3 hours of GPU time on a kernel that's
still running at 22:30 (started 19:46). Friction logged
(`tag: rule2-smoke-skip-realmlp-day3` — to be added).

## Pool weaknesses (deep)

### W1. Mechanism family monoculture
13 of 13 bases are gradient-boosted trees (LGBM, XGB, HGBC, CatBoost).
Same inductive bias: tree splits, axis-aligned regions, no global
linearity. 10 of 13 are essentially "trees of trees" with similar
hyperparams → consensus clones. ZERO non-tree models in the pool.

### W2. Categorical handling is the same across bases
All 13 bases handle Driver (887 levels) the same way (native cat or
TE). No base uses character/embedding-based representations. NN/EBM
would handle this differently.

### W3. No sequence-aware base
d3b explored sequence-FE (cum_pits, laps_since_last_pit, rolling-TE)
as features. But no sequence-aware MODEL (LSTM, attention, sliding
window). 97.4% of test has same-(Race, Driver) successors in test —
ungrasped structure.

### W4. Stint 2 blind spot is structural
All bases miss it the same way. Cause likely: a feature interaction
(e.g., "lap_since_last_pit × tyre_compound × race_position") that
GBDT trees can capture only with very deep interactions, but is
fundamentally additive in EBM or first-order in NN with embeddings.

### W5. The M5h LR meta over-emphasizes the consensus
LR meta on [raw, rank, logit] features × 13 bases = 39 features.
After fitting, the meta gives near-equal weight to the consensus
clones, drowning out the diverse bases (a_horizon's L1 is 0.154,
the lowest, despite being most diverse).

### W6. Probability calibration variance ignored
Different bases produce probabilities at different absolute scales
(e.g. cb_slow-wide-bag is more concentrated; e3_hgbc is more spread).
The LR meta's logit channel partially handles this but doesn't
correct the rank distortion.

## New hypotheses (for tomorrow's slots 9-10 and beyond)

### NH1. (Tonight) Minimal-orthogonal-basis stack as slot 9 candidate
Per the M5n sweep (just completed): a 3-4 base stack of [a_horizon,
b_lapsuntilpit, cb_slow-wide-bag, baseline] has Strat OOF 0.94808
(−23.5bp from M5h), but Spearman ρ=0.987 vs M5h test = STRUCTURALLY
different rank. Rank-shift mean 6260 rows. Submit-worthy because:
the 0.94991 LB is rank-locked by 10 redundant bases; a stack without
them produces a different ranking that may transfer to LB differently.
Expected LB outcome: anywhere in 0.946-0.951 range; the
information-per-slot is high.

### NH2. RealMLP / EBM / H1 / LR-FE as orthogonal-mechanism bases
All four are non-GBDT. Adding them to M5h pool (or to a minimal-basis
stack) could break the consensus. Pre-screen each by:
  - Spearman ρ vs M5h consensus (lower = more orthogonal)
  - Mean |new − M5h| on Stint 2 specifically (where the blind spot is)
The candidate that scores best on BOTH metrics is the slot-9 winner.

### NH3. (For Day 4) Sequence-aware base (LSTM / attention)
The (Race, Driver) lap-sequence structure is genuinely unexploited.
A small LSTM with Driver embedding (887 → 16 dim) + Compound +
LapNumber + per-lap features over the (Race, Driver) sequence could
capture state transitions that GBDTs can't. Kaggle GPU budget
permitting (Rule 13).

### NH4. (For Day 4) Per-blind-spot ensemble
M5h is uniformly wrong on Stint 2. Train a base SPECIFICALLY on
Stint 2 with Stint-2-aware features (TE on lap_since_last_pit ×
compound, rolling target rate). At inference, blend with M5h on a
weighted basis where the weight is high specifically on Stint=2 rows.
The d3c specialist failed because it was a SINGLE LGBM; an HGBC or
GA²M (EBM) specialist might succeed.

### NH5. Probability calibration via base-conditional rescaling
Per-base, fit isotonic on OOF (NOT inner-CV — the per-base calibration
isn't the overfit; the per-Race calibration was). Then re-stack the
calibrated bases. Each base's logit is then on a normalized scale,
and the LR meta's logit channel is more informative.

### NH6. Larger-pool pseudo-label refit
H1 retrained baseline_lgbm on augmented train. If H1 lifts (early
signal: +28bp on fold 0), repeat for ALL 13 bases — re-train the
entire pool on augmented train. Expected lift propagates through
the whole stack, not just one base.

### NH7. Stochastic blend
Average M5h's test predictions with multiple slightly-different
predictions (random base seed, random fold seed, random LR meta
seed) — bagging at the meta level. Cheap variance reduction; doesn't
help OOF much but might tighten LB.

## Slots-and-budget snapshot

  - 8/10 used today. LB best 0.94991 (4 different submissions).
  - Slots 9-10 reserved for tomorrow.
  - Headroom to top-5%: 35.4bp.

## Prioritized Day-4 plan

1. **Land RealMLP / H1 / EBM / LR-FE artifacts** (overnight).
2. **Diversity pre-screen** each new base via Spearman vs M5h and
   |new − M5h| on Stint 2.
3. **Build stacks**:
   - M5o = M5h + winner base (likely RealMLP if it lands)
   - M5p = M5n_3b + winner base (minimal orthogonal + new family)
4. **Submit slot 9** = best of {M5o, M5p, M5n_3b standalone}.
5. **Submit slot 10** = R2 hedge or RealMLP-direct (if its
   standalone is competitive).

## Pointers

- `audit/2026-05-04-d3-per-segment-oof.md` — per-Race/Stint/Year/Compound
- `audit/2026-05-04-d3-per-segment-analysis.md` — synthesis
- `audit/2026-05-04-d3-calibration.md` — calibration variants (null)
- `audit/2026-05-04-d3b-seqfe.md` — Step 2 sequence-FE (null)
- `audit/2026-05-04-d3-pool-disagreement.md` — diversity diagnostic
- `audit/friction.md` — load-bearing failure modes; consult before
  every new probe.
- `scripts/pre_submit_diff.py` — MANDATORY pre-submit check.
