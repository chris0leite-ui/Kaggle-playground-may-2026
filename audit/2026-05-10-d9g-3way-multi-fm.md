# Day-9g — 3-way multi-FM partition (REGRESSION vs d9f 2-way)

> Hypothesis: extending d9f's 2-way partition (FM_A driver-dynamics +
> FM_B race-context) to 3 partitions yields more orthogonal slots
> for the LR meta. Falsified — finer partition over-fragments the
> per-FM interaction surface.

## 3-way partition design

Domain-grounded split of 8 main-effect features into 3 disjoint sets:

- **FM_α "driver"**: D, C, S (driver + compound + stint, 3 cats, 3 pairs)
- **FM_β "race"**: R, Y (venue + year, 2 cats, 1 pair)
- **FM_γ "state"**: T_q5, Rp_q5, P_q5 (3 quintile bins, 3 pairs)

Same FM hyperparameters as d9c/d9f (k=8, 6 epochs, batch 8192, lr 0.05).

## Standalone — extreme per-base diversity but per-base too weak

| Model | Std OOF | ρ vs d9f PRIMARY | Notes |
|---|---:|---:|---|
| FM_α (D, C, S) | 0.79307 | **0.43537** | most-diverse cat-only |
| FM_β (R, Y) | 0.79174 | 0.71285 | only 1 pair → near-LR |
| FM_γ (T_q5, Rp_q5, P_q5) | 0.73921 | **0.33654** | **lowest single-base ρ in entire project** |
| (d9f FM_A: D,C,S,T) | 0.82505 | 0.487 (vs d9c) | 4 feats, 6 pairs |
| (d9f FM_B: R,Y,Rp,P) | 0.88438 | 0.861 (vs d9c) | 4 feats, 6 pairs |
| (d9c FM: 8 feats) | 0.92069 | 0.899 (vs d6 PRIMARY) | 8 feats, 28 pairs |

Pairwise ρ among 3-way partitions:
- α-β = **0.180** (almost orthogonal)
- α-γ = 0.461 (moderately correlated)
- β-γ = **−0.036** (essentially statistically independent — even slight anti-correlation)

The diversity numbers are extraordinary. FM_γ has ρ=0.337 vs PRIMARY,
which is by far the lowest single-base ρ in the entire project (R14
held the prior record at 0.444; FM_A in d9f was 0.487). β-γ are
essentially statistically independent. *In any sane diversity-driven
ensemble heuristic, these would be slot-worthy bases.*

## K=N stack experiments — all regressions

PRIMARY = d9f K=21 swap+multi-FM (Strat OOF 0.95073, LB 0.95031).
Pool keep = 16 + R6/R10/R7 (3) = 19 base.

| Stack | K | Δ d9f OOF | ρ | What's added |
|---|---:|---:|---:|---|
| S1 K=22 swap | 22 | **−0.46bp** | 0.99981 | drop d9f's FM_A+FM_B, add α+β+γ |
| S2 K=24 add | 24 | **−0.09bp** | 0.99995 | keep d9f FM_A+FM_B, add α+β+γ |
| S3 K=25 add all FMs | 25 | **−0.13bp** | 0.99993 | + d9c FM as well |

All 3 stacks are **worse than d9f K=21 PRIMARY**. The most
informative result is S1 K=22 swap (−0.46bp): even with same total
feature coverage as d9f, splitting into 3 weaker FMs lifts no signal.

L1 ranking diagnostics:
- **S1 (K=22 swap)**: zero d9g FMs in L1 top-15. The LR meta gives
  the entire α+β+γ trio combined L1 weight ≪ FM_A_d9f's solo weight.
- **S2 (K=24 add)**: only FM_β (L1=0.410) enters top-15. FM_A_d9f
  remains at L1=0.520. FM_α and FM_γ are demoted below 0.4.
- **S3 (K=25 add all FMs)**: FM_β at L1=0.441 (top-15), FM_A_d9f
  at L1=0.439. d9c FM and other 3-way bases below 0.4.

## Why 3-way doesn't work despite extreme ρ-diversity

**Standalone strength matters more than diversity past a threshold.**

The d9c→d9f progression worked because:
- d9c (1 FM, 8 feat, 28 pairs): std OOF 0.921, ρ 0.899
- d9f (2 FMs, 4 feat each, 6 pairs each): std OOF 0.825/0.884, ρ 0.487/0.861
- → both d9f FMs were "strong enough to be heard" in the LR meta
  despite lower ρ.

d9g breaks this:
- 3 FMs with 2-3 features each, 1-3 pairs each, std OOF 0.74-0.79
- Each FM is "too weak to add" — even with extreme ρ-diversity, the
  LR meta finds the marginal information delta is not worth a slot.

**Mathematical reading**: the LR meta optimizes a weighted sum of
per-base predictions. A base contributes its information to the
ensemble proportional to *both* its standalone predictive power *and*
its diversity. The product `(2 × strength × diversity)` matters, not
either alone. d9g's bases have great diversity but their strength
falls below the threshold where the meta can extract anything useful.

The 2-way partition (d9f) is the **information-theoretic sweet spot**:
each partition has enough features (4) to learn meaningful low-rank
interactions while remaining ρ-diverse enough that the meta routes
through both.

## Triage decision

**HOLD all d9g stacks.** d9f K=21 swap remains PRIMARY (LB 0.95031).
Three-way partition is over-fragmented; diversity numbers misleading.

The d9f 2-way is now confirmed as the local optimum across:
- d9c unified FM (1 partition): replaced by d9f
- **d9f 2-way (current PRIMARY)**: best
- d9g 3-way: regression
- d9d FM hparam sweep + bag (1 partition tweaks): TIE
- d9e FFM (richer parameterization, 1 partition): regression

## What to test next (post-d9g)

The factorization-machine family at single-feature-set level is
exhausted. Real next-step probes need either:

1. **Feature-augmented FM** — add new features (lag features, prev/
   next compound, cumulative race lap) to the FM input. Tests
   whether unified FM with richer features beats partitioned FMs.
   ~5 min CPU per attempt.
2. **DeepFM-lite** (FM + 1-hidden MLP head sharing embeddings) —
   adds non-linearity over the FM space. ~1h CPU.
3. **External-data Pirelli pit-window scrape** — orthogonal new
   signal, highest absolute EV (per main-branch agent's Day-11
   strategy critique). 1 day engineering.

## Pointers

- `scripts/d9g_multi_fm_3way.py` — 3-way builder + S1/S2/S3 stacks.
- `scripts/artifacts/d9g_3way_results.json` — full metrics.
- `scripts/artifacts/oof_d9g_FM_alpha/beta/gamma_strat.npy` — base
  predictions.
- `submissions/submission_d9g_S1_K22_swap_3way.csv`,
  `submission_d9g_S2_K24_add_d9f_plus_3way.csv`,
  `submission_d9g_S3_K25_add_all_FMs.csv` — all HELD (worse than
  PRIMARY).
