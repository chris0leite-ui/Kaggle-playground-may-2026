# 2026-05-12 — D12 Year-segmented specialist + adversarial-validation reweighting

Branch: `claude/review-major-improvements-NUhht`. No commits, no submits.
Pipeline: `scripts/d12_year_specialist_advweight.py`. Results JSON:
`scripts/artifacts/d12_year_specialist_advweight_results.json`.
Total wall: 1123s (5-fold each: Year-specialist 718s, AV-LR 17s,
e3_advweight 132s, year_advweight 97s, Part D stacks 159s; OMP_NUM_THREADS=2
to coexist with sibling agents).

## Hypothesis recap

P3 (`audit/2026-05-08-data-probe-results.md`): Year=2023 has 0.96% pit
rate vs ~28% in 2022/2024/2025; 31% of train+test are 2023. Hypothesis:
two specialists routed by Year compound a -45bp Year=2023 segment lift
to +3-10bp aggregate. P9: low-count drivers (~221) pit_mean ≈ 2023
base rate ⇒ likely 2023-only synthetics.

Adversarial-validation reweighting is an orthogonal angle: importance
weights `w_i = clip(p_test/(1-p_test), 0.1, 10)` realign the train
objective toward test marginal.

## Methodology

- Base learner: HGBC label-encoded Driver, native cat Compound/Race
  (mirrors `scripts/e3_hgbc_two_anchor.py`).
- 5-fold StratifiedKFold(seed=42), pinned anchor.
- Part A: M_active (Year ≠ 2023) + M_2023; route on val/test by
  `Year==2023`. Within each fold's tr indices, partition by Year and
  fit two specialists.
- Part B: AV classifier = LGBM 5-fold OOF on `is_test` over
  concat(train+test); compute `w` for train rows; retrain HGBC with
  `sample_weight=w`.
- Part C: Year-specialists trained with `sample_weight=w` per cohort.
- Part D: standalone OOF, Spearman ρ-vs-PRIMARY-test (and vs d9f K21
  LB-best test), min-meta gate (3-feat LR over {PRIMARY, candidate,
  |Δ|}, 5-fold OOF AUC), and K=22-add stack on the d9c_kn_stack pool.
- PRIMARY surrogate: `d9c_Sd_K20_swap_FM` (OOF=0.95070, LB=0.95029).
  `d9f_K21_swap` LB-best (0.95031) has only test array; ρ-vs-LB-best
  also reported.

## Results

### AV classifier — train/test are nearly indistinguishable

> **AV OOF AUC: 0.50191** (per-fold 0.50096–0.50378). Random would be
> 0.50; this is +1.9bp above noise. Train and test draw from the same
> joint distribution. Weight summary: mean=0.428, median=0.429,
> min=0.307, max=0.675, **0% clipped** at either bound. Weights live
> in a narrow [0.31, 0.67] band — far from the [0.1, 10] clip range —
> so they barely re-weight the train loss.

**Implication:** AV reweighting has no signal to exploit. The train
loss with `w` is essentially the train loss × 0.43, which is
equivalent to no reweighting up to a constant. The −5bp standalone
OOF drop in `e3_advweight` vs unweighted e3 (0.94878 vs 0.94870
documented in CLAUDE.md ladder) confirms it is a slight regularization
hit, not a domain-shift correction.

### Per-Year OOF AUC: routing HURTS the 2023 segment

| model | 2022 | 2023 | 2024 | 2025 | overall |
|---|---:|---:|---:|---:|---:|
| m5q | 0.91438 | **0.94609** | 0.92870 | 0.92897 | 0.95057 |
| PRIMARY (d9c_Sd) | 0.91456 | **0.94602** | 0.92892 | 0.92915 | 0.95070 |
| year_specialist | 0.91159 | **0.93556** | 0.92660 | 0.92646 | 0.94872 |
| e3_advweight    | 0.91148 | **0.93774** | 0.92664 | 0.92637 | 0.94878 |
| year_advweight  | 0.91147 | **0.93745** | 0.92641 | 0.92624 | 0.94863 |

**Surprising falsification.** The pool already extracts ≥0.94602 AUC
on Year=2023 — the 2023 cohort isn't the ROI source we hypothesised
from P3. Splitting train INTO 2023 vs non-2023 cohorts strips
cross-Year signal from M_2023 (cross-Year effects mediated by
TyreLife / Race / Position / Compound that are *not* 2023-specific
suddenly aren't visible to it). The 2023 specialist drops to 0.93556
— **−104.7bp** below PRIMARY on its own segment. Other Years also
regress because M_active loses the 31% of training rows that share
weather/compound/track-mix patterns with 2022/24/25 in a way the
single-model handled implicitly via Year-aware splits.

P3's "−45bp Year=2023 segment lift" was apparently mis-read — the
single-model 2023 segment AUC was already ABOVE the overall AUC
(0.94602 > 0.91438 for 2022), not below it. Year=2023 is structurally
**easier** to predict (near-zero base rate makes most rows trivially
"no pit"), not harder. The mode-collapse is real but the pool's Year
+ Compound + Race interactions handle it adequately.

### Standalone + ρ + min-meta gate + K=22 stack

| candidate | std OOF | ρ_test PRIMARY | ρ_test LB-best | min-meta Δ | K22 OOF | K22 pred-LB | ΔLB |
|---|---:|---:|---:|---:|---:|---:|---:|
| year_specialist | 0.94872 | 0.98385 | 0.98377 | **−4.54bp** | 0.95074 | 0.95033 | +0.35bp |
| e3_advweight    | 0.94878 | 0.99220 | 0.99159 | **−4.92bp** | 0.95070 | 0.95029 | −0.02bp |
| year_advweight  | 0.94863 | 0.98368 | 0.98360 | **−4.96bp** | 0.95072 | 0.95031 | +0.22bp |

Reference K=21 PRIMARY (no add): OOF=0.95069, ρ=0.99996, predLB=0.95028
(Δ −0.09bp baseline jitter — LR-meta refit reproduces PRIMARY within
a hair).

### Min-meta gate: ALL FAIL

All 3 candidates regress on the min-meta gate (3-feat LR over
{PRIMARY OOF, candidate OOF, |Δ|}) by 4.5–5.0bp vs PRIMARY OOF. Per
the d9 cohort precedent, min-meta failure means the candidate adds
no information PRIMARY doesn't already encode at min-feature meta.
Year-specialist is the lowest gate failure (−4.54bp) but still
clearly negative.

### K=22 stack-add with full LR meta is mildly positive for year_specialist

The full K=22 LR meta with rank+logit expansion can recover +0.35bp
OOF over PRIMARY by adding `year_specialist`. ρ_test=0.99955 ⇒
pred-LB +0.35bp = 0.95033, slightly above PRIMARY-LB 0.95029 (and
0.02bp above the d9f K=21 LB-best 0.95031). However:

- **Min-meta gate fails by −4.54bp.** Per CLAUDE.md d9c lessons (FM
  passed +0.18bp min-meta → +3bp LB; R5–R10 rules failed by −0.09 to
  −0.12bp → 0bp LB tie or regress), candidates that fail min-meta
  more than ~0.5bp negative do not consistently survive LB.
- **L1 ranking (K=22 with year_specialist):** rule_driver_compound
  0.898, e5_optuna_lgbm 0.674, realmlp 0.582, rule_year_race 0.572,
  b_lapsuntilpit 0.533 — `year_specialist` does not appear in top-5;
  it is being weighted as filler and the +0.35bp comes from the
  refit-jitter freedom of one extra column rather than new signal.
- **OOF→LB amplification:** historic d9c FM had +0.18bp min-meta → +3bp
  LB (×16.7 amplification, real signal). year_specialist has −4.54bp
  min-meta but +0.35bp K=22; an honest predicted ΔLB after
  min-meta-discount would be **negative**. Not submit-ready.

## Verdict

**ALL 3 CANDIDATES DEAD-LIST.** None passes min-meta. The K=22 +0.35bp
for year_specialist is a meta-refit artifact, not a real lift signal.

| candidate | submit-ready? | rationale |
|---|---|---|
| year_specialist | NO | min-meta −4.54bp; per-Year shows specialists strip cross-Year signal; std-alone −19.8bp |
| e3_advweight | NO | AV AUC 0.502 (no shift signal); weights ∈ [0.31,0.67]; min-meta −4.92bp; K=22 −0.02bp |
| year_advweight | NO | both Year-split AND AV defects compounded; min-meta −4.96bp |

### Falsification chain (for the running log)

1. **AV signal ≈ 0.** The s6e5 train/test split is i.i.d. — confirmed
   independently by U3 (CLAUDE.md state block) and now AV AUC 0.502.
   AV reweighting has no domain shift to correct.
2. **Year=2023 is not a hard segment.** P3 stated "structural mode
   collapse" but the pool's per-Year OOF on 2023 is **0.94602** (the
   *highest* of any Year cohort). 2023 is *easier* (near-zero base
   rate ⇒ sparse positive class ⇒ AUC ranking is lopsided in favor
   of the model). The proposed `−45bp Year=2023 segment lift` is
   not in the data.
3. **Specialist routing destroys cross-Year signal.** Splitting tr
   into 2023 / non-2023 cohorts removes the cross-Year regularization
   benefit of the unified model. M_2023's 0.93556 segment AUC is
   **−104.7bp** below the unified PRIMARY's 0.94602 on the same
   segment.

The hypothesis "2023 is a structural mode collapse that warrants a
specialist" is **falsified at the segment level**. The 2023 anomaly
is real (low base rate) but the existing pool's Year × Compound × Race
interactions already capture it sufficiently.

## Cost & coexistence

- 1123s wall on a 4-core box shared with 5–6 sibling agents at
  OMP_NUM_THREADS=2. CPU-only, no GPU.
- No submit (per task instructions).
- 6 OOF/test arrays + 1 results JSON saved.

## Pointers

- Hypothesis origin: `audit/2026-05-08-data-probe-results.md` (P3, P9)
- Pool reference: `scripts/d9c_kn_stack.py` (K=21 Sa = K=22 with
  `year_specialist` minus the candidate)
- Per-segment OOF precedent: `audit/2026-05-04-d3-per-segment-analysis.md`
- d9c FM precedent (real model-class lift, +0.18bp min-meta →
  +3bp LB): `audit/2026-05-09-d9c-fm.md`
- d9 simple-math residual cohort precedent (all fail min-meta, all
  regress LB): `audit/2026-05-09-d9-math-heuristics.md`

## Recommendation

Do not submit any of {year_specialist, e3_advweight, year_advweight}.
Mechanism family `year_segmented_specialist` and
`adversarial_validation_reweight` should be added to
`mechanism_families_explored` as **falsified** so future agents
don't re-attempt them under the 5-question pre-flight check
(Rule 16). The 2023 anomaly is genuine but does not warrant cohort
splits at this base-pool quality.
