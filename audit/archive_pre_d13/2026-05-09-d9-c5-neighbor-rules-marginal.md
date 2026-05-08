# 2026-05-09 — Day-9 C5 neighbor-rule extension: MARGINAL (K=20 TIE)

> Rule 16 5-question pre-flight applied. Class: rule_residual-on-raw +
> NEIGHBOR features (prev_compound, next_compound) → at risk of T1.3-mode
> collapse but predicted to be mitigated by genuinely orthogonal signal
> (neighbor features NOT in M5q's 14 raw cols). Verdict confirms the risk.

## Results

| Rule | Coverage | Std OOF | ρ vs M5q test | min-meta Δ M5q | K=20 L1 |
|---|---:|---:|---:|---:|---:|
| C5.1 prev_comp × Compound × stint_bucket | train 49.4% / test 49.1% | 0.94475 | 0.90896 | +0.19bp PASS ✓ | 0.297 |
| C5.2 next_comp × Compound | train 84.4% / test 84.6% | 0.94432 | **0.88841** | +0.37bp PASS ✓ | **0.599** (#5) |

K=20 stack vs K=18 PRIMARY anchor:
| Metric | K=18 anchor | K=20 (M5q + 4 d6 + 2 c5) | Δ | Gate |
|---|---:|---:|---:|---|
| Strat OOF | 0.95065 | 0.95065 | +0.04bp | **FAIL** (need +0.5bp) |
| ρ vs K=18 test | 1.0 | 0.99985 | n/a | **FAIL** (need <0.999) |
| pred-LB (anchor 0.95026) | 0.95026 | 0.95026 | 0bp | FAIL (need +0.5bp) |

**Verdict**: MARGINAL. Both rules pass minimal-meta with genuinely orthogonal
ρ but stack OOF ties anchor and ρ is in the tie-locked region.

## Mechanism analysis — load-bearing extension of Day-8 thesis

C5 was the cleanest test of "rule with signal NOT derivable from M5q's
raw features survives the LR-meta info ceiling". Verdict: **it does not**.

- **next_compound at ρ=0.888 is the MOST diverse base ever measured**
  in this comp (vs RealMLP 0.918, F1.2 R3 driver_compound 0.891).
- It earned **L1=0.599 (5th of 20)** in the meta — the LR meta values it
  meaningfully and routes weight to it.
- Despite both, **K=20 OOF tied K=18 to within 0.04bp**.

**Why**: the signal carried by (next_compound × Compound) is statistically
**redundant with d6 rule_driver_compound** (L1=0.891, top) which already
encodes per-Driver pit propensity. Drivers cluster strategy choices, so
the next_compound for a given (Driver, Race, Year, Stint) is
quasi-deterministic given Driver. The 0.37bp min-meta lift vs M5q
disappears when stacked alongside rule_driver_compound because the meta
already extracts that information through Driver-cohort weighting.

**Implication for the wider-step menu**: the d3-endgame thesis sharpens
again. Two new failure modes documented today:

1. **Orthogonality of the feature** (low ρ to M5q test) is necessary
   but **NOT sufficient**. The new feature must also carry signal not
   already extracted by ANY other rule in the stack. d6's 4 rules
   collectively saturate the pairwise + driver + year-race subspaces;
   anything inside is redundant.
2. **L1 weight is not OOF lift.** A new base getting top-5 L1 weight
   tells you the meta is rank-shifting toward it. But if the source
   information is already in another base, the rank shift is purely
   reallocational — total OOF stays put.

## Sub-1bp insights worth keeping

- prev_compound coverage is 49% (matches Stint ≥ 2 share); next_compound
  is **84%** (much higher than P5's 68% test estimate — counts train+test
  and ignores Year-overlap, which is actually fine since Compound is
  observed in test → leak-free lookup).
- prev_comp_stint_bucket rule-only AUC ≈ 0.778 (decent rule signal),
  next_comp_compound rule-only AUC ≈ 0.700.
- Total wall: 333s CPU (validates F1.2 template ports cheaply).

## Action items

- **Do NOT submit** `submission_d9_k20_neighbor.csv`. K=20 ties K=18
  anchor at ρ=0.99985 (would burn a slot for zero LB lift).
- **Hold artifacts** — `oof_d9_c5_*` and `oof_d9_k20_neighbor_strat.npy`
  saved for future stack experiments.
- **Pivot priorities** post-C5:
  1. **C1 SC-probability** (cross-Race generalization mechanism class —
     genuinely different from per-row neighbor features). Predict NULL
     by analogy unless the SC-prob feature induces cross-Race
     information sharing the GBDTs cannot capture from Race-as-categorical.
  2. **GPU**: TabM smoke (v2 pushed after pytabkit constructor fix; v1
     errored on `use_ls` kwarg). PROMOTE-or-HOLD gate is the next
     binary signal. Different model class is the only path that has a
     non-trivial chance of breaking the 18-pool ceiling.
  3. **Rule 16 update**: add a 6th question — "predicted residual
     ρ-orthogonality with EVERY existing base in the stack, not just
     PRIMARY". C5 passed Q4 (ρ vs PRIMARY) but failed the implicit
     Q4-bis (ρ vs every d6 rule). Codify it.

## Artifacts

- `scripts/d9_c5_neighbor_rules.py` — builder
- `scripts/artifacts/d9_c5_neighbor_results.json` — numerical detail
- `scripts/artifacts/oof_d9_c5_prev_comp_stint_bucket_strat.npy`,
  `scripts/artifacts/oof_d9_c5_next_comp_compound_strat.npy`,
  `scripts/artifacts/oof_d9_k20_neighbor_strat.npy` (+ test_*)
- `submissions/submission_d9_k20_neighbor.csv` — **HELD, do not submit**
