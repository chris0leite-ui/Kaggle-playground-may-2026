# 2026-05-06 — "do all" 4 probes session

PI: "wrap up findings, merge to main, then do all". Wrap-up + merge
done. Then ran 4 queued probes — all NULL. Net: 0 new LB advances,
1 falsification per probe (4 hypotheses ruled out).

## Probe results

| # | Probe | Cost | Standalone OOF | K=21+1 Δ at meta gate | ρ vs PRIMARY | Verdict |
|---|---|---:|---:|---:|---:|---|
| 1 | TE fold-leak audit on d2a/d3a | 8 min (read-only) | — | — | — | **CLEAN** |
| 2 | α-calibrated τ-resweep | ~6 min | n/a (PRIMARY recompute) | best τ=20k tied at OOF 0.95083 | 1.0 (identical) | **NULL** |
| 3 | Within-Race-Year quantile of LapTime_Delta | ~80 s | 0.94008 (−107 bp) | +0.20 bp | 0.9957 | **NULL/marginal** |
| 4 | Year×Stint sparse-LR | ~30 s + min-meta | 0.88164 (−692 bp) | +0.05 bp | 0.9958 | **NULL/marginal** |

## What each probe ruled out

**Probe 1 (TE fold-leak):** d2a + d3a both implement OOF discipline
correctly (per-fold `y[tr_idx]` only; inner-KFold for outer-train
TE; LapBin edges from RaceProgress = target-independent feature).
Hypothesis "silent fold-leak in TE bases inflates Strat OOF" RULED OUT.

**Probe 2 (α-calibrated τ-resweep):** Yesterday's audit-agent claim
that the per-fold-vs-full-train α asymmetry creates a sub-optimal
τ choice is FALSIFIED. Recomputed OOF using full-train counts at
α produces predictions IDENTICAL to PRIMARY at τ=20k (ρ=1.0). At
smaller τ values the "fix" actually regresses (amplifies per-segment
LR noise). Conclusion: d13e's τ=20000 is genuinely optimal.

**Probe 3 (within-Race LT_Δ quantile):** EDA Phase F showed
LapTime_Delta has +922 bp Strat→GroupKF single-feature leak.
Hypothesis: per-(Race, Year) quantile-rank normalization removes
the leak and creates a leak-robust signal addition. Result: at
the K=21 meta gate, the new feature lifts only +0.20 bp. The K=21
pool ALREADY absorbs the LapTime_Delta signal richly (via GBDTs,
FM, hier-meta); the de-leaked variant is mostly redundant. The
+922 bp leakage signal IS real but it's PUBLIC-LB load-bearing
(public is row-iid per U3) — removing it doesn't help public LB.

**Probe 4 (Year×Stint sparse-LR):** Round-2 critic identified
Year×Stint as the strongest FM field-pair magnitude (0.386). The
cohort axis was already known dead (d14 Path B Year×Stint NULL).
Hypothesis: the FEATURE axis (sparse-LR row-level interaction)
might still be alive. Result: standalone OOF 0.88 (very weak),
but ρ=0.844 (most-diverse base measured this session). At meta
gate, +0.05 bp — the LR meta gives the base modest weight but
the signal is largely redundant with K=21's existing pool.

## Cross-probe pattern

Three of the four probes share the same NULL pattern: the K=21
pool's outer LR meta has high absorption capacity for any single-
base addition that doesn't carry **structurally orthogonal** signal.
The friction tag from this morning's K=22 Path B failure
(`path-b-amp-needs-orthogonal-signal-not-meta-derivatives`) extends
here: even at the OUTER-LR layer (not just hier-meta), single-base
additions need orthogonal mechanism class to clear the gate.

What "orthogonal" means in this context (now empirically calibrated):
- ρ alone is NOT sufficient (Year×Stint sparse-LR ρ=0.844 was
  most-diverse measured this session, but min-meta added only
  +0.05 bp).
- The base must carry signal that CANNOT be reproduced as a convex
  combination of existing K=21 base predictions.

## State after session

- **PRIMARY unchanged:** d13e_compound_stint_tau20000 LB **0.95049**
- **Submissions used today:** 1 (K=22 Path B τ=100k → LB 0.95045)
- **Submissions used total:** 25/270
- **HEDGE candidates** (ρ<0.999, OOF within ~30 bp of PRIMARY):
  - `submission_path_b_K22_d12meta_tau100000.csv` — LB 0.95045
    (R7 eligible: −4 bp regress, flips 188 < 200)
  - `d12_lr_meta` standalone (held; OOF 0.95073)
  - `d13e_compound_stint_tau100000` (held; OOF 0.95081, ρ=0.999)
  - `path_b_K22_d12meta_tau20000.csv` (held; OOF 0.95092 but
    failed amp on LB)

## Implications for further probes

The "single-base addition" lever family is now well-mapped:
4-of-4 NULL (Day-13/14 alt-axis G1/G2'/G3/H1), 4-of-4 NULL today
(d6_rule_compound_stint, d10d, blend_rank_mean, within_race_lt_q5,
year_stint_sparse_lr — though to be precise within_race and
year_stint were marginal positive but well below useful threshold).

Lever families now considered exhausted within the K=21 + outer-
LR-meta + Path B hier-meta architecture:
- Single-base FE additions (LGBM/FM/sparse-LR class)
- Meta-derivative-as-base 2-level stacking
- α-calibration of hier-meta (no τ optimum shift)
- TE fold-leak source (no leak)

What remains (per CLAUDE.md `mechanism_families_explored`):
- Genuinely new model classes (TabPFN parallel branch unpromising;
  multi-task NN deferred per cost-EV; SCARF/contrastive untouched)
- External data integration (aadigupta failed at row-join; Pirelli
  scrape attempted at d12, status unknown — needs re-check)
- New target reformulation (already explored — d12 t12 family failed)
- Specialist meta-architectures (Driver-cluster Path B untested)

**For top-1% private:** the empirical hit rate of ~17% (now
4/27 = 15% across two sessions) is the binding constraint.
Breadth-first probing is correctly calibrated; the path forward
is more probes from genuinely-different families, not deeper
exploration of the K=21+meta architecture.
