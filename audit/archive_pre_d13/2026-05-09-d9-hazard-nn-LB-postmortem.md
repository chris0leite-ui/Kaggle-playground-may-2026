# 2026-05-09 — Day-9 hazard-NN K=19 LB POSTMORTEM (-73.5bp OOF→LB gap)

> Submitted `submission_d9_k19_hazard_nn_stack.csv`. **LB 0.94711.**
> Predicted LB 0.95367. Strat OOF 0.95446. Actual gap **−73.5bp** vs the
> calibrated −5bp pool gap. K=18 PRIMARY (LB 0.95026) **unchanged**;
> this submission REGRESSES PRIMARY by −31.5bp on public LB.
>
> This is a Rule 14 strategy-critic event (gap drift = 36× threshold).

## What happened (data-first)

| Metric | K=18 PRIMARY anchor | K=19 hazard-NN stack |
|---|---:|---:|
| Strat OOF | 0.95065 | **0.95446** (+38.1bp) |
| ρ vs M5q test | 0.99902 | 0.95941 |
| ρ vs K=18 test | 1.0 | 0.96029 |
| pred-LB (calibrated) | n/a | 0.95367 (+3.4bp vs K=18 LB) |
| **Actual LB** | **0.95026** | **0.94711 (−31.5bp vs K=18 LB)** |
| OOF→LB gap | −3.9bp | **−73.5bp** |

The `pred_lb` calibration scheme used a single ρ-bracketed offset from
prior submits (mostly ρ ≥ 0.99 stacks). At ρ = 0.96029 the offset was
extrapolated to −0.0004 (40bp). Actual offset: −0.0079. **The
calibration was roughly 20× too small in magnitude at this ρ regime.**

## Why the gap was so large — load-bearing finding

The hazard-target construction creates **multi-row label leakage** that
the existing pool only contains in much weaker form.

For a stint S with PitNextLap=1 at lap L, the hazard target for any row
r at lap r ≤ L is `bucket = L − r`. Per P6 of the data probes, **80%
of consecutive-lap pairs in the same (Race,Driver,Year,Stint) land in
DIFFERENT folds** under our StratifiedKFold(5, seed=42). So within
each fold:

- Training rows in stint S "encode" the timing of stint S's pit event
  (their hazard buckets are `L − r_train`).
- Val rows in stint S have their PitNextLap labels at lap L_val.
- The model trained on (encoded train hazard targets) learns
  "row at lap r with these features → predict pit at lap r + bucket".
- At val, the model's bucket-0 head fires correctly because the **same
  L was already encoded in training** (via train-row hazard buckets).

The existing pool's reformulation bases (`a_horizon`, `b_lapsuntilpit`)
have the same structural leak in principle, but:
- `a_horizon` shifts the binary target by a bounded horizon — minimal
  multi-row encoding;
- `b_lapsuntilpit` is a scalar regression target — one number per row,
  not 20 buckets.

The hazard NN's K=20 buckets per row carries 20× the multi-row label
information per (Race,Driver,Year,Stint) group. The `bfill` operation
that produces the hazard target literally propagates val rows' labels
backward to all earlier rows in the same stint.

## Why the K=18 pool was robust

K=18 = M5q (14 binary classifiers) + 4 d6 rule_residual bases. None of
the binary classifiers carries this multi-row leak. The 4 rules use
Bayesian-smoothed lookups whose keys (Compound × TyreLife,
Compound × Stint, Driver × Compound, Year × Race) do NOT depend on
within-stint future labels. Hence calibrated −3.9bp gap.

## Diagnosis confidence

- **Direction**: high. Multi-row label leakage at the hazard-target
  level is the dominant mechanism.
- **Magnitude (−73.5bp)**: high. The `bfill` operation directly leaks
  val-row PitNextLap into 80% of train rows (per P6).
- **Single-base hazard NN gap**: unknown (we did not submit
  `submission_d9_hazard_nn.csv`). Approximate via L1-weighted
  decomposition: stack share L1_haz/L1_total ≈ 0.32, so
  G_haz ≈ (−73.5 − 0.68 × −3.9) / 0.32 ≈ **−220bp**. If single-base
  hazard's true leak-free OOF is ~0.926 (vs measured 0.948), it
  would be below baseline and add zero to the stack.

## What survives

- M5q + d6 K=18 stack remains PRIMARY at LB 0.95026.
- Rule 16 stays. **Add Rule 16 Q7**: any base whose target/feature
  construction depends on FUTURE same-group labels has automatic
  ×0.1 EV downgrade unless trained on a leak-free fold scheme
  (GroupKFold by the stint group).
- The hazard NN architecture itself is fine; the FOLD STRATEGY for
  it is wrong. RealMLP-TD on binary target had no comparable issue
  because its target is the row's own PitNextLap, not a within-stint
  derived quantity.

## Action items for Day-10

1. **GroupKFold(Race,Driver,Year,Stint) hazard-NN re-run.** Run the
   exact same architecture/training, but with the outer CV grouping
   the entire stint into a single fold. This eliminates the
   bfill-based leak. Predicted leak-free OOF: ~0.93–0.94 (stripping
   the leak). Wall: same as current bag (~10 min T4).
2. **If GroupKFold OOF < 0.93**, the hazard NN is structurally
   redundant with the binary pool and is a dead lever. Pivot to:
   - **G4 SCARF/VIME pretrain on aadigupta1601 unlabeled** (different
     unlabeled corpus) — different inductive bias, no leak.
   - **TabM v3 with extended training** (the user's earlier question
     about under-training).
3. **Audit `b_lapsuntilpit` and `a_horizon`** the same way — quantify
   their leak-free OOF via GroupKFold. If they're materially
   over-credited too, the K=18 pool itself may have a couple bp of
   hidden leakage that's been masked by the matching gap structure.
4. **Update Rule 16 with Q7 (multi-row-label dependence test)** in
   CLAUDE.md.

## Submissions used

- 1/10 today. **9 slots remaining** before the 24h reset.
- Total comp: 15.

## Held / dead — DO NOT submit

- `submission_d9_k19_hazard_nn_stack.csv` — burned (LB 0.94711, dead
  weight). Do not resubmit.
- `submission_d9_hazard_nn.csv` (single-base hazard) — predict LB
  ~0.926 by L1 decomposition above. **Do not submit.**

## What this teaches us

The 8 prior nulls were all "rules don't lift the stack". Today's
result is qualitatively different: **a base that LIFTS the OOF
substantially can carry catastrophic OOF→LB gap if its target
construction propagates val-row labels into train-row supervision.**
The d3-endgame ceiling we documented (Day-7/8) was a CEILING ON OOF.
There's a separate, sharper ceiling on TRUSTABLE OOF lift — gated by
target-construction leak-freeness.

The framework's protections:
- Pre-submit-diff ρ check (PASSED, ρ=0.96): only catches duplicate
  submissions, NOT leakage.
- Min-meta + K=N+1 OOF gates (PASSED): only sensitive to LR-meta
  rank-shift, NOT to leak-induced OOF inflation.
- **The MISSING gate is**: does the candidate's target construction
  depend on within-fold-group labels? Rule 16 Q7 fixes this prospectively.

End of postmortem. Sober. Day-9 closes at 1 submission used,
0 LB lift, 1 high-information failure mode discovered.
