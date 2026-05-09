# 2026-05-09 — Decode the data generating process (7-step plan, v2)

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-plan-v2`
`scope: planning-only (no compute spent this session)`
`builds on: 2026-05-09-decode-NN-7step-plan.md (v1) + 2026-05-08-DGP-FINAL-summary.md`

> PI directive: plan to decode the data-generating process. Reconstruct the
> training data from the original data. Understand the host's pipeline,
> not improve LB. Note useful latent spaces, encoders, decoders.
>
> Constraint relaxed (per AskUserQuestion 2026-05-09): aadigupta1601 is
> available as ground-truth input.

## TL;DR

Goal: learn the host's forward map `f: orig_row → synth_row` well enough
that we can either (a) **reproduce** the synth dataset from orig, or
(b) **invert** any synth row back to its likely orig source row and read
its label. Six durable findings from the prior campaign already pin
roughly half of `f`. The plan below sequences six phases (A–F) over an
unconstrained timeline that, in order, refine the fingerprint, fit a
forward surrogate, train an inverse encoder, run a membership-inference
sanity check, ensemble the label posterior, and validate against held-out
orig pairs we can construct **today** for free using the literal-copy
trick.

The realistic ceiling is bounded by two physics: PitNextLap is itself
probabilistic at ~81% concordance with the next-lap PitStop event (A33),
so even a perfect inversion tops out near AUC 0.997; and the host's CTGAN
is 0.9993-distinguishable from off-the-shelf CTGAN (F6), so the
forward-surrogate disc-AUC will not collapse to 0.5. Both ceilings are
acceptable — this is a research/understanding task.

## Step 1 — Problem definition (what does "decode" mean operationally)

`f` is a stochastic map from the original 99k-row aadigupta1601 dataset
(call it `O`) to the host's 627k-row synth dataset (`S = train ∪ test`).
Decoding `f` has three nested operational targets, in increasing
difficulty:

1. **Forward fidelity.** A surrogate `G' : O → S'` such that a
   discriminator cannot distinguish `S'` from `S` on per-feature KS,
   joint distribution, and label-conditional structure. This is the
   "reconstruct training data from original" reading the PI asked for.
2. **Inverse fidelity.** An encoder `E : S → ΔO` (a distribution over
   orig rows) that maps every synth row back to its likely source row
   (or a soft pre-image set).
3. **Label fidelity.** A function `ℓ : S → [0,1]` that uses `E` and the
   orig labels to predict synth.PitNextLap. Implicit in all the prior
   work; not the main decode but a clean private-LB sanity check.

The outputs ranked by PI request: 1 > 2 > 3.

## Step 2 — Success metrics and gates

For 1 (forward):
- per-feature Kolmogorov-Smirnov distance on the 11 numeric columns,
  median ≤ 0.02 (current host KS ≈ 0.0; off-the-shelf CTGAN KS ≈ 0.13)
- per-cell `(Race, Year, Compound)` distribution match: KS ≤ 0.05
- 2-class disc-AUC `S` vs `G'(O)` < 0.85 (start floor; F6 says default
  CTGAN is at 0.9993, so anything < 0.95 already proves we found a
  better-fitting family/config)
- label-conditional `P(x | PitStop)` KS ≤ 0.05 (d18 f5 found PitStop is
  in the host's cond-vector — surrogate must reproduce that asymmetry)
- discrete-vocab match: surrogate's driver vocabulary recreates the
  31-active / 100-retired-uniform / 756-D-prefix-ghost partition (F2)

For 2 (inverse):
- top-K hit rate on a held-out oracle pair set, K ∈ {1, 10, 100}.
  We can construct ~960 oracle pairs **today** for free using P1c's
  4-tuple literal-copy trick (no new compute).
- agreement between three independent inverse signals (F3 density,
  F4 MIA, F5 OT plan) above chance — orthogonal validation.

For 3 (label):
- standalone OOF AUC of `ℓ`; gate at +0.5 bp on K=4+1 LR-meta. Even a
  null on the current ensemble is informative because it tells us the
  rank-lock subspace already absorbs the inversion signal.

## Step 3 — What we know, what we don't (inventory)

### Known structure of `f` (from F1-F6 + d14-d18 + P1-P11)

- **Per-row generation, not joint.** Each synth row is sampled
  independently. Within-stint sequence structure is not preserved
  (F1, P1 Q4: only 0.7% of synth stint groups have consecutive
  LapNumber).
- **Conditioning on `(Race, Year, Compound)` and on `PitStop`.**
  The first three are preserved exactly; `PitStop` enters the cond
  vector with class-conditional distortion (d18 f5 KS asymmetry).
- **Continuous columns are near-literal copies of orig.** 97.55% of
  synth `LapTime` values lie in the orig empirical set (d15); a
  4-tuple match between two synth rows agrees on `PitNextLap` 94.82%
  of the time (P1c). Mechanism: CTGAN's mode-specific normalisation
  re-emits empirical values; per-row label inheritance from the
  source row.
- **Categorical labels (Driver, Stint) are independently re-assigned.**
  Stint loses temporal meaning (F1, friction
  `synth-stint-label-is-fabricated-not-temporal`). Driver vocab is a
  fabricated 887 = 31 active + 100 retired-but-uniform + 756 D-prefix
  ghosts (F2).
- **Source heterogeneity in 2023.** 2023's portion has 0.96% pit rate
  vs 26-30% other years (F4). Most likely source: practice or
  qualifying sessions where pit is rare. Implies `O` is itself a
  mixture; the surrogate must condition on a year-source latent.
- **CTGAN-class but heavily host-customised.** Off-the-shelf SDV CTGAN
  on synth is 0.9993-distinguishable (F6, P3). The custom delta is
  somewhere in {cond-vector schema, mode-count per feature, training
  schedule, post-hoc filtering}.
- **Quantization grid.** Lap-counter columns are integer; LapTime,
  LapTime_Delta, Cumulative_Degradation, RaceProgress are float (F5).
  Confirms mode-specific normalisation rather than a post-quantisation
  step.
- **No id-cluster batching.** id-adjacent rows are no closer than
  random pairs (P1 Q2). Falsifies inversion-via-id.

### Unknown components of `f`

- Exact CTGAN-class config: hidden dims, number of training epochs,
  pac (packing for mode-collapse defence), generator/discriminator LR
  schedule, embedding dims for categorical columns.
- Number of modes per continuous column. SDV's default is 10. The
  host might have used a much higher count (gives sharper literal
  copies) or a fitted BGMM with its own concentration prior.
- Cond-vector schema beyond `(Race, Year, Compound, PitStop)`. Driver
  may also be in there (or it would not produce the 31 + 100 + 756
  partition — needs explicit testing).
- Whether the host applied any rejection sampling, post-hoc pit-rate
  re-balancing, or a custom loss term.
- Whether the 2023 source is aadigupta1601 itself (a slice we haven't
  noticed) or a separate practice/quali dataset bolted on. d18 f5 plus
  a 2023-only KS scan against orig will resolve this.

## Step 4 — Alternatives generated

Five mechanism families. None are mutually exclusive; the plan in Step 6
combines several.

| Family | Mechanism | Best for |
|---|---|---|
| **F1 forward surrogate** | Train CTGAN-class generators on `O` with a grid over hyperparameters and cond schemas; minimise disc-AUC vs `S`. | Goal 1 directly. Required input for F2. |
| **F2 inverse encoder** | Neural encoder `E: synth → ΔO` trained on simulated `(O, G'(O))` pairs from F1 (we know the true preimage of `G'(O)`). Apply to host `S`. | Goal 2. Standard GAN-inversion methodology. |
| **F3 density-ratio** | For each synth row `x`, compute `P(orig_i | x) ∝ p(x | orig_i) p(orig_i)` using a fitted conditional density on orig and a learnt similarity. | Goal 2 / 3 with no surrogate dependence. Robust if F1 fails. |
| **F4 membership-inference shadow** | Train shadow generators on splits of `O`; learn a discriminator that predicts which orig row likely seeded each synth row. CTGAN MIA is well-formalised in 2024–2025 literature. | Independent validation of F2 / F3. |
| **F5 optimal transport** | Solve the OT problem matching the synth empirical to the orig empirical under a learnt cost; the transport plan `T(synth_i → orig_j)` is the soft preimage. Sinkhorn divergence with feature-importance weighting. | Goal 2, novel angle, no prior null. |

## Step 5 — Scoring (with research-informed corrections)

| Family | P(useful signal) | Cost | What it gets us | Critical risk |
|---|---:|---:|---|---|
| F1 forward surrogate | 80% | 5–20 GPU-d | new disc-AUC floor + sanity on cond-vector | floor likely > 0.7 even with grid |
| F2 inverse encoder | 60% | 10 GPU-d | per-row preimage distribution | depends on F1 quality |
| F3 density-ratio | 70% | 3 d | preimage without depending on F1 | 14-D conditional density is brittle in the tails |
| F4 MIA shadow | 60% | 10 GPU-d | scores, not labels — orthogonal sanity | needs many shadow generators |
| F5 optimal transport | 50% | 2 d | soft preimage + a clean math object | OT in 14-D is well-defined but dimension-heavy |

Three corrections lifted from v1 of the plan that still apply:
- **PyTorch nondeterminism** kills any plan that tries to reproduce the
  host generator by exact-seed-replay across hardware/version stacks.
- **A33 ceiling**: even a perfect inversion gives ~0.997, not 1.0.
- **Public-LB ±12 bp noise band** swallows lifts under ~13 bp. Validation
  must be private-LB-aware (or held-out orig-style).

## Step 6 — Sequenced execution

The plan stages each phase so its result strictly informs the next, and
so each gate is a small enough commitment that we can pivot.

### Phase A — Refine the fingerprint (1–2 days, mostly CPU)

Pre-condition for everything: confirm or correct the F1–F6 facts.
Specifically:
- Run **P13** (CTGAN-on-orig defaults) to completion under exclusive CPU,
  measure disc-AUC; this is the missing data point from the WRAPUP doc.
- Run **P16** (CTGAN-on-orig with pac=1 + larger hidden). If disc-AUC
  drops materially below P13's, the host axis we are missing is config.
- Run **P17** (per-feature KS) once the two replays exist; this isolates
  which feature distributions the host fits best.
- Add **P18** (new): does the disc-AUC asymmetry for `PitStop` (d18 f5)
  vanish if we add `PitStop` to the cond vector of P13's CTGAN? If yes,
  we've nailed the cond-vector schema empirically.
- Add **P19** (new): is 2023's anomalous source aadigupta1601 itself or a
  bolt-on? Train CTGAN on `O minus 2023` and on `O ∩ 2023` separately,
  and measure disc-AUC contribution to host's 2023 slice.

Gate: P18 PASS or FAIL is the critical bit for the surrogate grid below.

### Phase B — Surrogate grid (3–7 days, GPU)

Hold cond-vector schema fixed at the Phase A winner. Sweep over:
- model family ∈ {SDV CTGAN, CTAB-GAN+, TVAE, CopulaGAN,
  TabDDPM (diffusion), Neural Spline Flows / RealNVP}
- mode count per continuous column ∈ {10, 30, 100, BGMM-fitted}
- discrete embedding dim ∈ {16, 64, 128}
- pac ∈ {1, 4, 10}
- training epochs ∈ {50, 200, 500}

Score: 2-class disc-AUC vs `S` (lower is better) and 6-tuple concordance
fraction vs `S` (higher is better). Halt the sweep when disc-AUC < 0.85
or after the budget; record the winner as `G'`.

### Phase C — Inverse encoder (8–14 days, GPU)

Train an encoder `E_θ : synth_row → 32-D latent → softmax(orig_index)`
on simulated `(O, G'(O))` pairs. Architecture: small Transformer
(2-block, hidden 128) for categorical columns + MLP for continuous +
contrastive head with an InfoNCE loss against a 4096-orig-row negative
pool. Validate on the **literal-copy oracle pairs** we construct for
free today (~960 pairs from P1c's 4-tuple match; expand by relaxing the
tuple to top-K nearest in 7-KS-low subspace).

### Phase D — MIA shadow models (15–21 days, GPU)

Train K=10 shadow CTGAN replicas, each on a different 80% draw of `O`.
For each row in `O`, label as in/out per shadow. Train a discriminator
on `(orig_row, synth_row)` features that predicts in/out membership.
Apply to `S`: high in-membership posterior identifies likely source rows.

### Phase E — Label-posterior ensemble (22–28 days)

Combine F2 + F3 + F4 + F5 outputs into a single per-synth-row
`P(orig_source = i | synth_row)` distribution. Read the orig labels
under that posterior; that's our `ℓ(synth)`.

### Phase F — Validation and write-up (29+ days)

- Held-out orig pair set (the 960 oracle pairs); measure top-K hit rate.
- Held-out year (2025 only) to test temporal generalisation.
- Compare reconstructed-synth's per-cell pit rates against host `S`.
- One LB submission of `ℓ(synth)` blended at 5% with K=4 PRIMARY purely
  as a sanity probe (not the goal of the work).

Each phase's output is committed to `audit/2026-05-09/` and
`scripts/dgp_v3/` with a one-line summary in `state/mechanism-ledger.md`.

## Step 7 — Anticipated failure modes and contingencies

| Risk | P | Diagnostic | Pivot |
|---|---:|---|---|
| Surrogate floor stays > 0.95 across the grid | 30% | Phase B disc-AUC trajectory | Drop F1 dependence; lean on F3 density-ratio + F5 OT (neither needs `G'`). |
| Memorisation is sparse (host regularised CTGAN) | 30% | MIA score-distribution mass piles near 0.5 | Re-aim at population-level statistics rather than per-row preimage. |
| 2023 ambiguity (mixed-source `O`) | 40% | Phase A P19 result | Treat 2023 as a separate generator; fit two surrogates. |
| Driver-ghost columns are pure noise | 60% | Phase A driver-vocab match | Drop Driver from the inverse encoder's input; use only physics columns. |
| A33 ceiling absorbs all label lift | 100% (bounded) | Reach 0.997 OOF | Accept; the decode is still the goal. |
| Compute budget bites | 50% | Phase B GPU-day burn | Cut model family list to {CTGAN, TabDDPM, NSF}; skip ablations. |

## Appendix A — Latent spaces worth attention

- **(L1) Mode-specific normalisation latent.** CTGAN's per-feature mode
  index. With a fitted BGMM(10) we can recover each synth row's likely
  mode for each continuous column → invert the normalisation back to a
  raw value drawn from the source mode. Already half-done in
  `scripts/d18_g_mode_id_ctgan.py`.
- **(L2) Cell latent `(Race, Year, Compound)`.** 26 × 4 × 5 = 520 cells.
  Preserved exactly. The natural partition all of phase B should
  condition on.
- **(L3) Year-source latent {race, practice/quali}.** Inferred from F4.
  Binary, deterministic-ish (2023 → practice; others → race). Adding
  this as a ninth column to `O` likely sharpens any surrogate.
- **(L4) Driver-vocab 3-cluster latent {active, retired-uniform, ghost}.**
  Deterministic from the F2 partition. Cardinality 3.
- **(L5) Inferred-stint latent `stint_start_imputed = LapNumber − TyreLife
  + 1`.** Per-row coherent identifier of cardinality ~5119 (orig stints).
  The "real" stint label that the synth re-shuffle destroyed.
- **(L6) PitStop class-conditional latent.** d18 f5 says `PitStop` enters
  the cond vector. The CTGAN therefore has a PitStop-conditioned
  generator head; the latent is binary but its effect on the continuous
  columns is large.
- **(L7) 7-KS-low feature subspace.** {TyreLife, Position, LapTime,
  Cumulative_Degradation, RaceProgress, LapTime_Delta, LapNumber}. The
  preimage-kNN-friendly subspace (already +1.88 bp standalone via
  d18_e2). The natural metric space for OT.
- **(L8) Tuple-fingerprint latent.** The 4-tuple `(LapTime, LapTime_Delta,
  RaceProgress, Cumulative_Degradation)`. P1c shows tuple match → 95%
  label concordance. This is the highest-precision per-row identity we
  have today.

## Appendix B — Encoders and decoders worth building

Split by their role in the pipeline.

### Decoders (forward; orig → synth)

1. **CTGAN-on-orig surrogate** (`G'`). Phase B core. Required as the
   teacher for F2.
2. **TabDDPM (diffusion)** alternative. SOTA on tabular synthesis; good
   chance of beating CTGAN's disc-AUC floor.
3. **Conditional Neural Spline Flow** on `(Race, Year, Compound, PitStop)`
   conditioning. Invertible by construction → tractable density `p(x | c)`
   without a separate density estimator. Doubles as input to F3.
4. **Cycle-AE decoder half**. Round-trip identity loss `|orig − E(G'(orig))|`
   on the literal-copy oracle pairs gives free supervision.

### Encoders (inverse; synth → orig)

5. **Contrastive synth→orig encoder** (Phase C core). InfoNCE loss
   against a 4096-orig negative pool; 32-D latent.
6. **Tabular VAE** with synth on the input side. Squeeze synth into a
   32-D latent and search nearest orig in latent space. Cheap baseline
   for the contrastive encoder.
7. **Density-ratio classifier**. A 2-class LightGBM that learns
   `p(orig | x) / p(synth | x)`. Per-row score is the soft preimage
   weight.
8. **OT plan as encoder**. Sinkhorn solver over 7-KS-low subspace with
   Mahalanobis cost; outputs a sparse synth × orig coupling matrix.
9. **MIA discriminator** (Phase D core). Per-row in/out posterior for
   each shadow generator.
10. **Mode-id encoder.** Apply a fitted BGMM per continuous column to
    every synth row; the mode-vector is a discrete latent. Combined with
    L2 it gives a 7 × |modes| × 520 cell-table that the host's CTGAN
    implicitly defines.

## Pointers

- v1 plan: `audit/2026-05-08/2026-05-09-decode-NN-7step-plan.md`
- final DGP campaign summary: `audit/2026-05-08/2026-05-08-DGP-FINAL-summary.md`
- WRAPUP from prior session: `audit/2026-05-08/2026-05-09-WRAPUP-decode-NN.md`
- prior chain-decomp / preimage / Path-B work:
  `audit/2026-05-07/2026-05-07-d18-dgp-decomp-batch.md`
- per-row tuple concordance evidence:
  `audit/2026-05-08/2026-05-08-p1c-tuple-concordance.md`
- existing dgp_v2 scripts (P1–P17): `scripts/dgp_v2/`
- proposed new directory for this plan's artifacts: `scripts/dgp_v3/`
- assumption ledger A29-A31, A33 in `ASSUMPTIONS.md`

## Open questions for PI before execution

1. Submission posture during decode: keep the 270-budget intact (the
   decode does not need LB rounds) or spend ~2 sanity probes per phase?
2. GPU budget: are Kaggle T4 ×2 kernels acceptable for Phase B's grid,
   or do we need a heavier instance?
3. Acceptance bar for "decoded": is the disc-AUC floor target (≤ 0.85)
   tight enough, or do we want a per-cell KS report as the success
   criterion instead?
