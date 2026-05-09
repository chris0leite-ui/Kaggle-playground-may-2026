# 2026-05-09 — Phase A3+A4: Q3+Q5 — synth-marginal alone does NOT recover host

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-loop-phase-A + marginal-falsified`
`script: scripts/dgp_v3/q3q4q5_ctgan_decomp.py`
`artifact: scripts/artifacts/dgp_v3_q3q4q5_disc_auc.json`

> Decisive falsification of the natural Phase A hypothesis: F8 (host
> uses a custom sampling marginal) is real but is NOT the dominant
> driver of F6's 0.9993 disc-AUC.

## Result

| Sample | disc-AUC vs synth | Notes |
|---|---:|---|
| Q3: SDV CTGAN-on-orig, default sampling | 0.9993 | Reproduces F6 exactly |
| Q5: SDV CTGAN-on-orig, **synth-marginal** sampling | 0.9993 | NO change |
| Δ (Q5 − Q3) | −0.00003 | within fold noise |

Per-year breakdown for Q5 sample: 2022 0.9991, 2023 0.9992, 2024
0.9988, 2025 0.9990. All effectively identical. Even on 2023 (the
"anomalous" year) the disc separates synth from CTGAN-replay
near-perfectly.

## What this rules out

**Hypothesis F8 alone**: that the host's only material customisation
is the sampling marginal `(Year, Compound, PitStop)`. This is now
falsified at the disc-AUC level. F8 *is* real (Q10 confirmed it
cleanly: synth oversamples PS=0 cells by ~1.5×, undersamples PS=1
cells by ~0.5×). But re-applying F8 to a default SDV CTGAN does not
close the disc-AUC gap.

## What's left as the dominant axis

Three remaining candidates for "what the host did differently":

1. **Model architecture class.** SDV CTGAN may not be the host's
   generator. Alternatives:
   - SDV TVAE
   - SDV CopulaGAN
   - SDV GaussianCopulaSynthesizer
   - Stand-alone TabDDPM (diffusion)
   - Normalising flow (NSF, RealNVP)
   - Custom CTGAN variant (CTAB-GAN+, etc.)

2. **Cond-vector schema.** SDV CTGAN auto-detects cond columns. The
   host may have explicitly forced (Driver, Race, ...) into cond,
   producing different generator dynamics.

3. **Hyperparameters within CTGAN.** Mode count per column (default
   10), embedding dim (default 128), pac (default 10), generator and
   discriminator architectures, training epochs (default 300; we ran
   10), and learning rate.

## Diagnostic strength of disc-AUC = 0.9993

Disc-AUC at 0.9993 is *not* "host has small fingerprint differences."
It is closer to "host's per-row joint distribution is structurally
different from SDV CTGAN's." A LightGBM with 200 trees can build a
nearly perfect classifier separating the two synth distributions —
something it could not do if the only difference were a re-weighting
of cells or a shift in column means.

The implication is that what the host calls "synthesise" is
structurally different from "default SDV CTGAN trained for 10 epochs
on orig". The architecture or training schedule is doing material
work.

## Honest reading

This was the optimistic outcome that didn't come through. The
plausible Bayes update on plan v2's plan G (model+cond grid) is to
**weight architecture and cond-vector axes over marginal axis**. F8 is
a clear cosmetic correction the host applied, but the load-bearing
axis lives in the generator's per-row dynamics.

## Phase B (next) — refined surrogate grid

In order, by cost:

1. **TVAE on orig** — ~10 min. Different latent encoder; tests
   whether the host might be a VAE.
2. **CopulaGAN on orig** — ~10 min. Closer to CTGAN family; tests
   for a copula-class fingerprint.
3. **GaussianCopulaSynthesizer on orig** — ~30 sec. Almost trivial;
   tests whether the host is a non-deep-learning synthesiser.
4. **CTGAN with explicit Driver+Race in cond** — ~10 min. Tests
   whether expanded cond-vector closes the gap.
5. **CTGAN with PAC=1 + larger hidden** — ~15 min. Within-CTGAN
   hyperparameter axis.
6. **TabDDPM on orig** — ~30 min, GPU recommended (CPU works but
   slow). Diffusion-class generator.

Gate: any candidate with disc-AUC < 0.95 is a hit; if multiple, take
the one with the lowest disc-AUC and the smallest per-feature KS
divergence on per-cell marginals.

## Pointers

- This audit
- `scripts/dgp_v3/q3q4q5_ctgan_decomp.py`
- `scripts/artifacts/dgp_v3_q3q4q5_disc_auc.json`
- v3 plan skeleton: `audit/2026-05-09/2026-05-09-plan-v3-PHASE-A-results.md`
- F6 origin: `audit/2026-05-08/2026-05-08-p3-ctgan-replay.md`

## Friction tag

`f8-marginal-recovery-does-not-close-disc-auc-gap` — promote. The
host's customisation is per-row-generator-class, not per-cell-mass.
F8 (marginal) and F6 (disc-AUC 0.9993) are independent axes.
