# 2026-05-09 — 7-step plan: decode the host's CTGAN

`branch: claude/find-dgp-research-ClsQE`
`tag: decode-nn-research-plan`

> Critical-thinking exercise: with unlimited time, how would we
> recover the host's exact synthesizer well enough to invert
> synth→orig and lift K=4 PRIMARY past its rank-locked ceiling?

## Step 1 — Problem definition

We have 627,305 synth rows from host's CTGAN. Recover f(x) → predicted
PitNextLap label from upstream orig source row. K=4 PRIMARY caps at AUC
0.954 (rank-lock). Orig held-out AUC is 0.99690 (d15). **Arithmetic
upper bound from perfect inversion: ~0.997 → 43 bp upside.**

## Step 2 — Success metric & constraints

- Private LB AUC > 0.97 (cleanly beats top-of-leaderboard 0.95476).
- Public LB sample noise is ±12 bp on a 20% draw → 13+ bp lift needed
  to be visible publicly.
- A33: PitNextLap is probabilistic (81% concordance with PitStop[L+1]);
  this caps achievable label recovery even from perfect orig lookup.
- aadigupta1601 banned per "no public CSV" PI directive — constrains
  most plans below.

## Step 3 — Research (literature + prior session)

Prior d14-d18 + P1-P11 nailed (selected):
- CTGAN-class confirmed (mean KS 0.1344 to off-the-shelf, d18 f1)
- PitStop in cond-vector (d18 f5 KS asymmetry)
- 97.55% LapTime literal copies; 95% tuple-concordance (d15, P1c)
- Disc AUC 0.9993 vs off-the-shelf default CTGAN (P3)
- Per-row conditional independence (d14)
- d18 e2 unsupervised preimage kNN +1.88 bp K=21+1 — capped low

External literature (just searched):
- **PyTorch reproducibility breaks across versions/hardware** even
  with identical seeds. CuDNN nondeterminism is a known issue.
  → Seed-search inversion is fragile.
- **CTGAN is empirically vulnerable to membership inference** —
  multiple 2024-2025 papers confirm. Higher accuracy ↔ greater
  privacy exposure.
- **Privacy reconstruction attacks on tabular GANs are formalized**
  (Alshantti 2025 SECURITY AND PRIVACY): identify training samples
  by minimizing proximity to synthetic records.
- **GAN inversion via learned encoder** is standard methodology
  (GAN Inversion: A Survey, 2021); learned f: synth → latent → orig.
- Walter Reade's public GitHub (`walterreade`) shows 6 repos —
  PyNotes, pyensemble, Kaggle-Solutions, right_whale_hunt, p2,
  scikit-learn fork. **No CTGAN/synthesis code disclosed publicly.**

## Step 4 — Alternatives generated

| Plan | Mechanism |
|---|---|
| A | Architecture+seed search for exact replication |
| B | Membership-inference shadow models |
| C | Proximity preimage with learned metric (extends d18 e2) |
| D | Supervised inversion encoder (orig, simulated_synth) pairs |
| E | Density-ratio Bayes update P(orig\|synth) |
| F | Cycle-consistent autoencoder synth↔orig |
| G | Massive arch+cond grid minimizing disc AUC vs host |
| H | DP-bound analytical upper bound on memorization |
| I | Side channels: read host's public code |
| J | B+D combined |

## Step 5 — Evaluation (with research-informed corrections)

| Plan | P(success) | Cost | Best AUC | Critical risk |
|---|---:|---:|---:|---|
| A. Seed search | <1% | 100 GPU-d | 0.997 | **PyTorch nondeterminism** kills exact replay across hardware/version stacks |
| B. Shadow MIA | 60% | 10 GPU-d | 0.96 | Scores, not labels |
| C. Learned-metric preimage | 80% | 2 d | 0.955 | d18 e2 already at +1.88 bp; ceiling close |
| **D. Supervised encoder** | 45% | 10 GPU-d | 0.97-0.98 | Surrogate fidelity |
| E. Density ratio | 30% | 5 d | 0.96 | 14-D density brittle |
| F. Cycle AE | 30% | 5 d | 0.96 | Loss landscape |
| **G. Arch+cond grid** | 70% | 20 GPU-d | unlocks D | Compute-heavy |
| H. DP analysis | 100% (analytic) | 1 d | 0 lift | Bound only |
| **I. Side channels** | 20% | 2 d | 0.997 if found | High asymmetry; cheap |
| J. B+D combined | 55% | 20 GPU-d | 0.97-0.98 | Highest engineered upside |

Three corrections from initial plan:
1. **Seed search isn't the killer.** Cross-version/hardware
   nondeterminism in PyTorch breaks exact replay. Earlier I
   overweighted this.
2. **A33's ceiling matters.** Even orig data is 0.997 not 1.0. Perfect
   inversion gives at most ~0.997.
3. **Public LB sample noise ±12 bp** swallows lifts under ~13 bp.
   Validation must be private-LB-aware (or held-out aadigupta-style).

## Step 6 — Plan / act

Sequenced execution:

| Phase | Days | Plan | What we get |
|---|---|---|---|
| 1 | 1 | I + H | Side-channel scan; DP upper bound |
| 2 | 2-30 | G | Arch+cond grid; min disc AUC surrogate |
| 3 | 31-60 | D | Supervised inversion encoder, simulated pairs |
| 4 | 61 | B | Membership-inference shadow models |
| 5 | 62-75 | Apply f to host | Hybrid: w·orig_lookup + (1-w)·K4 |
| 6 | 76-90 | Differential validation | 10× synth, retrain K=4, compare LB |
| 7 | 91+ | Iterate | Bigger surrogates, encoder ensembles |

## Step 7 — Anticipated failure modes

| Risk | Probability | Diagnostic |
|---|---:|---|
| Surrogate gap floor (disc AUC stays > 0.9) | 40% | Compare encoder accuracy on simulated vs host |
| Memorization sparse (regularized host CTGAN) | 30% | MIA score distribution mass |
| A33 ceiling absorbs lift | 100% (bounded) | Even perfect orig lookup gets only 0.997 |
| Public-LB ±12 bp noise band | 60% | Lifts < 13 bp invisible publicly |
| 2023 source heterogeneity (P9) | 40% | Two distinct DGPs in orig; inversion ambiguous |

## Honest revised takeaway

**With unlimited time, the realistic high-EV path is G → D → J**, not
seed-search. Expected lift OOF: 5-15 bp (from current PRIMARY 0.95403),
possibly invisible on public LB but real on private. The "+43 bp to
0.997" upper bound is essentially unreachable from this entry point.

**Cheapest first move is I + H** (side channels + DP analysis): pure
research, near-zero cost, either reveals the answer outright or tells
us the upper bound on what's possible.

The current execution attempt under PI's "no public CSV" constraint
is to substitute aadigupta1601 with synth-train itself in the
surrogate-training pipeline. That degrades plan G/D significantly
(surrogate trained on synth doesn't recover orig structure), but
preserves the framework.

## Pointers

- This audit
- Searches: PyTorch reproducibility, CTGAN MIA 2024-2025, GAN inversion
  survey, Alshantti 2025 privacy reconstruction
- `audit/2026-05-08/` — full prior DGP campaign

## Side-channel dump

Walter Reade's `walterreade` GitHub: 6 public repos, none synthesis-
related. No CTGAN config or preprocessing code disclosed. Past TPS
Feb 2021 was confirmed CTGAN openly but the pipeline code was not
published. Elizabeth Park's profile is staff-only (kaggle.com/
elizabethpark). Yao Yan ambiguous — multiple Kaggle profiles, no
clear synthesizer attribution.

Conclusion: side channels (Phase 1) yield ~zero. Decoding requires
compute-side analysis only.
