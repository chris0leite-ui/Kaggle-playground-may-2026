# 2026-05-09 — Executive summary: what we now know about how the data was made

`branch: claude/decode-data-process-5uLq3`
`tag: dgp-decoding-loop-EXEC-SUMMARY`
`audience: PI`

> Plain-English version. Detail in the seventeen `q*` audit/script
> files committed today. Read this first.

## What we did

We started this session with the prior team's six "DGP findings" and
a seven-step plan. We ran seventeen probes to extend, validate, and
in some places overturn that plan.

We were trying to answer one question: **how exactly does the host
turn aadigupta1601's 101,305-row file into the 627,305-row file
Kaggle ships us?**

## The headline answer

The host's data-making pipeline works roughly like this:

1. **Take aadigupta1601 as input.** Drop the column called
   `Normalized_TyreLife`. Drop the 66 rows with missing `Compound`.
   Keep 101,305 × 14 columns.
2. **Decide how many output rows to make per `(Year, Compound,
   PitStop)` cell.** This is the host's first big customisation —
   they use a sampling distribution that **halves the weight on
   `PitStop=1` cells** vs orig (so synth's pit rate is 0.20 instead
   of orig's 0.25). Inside `PitStop=0`, the `MEDIUM` compound is
   slightly upweighted, mostly in 2024.
3. **Inside each cell, fill in the rows by sampling from a per-cell
   continuous-value generator.** This is the unsolved part — see
   below. Whatever this generator is, it produces ~30 rows per
   `(Year, Compound, PitStop, Race, Stint, LapNumber)` cell vs
   orig's typical ~5, and 73 % of the synth values are *new*
   continuous numbers that orig does not have for that cell.
4. **Hand each row a Driver and a Stint label.** Driver is drawn from
   a vocabulary of 887 codes — the 31 codes that are actually in
   orig, plus 856 fabricated codes. The 856 split into 756
   `D###`-prefix ghosts and ~100 retired-driver three-letter
   abbreviations. Stint is similarly drawn. Both are drawn from a
   *structured* per-cell distribution, not uniformly. (The proof
   that the distribution is structured is the 14 percentage point
   drop in our match score when we move from uniform to per-cell
   sampling; see qF → qH.)
5. **Output `train.csv` and `test.csv`.** Drop `Normalized_TyreLife`.
   Test split is i.i.d. row-level (we already knew this).

## How well do we now match the host?

We measure the gap between any candidate generator and the host with a
LightGBM that's asked: "is this row from the host's file or from
this candidate?" A score near 1 means the candidate is easy to tell
apart from the host (bad). A score near 0.5 means the candidate is
indistinguishable (good).

| Candidate generator | gap score | reading |
|---|---:|---|
| Off-the-shelf SDV CTGAN (default) | 0.999 | trivial to spot |
| Same with synth's `(Year, Compound, PitStop)` weights | 0.999 | weight customisation alone is not the trick |
| Off-the-shelf SDV TVAE | 0.999 | not the trick |
| Off-the-shelf SDV GaussianCopula | 0.999 | not the trick |
| Off-the-shelf SDV CopulaGAN | not measured | pattern is conclusive |
| Raw aadigupta1601 (no synthesis) | 0.99 | orig vs synth differ a lot, mostly in `Driver` |
| Hand-built: orig values + synth weights + scrambled labels | 0.97 | beats every CTGAN we tested |
| Hand-built: + `Driver`/`Stint` drawn per-cell from synth | **0.83** | first big jump |
| Hand-built: + `Race` in the cell key | 0.79 | |
| Hand-built: + `Stint` in the cell key | 0.72 | |
| Hand-built: + `LapNumber` in the cell key | **0.72 ←best so far** | analytic ceiling |
| Sampling synth itself (theoretical lower bound) | 0.49 | what perfect mimicry looks like |

So we've moved from 0.999 (off-the-shelf CTGAN) to 0.72 (a hand-built
analytic pipeline that respects the structure we've decoded). The
remaining gap to 0.49 is the per-cell continuous-value generator —
**the unsolved residual.**

## What we know about the unsolved residual

The host's continuous-value generator produces:

- Values that are **closer to orig in feature space than orig is to
  itself**: synth's nearest-neighbour orig row has standardised
  distance 0.035, while orig's intra-pair NN distance is 0.048. The
  generator stays close to orig's manifold.
- Values that are **per-cell**: when we tried sampling continuous
  values from orig globally instead of per-cell, the gap score blew up
  to 0.99. The host respects cell boundaries.
- **New** values 73 % of the time: only 27 % of synth's
  `(Year, Compound, LapTime)` triples exist in orig with that exact
  `LapTime`. Most synth `LapTime` values are inventions for that cell.
- **No simple noise**: every `sigma > 0` we tried (global Gaussian,
  cell-scaled Gaussian) made the gap score worse, never better. The
  host doesn't perturb orig values with noise.
- **Not a per-cell BGMM**: a Bayesian Gaussian mixture per cell makes
  the gap score 0.86, worse than just sampling orig values.

Adding all this together, **the residual is a real per-cell-conditioned
generator** — it does the work of generating new continuous values
that look like they "belong" to a cell — but that generator is not in
SDV's library and is not any of the simple analytic pieces we tried.

The remaining candidates we haven't tested:

- **TabDDPM** (tabular diffusion). The most-likely candidate among
  modern tabular generators. Not pip-installable cleanly; would need
  to clone a research repo and run on GPU.
- **A normalising flow with cell conditioning** (RealNVP, NSF). Same
  story — research code, GPU.
- **GReaT** (LLM-based tabular generator). Heavy install. Probably
  overkill for a Playground competition.
- **Custom CTGAN variant** with very long training, larger embedding
  dim, or per-cell mode tuning. Defensible to try; ~30-60 min CPU per
  variant. Unlikely to drop below 0.95 given our 4-variant SDV sweep.

## What I think the host actually used

Best guess based on the evidence: **a CTGAN-class generator with the
cond-vector explicitly set to `(Year, Compound, PitStop, Race, Stint,
LapNumber)` and likely `(Driver)`**, trained for many more epochs
than 10 on aadigupta1601, possibly with a custom mode count or a
diffusion variant. The fact that no SDV variant we tested hit below
0.998 says the **specific** generator is not SDV-default, but the
*family* is consistent with a standard Kaggle Playground recipe.

Not a guess — **what the host did NOT do**:
- Did not literally copy orig rows (the per-row 6-tuple match rate is
  27 of 627,305).
- Did not add Gaussian noise to orig values.
- Did not use a global-marginal sample.
- Did not use SDV CTGAN out-of-the-box.
- Did not use TVAE or GaussianCopula out-of-the-box.

## Implications

**For decoding the DGP**: we have a structural model good to gap score
0.72, leaving a 0.22 gap. That gap is the per-cell continuous
generator, which we've isolated cleanly but not reproduced.

**For latent spaces / encoders / decoders that would help further**:

- The natural inversion latent is the `(Year, Compound, PitStop, Race,
  Stint, LapNumber)` cell key. We've shown this is the right
  conditioning axis.
- A density-ratio classifier per cell — orig (Y, C, PS, R, S, LapN)
  → predict P(y=1|x) on synth — should give a stronger d16-style
  base than the current `d16_orig_continuous_only`. **This is a free
  probe for the next session.**
- A normalising-flow trained on orig with the qM cell key as
  conditioning is the clean way to fit the residual generator.

**For the leaderboard goal** (not the focus of this work): the qM
cell-key insight gives a recipe to refactor `d16` so that its OOF
contribution is more diverse. Predicted lift is small (~0.5 bp at
K=4+1 LR-meta) because rank-lock saturates as we know.

## What's committed

- 17 probe scripts: `scripts/dgp_v3/q1`–`qR`
- 17 audit JSONs: `scripts/artifacts/dgp_v3_q*.json`
- 9 audit markdown docs: `audit/2026-05-09/` covering Q1+Q2 fingerprint,
  Q6+Q7 retraction, Q10 cell-marginal, Q3+Q5 marginal-not-the-axis,
  Phase B0 + Phase B+ + Phase B FINAL, plan v3, this exec summary
- 12 commits pushed to `claude/decode-data-process-5uLq3`

## One-line conclusion

**The host's data-generating pipeline is not a black box anymore — we
have a five-step structural model that closes 50 % of the gap to a
perfect-mimicry baseline, and the remaining 50 % is exactly the
per-cell continuous-value generator that creates 73 % new values per
cell.** The decode task has produced durable knowledge of how the
host works, even if we haven't reproduced the specific NN they used
to fill in continuous columns.
