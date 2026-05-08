# SUBMISSION-PROPOSALS — sparse-pool LB calibration

Per Rule 1 every `kaggle competitions submit` is single-shot, PI-approved.
This file pre-stages the candidates the Day-19 sparse-pool sprint will
generate, with rationales so the LB-vs-OOF calibration is interpretable
on landing.

Current PRIMARY: 27-base + Path-B Compound × Stint, τ=100k. OOF 0.95432,
LB 0.95368. Used 39/270 submissions.

The Day-18 PM postmortem already established: pool effective rank ≈ 2.88
of 24 (logit) / 1.78 of 24 (probability); K=10 = K=24 in plain LR-meta
OOF; LR-bank 6th confirmation of rank-lock at +0.022 bp; saturated-info
argument predicts NULL on most variants. **What's never been LB-tested:**
sparse-pool **+ Path-B amp** specifically. The user's hypothesis is that
removing redundancy gives the meta more rank to work with — *especially
through Path-B*.

---

## Decision rule for which sparse pool(s) to submit

Tier the candidates by the size of their OOF gap to PRIMARY. Submit only
those that are within the public-LB CI band (per Probe B: ±12 bp).

| Pool | What it tests |
|---|---|
| K=10 forward-select + Path-B C×S τ=100k | Does Path-B amp fire on E9's K=10 pick? |
| K=7 diversity-greedy + Path-B C×S τ=100k | Diversity-greedy at a meaningful capacity. |
| K=5 diversity-greedy + Path-B C×S τ=100k | Aggressive: 5 directions only. |
| K=3 diversity-greedy + Path-B C×S τ=100k | Sanity floor: lower bound on what 3 directions can do. |

---

## Proposed submission slate (PI sign-off needed)

The slate uses 1-3 of our remaining 231 submissions. Decision tree:

**A. If K=10_fwd + Path-B OOF beats PRIMARY by ≥1 bp:**
- Submit `submission_K10_fwd_pathb.csv`. **Highest EV.** This would
  invert the Day-18 conclusion that K=10 = K=24 (which was at *plain*
  LR-meta) — Path-B might amp differently on the de-redundant pool.

**B. If K=7_div + Path-B OOF is within −5 bp of PRIMARY:**
- Submit `submission_K7_div_pathb.csv`. **Calibration probe.** Even if
  it regresses on LB, we get the "OOF→LB gap on a deliberately-orthogonal
  pool" datapoint, which directly answers: does sparse-pool routing
  generalise to public LB the same way the K=27 dense pool does?

**C. If K=5_div + Path-B OOF is within −15 bp of PRIMARY:**
- Submit `submission_K5_div_pathb.csv`. **Aggressive diversity probe.**
  Same logic as B but with sharper separation — if K=5 lands within the
  bootstrap CI on LB despite OOF gap, the dense pool was over-correlating
  in a way the public-LB sample distribution doesn't reward.

**D. Bypass option — K=10_fwd + plain LR-meta:**
- The K=10 pool with plain LR-meta was characterized at OOF 0.953808
  (Day-18 PM, T2). Never LB-tested. A submit gives us a clean datapoint
  on "is Path-B amp doing real work on this pool size, or is plain
  LR-meta sufficient?" — a 7-way comparison piece. **Cost: 1 submission,
  high learning yield.**

---

## What we'd LEARN from each submission

| Submit | Wins (LB ≥ PRIMARY) | Loses (LB < PRIMARY) |
|---|---|---|
| K=10_fwd plain LR-meta (D) | The 27-base PathB amp was an over-fit to OOF; plain LR-meta on a sparse pool is enough → wrap-up | Day-18 finding holds; PathB amp adds real LB lift even at K=10 |
| K=10_fwd + PathB (A) | Path-B amp fires *better* on de-redundant pool → submit K=7 next | Falsifies user's "rank-collapse is the binding constraint" intuition |
| K=7_div + PathB (B) | Diversity beats raw OOF — *huge* finding; LB-OOF gap structurally favoured the sparse pool | Pool sparseness alone doesn't break ceiling; combine with H4 lookup or H8 stint features for next round |
| K=5_div + PathB (C) | Even 5 directions are enough for public LB → reframe entire approach | Quantifies the diversity-vs-AUC tradeoff at the meta |

---

## Submission hygiene per Rule 1, 27, 28

1. **Rule 27**: every submission must run `scripts/pre_submit_diff.py`
   against the previous submit to confirm Spearman < 0.999. Sparse-pool
   submissions that hit ρ ≥ 0.999 vs PRIMARY are aborted (LB will tie).
2. **Rule 1**: PI signs off on each `kaggle competitions submit`.
   Single-shot, no while/until loops.
3. **R7 hedge eligibility**: sparse-pool submissions that regress on
   public LB but are ≥99.9 ρ to a R5-eligible OOF-best candidate
   become hedge candidates for the final-window probe.

---

## Calibration value (this is the point)

Even a regressing submission is informative. Per `WEAKNESSES.md` W2,
the public-LB CI is ±12 bp wide. A K=5_div submission landing inside
the band would tell us:
- The diversity hypothesis isn't OOF-noise-resistant; the pool's
  apparent saturation is real.

A K=5 submission landing OUTSIDE the band (i.e., LB regression > 12 bp)
would tell us:
- Public LB rewards correlated-pool predictions in a way the bootstrap
  CI didn't anticipate; *that's* the structural risk we've been guessing
  about. Major reframe needed.

So: regardless of outcome, **at least one sparse-pool submission is
worth its budget cost** for the LB calibration itself. The slate above
is what I'd recommend for PI sign-off after OOF results land.
