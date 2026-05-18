# Path-B-specific vs pool-structural rank-lock

**Date opened**: 2026-05-18 (Day-19; R9 EOD)
**Status**: open

## The question

Does Path-B's per-segment LR refit limit the K=14 base-add value
to ONE specific operator, or is the rank-lock pool-architecture
agnostic?

## Why it matters

We've observed the K=13 pool absorbs NB4 and C1 at K=14 under the
SPECIFIC Path-B segmentation operator (DriverClass × Stint
τ=100k). The conclusion drawn at R9 EOD — "K=13+Path-B has reached
its structural ceiling for row features" — depends on whether the
rank-lock is:

A. **Pool-structural** (correct R9 conclusion): the K=14 base
   matrix itself has collinearity / redundancy that no meta-arch
   can extract additional signal from. In that case, R10 must
   pivot to mechanism expansion.

B. **Operator-specific**: only Path-B's per-(DriverClass × Stint)
   shrinkage absorbs the new base; a DIFFERENT meta operator
   (plain LR-meta, non-LR meta, different τ, different
   segmentation) might extract +0.05 to +0.20 bp lift from the
   same K=14 pool. In that case, the lock is at the OPERATOR
   layer, not the POOL layer, and we have a cheaper avenue than
   mechanism expansion.

## How to resolve

Re-run K=14 meta-arch sweep (the SAME experiments that closed C-axis
in 2026-05-14 / mechanism-ledger.md "Closed: Meta-architecture
redesign") but with NB4 (or C1) already on disk in the K=14 pool:

1. **Plain LR-meta** (no Path-B segmentation): does NB4 add Δ ≥ +0.02
   bp here when it regressed −0.022 bp under Path-B?
2. **τ sweep** {5k, 20k, 100k, 500k} under Path-B Compound × Stint
   on K=14: does any τ recover what τ=100k regressed?
3. **Path-B different segmentation** (Compound × FirstPitWindow
   was R8 marginal at +0.04 bp; Year × Stint at +0.05): does
   K=14 + NB4 lift more under those segmentations?
4. **Non-LR meta** (small XGBoost on K=14): is the absorption
   meta-LR specific?

## Estimated cost

~15-30 min CPU (all 4 metas are 1-2 min each on K=14). Cheap
relative to R10 mechanism-expansion (2-3 hr GPU for seq2seq).

## Decision rule

If ANY of the 4 alternative metas finds Δ ≥ +0.05 bp on K=14 + NB4
or K=14 + C1 (where K=13 + Path-B gave −0.022 / −0.045 bp), then:

- R9 conclusion was OPERATOR-SPECIFIC, not pool-structural.
- Mechanism-expansion (seq2seq / graph / survival) becomes
  parallel-track, not sole-track.
- Reopen NB4 / C1 candidates under the winning meta.

Else (all 4 metas absorb K=14 NB4/C1 similarly): R9 conclusion
holds, K=13 pool is truly saturated, commit fully to mechanism
expansion.

## Priority

Worth one R10 slot before committing to a 2-3 hr GPU build.
Recommend running this BEFORE R10 mechanism-expansion candidate A
(seq2seq transformer).
