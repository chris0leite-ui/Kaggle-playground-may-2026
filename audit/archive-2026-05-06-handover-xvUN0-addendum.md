# Archive — 2026-05-06 PM HANDOVER addendum from `claude/ml-handover-alignment-xvUN0`

> Archived from `HANDOVER.md` at 2026-05-06 by scribe on
> `claude/assess-synthetic-data-features-NYZuK` at merge-to-main time.
> Per parallel-branch convention (CLAUDE.md Rule 15 / WRAPUP.md
> "Parallel-branch convention"), per-branch `Day-N PM <slug>`
> addenda are consolidated by the merge-target scribe.
>
> Actionable items (α-resweep, d12_lr_meta single-candidate ablation)
> are captured in HANDOVER §"Remaining live moves (Day 15)" PURSUE
> list. The K=22 + d12_lr_meta SUBMITTED LB 0.95045 (-4 bp REGRESS)
> finding is captured in CLAUDE.md `mechanism_families_explored` as
> `two_level_stacking_meta_as_base` and friction tag
> `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`.

## Original content

PI redirect: experimentation culture; many small probes; BOTE-first;
"the solution is probably simple, maybe a code-quality fix".

**Built:** experimentation harness — `scripts/probe.py` (`bote` +
`gate`), `scripts/probe_min_meta.py` (K=21+N stack-add gate),
`scripts/probe_blends_K21.py`, `scripts/probe_rho_inventory.py`.
Rule 19 added to CLAUDE.md codifying BOTE-first / gate-after.

**Cheap probes (all via harness):**

1. **α-asymmetry verification.** OOF uses fold-train counts in α=n/(n+τ);
   test uses full-train counts. Bayesian-correct shrinkage, NOT a fixable LB
   cap. **PURSUE**: α-calibrated τ-resweep (~30 min).

2. **K=21 simple-blend probe.** mean/gmean/rank_mean/trimmed all regress
   19–32bp standalone vs PRIMARY. LR-meta-stays-best CONFIRMED.

3. **ρ inventory of 22 held candidates.** Best near-tie HEDGE: **`d12_lr_meta`**
   (OOF 0.95073, ρ=0.996, flip ratio 0.297).

4. **K=21 + d6_rule_compound_stint min-meta.** Δ −0.020bp NULL (already absorbed).

5. **K=21 + 3 (`d12_lr_meta` + `d10d_leak_corrected_meta` + `blend_rank_mean_K21`).**
   **Δ +1.298bp OOF** (0.95073 → 0.95086). `d12_lr_meta` dominates. **First non-NULL.**

**Open candidates (NOT YET RUN, BOTE-graded):**
- α-calibrated τ-resweep on PRIMARY hier-meta (PURSUE; ~30 min).
- `d12_lr_meta` single-candidate ablation (was in flight at session-end).
- Within-Race quantile-rank of LapTime_Delta as FM input (DEFER; H5 z-score leak fix needed).
- Per-Driver historical pit rate smoothed EB (DEFER; ~10 min).
- Year×Stint sparse-LR / FM partition (DEFER; ~30 min).
