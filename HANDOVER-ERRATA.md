# HANDOVER-ERRATA — 2026-05-08

Errata against `HANDOVER.md` (and the state docs it links to) discovered
during the Day-19 "understand the problem better" session. Each entry
points at the source claim, what's wrong, and the supporting evidence.
This file is meant to be read alongside `HANDOVER.md` until the next
"prepare handover" rewrites the brief.

Communication-friction tag introduced this session:
**`open-axis-overstated`** — the handover lists axes as "open" or
"untouched" that have actually been probed and falsified. Same pattern
across `state/current.md`, `state/hypothesis-board.md`, and `HANDOVER.md`.

---

## E1 — "Sequence-level fingerprinting is the only structurally-orthogonal axis remaining"

Source: `HANDOVER.md` § "What axes are still open" item 1; mirrored in
`state/current.md` "Sequence-level fingerprinting (A1, untouched)" and
`state/hypothesis-board.md` "Open priorities" item 1.

**Wrong.** Three sequence/cross-row probes have already run and were
absorbed or falsified:

| Probe | Result | Source |
|---|---|---|
| Day-16 H1 GRU sequence on (Driver, Race) lap windows | std OOF 0.93066, ρ=0.919, **−0.043 bp NULL** at K=22+1 LR-meta gate | `audit/2026-05-16-d16-virgin-axes-results.md` |
| Day-17 cross-driver field-state aggregates (`probe_field_state.py`) | **−0.015 bp NULL** at K=24 meta gate | `audit/archive-2026-05-07-handover-pm-sections.md` |
| Day-? `probe_combined_lead_lag` (combined-frame lag/lead within (Race, Driver)) | combined-frame premium **−0.36 bp NEGATIVE** vs train-only; total LL +2.18-2.55 bp inside fold-noise band | `audit/decisions.jsonl` outcome row |

The actual remaining-open axis from the Day-16 synthesis is
**meta-architecture redesign**, not sequence-level base addition. The
handover gets this backwards by listing the closed sequence axis first
and not surfacing the meta-arch axis at all.

## E2 — "Predicted +1 to +3 bp" for sequence axis

Source: `HANDOVER.md` items 1 and 2; identical band in
`state/hypothesis-board.md`.

**Ungrounded.** No audit reference is cited for these prediction bands.
Cross-checking `audit/decisions.jsonl` and `state/calibration-ladder.md`:
the closest sequence-class precedent is the d16 GRU which delivered
−0.043 bp at meta. The +1 to +3 bp band is not anchored to any
calibration row.

## E3 — "RealMLP with 24 ensembles predicted +1 to +3 bp standalone"

Source: `HANDOVER.md` item 2.

**Ungrounded.** RealMLP-yekenot at n_ens=4 (h1d) is in the stack at OOF
0.95257. Going from n_ens=4 to n_ens=24 is a 6× ensembling cost
multiplier; classical NN ensembling theory gives diminishing returns
with sqrt(n_ens) of standard error reduction. The +1 to +3 bp band has
no calibration anchor in the ladder.

## E4 — Treating OOF→LB gap as "structural"

Source: `state/calibration-ladder.md` repeatedly logs "realised gap −6.4
bp" / "realised gap −6.1 bp"; `state/current.md` lists a −6.4 bp gap
as if it were a tracked invariant.

**Refuted by Probe B (this session).** Bootstrapped 95% CI of a random
20% public draw on PRIMARY OOF is [0.95309, 0.95550] around the full
OOF 0.95432. The observed LB 0.95368 is comfortably inside the band.
The "consistent gap" is one realization of the public-split lottery,
not a tracked overfit signal.

Implication: the team's worry about overfitting public LB is *not*
empirically supported. Adding bases / refining the meta is unlikely to
be wrecked by a private-LB gap, *unless* the test partition is
selected non-randomly — which AV-AUC=0.502 already says it isn't, at
least at row level.

## E5 — "The synthesiser broke within-stint sequence coherence"

Source: `state/current.md` item under "Sequence-level fingerprinting";
`state/hypothesis-board.md` item 1.

**Wrong about the mechanism.** Probe C (this session): within-stint
*physical* constraints (Compound constancy, TyreLife monotone, LapNumber
strictly increasing) are preserved at ≥99.99% in synthetic data. The
synthesiser does NOT inject physical violations.

What it does instead: **temporal downsampling.** Synthetic stints have
mean length 3.87 laps vs 19.80 in original; synthetic
consecutive-rows-at-gap=1 fraction is 27.98% vs 99.60% in original.
Synthetic "stints" are sparse subsets of underlying stints, not
coherence-broken ones. Any feature engineered around "detect-coherence-
break" rests on a wrong premise.

Re-deriving the original sequential structure (lap-by-lap) from the
synthetic snapshot is not possible without an external source (FastF1).

## E6 — "External data axis closed"

Source: `state/current.md` item D.

**Partially wrong.** What's closed: aadigupta original (already in the
stack), debashish historical priors (pre-flight rejected), FastF1 hard-
join (1.4% match rate cap). What's *not* closed and not surfaced:
**FastF1 soft features** at the (Race, Year) or (Compound, Year) level
that don't require row-level matching. These would survive the synthetic
driver-code remap because they're aggregates, not joins. Not pursued, no
audit refutation.

## E7 — Public-notebook scan claim "missing feature classes"

Source: this session's Probe D suggested "interaction TE" and
"cross-driver same-(Race, Lap) field-state" were missing.

**Wrong.** Both have been tested:

- Interaction TE (`make_features_A` with Driver×Race×Year, Driver×Race,
  Driver×Compound, etc.): **leaky.** OOF 0.94970 standalone → submitted
  alone LB **0.94107** (gap −863 bp). K=22 LR-meta-add OOF 0.95404 →
  submitted LB **0.94933** (−126 bp vs PRIMARY).
- Cross-driver field-state aggregates: **NULL** at K=24 meta (−0.015 bp).

The public kernel `s6e5-driver-s-high-driver-feature-eng` claiming OOF
0.95994 via interaction-TE + Transformer is consistent with the same
target-leakage signature the team caught in Day-17 audit. Treat its OOF
as untrusted.

---

## Pattern across all of E1-E7

The handover's "open axes" framing has a **systematic optimism bias**:
items are listed as open or predicted to lift +N bp without back-
references to falsifying probes already in `audit/`. Six of seven items
had a counter-evidence trail in the same repo.

Friction tag for next postmortem:
- `handover-open-axes-overstated`: handover writes the *un-tried*
  version of an idea even after the *tried* version has been falsified.

Mitigation: every "open axis" line in `HANDOVER.md` should cite either
(i) a probe that was attempted but ran out of compute, or (ii) a
specific variant that was NOT tried, with the variant differentiated
from the tried-and-failed version.
