# Grilling round 5 — drilling into "calibration as trust mechanism"

After [round 4](./2026-05-06-grilling-round-4.md) and the
[verification + synthesis](../thoughts/2026-05-06-bote-fresh-and-calibration-as-trust.md).
PI committed to calibration-of-BOTE-estimates as the at-scale trust
mechanism. Three follow-ups: two on calibration mechanics, one
carried over.

---

## F1.6 — Which calibration stage(s) do you want to track?

`probe.py bote` outputs both **predicted OOF Δ** and **predicted LB Δ**.
The calibration ladder shows that for FM-class probes, OOF→LB is
miscalibrated 8-300× and once direction-flipped. So calibration is
necessarily two-stage:

- **Stage 1: BOTE → OOF Δ.** Agent's prediction quality. Measurable
  immediately after a probe runs.
- **Stage 2: OOF Δ → LB Δ.** Leakage / generalization gap.
  Currently treated as a CV-design problem (Strat vs GroupKF), but
  empirically appears as systematic LB *under*-prediction for
  FM-class.
- **Coupled: BOTE → LB Δ (end-to-end).** What you actually care about,
  but mixes both sources of error.

Question:

- Track only Stage 1?
- Track Stage 1 + Stage 2 separately, with per-family priors for
  Stage 2 (FM-class amplification ≠ GBDT-class)?
- Track end-to-end (Stage 1 × Stage 2) and accept losing diagnosability?

**Why I'm asking.** Determines what data each probe must log and what
"BOTE was right" actually means. Pre-commits you to either richer
per-probe metadata or simpler-but-coarser dashboards.

> **PI answer (2026-05-06).** PI did not pick a stage. Pushed back on
> the framing: predictions are intrinsically hard for novel things,
> and OOF→backtest→live is "the typical [problem] in my work." Both
> stages will always be there.
>
> PI moved the trust mechanism from outcome-rate to **decision
> quality**. See [concepts/decision-quality-vs-outcome-quality.md](../concepts/decision-quality-vs-outcome-quality.md).
> Re-drilled in [F1.6.1](./2026-05-06-grilling-round-6.md#f161) on
> decision-time information capture (which is required regardless of
> which stage you'd track).

---

## F1.7 — How to measure SKIP-recall

Calibration plan covers **PURSUE-precision** trivially (probe ran,
ground truth available). But **SKIP-recall** — fraction of skipped
probes that *would have* paid off — has no automatic ground truth:
you didn't run the probe.

Three plausible strategies (not mutually exclusive):

- **Periodic sample-runs.** Randomly run a small fraction of SKIPs
  for ground truth. Pros: unbiased recall estimate. Cons: expensive
  (defeats the BOTE skip).
- **Adversarial / counter-prior probes.** Deliberately PURSUE probes
  the prior says SKIP, expecting most to fail. Pros: tightens
  FAMILY_PRIORS. Cons: still expensive.
- **Implicit recall via sibling agents.** Other branches / comps may
  pursue what this branch SKIPped; their results retro-feed
  calibration. Pros: free. Cons: only works if the skipped families
  show up elsewhere.

Plus the null option: **don't measure SKIP-recall**, and accept
the calibration only watches the PURSUE side of the matrix.

**Why I'm asking.** Without one of these, "decision calibration" is
half-blind. PI may legitimately accept the half-blind position — but
should be conscious of it.

> **PI answer (2026-05-06).** Rejected the precision/recall framing.
> Proposed alternative metric: **cumulative compute waste that, in
> postmortem, was avoidable**. Self-flagged that this is hindsight-
> biased. Landed on: **transparent decisions + a notion of "good
> decision"** is the actual ask, not a success rate.
>
> Drilled into the decision-quality framing in
> [`concepts/decision-quality-vs-outcome-quality.md`](../concepts/decision-quality-vs-outcome-quality.md).
> Re-drilled in [F1.7.1](./2026-05-06-grilling-round-6.md#f171) on
> rule extraction as a deliberate post-mortem step.

---

## F1.5 (carried over) — Bipartite CLAUDE.md ownership

Still open from [round 4](./2026-05-06-grilling-round-4.md#f15--bipartite-claudemd-concretely).
Which numbered rules feel **yours**, which feel agent-drafted, which
would you delete? F-2 today added empirical evidence Rule 19/BOTE
itself was agent-drafted, so the question now has a concrete starting
example.

> **PI answer (2026-05-06).** Deferred. PI plans to remove "many things"
> from CLAUDE.md, will redo and check it, will monitor more closely
> going forward. No structural pre-commitment. Closed for now; will
> re-open empirically as PI's edits arrive.

---

## Parked

- **Cross-comp calibration store.** Per-comp calibration sample size
  (~100-300 datapoints) is too noisy alone. Trust mechanism likely
  needs to aggregate across comps. Out of scope this round; will
  return when PI is ready to design the calibration log's lifetime.
- **`pi_override` field** on hypothesis-board / audit entries —
  suggested but not actioned. If implemented, would let aggregate
  calibration distinguish rule-driven kills from PI-driven kills.
