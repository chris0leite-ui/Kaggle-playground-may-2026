# ASSUMPTIONS.md — what we've assumed vs measured

A live ledger of every claim our agents act on. Each row tags the claim
with **strength** (how it was established) and **status** (whether it
still holds at last check).

Strength tiers:
- `MEASURED` — direct empirical probe with artifact reference.
- `INFERRED` — derived from MEASURED facts via a non-trivial inference;
  the inference itself can be wrong.
- `ASSUMED` — taken on faith, no probe, no derivation. Treat with
  suspicion.
- `FALSIFIED` — was MEASURED or INFERRED and then refuted.

Status tiers:
- `live` — currently used as a load-bearing claim.
- `stale` — last verified > 7 days ago; re-check before relying.
- `dropped` — falsified or no longer relied on.

Process rules for this file:
1. Every load-bearing claim that drives a strategic decision goes here.
2. Each entry must cite the artifact. "We tried it" without a path is
   worth `ASSUMED`, not `MEASURED`.
3. A claim is re-checked at every postmortem (Rule 14 trigger) and at
   handover prep. Update `last_checked`.
4. When `state/current.md` or `HANDOVER.md` says X, and X is not in
   this file, that is a friction event — log it and add a row.

---

## Session note 2026-05-08

Initial assumption audit done as part of the "understand the problem
better" probe. Session probes A/B/C are referenced below.

| # | Claim | Strength | Status | Source / Evidence | Last checked |
|---|---|---|---|---|---|
| A1 | PRIMARY is K=27 stack + Path-B Compound × Stint hier-meta, τ=100k, OOF 0.95432, LB 0.95368 | MEASURED | live | `state/calibration-ladder.md` row "27-base v4+h1d+DGP-class"; `scripts/artifacts/oof_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000_strat.npy` confirms col-1 AUC = 0.95432 | 2026-05-08 |
| A2 | OOF→LB gap of −6.4 bp is **sampling noise**, not structural overfit | MEASURED | live | This session Probe B: bootstrapped 20% CI [0.95309, 0.95550] around 0.95432; observed 0.95368 inside band | 2026-05-08 |
| A3 | Public LB and train are row-iid; AV-AUC = 0.502 | MEASURED | live | `comp-context.md` U3 probe; pre-baseline gate doc | 2026-05-04 |
| A4 | Train and original ARE distinguishable at sequence level (stint length, lap-gap distribution) — synthesiser temporally downsamples | MEASURED | live (NEW) | This session Probe C: synth stint mean 3.87 vs orig 19.80; gap=1 frac 27.98% vs 99.60% | 2026-05-08 |
| A5 | PRIMARY's residual loss is concentrated in INTERMEDIATE / WET compound rows | MEASURED | live (NEW) | This session Probe A; 8 worst (Compound × Stint × position) cells all rain-condition; AUC 0.68–0.86 vs global 0.954 | 2026-05-08 |
| A6 | Per-row FE on the 14 raw columns is dead (residual variance ≈ marginal variance) | MEASURED | live | Five separate probes per `state/hypothesis-board.md` "load-bearing" item 2 | 2026-05-07 |
| A7 | Target reformulations (`inv-laps`, `pit-horizon`, `reverse-cumulative`, `stint-progress`) are leaky under standard CV unless aggregates are refit per fold | MEASURED | live | `audit/2026-05-06-target-reform-leakage-audit.md`; collapse rates 88-100% | 2026-05-06 |
| A8 | K=22 + Path-B Compound × Stint is the local optimum among 9 tested meta variants | MEASURED | live | `state/current.md`; "Nine variants tested across Days 14-19" | 2026-05-07 |
| A9 | The K=22 LR-meta is rank-locked: any standalone OOF-computable base is absorbed | INFERRED | live | 5 cross-confirmations per `audit/2026-05-16-d16-virgin-axes-results.md` F4. Strong inference but doesn't rule out base-classes that violate the absorption argument's premises | 2026-05-08 |
| A10 | Sequence-level fingerprinting is +1 to +3 bp open lift | ASSUMED | dropped | `HANDOVER.md` item 1; no calibration anchor; the only sequence-class precedent (d16 GRU) was −0.043 bp NULL | 2026-05-08 |
| A11 | RealMLP n_ens=24 is +1 to +3 bp standalone | ASSUMED | live (low confidence) | `HANDOVER.md` item 2; classical sqrt(n_ens) law gives ≤ 1 bp from variance reduction alone; lift would have to come from a ceiling effect we haven't tested | 2026-05-08 |
| A12 | Per-Year CatBoost specialists are ±2 bp | ASSUMED | live (low confidence) | `HANDOVER.md` item 3; cited finding "Day-12 found 2023 was the easiest year" but doesn't translate directly to per-Year specialist lift band | 2026-05-08 |
| A13 | FastF1 hard-join is capped at 1.4% match rate by synthetic driver codes | MEASURED | live | `audit/decisions.jsonl` h2_fastf1_external_join; pre-flight | 2026-05-07 |
| A13b | FastF1 *soft* features (e.g., (Race, Year, Compound) aggregates that don't need driver-row matches) are also closed | ASSUMED | live (low confidence) | Not separately probed; `state/current.md` says "External data axis: closed" but the closure argument is hard-join-specific | 2026-05-08 |
| A14 | The synthesiser preserves within-stint physical constraints (Compound constancy, TyreLife monotonicity, LapNumber strictly increasing) | MEASURED | live (NEW) | This session Probe C; ≥99.99% on all three | 2026-05-08 |
| A15 | The synthesiser broke within-stint sequence coherence (the assumption used to motivate sequence-level fingerprinting) | FALSIFIED | dropped | This session Probe C; mechanism is downsampling, not coherence-break | 2026-05-08 |
| A16 | Public LB stability is "stable"; the Path-B amp transfers to private | ASSUMED | live (low confidence) | `comp-context.md`; assumes private split has same row-level structure as public; testable only when comp ends | 2026-05-04 |
| A17 | Top-5% boundary at 0.95405 is reachable by closing OOF→OOF lift | INFERRED | live | Given A2: random LB sample variance is ~12 bp at 95% CI, so a small OOF gain × public lottery could plausibly hit top-5% even without OOF reaching 0.95405 | 2026-05-08 |
| A18 | The leader's score 0.95476 implies a single-mechanism gap of ~10 bp | INFERRED | live (low confidence) | Multiple unidentified mechanisms could compose to that lift; the inference that "FastF1 hard-join is the only path to top-5" is itself an assumption | 2026-05-08 |
| A19 | The 27 bases are sufficiently diverse to saturate the meta | INFERRED | live | Five probes show new bases get absorbed. Doesn't rule out a base of a *truly* novel class (e.g., one trained on a different cost function entirely) | 2026-05-07 |
| A20 | The synthetic-data DGP is "conditionally near-independent per row" | INFERRED | live | "Five separate probes confirmed" per `state/hypothesis-board.md`. The probes test residual variance after conditioning on the 14 raw columns. Doesn't rule out signal in conditioning structure not captured by the 14 cols | 2026-05-07 |
| A21 | Public-notebook scan reflects the ceiling for what others have published | MEASURED at-time-of-scan | stale | Last scan in this session, 8 kernels. Top-5%-reaching kernels may exist but not be public, and new kernels are published throughout the comp | 2026-05-08 |

## How to read this for strategy

**Strong (MEASURED, live):** A1, A2, A3, A4, A5, A6, A7, A8, A13, A14.
These are the bedrock; build on them.

**Inferred but not falsifiable cheaply:** A9, A17, A18, A19, A20. We
treat them as load-bearing because we have no better framing, but each
has a low-cost re-check that hasn't been scheduled (e.g., A19 is
testable by training on a deliberately mis-specified objective and
seeing if it routes through).

**Low-confidence / assumed (treat with suspicion):** A11, A12, A13b,
A16, A18. A11 and A12 are the two listed top open priorities in the
handover — both rest on ungrounded prediction bands.

**Dropped (do not act on):** A10, A15. These were the load-bearing
claims of the handover's "open axes" — both refuted this session.

## What the handover should say if A10, A15 are dropped

The actually-open axes given the dropped claims are:
1. **Targeted modelling on rain-condition rows** (Probe A) — the
   PRIMARY's residual loss is structurally concentrated in INTERMEDIATE
   / WET cells; a rare-class specialist could plausibly close some of
   that. NOT in any current "open axes" list.
2. **Meta-architecture redesign beyond Compound × Stint** — actually
   listed in `audit/2026-05-16-d16-virgin-axes-results.md` synthesis
   but missing from `HANDOVER.md`.
3. **FastF1 soft features at non-driver-row resolution (A13b)** —
   not separately probed.
4. **Wrap-up / hedge-ladder / submission-budget burn** per Rule 12.
