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
| A5 | PRIMARY's residual loss is concentrated in INTERMEDIATE / WET compound rows | MEASURED, but residual is INTRINSIC | live (refined) | Probe A: 8 worst (Compound × Stint × position) cells AUC 0.68–0.86 vs global 0.954, but those cells are ≤1k rows each. Probe `probe_rain_specialist.py`: a single-LGBM specialist on all 18,737 rain rows hits AUC 0.92641 vs PRIMARY's 0.94157 on the same rows (−152 bp); mixed-prediction global AUC regresses 4.17 bp. PRIMARY already extracts near-ceiling via cross-Compound transfer. | 2026-05-08 |
| A22 | Rain-specialist axis is closed — a fresh model trained on rain only loses cross-Compound transfer signal | MEASURED | live (NEW) | `probe_rain_specialist.py`; specialist −152 bp on its own segment, no path to meta-gate lift | 2026-05-08 |
| A23 | Meta-architecture redesign at K=27 via segmentation cross is exhausted (DROPPED in favour of A9b) | superseded | dropped | Both alt segmentations this session at-or-below PRIMARY across all τ; combined with prior 9 variants → 11+ tested. See A9b for the canonical row | 2026-05-08 |
| A24 | Rain-row sample-weighting in a global single-model preserves transfer better than a specialist (the "preserves transfer" intuition motivating Path 2) | FALSIFIED | dropped | `probe_rain_weighted.py`. K=2 meta NULL at weights {1,3,5,10}; even unweighted single LGBM is −70 bp on rain vs PRIMARY because PRIMARY's pool is multi-class and a single LGBM can't replicate that. The "preserves transfer" framing was wrong: PRIMARY's transfer comes from the multi-base pool, not from training-on-everyone | 2026-05-08 |
| A6 | Per-row FE on the 14 raw columns is dead (residual variance ≈ marginal variance) | MEASURED | live | Five separate probes per `state/hypothesis-board.md` "load-bearing" item 2 | 2026-05-07 |
| A7 | Target reformulations (`inv-laps`, `pit-horizon`, `reverse-cumulative`, `stint-progress`) are leaky under standard CV unless aggregates are refit per fold | MEASURED | live | `audit/2026-05-06-target-reform-leakage-audit.md`; collapse rates 88-100% | 2026-05-06 |
| A8 | K=22 + Path-B Compound × Stint is the local optimum among 9 tested meta variants | MEASURED | live | `state/current.md`; "Nine variants tested across Days 14-19" | 2026-05-07 |
| A9 | The K=22 LR-meta is rank-locked: any standalone OOF-computable base is absorbed | INFERRED | live | 5 cross-confirmations per `audit/2026-05-16-d16-virgin-axes-results.md` F4. Strong inference but doesn't rule out base-classes that violate the absorption argument's premises | 2026-05-08 |
| A9b | At K=27, alternative Path-B segmentations (Compound × Stint × Year, Compound × RaceProgress-bin) are all NULL or regressing vs PRIMARY (Compound × Stint, τ=100k) | MEASURED | live (NEW) | This session, `scripts/probe_path_b_alt_segs.py`. cs_y delta vs PRIMARY: −2.25 / −0.61 / −0.16 bp at τ ∈ {5k, 20k, 100k}. c_rp delta vs PRIMARY: −1.71 / −0.65 / −0.16 bp. At τ=100k, PRIMARY's hier-meta lifts only +0.03 bp over a plain global K=27 LR-meta, so the "Path-B amp" is essentially gone at this pool size. Combined with the d14 Year/YxStint/Race sweep (9 variants null at K=21), d18 Compound×Year (null at K=24), and Yao/Vehtari covariance variant (null), the per-segment-stacker family has now been tested across **11+ variants** all at-or-below PRIMARY OOF | 2026-05-08 |
| A9c | Path-B amp is **+0.0–0.4 bp at every pool size we've tested** (K=3, K=5, K=7, K=10, K=27). The "Path-B amp" framing the team has used since Day-13 is statistical noise within the +0.04 bp range, NOT a real meta-architecture lift | MEASURED | live (NEW) | `scripts/probe_pool_structure.py`. K=3 Δ(B−P) = −0.02; K=5 = +0.16; K=7 = +0.32; K=10 = +0.24; K=27 = +0.34. The Day-13 +18 bp claim was pool-specific and never reproducible. Implication: any future attempt to "improve Path-B" is fitting noise | 2026-05-08 |
| A25 | The K=27 pool's effective rank is **3.23 in logit, 1.63 in probability** — the same dimensionality as K=24 was. The d18 "DGP-class" bases (e2_preimage_knn, f2_constraint, chain_decomp) added ~0.35 to logit eff-rank, not a new direction | MEASURED | live (NEW) | `scripts/probe_pool_structure.py`. Top 5 logit components capture 93% variance. Top redundant pair: f2_constraint ↔ HGBC-class at ρ=+0.985 — f2_constraint is structurally an HGBC clone, not a "DGP" base | 2026-05-08 |
| A26 | A K=10 forward-greedy sparse pool has the **same LB performance as K=27 PRIMARY** within 1.2 bp. The 17 extra bases in K=27 buy us exactly 1.2 bp of OOF, exactly 1.2 bp of LB | MEASURED | live (NEW) | `submissions/submission_K10_fwd_pathb.csv` LB **0.95356** vs PRIMARY 0.95368 (Δ −1.2 bp). OOF Δ −1.19 bp → LB Δ −1.2 bp = **exact transfer at 0.1-bp precision**. | 2026-05-08 |
| A26b | A K=4 forward-greedy sparse pool (4 bases: yekenot-RealMLP, p1_cb_v4, f1_hgbc_deep, d16_orig_continuous_only) gets within **1.7 bp of PRIMARY on LB** despite OOF Δ −2.93 bp. The 23 unused bases buy 1.7 bp of LB | MEASURED | live (NEW) | `submissions/submission_K4_fwd_pathb.csv` LB **0.95351** vs PRIMARY 0.95368. OOF Δ −2.93 → LB Δ only −1.7 (LB *better* than OOF predicted by ~1.2 bp). | 2026-05-08 |
| A27 | OOF → LB transfer at this pool's modeling scale is **roughly 1:1 with ~1 bp of sample noise** at this scale. The K=10 case was within 0.1 bp; the K=4 case was off by 1.2 bp in our favour. **Refines morning A27**: don't treat OOF→LB as exact at sub-bp precision | MEASURED | live (NEW, refined) | A26 + A26b two datapoints. Implies: cross-submission LB deltas trust to ~±1 bp, not ±0.1 bp. The bootstrap CI ±12 bp is for absolute LB level | 2026-05-08 |
| A28 | At K=10+1 (sparse pool), the d16 GRU sequence base absorbs identically to its K=22+1 result. Δ at K=10+1 = −0.045 bp (vs original Δ −0.043 bp at K=22+1) | MEASURED | live (NEW) | `scripts/probe_exp1_gru_retest.py`. Rank-lock at the LR meta is **pool-size-independent**: a base whose prediction lies in the existing logit subspace gets absorbed regardless of how dense the pool is. The "dense pool concealing signal" hypothesis is FALSIFIED for the GRU candidate; strong prior to skip rerunning field-state / H9 / lead-lag | 2026-05-08 |
| A29 | Rank-lock is at the **logit-direction level**, NOT at the rank-correlation level. Three structurally-different inductive biases (LambdaRank per-stint, inter-stint memory features, stint-completion dual-head) all produce predictions with low Spearman ρ to K=10 (0.41–0.73) but ALL absorb at K=10+1 LR meta within ±0.05 bp | MEASURED | live (NEW) | `scripts/probe_exp{2,3,4}_*.py`. EXP-2 LambdaRank: ρ 0.73, Δ +0.042. EXP-3 inter-stint: ρ 0.47, Δ −0.011. EXP-4 dual-head: ρ 0.41, Δ −0.014 / +0.035. The K=10 [P, rank, logit] expansion (30 features) can reconstruct any new base's logit as a linear combination — different rank info doesn't add a logit dimension | 2026-05-08 |
| A30 | The 3-D logit subspace of the K=10 / K=27 pool is the **information ceiling of the 14-feature row-level data under the LR-meta family**. Breaking it requires either (i) data beyond the 14 columns, or (ii) a non-LR meta architecture | INFERRED | live (NEW) | A29 + A25 + EXP-2/3/4 NULL across 3 distinct task framings. The inference is strong: every task framing, model class, and feature engineering recipe at row level has been tested null at K=10+1. Doesn't rule out a nonlinear meta-projection (gradient boosting on predictions or neural meta-learner) — that's the only architecturally untested avenue | 2026-05-08 |
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

## What the handover should say if A10, A15, A23 are dropped

The actually-open axes given the dropped claims are:
1. ~~**Targeted modelling on rain-condition rows** — closed by A22
   (single-model specialist −152 bp on rain segment).~~ A richer
   specialist (full-pool retrain on rain only) is theoretically possible
   but the cross-Compound transfer evidence makes it unlikely to lift.
2. ~~**Meta-architecture redesign beyond Compound × Stint** — closed by
   A9b. Two more variants tested null this session (Compound × Stint ×
   Year, Compound × RaceProgress-bin); brings per-segment-stacker family
   to 11+ variants all at-or-below PRIMARY. At τ=100k, PRIMARY's lift
   over a plain global K=27 LR-meta is +0.03 bp — the "Path-B amp" is
   essentially gone at this pool size.~~ Untested variants that violate
   the LR-routing premise: nested hierarchy (per-Compound, then per-Stint
   within), non-LR per-segment models (e.g., per-segment XGBoost head),
   non-Gaussian shrinkage. None of these was tested.
3. ~~**Rain-row sample-weighting in a global model** — closed by A24
   (this session): K=2 LR-meta NULL at all weights {1,3,5,10}.~~
4. **FastF1 soft features at non-driver-row resolution (A13b)** —
   not separately probed. Distinct from FastF1 hard-join (capped at
   1.4% match rate).
5. **Wrap-up / hedge-ladder / submission-budget burn** per Rule 12.
6. **Scheduled assumption-recheck loop** — every entry in this file
   tagged `live (low confidence)` should be re-checked at every
   postmortem and at handover prep.
