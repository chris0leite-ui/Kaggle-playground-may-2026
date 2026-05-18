# HANDOVER

Next-session brief. **PI says "handover"** → agent reads this file
and proceeds. **PI says "prepare handover"** → agent rewrites it
following `WRAPUP.md` section B.

This file is rewritten (not tail-appended) every wrap-up. Prior
versions: `audit/archive-YYYY-MM-DD-handover-*.md`.

---

## Where we are

**Active PRIMARY: rank-blend 70/30 K=11 + K=9. LB 0.95386.**
Unchanged from 2026-05-14. Top-5% gap −1.9 bp; leader gap −9.0 bp.

Submissions: **42 / 270** total; **0 used 2026-05-18**. Comp-day
**18 of 31**; days remaining **13**.

File: `submissions/submission_blend_K11_K9_w_70_30.csv`.

## 2026-05-18 session — Tier-A batch and Research-loop done

The plan-mode session approved (a) Strategy-critic + Research-loop
before any compute, and (b) the cheap Tier-A feature batch as the
primary compute. Strategy-critic was deferred (needs K=11 OOFs which
are NOT in the 2026-05-08 artifact snapshot); Research-loop ran in
full; Tier-A batch ran on a K=4+1 proxy gate.

**Net result: NULL.** No mechanism cleared the +0.5 bp G1 threshold.

### What was probed

| Pick | Class | Smoke (50k×1) | 5-fold OOF | K=4+1 Δ | ρ_test | Verdict |
|---|---|---:|---:|---:|---:|---|
| a2_2_mandatory_compound_rule | feature | +9.3 bp | 0.94577 | +0.302 bp | 0.9832 | WEAK |
| a3_1_rank_sorted_gaps (NFL BDB) | feature | -2.5 bp | 0.94548 | +0.337 bp | 0.9833 | WEAK |
| C3 per-(Race, LapNumber) shrinkage | post-process | n/a | n/a | -3.9 to -188 bp | n/a | FALSIFIED |
| C4 UID magic-features (IEEE-CIS) | base | -16.2 bp | n/a | n/a | n/a | smoke FAIL |
| a2_4 / a2_6 / a2_7 / a3_2 / a3_3 | feature | -18.6 to +0.4 bp | not run | n/a | n/a | smoke-skip |

Both K=4+1 gates landed in REGRESSION_RISK band (ρ < 0.999); per
Rule 27 they're abort-or-PI-authorise-override candidates. Below the
+0.5 bp G1 threshold makes the call easy: no submission.

### Research-loop output

Three parallel agents wrote `audit/research/2026-05-18-{notebooks,
prior-comp, domain,research}.md`. After dedup vs ledger, 4 new
candidates emerged. Two are now exhausted (C3 falsified, C4 smoke-
failed). Two remain queued:

- **C1 — per-Race OpenF1 join** (~45 min CPU; predicted +0.1 to
  +0.5 bp). Different join key than the 1.4%-Driver-cap explored
  earlier. Fold-safe.
- **C2 — DAE with swap-noise on combined train+test** (~2-3 hr
  Kaggle T4; predicted +0.2 to +0.8 bp after saturation discount).
  Distinct from d15b vanilla DAE. AV-safe per AV-AUC=0.502. Porto
  Seguro 1st-place precedent.

## Why the Tier-A batch nulled

- **Rank-lock at K=4 LR-meta is robust.** The 3-D logit subspace
  (A25 / A30) absorbs both the FIA-regulation feature class
  (a2_2 mandatory_compound_rule) and the permutation-invariant
  cross-actor feature class (a3_1 rank-sorted gaps, NFL BDB pattern).
  These were the strongest research-grounded picks; both produce a
  standalone single-LGBM lift but absorb at the meta to +0.3 bp.
- **C3 shrinkage falsified across 3 variants.** Per-(Race, LapNumber)
  shrinkage toward pred-mean / target-rate / per-Race mean all
  collapse ranking. Same failure mode as the Day-15 simple K=21
  blending; per-group smoothing trades row-level discriminability
  for group means.
- **C4 smoke-failed due to subsample-breaking-UID-groups.** Smoke
  is non-evaluable for UID-grouped FE; full run requires 30-60 min
  CPU and saturation discount makes EV marginal.

## Strategic posture for next 13 days

The K=4 LR-meta 3-D logit subspace is empirically stable against
the strongest research-grounded row-FE picks of the May-2026 menu.
**Confirms the 2026-05-14 audit's "Bayes-optimal ceiling on row
features" interpretation.** Lift options:

1. **C2 swap-noise DAE on combined train+test** — highest predicted
   lift in the research scan. Distinct from d15b (vanilla DAE).
2. **C1 OpenF1 per-Race scalar features** — different join key.
3. **EXP-9 gap-aware sequence transformer** — directly attacks the
   W3 synth-downsampling thesis. Final-window reserve.
4. **Multi-seed bagging of full K=11 pipeline** — variance-reduction
   stack-add (+0.1 to +0.3 bp tie-band; ~12 hr).

## 🔴 Critical: held submissions — DO NOT submit

Day-17 strict fold-safe audit collapsed all target-reformulation
single-add results 88-100% (Rule 24 origin). Files still on disk
but invalidated:

- `path_b_K22_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K25_megapool_tau{5k,20k,100k}.csv`
- `path_b_multilevel_τ_*.csv`

Origin: `audit/2026-05-06-target-reform-leakage-audit.md`.

## Read order on session start

1. `CLAUDE.md` — rules + pointers.
2. `state/current.md` — current PRIMARY, LB ladder, axes status.
3. **THIS FILE** — held-submissions warning above; the 2026-05-18
   Tier-A null result.
4. `audit/2026-05-18-tier-a-batch.md` — full breakdown of today's
   probes; pointers to research artifacts.
5. `audit/research/2026-05-18-research.md` — C1/C2/C3/C4 synthesis;
   C3 and C4 closed this session; C1 and C2 remain pending.
6. `audit/2026-05-14-overnight-iteration.md` — prior plateau audit.
7. `audit/friction.md` — current-week friction summary.

For detail when needed:

- `state/mechanism-ledger.md` — every mechanism family probed.
- `state/calibration-ladder.md` — OOF / LB anchors per family.
- `state/hypothesis-board.md` — open / killed list.
- `glossary.md` — abbreviations, frozen short-codes.

## Operational fixes for next session

- **Bootstrap KGAT_ token handling.** `bootstrap.sh` should detect
  a KGAT_-prefixed `KAGGLE_API_TOKEN` and explicitly UNSET
  `KAGGLE_USERNAME` and `KAGGLE_KEY` before invoking the Kaggle CLI.
  Setting all three simultaneously causes the CLI to try basic-auth
  with username/key (KGAT_ rejected), which 403s on private
  datasets and 401s on kernels.

- **Push K=11+K=9 OOFs to Kaggle artifact dataset.** The 2026-05-08
  snapshot is 10 days stale. Without this, future sessions can only
  gate against K=4 + Path-B (3.5 bp behind actual PRIMARY).

## Empirical transfer bands (Rule 27, unchanged)

| Band | ρ_test vs PRIMARY | Expectation |
|---|---|---|
| TIE_ZONE | ≥ 0.9999 | LB ties within ±0.05 bp |
| OK transfer | 0.999 ≤ ρ < 0.9999 | Sub-bp to few-bp LB movement |
| REGRESSION_RISK | < 0.999 | Wide-ρ adds overfit CV patterns |

Today's probes (a2_2 ρ=0.9832, a3_1 ρ=0.9833) sit deep in
REGRESSION_RISK band. The K=4 LR-meta + new-base ρ pattern is
remarkably uniform across mechanism families: every base in the
K=4-era falls in the 0.98 range, suggesting the meta-layer
transformation puts everything in essentially the same
neighbourhood of K=4's predictions.

## Next-session first actions (EV / cost order)

1. **C1 OpenF1 per-Race join** (~45 min). Lowest cost, novel join
   key. If +0.5 bp at K=4+1 → submit.
2. **Push K=11 OOFs to artifact dataset** (~10 min Kaggle dataset
   version). Unblocks K=11+1 gating for all future sessions.
3. **C2 swap-noise DAE on Kaggle T4** (~2-3 hr GPU). Porto Seguro
   1st-place precedent; highest predicted lift in the menu.
4. **Hill-climb 3-way / 4-way blend** of LB-confirmed PRIMARYs +
   R5 hedge candidates (~10 min; pure post-process).
5. **If C1/C2 are NULL**: trigger EXP-9 (gap-aware sequence
   transformer) on Kaggle T4×2.

## Falsified / dead — do not retry (additions 2026-05-18)

- **C3 per-(Race, LapNumber) Bayesian shrinkage** (3 variants:
  pred-mean / target-rate / per-Race coarser). All regress at every
  weight. Per-group shrinkage of row-level ranking collapses AUC.
  See `scripts/probe_c3_race_lap_shrinkage.py`.
- **C4 UID magic-features** (smoke-only) — smoke regression -16.2 bp
  due to subsample-breaking UID groups; full-run not justified on
  EV given Tier-A null pattern. Code preserved at
  `scripts/probe_c4_uid_magic.py` for future retry on full data if
  C1/C2 also null.

## Hedge ladder (R5 / R7 final-window candidates) — unchanged

- K=4 + Path-B C×S τ=100k (LB 0.95351) — clean reference base.
- K=9 qAX + Path-B τ=20k (LB 0.95375) — slim-kNN solo.
- K=27 + Path-B τ=100k (LB 0.95368) — pre-sparse-pool PRIMARY.
- `d15b_path_b_K22_dae_only_tau{20k,100k}` — Day-15 PRIMARY runner-up.
- `path_b_K5_rf_yekenot_tau{5k,20k,100k}` — Forest-base τ-sweep.
