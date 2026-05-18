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

## 2026-05-18 session — Tier-A null + Round-2 9-of-9 null

Two execution rounds today. Round 1: original Tier-A FE batch
+ formal Research-loop. Round 2: PI-requested larger-step-size
iteration via `problem-solving.md` step-1 re-entry + two fresh
Opus personas (10 Wild Options + Junior ML Engineer).

**Net of both rounds: 11 distinct mechanism classes tested today;
ALL nulled or regressed at the K=4+1 gate (the operative gate
because K=11 OOFs are not in the 2026-05-08 artifact snapshot).**

### Round-1 picks (4)

| Pick | Type | Result |
|---|---|---|
| a2_2_mandatory_compound_rule | feature add | K=4+1 +0.302 bp ρ=0.983 WEAK |
| a3_1_rank_sorted_gaps | feature add | K=4+1 +0.337 bp ρ=0.983 WEAK |
| C3 per-(Race, LapNumber) shrinkage | post-process | FALSIFIED 3 variants |
| C4 UID magic-features | base | smoke FAIL -16.2 bp |

### Round-2 picks (9 — all NULL/REGRESS)

| Pick | Mechanism class | Result |
|---|---|---|
| P0.1 RRF blend (k=60) | post-process | K=4+1 +0.060 bp ρ=0.983 |
| P0.2 Trimmed-rank (1,1) | post-process | same as above |
| P0.3 Stint-cap multiplier (15 cells) | post-process | every cell regressed |
| P1.1 LightGBM `rank_xendcg` meta | meta loss | -9.5 bp REGRESS |
| P1.2 SGD hinge pairwise meta | meta loss | -21.2 bp REGRESS |
| P1.3 Torch AUC-surrogate MLP | meta loss | -63.2 bp REGRESS |
| P2.1 Per-Driver random-effect (12 cells) | meta enhancement | -73 to -363 bp REGRESS |
| P2.2 Per-bin conformal widths | meta feature | +0.012 bp TIE |

**Major closed axis from Round 2: LR with log-loss is loss-OPTIMAL
at the K=4 meta layer.** Three AUC-aligned losses (LGBM rank,
SGD pairwise, torch surrogate) all underperform LR. The
[P, rank, logit] expansion + log-loss training extracts both
ranking AND calibration; pairwise losses discard the calibration
without adding new ranking info.

## Strategic state

Three independent axes are now closed (Round 2):
1. Loss-class diversity at meta layer (LR optimal).
2. Per-actor (Driver) heterogeneity capture (BLUP adds noise).
3. Uncertainty quantification meta-features (per-bin std ties).

Combined with the Tier-A null, **the post-Tier-A axis is closed
for row-feature + row-prediction-meta mechanisms** on this comp's
K=4 + Path-B base. Per Rule 4's escape clause, we have run the
Research-loop (2026-05-18) and rotated 2 personas; the team can
defensibly state "row-feature ceiling reached."

Remaining lift sources are **outside** the row-feature space:
- **C2 swap-noise DAE on combined train+test** (~2-3 hr Kaggle T4).
  Porto Seguro 1st-place. d15b vanilla DAE lived in hedge at
  +0.79 bp K=22+1; swap-noise variant is structurally different.
  **Highest-EV remaining pick.**
- **C1 OpenF1 per-Race scalar join** (~45 min). 26-Race-level
  external join; novel join key.
- **EXP-9 Gap-aware sequence transformer** (~4-6 hr T4×2). Final-
  window reserve.
- **A3 TabDDPM diffusion imputation** (~2-3 hr T4). Speculative.
- **B2 GraphSAGE Driver-Race-Compound tripartite** (~2-3 hr T4).
  Speculative.

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
2. `state/current.md` — current PRIMARY, LB ladder.
3. **THIS FILE** — held-submissions warning above + today's
   11-of-11 null.
4. `audit/2026-05-18-round-2-execution.md` — full breakdown of
   Round-2 9 picks; specific findings on AUC-loss-class closure.
5. `audit/2026-05-18-plateau-brainstorm.md` — persona output
   + filter rationale.
6. `audit/2026-05-18-tier-a-batch.md` — Round-1 Tier-A null
   breakdown.
7. `audit/research/2026-05-18-research.md` — Research-loop
   synthesis (C1/C2 still pending).

## Operational fixes (carry-over)

- **Bootstrap KGAT_ token handling.** `bootstrap.sh` should
  detect a KGAT_-prefixed `KAGGLE_API_TOKEN` and UNSET
  `KAGGLE_USERNAME` + `KAGGLE_KEY` before invoking the Kaggle
  CLI. Setting all three causes the CLI to try basic-auth with
  username/key (rejected for KGAT_), 403s on private datasets.

- **Push K=11+K=9 OOFs to Kaggle artifact dataset.** 2026-05-08
  snapshot is 10 days stale. Without this, sessions can only
  gate at K=4+1 (3.5 bp behind actual PRIMARY).

- **LightGBM `LGBMRanker` group-size cap.** Hard limit 10000 rows
  per group. Future ranking-loss probes need bucketed groups
  (5000 is safe).

## Empirical transfer bands (Rule 27, unchanged)

| Band | ρ_test vs PRIMARY | Expectation |
|---|---|---|
| TIE_ZONE | ≥ 0.9999 | LB ties within ±0.05 bp |
| OK transfer | 0.999 ≤ ρ < 0.9999 | Sub-bp to few-bp LB movement |
| REGRESSION_RISK | < 0.999 | Wide-ρ adds overfit CV patterns |

Today's K=4+1 ρ values cluster at 0.982-0.983 — deep in
REGRESSION_RISK band across all 5 picks that produced meta-OOFs.
This is the K=4-era ρ signature; any K=4 +1 base lands at this
ρ because the meta-layer transformation puts everything in
essentially the same neighbourhood of K=4's predictions.

## Next-session first actions (EV / cost order)

1. **Push K=11+K=9 OOFs to Kaggle artifact dataset** (~10 min)
   — unblocks all future K=11+1 gating.
2. **C2 swap-noise DAE on combined train+test** (~2-3 hr Kaggle
   T4) — highest-EV remaining mechanism class.
3. **C1 OpenF1 per-Race scalar join** (~45 min) — novel external
   join key; sub-bp expected.
4. **Hill-climb 3-way / 4-way blend** of LB-confirmed PRIMARYs
   + R5 hedge candidates (~10 min) — pure post-process variance
   reduction.
5. **Final-window posture** — if C2 / C1 null, begin R5/R7 hedge
   ladder preparation. 12 days remain.

## Falsified / dead — additions 2026-05-18

Round-1 additions (carried from earlier handover):
- C3 per-(Race, LapNumber) shrinkage (3 variants).
- C4 UID magic-features (smoke fail; not pursued at full).

Round-2 additions:
- **RRF blend** (k=30/60/100) — meta absorbs at |w|<0.17.
- **Trimmed-rank blend** (trim=1,1) — meta absorbs.
- **Hand-coded stint-cap multiplier** (15 cells) — every cell
  regresses; Path-B already routes Compound × Stint optimally.
- **LightGBM `rank_xendcg` meta** (bucketed groups of 5000) —
  -9.5 bp; closes loss-class novelty test.
- **SGD hinge pairwise meta** (200k pairs/fold) — -21.2 bp.
- **Torch MLP with smooth-AUC surrogate** (30 epochs) — -63.2 bp.
- **Per-Driver random-intercept BLUP shrinkage** (6 lambdas) —
  all -73 to -188 bp.
- **Per-Driver random-slope on RaceProgress** (6 lambdas) —
  all -73 to -363 bp.
- **Per-(Compound, Stint) conformal-like width meta-features**
  (4 widths added) — +0.012 bp tie.

## Hedge ladder (R5 / R7 final-window candidates) — unchanged

- K=4 + Path-B C×S τ=100k (LB 0.95351) — clean reference base.
- K=9 qAX + Path-B τ=20k (LB 0.95375) — slim-kNN solo.
- K=27 + Path-B τ=100k (LB 0.95368) — pre-sparse-pool PRIMARY.
- `d15b_path_b_K22_dae_only_tau{20k,100k}` — Day-15 PRIMARY
  runner-up.
- `path_b_K5_rf_yekenot_tau{5k,20k,100k}` — Forest-base τ-sweep.
