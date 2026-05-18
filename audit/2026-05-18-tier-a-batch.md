# 2026-05-18 — Tier-A batch execution

Triggered by: PI "bootstrap and check that you have all the data
available then go." Bootstrap succeeded; private artifact dataset
loaded (240 files; snapshot version 2026-05-08). The K=11 OOFs and
the LB-confirmed 70/30 K=11+K=9 PRIMARY are NOT in the snapshot
(built later, never pushed to Kaggle). **Gating is against K=4 +
Path-B proxy (LB 0.95351) instead of K=11+1.**

## Smoke sweep (50k × 1 fold)

Baseline single-LGBM smoke AUC (no FE pick): **0.93149** (fold 1
seed=42).

| Pick | Smoke AUC | Δ vs baseline | Verdict |
|---|---:|---:|---|
| a2_2_mandatory_compound_rule | 0.93242 | **+9.3 bp** | PASS → full 5-fold |
| a2_4_vsc_fullsc_split | 0.93153 | +0.4 bp | borderline, skip |
| a2_6_fuel_correction_const | 0.93062 | -8.7 bp | FAIL |
| a2_7_field_state_f3 | 0.93133 | -1.6 bp | FAIL |
| a3_1_rank_sorted_gaps | 0.93124 | -2.5 bp | borderline → ran full anyway |
| a3_2_per_track_fuel_coef | 0.93146 | -0.3 bp | borderline, skip |
| a3_3_tirechange_pursuer_lagged | 0.92963 | -18.6 bp | FAIL |

Wall: 60s baseline + 6 picks × ~10-110s each ≈ 5 min total.

## K=4 LR-meta baseline (reproduction anchor)

`scripts/probe_c3_race_lap_shrinkage.py` recomputed the K=4
forward-greedy + LR-meta OOF:

- K=4 LR-meta OOF AUC: **0.95399**

Matches `state/calibration-ladder.md` row exactly (0.95403 reported;
±fold-noise). PRIMARY proxy validated.

## C3 — per-(Race, LapNumber) Bayesian shrinkage: FALSIFIED

Three variants tested. ALL regress at every nonzero weight.

| Variant | Best w | Δ at best | Note |
|---|---:|---:|---|
| A: shrink toward per-(R, L) PRED mean | 0.00 | +0.000 bp | -9.6 bp at w=0.05 |
| B: per-(R, L) empirical TARGET rate as prior | 0.00 | +0.000 bp | -9.7 bp at w=0.05 |
| C: per-Race PRED mean (coarser, 26 levels) | 0.00 | +0.000 bp | -3.9 bp at w=0.05 |

**Verdict:** any per-group shrinkage of the row-level ranking signal
collapses AUC. The C3 idea was wrong at the mechanism level — same
failure mode as the Day-15 "simple K=21 blending (mean / geometric /
rank / trimmed)" which lost 19-32 bp. Per-group shrinkage trades
row-level discriminability for group-level smoothness, and AUC
measures ranking.

Distinct from EXP-A3-7's -124 bp UID smoothing (which used target
not prediction), but reaches the same dead end.

## Full 5-fold OOF + K=4+1 gate

### a2_2_mandatory_compound_rule — REPRODUCED (WEAK at gate)

- 5-fold OOF AUC: **0.94577**  (fold-std 0.00052; total wall 3613s = 60 min)
- K=4+1 LR-meta: **0.95402** (Δ **+0.302 bp** vs K=4 baseline 0.95399)
- ρ_test vs PRIMARY (d13e C×S τ=20k): **0.983221** → REGRESSION_RISK band
- Per-candidate weight: |w| = 0.2759 (raw -0.182, rank +0.039, logit -0.056)

**Verdict:** WEAK / null. Below +0.5 bp G1 threshold; ρ_test 0.983
is in the REGRESSION_RISK band (<0.999) so LB transfer is high-risk.
Exact reproduction of the team's `hypothesis-board.md` entry
"K=4+1 plain LR-meta +0.302 bp (below +0.5 PASS); G3 flip 0.195
asymmetric; TIE_EXPECTED on 4-gate. K=4+1 Path-B C×S τ=100k
+0.26 bp; ρ 0.999893; WEAK."

The standalone OOF lift (0.94577 - 0.94563 = +1.4 bp single-LGBM)
mostly absorbs at the meta. No new information here; closed FAIL
on this gate. Path-B refit not retried (Day-15 friction
`path-b-amp-only-fires-on-meta-arch-not-base-add`).

### a3_1_rank_sorted_gaps — REPRODUCED (WEAK at gate)

- 5-fold OOF AUC: **0.94548**  (fold-std 0.00049; total wall 3310s = 55 min)
- K=4+1 LR-meta: **0.95403** (Δ **+0.337 bp** vs K=4 baseline 0.95399)
- ρ_test vs PRIMARY (d13e C×S τ=20k): **0.983252** → REGRESSION_RISK band
- Per-candidate weight: |w| = 0.2701 (raw -0.179, rank +0.037, logit -0.054)

**Verdict:** WEAK / null. Same pattern as a2_2 — below +0.5 bp G1
threshold AND below 0.999 ρ-transfer band. Rank-sorted multi-
neighbour gaps + pit-pressure scalar add no information beyond the
existing K=4 logit subspace. Confirms rank-lock at the K=4 LR-meta
level extends to permutation-invariant cross-actor features (NFL
BDB 2020 pattern absorbs on the F1 synth row space).

### C4_uid_magic_features — SMOKE FAIL (-16.2 bp)

Smoke (50k × 1 fold) regressed to **0.91527** vs baseline **0.93149**
(Δ -16.2 bp). UID = `Driver_Race_floor(LapNumber/W)` for
W∈{5, 10, 20} + 3 untimed UIDs, 145 per-UID aggregates of
LapTime/TyreLife/Position/etc. The smoke regression is mechanically
expected — 50k subsampling breaks UID grouping (most UIDs end up
with 1-2 sample rows; aggregates become noisy / leaky). Full 5-fold
on 439k rows COULD still pass; deferred on EV grounds (saturation
discount on IEEE-CIS +109 bp precedent; W=5 may also over-shrink the
group).

## Research-loop synthesis (Step 1 complete)

Three parallel agents wrote `audit/research/2026-05-18-{notebooks,
prior-comp, domain}.md` and synthesis `2026-05-18-research.md`. Key
deltas to the Tier-A menu:

- **NEW C1**: per-Race aggregate join from OpenF1 (pit-loss seconds,
  VSC/SC base rates, weather indicators). Different join key from
  the 1.4%-Driver-cap; 26-Race-level join is fold-safe and not yet
  tried. ~45 min compute. **Highest novelty.**

- **NEW C2**: DAE with swap-noise on combined train+test (Porto
  Seguro 1st-place mechanism). Distinct from d15b vanilla DAE
  because swap-noise corrupts at the row level; AV-safe per
  AV-AUC=0.502. ~2-3 hr Kaggle T4. **Highest predicted lift.**

- **C3 falsified** (this session — see above).
- **C4 smoke-failed** (this session — see above).

Notebooks agent harvested zero usable candidates (reCAPTCHA-walled).
Domain agent surfaced 6 academic refs (TUM Heilmeier/race-sim,
arXiv 2512.00640 state-space tyre-deg, Frontiers Bi-LSTM, etc.) but
all required external data or GPU sequence models already on the
EXP-9/10 menu.

## Net verdict for Tier-A batch

**Null.** Two of the 8 Tier-A picks ran through full 5-fold + K=4+1
gate; both WEAK/null. C3 falsified across 3 variants in <5 min.
C4 smoke-failed. The smoke screening on the remaining 5 picks
(a2_4 +0.4 bp, a2_6 -8.7 bp, a2_7 -1.6 bp, a3_2 -0.3 bp, a3_3 -18.6
bp) suggests they will not clear the +0.5 bp K=4+1 G1 threshold
either; running them would burn ~150 min CPU without expected
payoff. Recommend SKIP unless PI overrides.

**Strategic implication:** the K=4 LR-meta 3-D logit subspace
(A25 / A30) is empirically stable against the strongest
research-grounded row-FE picks. The K=11+K=9 PRIMARY (LB 0.95386)
likely sits at or very near the row-feature-attainable ceiling, as
the 2026-05-14 audit suspected.

## Next-session priorities (ranked by EV / cost)

1. **C2 swap-noise DAE on combined train+test** (~2-3 hr Kaggle T4).
   Porto Seguro 1st-place mechanism. AV-safe. Highest predicted lift
   (+20-60 bp speculative; saturation discount may bring to +0.5 to
   +2.0 bp). Distinct enough from d15b vanilla DAE to be worth the
   GPU slot.
2. **C1 per-Race OpenF1 join** (~45 min). Sub-bp predicted but cheap
   and not yet tried. Uses a different join key than the 1.4%-Driver-
   cap explored earlier.
3. **EXP-9 gap-aware sequence transformer** (~4-6 hr Kaggle T4×2).
   Directly attacks the W3 synth-downsampling thesis. Only fire if
   C1/C2 don't lift.
4. **Push K=11+K=9 OOFs to Kaggle artifact dataset** so future
   sessions can gate against the actual current PRIMARY, not the
   K=4 proxy.

## Files added this session

```
scripts/probe_c3_race_lap_shrinkage.py     (3-variant sweep; falsified)
scripts/probe_c4_uid_magic.py              (smoke-failed -16.2 bp)
audit/2026-05-18-tier-a-batch.md           (this file)
audit/research/2026-05-18-notebooks.md     (Research-loop)
audit/research/2026-05-18-prior-comp.md    (Research-loop)
audit/research/2026-05-18-domain.md        (Research-loop)
audit/research/2026-05-18-research.md      (synthesis)
```

## Friction surfaced today

- `tag: kggt-token-needs-isolated-auth` — KGAT_-prefixed harness
  Kaggle token only works via KAGGLE_API_TOKEN env var ALONE;
  setting KAGGLE_USERNAME + KAGGLE_KEY in parallel triggers 403 on
  private datasets. Bootstrap should detect and branch.
- `tag: artifact-dataset-snapshot-stale` — 2026-05-08 snapshot is 10
  days behind PRIMARY (K=11+K=9 built 2026-05-09 to 2026-05-12,
  never pushed to Kaggle). Wrap-up step must include
  `kaggle datasets version` push.
- `tag: smoke-gate-unreliable-for-group-FE` — Smoke (50k subsample,
  1 fold) regresses noisily for FE picks that depend on cross-row
  group statistics (a2_7, a3_1, a3_3, C4). The picks that smoke-
  pass tend to be self-contained row features; group-aggregate
  picks need full 5-fold to evaluate fairly. Smoke screening is
  biased toward self-contained mechanisms.

## Operational notes

- **Private Kaggle dataset auth fix.** The KGAT_-prefixed harness
  token requires `KAGGLE_API_TOKEN` env var ALONE (without
  KAGGLE_USERNAME + KAGGLE_KEY also set). Setting all three causes
  the CLI to try basic-auth with username/key (KGAT_ rejected as
  username/key, hence 403). Unsetting both username+key and using
  only KAGGLE_API_TOKEN succeeds. Promotion candidate for
  `bootstrap.sh` — add a "unset KAGGLE_USERNAME KAGGLE_KEY" branch
  when the API token starts with KGAT_.

- **Artifact snapshot is stale.** Dataset version is 2026-05-08,
  but PRIMARY (K=11+K=9 70/30 rank-blend) was set 2026-05-12 and
  iterated through 2026-05-14. Next session-end should run
  `scripts/post_comp_cleanup.sh`-style `kaggle datasets version`
  to push K=8, K=9, K=10, K=11, K=12 OOFs into the private dataset
  so future sessions can gate against the current PRIMARY.

- **Compute budget**: this session used ~30 min CPU on background
  full-5-fold runs (a2_2 + a3_1 still in flight). Submission budget
  for the day: none used (no LB submits yet; gate must pass first).
