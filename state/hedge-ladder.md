# Final-window R7d hedge ladder

For the final-3-day window **2026-05-28 → 2026-05-31** per CLAUDE.md
Rule R5d (final-window OOF-best regression probe) and prior-comp
postmortem default R2d (PRIMARY = best public LB; HEDGE = best OOF
regressed ≤30 bp on public).

Promoted out of `state/current.md:117-127` on 2026-05-19 (R10
hedge-prep pivot) so the final-window agent can load this without
parsing all of `current.md`, and so PRIMARY-change rewrites of
`current.md` don't clobber ladder edits.

## Ladder slate (post-R10, snapshot 2026-05-19)

| Rank | Mechanism | Submission CSV | OOF | LB | ρ vs PRIMARY | Status |
|------|-----------|----|----|----|--------------|--------|
| **PRIMARY** | R7.1 K=13 + Path-B DC×S τ=100k | `submissions/submission_K13_pathb_driverclass_stint_tau100000.csv` | 0.954471 | **0.95389** | 1.000 | LB-confirmed |
| HEDGE 1 | R7.2 R7.1 + 5-seed fold-fit bag | `submissions/submission_K13_dcs_pathb_foldbag.csv` | 0.954497 | **0.95389** | TIE_ZONE (tied LB) | LB-confirmed |
| HEDGE 2 (proposed) | R8 60/20/20 blend (R7.1 + DriverTier + RaceCluster) | `submissions/submission_R8_blend_60_20_20_r71_dt_rc.csv` | 0.954548 | TBD | TIE_ZONE (.npy ρ=0.99997) | staged; Kaggle CLI 401 |
| HEDGE 3 (proposed) | R10 R7.2 + K27 arith 75/25 | `submissions/submission_R10_blend_R72_K27_arith_75_25.csv` | 0.954489 | TBD | **OK band (.npy ρ=0.999882)** | staged; Kaggle CLI 401 |
| HEDGE 4 | R5.2 K=13 + Path-B C×S τ=100k | `submissions/submission_K13_seghmm_pathb_tau100000.csv` | 0.954460 | 0.95387 | OK band (operator diversity) | LB-confirmed |
| HEDGE 5 (optional) | R10 R7.2 + R6.1 arith 80/20 | `submissions/submission_R10_blend_R72_R61_arith_80_20.csv` | 0.954500 | TBD | TIE_ZONE (.npy ρ=0.999963) | staged; Kaggle CLI 401 |

## Final-window submission schedule (Days 28-31)

- **Day 28 (2026-05-28)**: reconfirm PRIMARY + HEDGE 1 LB scores
  (smoke check against any private-LB drift since 2026-05-18); spend
  3 slots if recent LB data is stale.
- **Day 29 (2026-05-29)**: submit HEDGE 2 + HEDGE 3 if not already
  LB-confirmed; spend 1 slot on one experimental variance hedge
  (e.g., logit-mean of PRIMARY + HEDGE 1) at low priority.
- **Day 30 (2026-05-30)**: lock final selections. Confirm with PI
  the (PRIMARY, HEDGE) pair via `ExitPlanMode`-equivalent sign-off.
- **Day 31 (2026-05-31)**: select 2 final submissions on Kaggle
  before competition close. Verify CSV match via diff vs the
  LB-confirmed reference.

## Decision rule (R2d / R5d / R7d)

- **Final PRIMARY** = highest LB-confirmed.
- **Final HEDGE** = highest OOF such that:
  - LB regression ≤ 30 bp vs PRIMARY's LB, AND
  - ρ vs PRIMARY < 0.9999 (true diversity, not just-a-tied-variant), AND
  - flip count ≥ 200 OR explicit PI sign-off (per Rule R7d override).
- **Override-mechanism rule (R7d)**: flip count <200 → HEDGE only,
  never PRIMARY-swap. Flip count >200 needs PI sign-off before
  PRIMARY-swap.

## Held-back submissions — DO NOT SUBMIT (provenance: state/current.md:104-115)

Day-17 strict fold-safe audit collapsed all target-reformulation
single-add results 88-100%. CSVs still on disk:

- `path_b_K22_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv`
- `path_b_K25_megapool_tau{5k,20k,100k}.csv`
- `path_b_multilevel_τ_*.csv` (5 configs, all null anyway)

Origin: `audit/2026-05-06-target-reform-leakage-audit.md`.

## Older hedge candidates (informational; not on active slate)

| Mechanism | LB | Notes |
|-----------|----|-------|
| K=27 + Path-B τ=100k | 0.95368 | Pre-sparse-pool PRIMARY; bigger pool diversity |
| K=9 qAX (slim-kNN) + Path-B τ=20k | 0.95375 | Slim-kNN-only diversity |
| K=4 + Path-B C×S τ=100k | 0.95351 | Clean reference base |
| 2026-05-12 70/30 K=11+K=9 rank-blend | 0.95386 | First cross-mechanism error-cancellation lift |

Of these, the 0.95375 K=9 qAX and 0.95386 70/30 blend are most useful
for HEDGE-class private-LB variance hedge (lowest ρ vs PRIMARY-class
in pool); the K=27 and K=4 are reference anchors, not active candidates.

## Update log

- 2026-05-19 R10: Created from `state/current.md:117-127`. R8 60/20/20
  promoted from HANDOVER held-state to HEDGE 2 candidate (Phase A
  submit pending Kaggle CLI auth fix). R10 blend-operator sweep
  HEDGE 3 slot reserved.
