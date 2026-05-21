# Final-window R7d hedge ladder

For the final-3-day window **2026-05-29 → 2026-05-31** per CLAUDE.md
Rule R5d (final-window OOF-best regression probe) and prior-comp
postmortem default R2d (PRIMARY = best public LB; HEDGE = best OOF
regressed ≤30 bp on public).

Re-anchored 2026-05-21 PM on **R25 PRIMARY** (LB 0.95402) after Day-21
inventor track. PRIMARY swap chain R7.1 → R12-2 → R13 → R14 → R15 → R25.
Day-21 lifted +0.05 bp via submission-level rank-blend (K=18 0.89 + R21
0.11), bypassing the Path-B LR-meta saturation on cohort-listwise.

## Ladder slate (snapshot 2026-05-21 PM — R25 PRIMARY set)

| Rank | Mechanism | Submission CSV | OOF | LB | ρ_test vs R25 | Status |
|------|-----------|----|----|----|--------|--------|
| **PRIMARY** | **R25 rank-mean: K=18 × 0.89 + R21 CB YetiRank × 0.11** | `submission_R25_K18_R21_rankmean_w89.csv` | **0.954508** | **0.95402** | 1.000 | **LB-confirmed 2026-05-21** PRIMARY; +0.05 bp vs R15; 351 flips vs K=18 (R7d-eligible) |
| **HEDGE 0** | **R27 4-way: K=18 0.70 + R21 0.12 + K=27 0.10 + R7.2 0.08 rank-mean** | `submission_R27_4way_K18_R21_K27_R72_70_12_10_08.csv` | 0.954519 | 0.95402 | 0.9999 TIE | **LB-confirmed 2026-05-21** TIED PRIMARY; structurally distinct 4-way; final-day hedge candidate |
| HEDGE 1 | R17/R18 K=18 + R17 listwise + Path-B DCS | `submission_K18_pathb_driverclass_stint_tau100000.csv` | 0.954499 | 0.95398 | 0.99988 vs R25 | LB-confirmed; pre-blend K=18 anchor |
| Prior PRIMARY | R15 K=17 + Path-B DC×S τ=100k | `submission_R15_K17_xendcgbase_pathb_dcs_tau100000.csv` | 0.954490 | 0.95397 | 0.99967 vs R25 | LB-confirmed; pure Path-B without R21 leg |
| HEDGE 1 | R14 K=16 (+ TabM) + Path-B DC×S | `submission_R14_K16_tabm_pathb_dcs_tau100000.csv` | 0.954487 | 0.95395 | 0.9999 TIE | LB-confirmed; "remove xendcg" ablation |
| HEDGE 2 | R13 K=15 (+ cb_stint_completion) + Path-B DC×S | `submission_R13_K15_cbh_cbsc_pathb_dcs_tau100000.csv` | 0.954485 | 0.95393 | 0.9998 OK edge | LB-confirmed; "remove TabM+xendcg" ablation |
| HEDGE 3 | R12-2 K=14 (+ cb_horizon) + Path-B DC×S | `submission_R12_cb_horizon_K14_pathb_dcs_tau100000.csv` | 0.954475 | 0.95392 | 0.9998 OK edge | LB-confirmed; ablation chain |
| **HEDGE 4** | **R7d_1**: R15+R7.2+K27 (0.5/0.3/0.2) rank_mean | `R7d_1_R15_R72_K27_50_30_20_rankmean.csv` | 0.954502 | **0.95392** | **0.99985 OK** | **LB-confirmed 2026-05-20**; three-axis (mech + variance + R15) |
| **HEDGE 5** | **R7d_2**: R15+R5.2+K27 (0.6/0.3/0.1) arith | `R7d_2_R15_R52_K27_60_30_10_arith.csv` | 0.954497 | **0.95394** | **0.99990 OK** | **LB-confirmed 2026-05-20**; segmentation + mech axis |
| **HEDGE 6** | **R7d_3**: R15+K27 75/25 rank_mean | `R7d_3_R15_K27_75_25_rankmean.csv` | 0.954495 | **0.95394** | **0.99989 OK** | **LB-confirmed 2026-05-20**; pure-mechanism hedge |
| HEDGE 7 | R7.1 K=13 + Path-B DC×S τ=100k | `submission_K13_pathb_driverclass_stint_tau100000.csv` | 0.954471 | 0.95389 | 0.9997 OK | LB-confirmed; pre-K=14 anchor |
| HEDGE 8 | R7.2 R7.1 + 5-seed fold-fit bag | `submission_K13_dcs_pathb_foldbag.csv` | 0.954497 | 0.95389 | 0.9996 OK | LB-confirmed; variance-reduction leg |
| **HEDGE 9** | **R7d_4**: R8 60/20/20 multi-seg (R7.1+DT+RC) | `submission_R8_blend_60_20_20_r71_dt_rc.csv` | 0.954548 | **0.95389** | TBD | **LB-confirmed 2026-05-20**; multi-segmentation hedge |
| HEDGE 10 | R5.2 K=13 + Path-B C×S τ=100k | `submission_K13_seghmm_pathb_tau100000.csv` | 0.954460 | 0.95387 | 0.9994 OK | LB-confirmed; Compound×S segmentation diversity |
| HEDGE 11 | R10 R7.2+K27 arith 75/25 | `submission_R10_blend_R72_K27_arith_75_25.csv` | 0.954489 | 0.95387 | 0.99988 OK | LB-confirmed; pre-R15 cross-mechanism hedge |

## Diversity axis coverage

| Axis | Coverage |
|------|----------|
| Pool-size class | K=13 (HEDGE 7/8/10), K=14 (HEDGE 3), K=15 (HEDGE 2), K=16 (HEDGE 1), K=17 (PRIMARY) |
| Segmentation class | DriverClass×Stint (PRIMARY + most), Compound×Stint (HEDGE 10), multi-seg DT+RC (HEDGE 9) |
| Mechanism class | Standard pool (PRIMARY chain), wide-pool K=27 (HEDGE 4/5/6/11) |
| Variance class | Single-seed (most), 5-seed fold-bag (HEDGE 8 + leg of HEDGE 4) |
| Loss class | Logloss (PRIMARY), + xendcg-as-base (PRIMARY R15) |

## R7d HEDGE selection rules (R2d / R5d / R7d)

- **Final PRIMARY** = highest LB-confirmed (= R15 0.95397 unless updated).
- **Final HEDGE** = highest OOF such that:
  - LB regression ≤ 30 bp vs PRIMARY's LB ✓ (worst is HEDGE 9/7/8/10/11 at −0.8 to −1.0 bp)
  - ρ_test vs PRIMARY < 0.9999 ✓ (HEDGE 1/14 are at 0.9999 edge)
  - flip count ≥ 200 OR explicit PI sign-off (per Rule R7d override)
- **Override-mechanism rule**: flip count <200 → HEDGE only, never PRIMARY-swap.
  Flip count >200 needs PI sign-off before PRIMARY-swap.

## Today's blend-sweep finding (closes blend-op axis)

13,720 candidates analyzed across 8 ingredients × 4 operators × 2-to-4-way
simplex grid (`audit/2026-05-20-blend-sweep.log`, JSON
`scripts/artifacts/probe_blend_harness.json`). All 4 LB-tested OK-band
candidates regressed −0.3 to −0.8 bp vs R15. The R10 HEDGE 11 datapoint
(−0.02 bp at ρ=0.99988) was a quantization outlier, not a transferable
pattern.

**Conclusion**: blend-operator axis closed. No further blend-op probes
should be expected to lift LB beyond R15 PRIMARY at the 5-decimal
quantization. R7d ladder is now well-populated (11 LB-confirmed hedges,
5 diversity axes covered).

## Final-window submission schedule (Days 28-31)

- **Day 28 (2026-05-28)**: re-verify PRIMARY R15 + top-3 HEDGEs (1/4/5)
  LB against any private-LB drift. Spend ≤3 slots.
- **Day 29 (2026-05-29)**: final selection candidates probe. PI signs off
  on the (PRIMARY, HEDGE) pair.
- **Day 30 (2026-05-30)**: lock selections.
- **Day 31 (2026-05-31)**: select 2 final submissions on Kaggle before
  competition close. Verify CSV match via diff vs LB-confirmed reference.

## Provisional final pair (subject to Day 30 PI sign-off)

- **PRIMARY**: R15 (LB 0.95397) — public-best.
- **HEDGE**: HEDGE 4 (R7d_1, LB 0.95392) — cross-mechanism (K=27) +
  variance (R7.2) + R15 anchor at OOF +0.120 bp. Three-axis structural
  distinctness; if private LB differs from public, this is the highest-EV
  recovery candidate. OOF-best of the OK-band slate.

## Held-back submissions — DO NOT SUBMIT

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
| K=27 + Path-B τ=100k standalone | 0.95368 | ρ=0.9979 REGRESSION_RISK standalone; valuable as blend leg only |
| K=9 qAX (slim-kNN) + Path-B τ=20k | 0.95375 | Slim-kNN-only diversity |
| K=4 + Path-B C×S τ=100k | 0.95351 | Clean reference base |
| 2026-05-12 70/30 K=11+K=9 rank-blend | 0.95386 | First cross-mechanism error-cancellation lift |

## Update log

- 2026-05-19 R10: Created from `state/current.md:117-127`. R8 60/20/20
  promoted from HANDOVER held-state to HEDGE 2 candidate. R10 blend-op
  sweep HEDGE 3 slot reserved.
- 2026-05-19 PM: HEDGE 3 LB-confirmed **0.95387** (-0.02 bp vs PRIMARY
  R7.1 0.95389). Single OK-band candidate LB-validated.
- 2026-05-19 PM (later): PRIMARY chain swap R7.1 → R12-2 → R13 → R14 → R15.
- **2026-05-20 R7d build**: ladder re-anchored on R15 PRIMARY (LB 0.95397).
  Blend-op sweep ran 13,720 candidates; 4 OK-band hedges LB-confirmed
  (HEDGE 4 0.95392, HEDGE 5 0.95394, HEDGE 6 0.95394) + R8 multi-seg
  finally LB-validated (HEDGE 9 0.95389). Blend-op axis closed: no
  candidate lifts beyond R15 at 5-decimal quantization.
