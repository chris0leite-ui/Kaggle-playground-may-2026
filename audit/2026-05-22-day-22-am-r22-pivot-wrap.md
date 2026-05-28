# Day-22 AM (2026-05-22) — R22 pivot wrap

Comp-day 22 of 31; 9 days remaining; final-3-day lock window opens 2026-05-29.

## Session arc

1. Last commit `2cd106b R22 public-notebook IDEA-scan — external-data
   augmentation finding` (pushed at end of prior session). PI prompted
   "go" → start executing on the proposed real-driver cohort specialist.
2. **R32 session-start fetch + ISSUES.md read.** Branch
   `claude/research-improvements-jjI84` clean, no upstream changes.
3. **R18 leaf-claim check** — searched scripts/ for prior real-F1 work.
   Found `scripts/probe_r16_real_f1_specialist.py` and audit
   `audit/2026-05-20-round-16-real-f1-specialist.{log,json}` from
   2026-05-20. **R16 already null-tested a comp-only real-F1 specialist.**
4. **R26 precedent surfaced to PI** before claiming the leaf. Verdict
   table from R16:

   | Metric | R16 |
   |---|---|
   | Slice AUC standalone | 0.93966 vs PRIMARY's slice 0.94672 (-70.6 bp) |
   | Replace-at-slice combined | 0.95184 (-26.5 bp) |
   | LR-meta blend combined | 0.954488 (-0.026 bp) |
   | ρ vs R15 PRIMARY | 0.999997 |
   | Verdict band | TIE_ZONE |

   Two ways the proposed plan differs from R16:
   (a) Tighter cohort: my 31 real-F1 drivers / 30K rows (6.9%) vs R16's
       "doesn't start with D" heuristic / 168K rows (38%; contaminated).
   (b) External augmentation: R16 trained only on comp's is_real==1
       subset; proposal adds 101K ext rows on top.

   Friction `external-data-arch-bag-redundant-when-shared-training-data`
   and 4-5 cross-confirmations of `rho-alone-insufficient-for-meta-utility`
   add precedent weight.

5. **PI decision (1st AskUserQuestion):** pivot to "augment one PRIMARY
   base — refit one LGBM in K=18 pool with comp+ext concat; stack-add via
   Path-B". Closer to arunklenin's recipe; bypasses specialist-meta
   absorption issue.

6. **R19 BOTE on the pivoted plan** (`audit/decisions.jsonl` row):
   ```
   ext_aug_lgbm_K19
     family: external_data_aggregate  P(useful): 0.15
     bp band (P/M/O): 0.0 / 0.3 / 1.0
     cost: 45 min  → expected LB: +0.04 bp
     PI-predicted LB: +0.30 bp  (Δ vs agent: +0.26 bp)
     cost-efficiency: 0.001 bp/min
     verdict: SKIP
     Q6 metric_aligned: True
   ```
   BOTE SKIP threshold is 0.01 bp/min; we are 10× below.

7. **2nd AskUserQuestion** with explicit precedent-pricing. PI returned
   "No preference" → agent discretion. Picked option D (cheapest
   information-bearing diagnostic: 10-min matched-subset AV-AUC).

8. **PI message: "wrap up"** before diagnostic could run. Halted.

## Verdict

No new artifacts produced. No leaf claimed. No mechanism family advanced
or falsified. The BOTE in `audit/decisions.jsonl` is the only persistent
artifact from this session.

**Net contribution to the comp:** documented R16 precedent loud enough
that future "real-F1 cohort" / "ext-aug specialist" proposals will short-
circuit. Sealed PI vs agent +0.04 vs +0.30 BOTE prediction for later
calibration. 3 new frictions logged.

## Open question for next session

The PI pivot to "augment one base + Path-B stack-add" was rejected by
BOTE but **never actually tested**. Two cheap paths remain:

- (a) The 10-min matched-subset AV-AUC diagnostic (was about to run when
  PI said wrap). If AV-AUC < 0.55 on (comp real-F1 rows) vs (ext rows),
  ext is clean augmentation for the cohort → BOTE band probably revises
  upward.
- (b) Pivot fully to hedge-ladder R5d/R7d validation, which is priority
  #2 in `state/current.md`. 8 days to final-window lock. PRIMARY+HEDGE
  private-LB regression-risk probe is overdue.

## Decisions log (this session)

```
ext_aug_lgbm_K19 | external_data_aggregate | SKIP | agent +0.04 | PI +0.30
```

PI-vs-agent gap +0.26 bp open. No outcome to pair against yet (probe
didn't run).

## Frictions added (3)

- `r22-finding-overlooked-r16-precedent`
- `pi-vs-agent-bote-7x-gap-on-ext-aug`
- `wrapup-step-3-CLAUDE-md-current-state-stale`

## Calibration snapshot (R26 / step 5b)

```
name                                     family                         actual    agent       PI  agent_err   pi_err
h3_id_shift_row_position                 single_base_fe_addition         +0.00    +0.60    +0.00      +0.60    +0.00
h2_fastf1_external_join                  external_data_aggregate         +0.00    +3.60    +5.00      +3.60    +5.00
h1_yekenot_realmlp_recipe                new_model_class                 +0.00   +27.00    +0.00     +27.00    +0.00
d18_path_b_K23_d16_d18_tau20000          external_data_aggregate         +6.00    +1.26    +3.00      -4.74    -3.00
h1_yekenot_realmlp_recipe                new_model_class                +19.60   +27.00    +0.00      +7.40   -19.60
probe_combined_lead_lag                  single_base_fe_addition         +0.00    +0.03    +0.00      +0.03    +0.00
probe_field_state                        single_base_fe_addition         +0.00    +0.03    +0.00      +0.03    +0.00
probe_field_state                        single_base_fe_addition         +0.00    +0.03    +0.00      +0.03    +0.00
d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau10 external_data_aggregate         +1.40    +0.34        –      -1.06        –
d19_historical_priors_debashish          external_data_aggregate         +0.00    +0.20    -1.00      +0.20    -1.00
b2_xgb_v4_K27_verify                     pool_addition_redundant         +0.14    +0.03        –      -0.11        –
a5_lgbm_v4_fs_K27_proxy                  single_base_fe_addition         -0.11    +0.10        –      +0.21        –
c1_yao_vehtari_path_b_K27                meta_arch_redesign              -0.47    +1.20        –      +1.67        –
svm_kernel_smoke_1e                      new_model_class                 -0.05    +0.10    +0.50      +0.15    +0.55
a3_7_uid_smoothing                       code_fix_calibration          -124.00    +0.30        –    +124.30        –
```

Today's ext_aug_lgbm_K19 decision: PI +0.30 vs agent +0.04 (gap +0.26).
No outcome (probe not run); cannot pair yet.

Of paired rows where PI predicted: agent err signed sum ≈ +37 (over-
predict bias), PI err signed sum ≈ -17 (closer to truth on net, but
larger spread). PI override count this session: 0 (no override applied
on the SKIP verdict; PI granted discretion via "no preference"). Last
session also 0 → if next session also 0, postmortem flags `pi-stamp-risk`
in HANDOVER per WRAPUP step 5b.
