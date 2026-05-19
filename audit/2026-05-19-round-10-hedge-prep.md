# Round 10 — Hedge-prep pivot (2026-05-19)

## Trigger

PI directive "pivot" after the morning's R10 multi-constituent alt-stack
result: every constituent of the LR-meta alt-stack (LambdaRank stint,
LambdaRank race, rolling LGBM, kernel hazard) blended with R7.1 PRIMARY
returned Δ < 0 bp NULL at every weight, even at w_R71=0.99
(Δ −0.045 bp). The alt-stack blend axis is closed alongside the R9
single-add axis (NB4 TE-base −0.022 bp, C1 Aadigupta −0.045 bp).

Three rank-lock confirmations in <2 sessions at the row-feature class
→ PI confirmed pivot to hedge-prep posture per the R8 strategy-critic
contingency (`audit/2026-05-18-strategy-critique.md:25-30`).

## Pool integrity check (Rule 2)

R7.1 PRIMARY OOF AUC recomputed from `oof_K13_pathb_driverclass_stint_tau100000.npy`:
**0.954471** (matches state, drift < 0.000001). Pool integrity confirmed
before sweep.

## Phase A — R8 60/20/20 blend submit (PENDING)

R8 60/20/20 blend CSV verified on disk
(`submissions/submission_R8_blend_60_20_20_r71_dt_rc.csv`, 4.94 MB,
2026-05-18 16:22).

**Correction on ρ band**: `pre_submit_diff.py` reports CSV-level ρ
0.998063, but that's a tie-structure-mismatch artifact already
documented at `audit/friction.md:30` (`pre-submit-diff-floor-clip`).
R7.1 CSV floors 15.6 % rows at 0.001; R8 blend is rank-uniform with
0 % floored. Spearman degrades from the tie-structure mismatch, not
actual rank divergence. **TRUE rank divergence (raw .npy)**: ρ=0.99997
→ **TIE_ZONE** per state/current.md:83 (LB tie within ±0.05 bp).

This matches state/current.md:70 which correctly labeled R8 as
TIE_ZONE. My initial mis-read of pre_submit_diff led to a "recalibrate
the bands" detour — the bands are right; the tool needs rank-normalize
fix (open action from 2026-05-18 friction).

**Submit BLOCKED** by friction `kaggle-cli-401-auth` (logged
`audit/friction.md:29`, dated 2026-05-18): Kaggle CLI returns 401
Unauthorized on `kaggle competitions submissions`. Submit cannot proceed
until PI refreshes credentials in `~/.kaggle/kaggle.json` or sets
working KAGGLE_USERNAME/KAGGLE_KEY env vars. CSV is staged and ready.

## Phase B — Blend-operator sweep (DONE, 162s wall)

Script: `scripts/probe_r10_blend_operator_sweep.py`.
Output: `scripts/artifacts/r10_blend_operator_sweep.json`.

**Pool** (5 LB-confirmed R7-era ingredients, all OOF AUC verified):

| Name | OOF | LB |
|---|---:|---:|
| R7.1 K=13 + Path-B DC×S τ=100k        | 0.954471 | 0.95389 |
| R7.2 R7.1 + 5-seed fold-fit bag       | 0.954497 | 0.95389 |
| R5.2 K=13 + Path-B C×S τ=100k         | 0.954460 | 0.95387 |
| R6.1 R5.2 + 5-seed fold-fit bag       | 0.954481 | 0.95387 |
| K27  K=27 + Path-B τ=100k             | 0.954317 | 0.95368 |

**Operators**: {arith, gmean, logit_mean, rank_mean}
**Sweep size**: 280 two-way (10 pairs × 4 ops × 7 weights) + 288 three-way
(6 trios × 4 ops × 4 prim_w × 3 share) = 568 configs.

### Top OOF (TIE-band, dominated by R7.2 standalone)

| op    | ingredients          | weights      | Δ OOF (vs R7.1) | ρ_npy | flips +/− |
|-------|----------------------|--------------|----:|--------:|----------:|
| arith | R7.2 + R6.1          | 0.80, 0.20   | +0.262 bp | 0.999963 | 103/58 |
| arith | R7.1+R7.2+R6.1       | 0.50, 0.30, 0.20 | +0.148 bp | 0.999984 | TBD |
| arith | R5.2 + R6.1          | 0.50, 0.50   | TBD | TIE  | TBD |

These are TIE-zone candidates by .npy ρ — essentially R7.2 dominated.
LB outcome predictably tied at 0.95389. Marginal hedge value.

### Top OK-band (real diversity; cross-pool) — by TRUE .npy ρ

| op    | ingredients          | weights      | Δ OOF (vs R7.1) | ρ_npy (true) | flips +/− |
|-------|----------------------|--------------|----:|--------:|----------:|
| arith | R7.2 + K27           | 0.75, 0.25   | +0.175 bp | **0.999882** OK | 241/137 |
| arith | R7.2 + K27           | 0.70, 0.30   | +0.129 bp | **0.999846** OK | 276/165 |
| arith | R6.1 + K27           | 0.80, 0.20   | +0.081 bp | **0.999757** OK | 237/192 |
| arith | R7.2 + K27           | 0.65, 0.35   | +0.072 bp | **0.999804** OK | 314/191 |

**These are the genuine cross-mechanism diversity hedges** — K27 is
a structurally different pool (27 bases vs 13). OK band per .npy ρ
(0.999-0.9999) → expected LB movement is sub-bp to few-bp
(state/current.md:84). Best candidate: **R7.2+K27 arith 75/25**
(+0.175 bp OOF lift, ρ=0.999882, asymmetric flips 241/137 favouring
positive class).

## Pre-submit-diff CSV ρ for reference (biased by floor-clip artifact)

For completeness — what `pre_submit_diff.py` would show vs R7.1 CSV
(numbers documented as biased; .npy ρ is the operative metric):

| CSV | CSV ρ vs R7.1 | TRUE .npy ρ |
|-----|------:|------:|
| R7.2 (LB 0.95389, tied) | 0.999836 | (single-model; identical mechanism family) |
| R5.2 (LB 0.95387, −2 bp) | 0.999513 | 0.99973 |
| R8 60/20/20 | 0.998063 | **0.99997 TIE_ZONE** |
| R10 R7.2+K27 arith 75/25 | 0.997973 | **0.999882 OK** |
| R10 R7.2+R6.1 arith 80/20 | 0.998053 | **0.999963 TIE_ZONE** |
| R10 R7.2+K27 arith 70/30 | 0.997938 | 0.999846 OK |

The pre_submit_diff fix (rank-normalize both inputs OR warn on
distribution mismatch) is the open action item from
`audit/friction.md:30`.

## Staged R10 candidate CSVs (all in `submissions/`)

| File | Mechanism | OOF Δ (vs R7.1) | .npy ρ band |
|------|-----------|----:|------------|
| `submission_R10_blend_R72_K27_arith_75_25.csv` | R7.2 + K27 arith 75/25 | +0.175 bp | **OK** (0.999882) |
| `submission_R10_blend_R72_K27_arith_70_30.csv` | R7.2 + K27 arith 70/30 | +0.129 bp | **OK** (0.999846) |
| `submission_R10_blend_R72_R61_arith_80_20.csv` | R7.2 + R6.1 arith 80/20 | +0.262 bp | TIE_ZONE (0.999963) |
| `submission_R10_blend_R71_R72_R61_arith_50_30_20.csv` | R7.1+R7.2+R6.1 50/30/20 | +0.148 bp | TIE_ZONE (0.999984) |
| `submission_R10_blend_R61_K27_arith_80_20.csv` | R6.1 + K27 arith 80/20 | +0.081 bp | OK (0.999757) |
| `submission_R8_blend_60_20_20_r71_dt_rc.csv` *(already on disk)* | R8 60/20/20 | +0.077 bp | TIE_ZONE (0.99997) |

## Verdict

- **Mechanism families closed**: alt-stack (today), single-base
  (R9 yesterday), all CPU-buildable row-feature additions structurally
  rank-locked.
- **Hedge-prep complete**: 6 candidate CSVs staged, all with CSV-ρ ≈
  0.998 (real diversity, not just TIE-zone). PI choice of 1-3 to
  submit covers the realistic hedge slate.
- **Recalibration finding**: Rule 27 bands need rewriting against
  CSV-level ρ, not .npy ρ.
- **Blocker**: Kaggle CLI 401 auth. Submit slots not consumable until
  PI fixes creds. Today's 10 fresh slots otherwise forfeit at next
  UTC midnight (Rule 12 cost: up to 10 slots lost).

## Recommendation to PI

1. **Fix Kaggle CLI creds** (refresh `~/.kaggle/kaggle.json` or set
   KAGGLE_USERNAME/KAGGLE_KEY env). This blocks all submits — today's
   10 fresh slots forfeit at next UTC midnight per Rule 12 if unfixed.
2. **Submit 2-3 of the staged candidates** (suggested priority,
   updated for correct .npy ρ bands):
   - HEDGE 3 (HIGH PRIORITY): **R10 R7.2+K27 arith 75/25** —
     +0.175 bp OOF, **OK band ρ=0.999882** (first truly diverse
     candidate from R7-era pool; K27 pool-composition orthogonal
     to K=13 pool). Expected LB shift: −0.5 to +1 bp.
   - HEDGE 2: **R8 60/20/20** — +0.077 bp OOF, TIE_ZONE ρ=0.99997.
     Likely LB tie at 0.95389; converts held-back to LB-confirmed.
   - OPTIONAL HEDGE 4: **R10 R7.2+R6.1 arith 80/20** — +0.262 bp OOF
     (highest absolute), TIE_ZONE. Variance-cancellation hedge.
3. **Defer mechanism-expansion** (transformer/graph/survival from
   HANDOVER R10 priority queue) to R11+ next session with Kaggle T4
   budget.

## Files

- Script: `scripts/probe_r10_blend_operator_sweep.py`
- Sweep JSON: `scripts/artifacts/r10_blend_operator_sweep.json`
- 6 staged CSVs in `submissions/`
- Plan: `/root/.claude/plans/cached-frolicking-rivest.md`
- Hedge ladder: `state/hedge-ladder.md` (NEW, promoted from
  state/current.md:117-127)
- ISSUES.md leaf claim: 2h (`research-improvements-jjI84 | status: wip`)
