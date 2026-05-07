# Probe spec: Path-B hier-meta with (Compound × Year) segmentation

> Branch `claude/logistic-regression-ensemble-0PNkA`. Planning doc for
> next-session probe. PI committed to (b): skip submission, write up the
> per-(Compound × Year) Path-B as the next move. **Not yet executed.**

## Hypothesis

The s6e5 DGP has **year-conditional structure** (especially 2023) that
the current Compound × Stint segmentation does not capture.
Re-segmenting the Path-B hier-meta cross from `(Compound, Stint)` to
`(Compound, Year)` should fire the Path-B amplification mechanism.

## Motivation (load-bearing evidence)

From `audit/2026-05-07-lr-leverage-six-probes.md` Probe 5 — per-segment
mega LR with **(Compound × Year)** as the partition (not Stint):

| Cell | n | global mega AUC | per-segment AUC | Δ bp |
|---|---:|---:|---:|---:|
| MEDIUM_2023 | 58264 | 0.8176 | **0.9257** | **+1081** |
| SOFT_2023 | 15457 | 0.8177 | 0.9042 | +865 |
| HARD_2023 | 60996 | 0.8573 | 0.9298 | +725 |
| SOFT_2024 | 5652 | 0.8051 | 0.8402 | +351 |
| INTERMEDIATE_2024 | 8440 | 0.8480 | 0.8753 | +272 |
| MEDIUM_2024 | 59548 | 0.8726 | 0.8991 | +265 |
| (other 14 cells, mostly +20 to +180; small cells regress) | | | | |
| **Global** | 439140 | 0.92776 | **0.93385** | **+60.8** |

**Key reading:** the 2023 cohort cells alone account for ~+650 bp of the
60.8 bp aggregate lift. Pooled-coefficient global LR can't represent
2023's segment-specific structure. **This is exactly the kind of
DGP-conditional signal that Path-B amp captures**, when the segmentation
matches the underlying conditioning variable.

## Why Path-B Compound × Year specifically (not Stint)

Existing Path-B PRIMARY uses `(Compound, Stint)` τ=100k:
- d13e_compound_stint_tau20000 LB **0.95049** (8× LB amp on +0.30bp OOF)
- d13_path_b_stint_tau100000 LB **0.95041** (11.6× amp)
- d17_path_b_K23_v4_h1d_tau100000 LB **0.95354** (current PRIMARY)

Friction `path-b-amp-only-fires-on-meta-arch-not-base-add`: amp fires
on **meta-arch redesigns** (segmentation cross changes). Path-B Year
on K=24 is exactly that — same K=24 pool, different segmentation cross.

CLAUDE.md "Hypothesis board" line:
> Alternative seg crosses (Year×Compound, Compound×TyreLife_q5,
> Driver_clustered×Stint). Highest tail EV; only Path-B-amp-eligible
> axis per friction tag.

Probe 5 just gave us the **strongest empirical motivation yet** for
Year as a segmentation axis: per-LR cohort lift dominates 2023.

## Probe specification

### Mechanism

For each `(Compound, Year)` segment, fit a per-segment LR-meta on the
K=24 GBDT OOF columns (expanded to [P, rank, logit]). Shrink each
segment's coefficients toward the global LR-meta coefficients with
empirical-Bayes Gaussian weight α = n_seg / (n_seg + τ).

Per d13e template: `scripts/d13e_path_b_compound_stint.py` is the
reference implementation. Adapting it:

```python
# In d13e: segment by (Compound, Stint)
segments_train = list(zip(train["Compound"].values, train["Stint"].values))

# Probe: segment by (Compound, Year)
segments_train = list(zip(train["Compound"].values, train["Year"].values))
```

That's the only mechanism change.

### Pool

K=24 = K=21 + d16_orig_continuous_only + p1_single_cb_v3_gpu + d17_h1d_yekenot_full
(matches current PRIMARY's pool composition).

### τ sweep

- `tau ∈ {5000, 20000, 100000, 500000}` (matches d13e default sweep)
- 5k = local-leaning, 500k = global-leaning, 20k/100k = canonical

### Segment count

- 4 Compounds × 5 Years (2022, 2023, 2024, 2025, +WET as a Compound but
  WET sometimes absent in some years) = up to 20 cells
- Per probe-5 evidence: ~17 cells with n ≥ 1k (skip <1k like d13e
  min_rows constraint)

### Cost

- ~15 min CPU (matches d13e wall, same K=24 pool, same τ sweep, similar
  segment count)
- 5-fold StratKF (SEED=42, matching all existing artifacts)
- 4 OOF + test pairs saved (one per τ)

## Pre-run checklist (Rule 19 BOTE + Rule 6Q + Rule 26)

### 6-Q check (CLAUDE.md Rule 16)

| Q | Answer |
|---|---|
| 1. Mechanism family in `mechanism_families_explored`? | YES — `path_b_cohort_sweep_d14` family explored. d14 sweep ran Year/YxStint/Race × τ; ALL 9 variants <PRIMARY OOF. **BUT** d14 was on K=21 pre-CB pool; this probe is on K=24 with d16/v3/v4/h1d which has different OOF structure. Re-test justified. |
| 2. Class? | meta_arch_redesign (segmentation cross). NOT base-add. NOT meta-derivative-of-meta. Path-B-amp-eligible per friction. |
| 3. Predict standalone OOF | K=24 baseline meta 0.95385 → Path-B Compound×Year 0.95390-0.95410 (predicted +0.5-2.5 bp OOF) |
| 4. Predict ρ vs PRIMARY | 0.985-0.995 (close to PRIMARY since same pool, just re-segmented) |
| 5. Cite closest precedent | d13e Compound × Stint τ=20k: K=22 OOF 0.95083 (+0.30 bp over canonical) → LB 0.95049 (+8 bp). Probe-5 LR analog gives much larger standalone lift (+60 bp) but Path-B shrinks toward global so realised will be much smaller. |
| **6. Metric-aligned?** | **YES** — LR-meta on logits is BCE, row-AUC matches BCE ranking. No Q6 downgrade. |

All 6 answer coherently → no EV downgrade.

### BOTE entry (Rule 19a)

```
python scripts/probe.py bote pathb_compound_year_K24 \
    --family meta_arch_redesign \
    --cost_min 15 \
    --metric-aligned true \
    --pi-predicted-lb-bp <PI_TO_FILL>
```

Per family prior + this evidence, agent EV preview:
- LB Δ midpoint **+3 bp** (range −1, +3, +10)
- Confidence: medium-high (probe 5 is strong evidence; d14's null
  evidence weakens it but on a different pool)
- Family prior: meta_arch_redesign p=0.30, base midpoints (1, 4, 8) bp
  per Rule 19f calibration.
- This evidence raises p from 0.30 to ~0.45-0.55 IMO.

### Sealed-prediction protocol (Rule 26a)

**PI to commit FIRST** (before agent reveals BOTE):
- Sealed PI prediction: LB Δ in bp = ?
- One-line rationale: ?

Agent reveals BOTE only after PI commits. Both go to
`audit/decisions.jsonl` via `--pi-predicted-lb-bp`.

### Devil's-advocate (Rule 26c)

Counter-arguments against this probe firing:
1. **d14 sweep already ran Year**: 9 variants of cohort axes ALL <PRIMARY.
   Why would Year fire here? Counter-counter: d14 was on K=21 (no CB v4,
   no h1d). The K=24 pool's OOF correlation structure is different;
   d16_cont_only and h1d both have Year-conditional patterns absent in
   K=21.
2. **Probe 5 lift is at LR-class, not GBDT-meta-class.** GBDT pool
   already sees Year as a feature; per-Year specialists may not lift the
   GBDT-pool meta the way they lift LR. Counter-counter: even GBDTs
   trained on raw "Year" don't learn cohort-conditional META coefficients;
   the lift is on routing, not on prediction. Path-B meta should
   capture the routing.
3. **Realized Path-B amp has been small recently.** d15b DAE Path-B
   was 1.4× amp (`path-b-amp-only-fires-on-meta-arch-not-base-add`). d17
   v4+h1d Path-B was friction re-confirmed at +0.12 bp. The amp axis
   may be saturated. Counter-counter: those were Compound × Stint
   re-runs. Year is genuinely new segmentation cross on this pool.

If 2-of-3 devil's-advocate points are right, expected LB Δ ≤ 0 bp.

## Stop-conditions (Rule 19b)

Run gate after artifacts via:
```
python scripts/probe.py gate pathb_compound_year_tau20000 \
    --oof scripts/artifacts/oof_pathb_compound_year_tau20000_strat.npy \
    --test scripts/artifacts/test_pathb_compound_year_tau20000_strat.npy
```

PASS criteria for SUBMIT recommendation:
- OOF Δ ≥ +0.5 bp vs current PRIMARY OOF (0.95385)
- ρ vs PRIMARY 0.99 ≤ ρ ≤ 0.998 (orthogonality without total mismatch)
- G3 flip ratio ≥ 0.5
- Predicted LB Δ ≥ +1 bp at the gate's calibration band

If any τ ∈ {5k, 20k, 100k, 500k} passes all gates, that τ becomes the
submission candidate. PI sign-off then submit.

## Implementation outline (next-session work)

1. **Copy** `scripts/d13e_path_b_compound_stint.py` →
   `scripts/d18_path_b_compound_year.py`
2. **Modify** the `segments_train` line:
   ```python
   # OLD: zip(train["Compound"], train["Stint"])
   segments_train = list(zip(
       train["Compound"].astype(str).values,
       train["Year"].astype(str).values,
   ))
   ```
3. **Update** the K=21 pool list to K=24 (add `d16_orig_continuous_only`,
   `p1_single_cb_v3_gpu`, `d17_h1d_yekenot_full`)
4. **Update** `min_rows` to 1000 (matches d13e; smaller cells like
   WET_2022 n=1299 will skip)
5. **Set** save names to `pathb_compound_year_tau{TAU}` for the 4 τ
6. **Run** end-to-end (~15 min)
7. **Gate** each τ with `probe.py gate`
8. **PI sealed-prediction → submit if PASS**

## Files this depends on

- `scripts/d13e_path_b_compound_stint.py` — copy template
- `scripts/probe.py` — bote + gate harness
- `scripts/artifacts/oof_d17_K24_d18pool_h1d_strat.npy` — current PRIMARY ref for ρ
- 24 K=24-pool OOF artifacts in `scripts/artifacts/oof_*_strat.npy`

## Predicted outcomes (sealed)

Three scenarios:

**Best case (~15% probability):** Path-B amp fires on Compound × Year.
Best τ delivers OOF +5 bp / LB **+10 to +20 bp**. Top-5% breached
(0.95354 + 15 = 0.95504, top ~30).

**Mid case (~50% probability):** Modest amp. Best τ delivers OOF
+1 bp / LB +2 to +5 bp. PRIMARY-replace at LB ~0.95360 (move to
top ~80 from #98).

**Bad case (~35% probability):** d14 history wins. All τ NULL or
−2 bp regression. Path-B amp confirmed dead on this pool. Adds another
cross-confirmation of `path-b-amp-only-fires-on-meta-arch-not-base-add`
"unless cross-axis is novel and DGP-conditional".

## What we'd learn either way

- **Positive**: Year-cohort structure is a Path-B-amp-eligible axis that
  Stint-segmentation missed for 17 days. New friction tag origin.
- **Null**: Refines the Path-B amp friction with one more axis tested.
  d14 sweep already had Year-on-K=21 NULL; this would be Year-on-K=24
  NULL — completes the matrix.
- **Either way**: `audit/decisions.jsonl` calibration entry with PI
  sealed prediction, agent BOTE, actual LB.

## Friction-tag updates (anticipated)

If positive: new tag `pathb-amp-fires-on-cohort-conditional-DGP-axes`.
If null: extends `path-b-amp-only-fires-on-meta-arch-not-base-add` with
the qualifier "and only on Stint axis on s6e5; Year NULL on K=24".
