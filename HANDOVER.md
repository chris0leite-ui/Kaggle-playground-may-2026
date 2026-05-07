# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are (Day-16 PM, 2026-05-06 evening)

- **PRIMARY** = `d16_path_b_K22_continuous_only_tau20000` LB **0.95089**
  (Day-17 advance via `claude/autoencoder-synthetic-data-pEMB6`, scored 2026-05-07).
  K=22 = K=21 + `d16_orig_continuous_only` (orig-LGBM on 7 features the
  synthesizer left marginal-aligned per Phase-1 KS-divergence diagnostic).
  Mechanism: selective-feature-restriction transfer; not target-derived.
- _Previous PRIMARY:_ `d15b_path_b_K22_dae_only_tau20000` LB **0.95059**
  (Day-15 PM via DAE swap-noise → LGBM-on-latent). Both DAE-class and
  selective-feature-restriction-transfer signals are legitimate (no
  target-label leakage).
- **Gap to top-5%** (0.95345): −25.6 bp.
- **Top of LB ~0.955** (PI observation, end of session): leaders likely
  use FEW or a SINGLE model with a structural mechanism we haven't found
  yet. Stacking-with-target-derived-bases was chasing inflated OOF
  (see leakage section below).
- **Submissions used total:** 28/270.
- **Branches active recently:**
  - `claude/read-handover-lA8Nr` — Day-16 virgin-axes, 11 probes,
    4 NULL / 1 falsified / 3 KILLED / 2 parked / 1 marginal (no advance).
  - `claude/ml-handover-alignment-xvUN0` — harness + target-reformulation
    thesis **falsified via strict-OOF audit**.
  - `claude/autoencoder-synthetic-data-pEMB6` — d16 cont_only PRIMARY
    advance (LB 0.95089, +3.0 bp) + d17 Phase 0/A in flight.

## 🔴 CRITICAL — held candidates INVALIDATED

End-of-day strict-OOF audit on this branch **collapsed all
target-reformulation single-add results 88-100%**:

| candidate | original Δ at K=21+1 | strict-OOF Δ | collapse |
|---|---:|---:|---:|
| reverse_cum | +4.867 bp | −0.005 bp | 100% |
| pit_horizon | +3.191 bp | +0.302 bp | 90% |
| inv_laps_until_pit | +1.899 bp | +0.234 bp | 88% |
| Joint K=21+3 | +7.667 bp | +0.275 bp | 96% |

**Bug:** `compute_targets()` in `scripts/probe_target_reform.py` and
`_v2.py` aggregates per (Driver, Race, Year) group using ALL train
labels — leaking val-row labels into tr-row regression targets via
`total_pits` + `cumsum`. New friction tag
`target-construction-layer-leakage`. Same family as
`d12_lr_meta` 2-level stacking (LB regress on +1.348 bp inflated OOF).

**Held candidates DO NOT submit:**
- `path_b_K22_invlaps_tau{5k,20k,100k}.csv` — 88% leaky
- `path_b_K23_dae_invlaps_tau{5k,20k,100k}.csv` — partially leaky (inv_laps component)
- `path_b_K25_megapool_tau{5k,20k,100k}.csv` — 96% leakage mirage
- `path_b_multilevel_τ_*.csv` — 5 configs NULL anyway

**Held candidates safe (no target-leakage):**
- `d15b_path_b_K22_dae_only_tau{20k,100k}.csv` (PRIMARY + close-second)
- `path_b_K22_d12meta_tau100000.csv` (LB 0.95045, R7-eligible HEDGE)
- `d15c` (ExtraTrees), `d15d` (LGBM-on-KNN) — R5 HEDGE only

Audit: `audit/2026-05-06-target-reform-leakage-audit.md`.

## Read order on session start

1. `CLAUDE.md` — state block + Rules 1-19 (Rule 19 = experimentation harness)
2. `audit/2026-05-06-target-reform-leakage-audit.md` — **load-bearing** —
   strict-OOF audit collapse table
3. `audit/2026-05-16-d16-virgin-axes-results.md` — Day-16 virgin-axes
   (11 probes, all NULL / falsified / parked)
4. `audit/friction.md` — top tags `target-construction-layer-leakage`,
   `path-b-amp-only-fires-on-meta-arch-not-base-add`,
   `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`,
   `lr-meta-rank-lock-strong-anchor`
5. `scripts/probe.py` — `bote()` + `gate()` harness (Rule 19)
6. `scripts/probe_min_meta.py` — K=21+N stack-add gate
7. `scripts/probe_target_reform_strict_oof.py` — strict-OOF audit pattern
8. `scripts/pre_submit_diff.py` — MANDATORY before submit


## Falsified or dead — do NOT retry

See `ISSUES.md ## Falsified or dead` (full list). Highlights:
- **target_reformulation_invlaps / pit_horizon / reverse_cum / stintprog**
  — all leaky; strict-OOF audit 88-100% collapse
- **path_b_K22_invlaps_*, path_b_K23_dae_invlaps_*, path_b_K25_megapool_***
  — all built on leaky targets
- **multi_level_path_b_4tier** — 5 configs NULL
- **Day-16 virgin-axes** — 11 of 11 NULL/falsified/killed
- TabPFN v2.5/v2.6, FM-aug16+, drop-GBDT pool refactor, simple K=21
  blends, α-calibrated τ-resweep, multi-target NN, masked-column
  self-prediction (DGP-residual)


## Operating rules (load-bearing)

1. **Pre-submit-diff before EVERY submit**; ρ < 0.999 mandatory.
2. **Strict-OOF audit any per-group y-derived target before submission**
   (`tag: target-construction-layer-leakage`).
3. **Per-row feature engineering is dead**
   (`tag: synthetic-dgp-conditionally-near-independent`).
4. **ρ alone NOT sufficient for meta-utility** (5 cross-confirmations).
5. **Path B amp does NOT fire on base-adds** (1.4× realised, not 6-11.6×;
   `tag: path-b-amp-only-fires-on-meta-arch-not-base-add`).
6. **Path B amp REQUIRES orthogonal signal** (meta-derivatives FAIL;
   `tag: path-b-amp-needs-orthogonal-signal-not-meta-derivatives`).
7. Strat-only Day-3+ (R1) for primary OOF; public LB row-iid per U3.
8. Cap ≤3 concurrent CPU-heavy probes; schedule cheap probes first.



---

## Archived per-branch PM detail (Day-15/16/17)

Per Rule 5 file-size guard (HANDOVER cap ≤ 150 lines), the detailed
per-branch PM sections from Day-15/16/17 have been moved to:
`audit/archive-2026-05-07-handover-day15-17-pm-sections.md`

Load-bearing summaries remain in `## Where we are` and `## 🔴 CRITICAL`
above. Read the archive only for archaeology / detailed reproductions.

## Day-18 PM ensemble-logistic-regression-research-MbLKu (LR-diagnostic + meta-arch test)

**0 LB submits this session.** Research-loop session per PI L2
("learn, don't chase bp"). 12 experiments across 3 arcs + 2 Tier-1
meta-arch tests. PRIMARY unchanged: `d17_K24_d18pool_h1d` LB 0.95345.

### Three load-bearing findings

1. **Pool eff_rank ≈ 3 of 24** (E1 entropy + E9 forward-select cross-
   confirmed). K=10 = K=24 in OOF AUC. **14 of 24 bases dead weight.**
   `cb_slow-wide-bag` first negative-marginal pick; predicted by E1
   redundancy and E2 miscalibration independently.
2. **Stint is the dominant interaction hub** (E6). 9 of 10 top
   cell-residual pairs include Stint. PRIMARY's per-cell residuals
   on those pairs <1% — GBDT pool fully saturates the (Stint × *)
   DGP information. Adding the 9 Stint-cross interactions to LR
   lifts +123 bp standalone (A2 vanilla → rich).
3. **Representation-only diversity is meta-null on a saturated info
   space** (A2 + A4 cross-confirmed). A2_rich ρ=0.71 (lowest base ever),
   A4 ρ=0.75 — both Δ ≤ +0.04 bp on K=10. **6 cross-confirmations of
   `rho-alone-insufficient-for-meta-utility`.**

### Tier-1 tests executed

- **T2 K=10 PRIMARY artifact built**. OOF 0.95381, ρ_test 0.99913 vs
  d18 K=24. Δ = −0.4 bp (within sub-bp noise). No-cost simplification
  candidate; not LB-submitted (PI hold).
- **T1#3 Path-B 3 alt segmentations × 3 τ on K=10**. ALL 9 within
  sub-bp of K=10 baseline; best S3 Driver_freq_q4 × Stint τ=20k +0.27
  bp. **Path-B amp does NOT fire on K=10.** Refines friction:
  `path-b-amp-requires-large-redundant-pool-not-saturated-pool` —
  Path-B was a redundancy-re-allocation mechanism, not a
  new-information mechanism.

### Durable deliverables (carry across comps)

- **`.claude/skills/kaggle-comp/lr-diagnostics.md`** — skill entry
- **`.claude/skills/kaggle-comp/templates/scripts/lr_diag/`** — 10
  Python scripts + README, drop-in for any tabular comp's Day-1
- 3 audits `audit/2026-05-07-lr-diagnostics-arc{A,B,C}.md`
- 5 friction tags + 4 mechanism-family entries codified

### NEXT-SESSION PRIORITY (post-execution view)

In-pool research empirically exhausted. Three honest options ranked:

1. **T3 Pirelli external data scrape** (only path with new-information
   mechanism). PI scope sign-off: NOT a public CSV; a structured
   external scrape of historical pit-window aggregates per
   (Compound, Race, Year). Aggregate-prior join, not row-join.
   ISSUES leaf 4a; untouched. Tier-2 EV +0.5 to +3 bp per Day-8
   research.
2. **Wrap s6e5** — top-5% achieved (LB 0.95345 = 0.0 bp gap). Durable
   wins shipped (skill suite + 9 codified findings). Reserve compute
   for next Featured comp Day-1.
3. **T1#1 non-Gaussian shrinkage** (4-8 h CPU). Predicted null at high
   confidence per saturated-info argument; same Path-B mechanism that
   already failed in T1#3. Run only if PI wants completeness on
   meta-arch family.

**Recommendation: option 1 (T3) or 2 (wrap).** Option 3 is low-EV
discipline-only.

### Files added this session

- `audit/2026-05-07-chris-deotte-lr-stacker-research.md` (research note)
- `audit/2026-05-07-lr-diagnostics-arc{A,B,C}.md` (3 arc audits)
- `audit/2026-05-07-postmortem-lr-diagnostic-expedition.md`
- `audit/archive-2026-05-07-handover-day15-17-pm-sections.md` (file-cap)
- `scripts/lr_diag_e{1,2,4,5,6,8,9}_*.py` + `lr_diag_a{2,4}_*.py`
- `scripts/t2_k10_primary.py` + `t1_3_segmentation_crosses.py`
- `.claude/skills/kaggle-comp/{lr-diagnostics.md,templates/scripts/lr_diag/}`
- 12 JSON results + 6 OOF/test artifact pairs in `scripts/artifacts/`

### Submissions used

0/10 today. Total: 35/270.
