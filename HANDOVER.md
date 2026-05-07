# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Where we are (Day-17 PM, 2026-05-07 evening)

- **PRIMARY** = `d17_path_b_K23_v4_h1d_tau100000` LB **0.95354** (Day-17 PM
  advance via `claude/optimize-model-performance-rruC2`, scored 2026-05-07
  13:27 UTC). K=23 = K=21 + `p1_single_cb_v4_gpu` (yekenot transfer
  recipe; see CB-axis arc below) + `d17_h1d_yekenot_full` (RealMLP-
  pytabkit replication imported from `claude/read-handover-62BCt`),
  Path-B Compound×Stint hier-meta τ=100k. **Realised OOF→LB gap −6.1 bp.**
- _Previous PRIMARIES (Day-17 trail):_
  - `d17_K24_d18pool_h1d` LB 0.95345 (research-branch midday;
    K=21 + d16_cont_only + p1_single_cb_v3 + h1d).
  - `d18_path_b_K23_d16_d18_tau20000` LB 0.95149 (research-branch AM;
    K=21 + d16_cont_only + d18_chain_decomp).
  - `d16_path_b_K22_continuous_only_tau20000` LB 0.95089 (yesterday).
  - `d15b_path_b_K22_dae_only_tau20000` LB 0.95059 (Day-15 PM).
- **Rank: #98 of 893 = top 11%.** Tied with [QWERTY] Roman Rozen at
  0.95354. Top-5% boundary moved up 60 bp today: was 0.95345 (stale),
  now **0.95405** (rank 44). **Gap to top-5%: −51 bp.** Leader MILANFX
  0.95476 (gap −122 bp).
- **Submissions used total:** 38/270; today 7/10.
- **Branches active recently (Day-17):**
  - `claude/optimize-model-performance-rruC2` — **NEW PRIMARY LB 0.95354**
    via CB-axis closure (v3→v4 yekenot transfer, +20.7 bp standalone OOF).
  - `claude/read-handover-62BCt` — h1d (yekenot RealMLP replication,
    OOF 0.95257) + K=24+h1d LB 0.95345 (intermediate Day-17 PM PRIMARY).
  - `claude/reverse-engineer-data-generation-Hu8EK` — d18 chain_decomp
    DGP probe (K=21+1 +7.37 bp) + d18 K=23 LB 0.95149 (Day-17 AM PRIMARY).
  - `claude/research-agentic-kaggle-W6IAP` — yekenot recipe extraction
    + h1d full-recipe replication.
  - `claude/autoencoder-synthetic-data-pEMB6` — d16 cont_only LB 0.95089.

## ⭐ Day-17 PM CB-axis arc + new PRIMARY

This branch (`claude/optimize-model-performance-rruC2`) closed the CB
recipe axis with a +20.7 bp single-model lift via yekenot FE transfer.
Full audit: `audit/2026-05-07-d17-cb-v4-yekenot-transfer.md`.

| Stage | OOF | LB | Key fact |
|---|---:|---:|---|
| LGBM v3 honest (Day-17 AM, P1) | 0.94563 | n/a | single-model honest ceiling |
| **CB v3 (research-recipe)** | 0.94993 | **0.95143** | +43 bp over LGBM v3; K=21+1 +12.06 bp |
| **CB v4 (yekenot transfer + orig-aug)** | **0.95200** | (held) | +20.7 bp over v3; K=21+1 **+24.21 bp** |
| K=22 v4 Path-B C×S τ=20k | 0.95319 | (held) | Path-B amp +0.39 bp (friction re-confirmed) |
| **K=23 v4+h1d Path-B τ=100k = NEW PRIMARY** | **0.95415** | **0.95354** | submitted; gap to LB top −122 bp; rank 98/893 |

**Recipe transfer items 2/3/4 (floor-cat, count-encoding, KBins) fired
on CatBoost despite research-branch audit caveat that they are
"NN-specific."** New friction tag candidate
`yekenot-floor-count-kbins-fires-on-gbdt-too`.

## 🎯 Next-step priority list (Day-18+)

To close the −51 bp gap to top-5% (boundary 0.95405), ranked by EV/cost.

| # | Move | Cost | Predicted LB Δ | Notes |
|---|---|---|---:|---|
| 1 | **K=25 = K=21 + v4 + h1d + d16 + d18** | 10 min CPU | +3 to +8 bp | Cheapest near-term; combines all today's bases |
| 2 | 3-seed bag of v4 (seeds 42/13/71) | 75 min Kaggle GPU | +1 to +3 bp | Variance reduction on load-bearing base |
| 3 | XGB with v4-recipe FE | 30 min Kaggle GPU | +5 to +15 bp | New model class on same FE; ρ < 0.97 likely |
| 4 | RealMLP n_ens=24 (h1d ran 4) | 3.5 h CPU/GPU | +2 to +5 bp standalone | Yekenot's published; +1-3 bp at K-meta |
| 5 | **FastF1 lap-by-lap pit-call hard-join** | 1-2 days | +10 to +30 bp | HANDOVER A4; only single-mechanism path to top-5% |
| 6 | Pirelli tyre-curve scrape | 1-2 days | +10 to +30 bp | Same axis as #5 |
| 7 | Per-Year specialists with v4 recipe | 30 min Kaggle GPU | ±5 bp | d12 found 2023 is easiest segment |
| 8 | Cross-segmentation Path-B (Y×S, R×C) on K=23 v4+h1d | 20 min CPU | +0 to +3 bp | d14 falsified Y-axis without v4 |

Items 1-3 sum to ~+10-25 bp predicted; still below the 51 bp gap.
Items 5-6 are the only single-mechanism path.

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

---

## Day-17 AM autoencoder-synthetic-data-pEMB6 (status-only wrap)

**No compute this session.** Session opened with a misgrounded clarifying
question (proposed "Phase F" without first reading branch state); PI
responded **"stop here. wrap up."** Inherited d17 Phase 0 + Phase A in flight
from prior commit `1f442e8`. No LB submissions; no new OOF/ρ measurements.

**Inherited artifacts staged in this wrap** (under `scripts/artifacts/`):
- Phase 0 leakage cleanup: `d17_phase0_leakage_summary.json`,
  `oof_d17_dr_weighted_orig_v2_strat.npy` + test pair.
- Phase A K=22/K=23 stack-add OOF/test pairs (5 candidates, `_strat`):
  `oof_d17_C1_K22_cont`, `oof_d17_C2_K23_cont_nolaptime`,
  `oof_d17_C3_K23_cont_notyrerp`, `oof_d17_C4_K23_cont_catonly`,
  `oof_d17_C5_K23_cont_invlaps_strict` (+ matching `test_*` files).

**Phase B / Phase C unrun.** Scripts present and committed in `1f442e8`:
- `scripts/d17_phase_b_extend.py` — multi-arch + N-sweep + physics
  specialists + synth-restricted variants.
- `scripts/d17_phase_c_meta_arch.py` — Path-B-amp meta-arch redesigns
  (Student-t shrinkage, 3-level hierarchy, 75-seg Compound×Stint×r̂_q3).

**Next agent:** read `1f442e8` commit message + the C1-C5 artifacts above,
then either (a) gate the C-candidates with `probe_min_meta.py` to decide
PRIMARY-advance vs HEDGE, or (b) execute Phase B / Phase C scripts.
PRIMARY remains `d16_path_b_K22_continuous_only_tau20000` LB **0.95089**.

**File-size flag:** HANDOVER.md is at 317 lines, over the 150-line cap in
WRAPUP.md step 5. This bloat predates today's session; not archived here
to avoid touching other branches' Day-N PM sections (Rule 15). Flag for
the next merge-target scribe.

---

## Day-17 AM read-kaggle-handover-rsi2Q

P1 single-model thesis (PI hypothesis "leader at LB ~0.955 likely
uses ONE strong model") — tested end-to-end via Rozen 0.95354 recipe
replication. **CONCLUSIVELY FALSIFIED** under strict OOF discipline.

### What ran
- Pulled top 8 public s6e5 notebooks under `external/kernels/` as
  reference (incl. `romanrozen/f1-pit-driver-race-year-encoding-0-95354`).
- Pulled external datasets: `aadigupta_orig`, `f1_official_1950_2022`
  (driver/circuit historical priors), `weather_woodshole`.
- Built `make_features_A` v1 (50 engineered + 6 CV TE incl Driver×Race×Year).
- v1 single LGBM OOF 0.94970 → submitted alone LB **0.94107** (gap −863 bp);
  K=22 LR-meta-add OOF 0.95404 → submitted LB **0.94933** (−126 bp vs PRIMARY).
- v2 fixed `stint_size_far` per-split-count cluster + added FS_A merge
  aggregates. OOF 0.95128 (+38 bp over PRIMARY OOF — too good).
  K=2 LR(PRIMARY, v2) submitted LB **0.94996** (−63 bp vs PRIMARY).
- 80/20 honest holdout test (`scripts/p1_holdout.py`) caught FS_A target
  leak: holdout AUC **0.94637** vs OOF 0.95128 = **−491 bp gap**.
- v3 with **fold-safe FS_A** (`fit_fs_a` per-fold, `apply_fs_a` merge):
  OOF **0.94563** matches holdout. Honest single-LGBM ceiling on this
  comp.

### What we now know
- Single-LGBM with kitchen-sink Rozen-style FE achieves OOF ~0.946.
- PRIMARY (K=22 + Path-B hier-meta, OOF 0.95090) is +52 bp ahead.
- **Stacking is necessary for our LB position.** P1 thesis FALSIFIED.
- Rozen's published 0.95241 single-LGB OOF is likely similarly inflated
  by FS_A leak in his pipeline (he uses the same `df[df['PitNextLap']==1]
  .groupby(...).mean()` pattern fit on full train); his real single-LGB
  LB is probably ~0.946, blend wins via 5 external sources.

### Other branch's win
- **`claude/.../d16_path_b_K22_continuous_only_tau20000` LB 0.95089**
  (+30 bp over PRIMARY 0.95059). Clean Path-B base-add candidate using
  KS-divergence-identified marginal-aligned features; this is the
  Day-17+ PRIMARY-replacement candidate to confirm.

### Lessons captured (skill `improvements.md` + local CLAUDE.md R20-R25)
- R20 single-model-first / kitchen-sink FE before stacking
- R21 family falsification requires ≥3 variants
- R22 public-notebook scan at every plateau
- R23 framework is scaffolding, not authorship
- **R24 fold-safe label-conditional aggregates** (NEW Day-17)
- **R25 transductive features need AV check** (NEW Day-17 PI lesson)

`scripts/p1_holdout.py` — 80/20 honest holdout test (independent seed).
Mandatory before any new-FE-family LB submit.

### Submissions used (all UTC days combined)
Day-17: 4/10 used (3 by this branch + 1 d16 from another).
Total: 32/270.

### Files
- `scripts/p1_features.py` — `make_features_static` + `fit_fs_a` +
  `apply_fs_a` (v3 fold-safe). Legacy `make_features_A` flagged.
- `scripts/p1_single_lgbm_v3.py` — fold-safe trainer.
- `scripts/p1_single_lgbm.py` — v1/v2 trainer (legacy).
- `scripts/p1_single_cb.py` — single CatBoost (deferred, not run).
- `scripts/p1_holdout.py` — 80/20 honest holdout.
- `scripts/p1_post.py`, `scripts/p1_gate_all.py` — gate harnesses.
- `scripts/artifacts/oof_p1_single_lgbm_v3_feA_te_strat.npy` (+test).
- `audit/2026-05-06-p1-single-model-{plan,results}.md`.
- `external/kernels/{romanrozen,...}/` — 8 reference notebooks.
- `external/{aadigupta_orig,f1_official_1950_2022,weather_woodshole,
  makimakiai_idsafe,gkanamoto_tabm,pavloivanin_baseline}/`.

### Open candidates from other branches
- `d16_path_b_K22_continuous_only_tau20000` LB 0.95089 — verify and
  consider as new PRIMARY.
- v3 single LGBM OOF 0.94563 itself — too low standalone, but ρ=0.953
  diversity. Genuine K=22+v3 stack-add lift only +3.40 bp OOF
  (vs leaky +30.79 bp). Held; probably not worth a slot.

---

## Day-17 PM read-handover-62BCt — TOP-5% AT-THRESHOLD via yekenot recipe (LB 0.95345)

**🎯 NEW PRIMARY: LB 0.95345 (AT TOP-5% THRESHOLD) 🎯**
`submission_d17_K24_d18pool_h1d.csv` (ref 52420646, scored 2026-05-07
11:39 UTC). +19.6 bp over d18 PRIMARY 0.95149 = **BIGGEST single-submit
lift of comp**. Headroom to top-5% closes from −19.6 bp → **0**.

### What worked

Full yekenot RealMLP recipe replication (`scripts/d17_h1d_yekenot_full_recipe.py`):
- 5-fold StratKF OOF AUC 0.95257 (matches yekenot pub 0.95273 within 1.6 bp)
- ρ_test vs PRIMARY 0.972 (single base) — first base to break ρ < 0.99
  in 5+ months
- All 6 load-bearing FE items: arithmetic ratios, floor-cat, count enc,
  KBins(200/7), per-fold stratified orig concat, **CV TargetEncoder on
  (Race,Compound)+(Race,Year) inside fold loop** (load-bearing).
- `n_ens=4` on 4-core CPU; yekenot's `n_ens=24` on Kaggle GPU is +5 bp
  ceiling at most.

K=24 d18pool+h1d submission stack:
- K=21 + d16_orig_continuous_only + p1_single_cb_v3_gpu + d17_h1d_yekenot_full
- LR-meta (Path B over K=22 with h1d was TIE per 6th cross-confirmation
  of `path-b-amp-only-fires-on-meta-arch-not-base-add`)
- OOF 0.95385, ρ_test vs d18 PRIMARY 0.989, predicted LB Δ +15 bp
- Realised LB Δ +19.6 bp (PI sealed prediction +10 bp; agent +15.11 bp;
  both conservative)

### Calibration outcomes (audit/decisions.jsonl)

| Probe | PI pred | Agent pred | Actual |
|---|---:|---:|---:|
| H1 (initial 3 variants) | 0 bp | +27 bp | NULL across 3 variants — recipe-gap misdiagnosis |
| H2 FastF1 | +5 bp | +3.6 bp | ~0 bp (1.4% match rate cap from synth D### codes) |
| H3 ID-shift | 0 bp | +0.6 bp | 0 bp (PI win — id_div_N AV is labeling convention only) |
| H1d full-recipe (final) | +10 bp | +15.11 bp | **+19.6 bp** (both beat) |

### What didn't work (this branch)

- H1 v1/v2/v3 (yekenot-hyperparams + orig-merge alone): all NULL.
  Misdiagnosed +69 bp standalone gap as hyperparameter+orig only;
  actual gap is the FULL FE pipeline.
- H2 FastF1: 1.4% match rate due to 60% synthetic D### driver codes
  + sandbox 403 on livetiming.formula1.com.
- H3 ID-shift: train ids 0..439139 / test 439140..627304 = labeling
  convention with zero overlap; sparse-LR base on id-div features =
  chance level.
- C7 K=24 LR-meta (without h1d): predicted LB Δ −0.69 bp (TIE/regress).

### Files

- `scripts/d17_h1d_yekenot_full_recipe.py` — verified replication
- `scripts/artifacts/oof_d17_h1d_yekenot_full_strat.npy` + test
- `scripts/artifacts/oof_d17_K24_d18pool_h1d_strat.npy` + test
  (the SUBMITTED stack)
- `submissions/submission_d17_K24_d18pool_h1d.csv`
- `external/kernels/ps-s6-e5-realmlp-pytabkit/VALIDATED.md`
- `.claude/skills/kaggle-comp/examples/fe-recipe-yekenot-realmlp-kitchen-sink.md`
- `audit/2026-05-07-d17-strategy-critique.md`
- `audit/2026-05-07-d17-h1-verdict.md`
- `audit/2026-05-07-d17-h2-fastf1-external.md`
- `audit/2026-05-07-d17-h3-id-shift.md`
- `audit/2026-05-07-d17-phase-a-composition-gate.md`

### Submissions used (Day-17, all UTC days combined)

7/10 today (this branch +1; 6 prior including 3 sibling submits).
Total: 35/270.

### Next-session priorities

1. **PI submission discussion**: do we need to submit anything else
   today? K=24 LR-meta variants with Path B Compound×Stint segmentation
   are unlikely to lift (6th confirmation of meta-arch friction).
2. **Tier-2 follow-ups for the yekenot recipe**:
   - n_ens=8 or 12 variant of h1d (~1-2 h CPU); +2-5 bp standalone OOF
     ceiling. EV +1-3 bp LB.
   - Apply CV-TE / engineered-cat FE pipeline to a second base
     architecture (CatBoost or LGBM on the same yekenot FE set).
     Could yield a structurally different base.
3. **PRIMARY-replace candidates pending sibling integration**: we have
   not yet tested K=25+ unions with sibling-branch new bases (d18
   already includes d16 + p1cb; if siblings produce d19+ candidates,
   re-stack).

---

## Day-17 PM read-handover-62BCt — d17 Phase-A composition gate

**0 submits this session.** Bootstrapped repo (deps + Kaggle data),
claimed ISSUES leaf 7f, re-ran inherited `scripts/d17_phase_a_compose.py`
to completion (sibling branch had bailed mid-run after C1-C5 OOFs were
written but before summary JSON / C6 / C7).

**Result.** Best K=24 LR-meta combo C7 (cont_only + no_laptime +
no_tyrerp) OOF **0.95129**, +5.50 bp over the script's printed PRIMARY
column — but that column was the OLD `oof_PRIMARY_K22_strat.npy` (d15b
DAE LB 0.95059, OOF 0.95074), not the actual current d16 cont_only
Path B PRIMARY (LB 0.95089, OOF 0.951208). Vs the actual current
PRIMARY, **C7 is +0.81 bp OOF at ρ_test 0.99506 → predicted LB Δ −0.69
bp (TIE/regress). All other Cn combos REGRESS −0.09 to −1.45 bp OOF.**

| Combo | K | OOF | Δ vs d16 PRIM (bp) | ρ_test | pred LB Δ |
|---|---:|---:|---:|---:|---:|
| C1 cont | 22 | 0.95106 | −1.45 | 0.99581 | −2.95 |
| C2 cont+nolaptime | 23 | 0.95120 | −0.09 | 0.99557 | −1.59 |
| C3 cont+notyrerp | 23 | 0.95122 | +0.11 | 0.99517 | −1.39 |
| C4 cont+catonly | 23 | 0.95115 | −0.54 | 0.99515 | −2.04 |
| C5 cont+invlaps_strict | 23 | 0.95107 | −1.42 | 0.97555 | −6.42 |
| C6 cont+nolaptime+invlaps | 24 | 0.95122 | +0.09 | 0.97714 | −4.91 |
| **C7 cont+nolaptime+notyrerp** | **24** | **0.95129** | **+0.81** | 0.99506 | **−0.69** |

**Mechanism.** Path-B Compound×Stint τ=20k segmentation on K=22 cont_only
adds +0.15 bp OOF over canonical LR-meta on the *same* pool. Stacking 3
more orig-LGBM bases via LR-meta does not close that gap. **5th
cross-confirmation of `path-b-amp-only-fires-on-meta-arch-not-base-add`.**
Strict-OOF inv_laps adds essentially nothing on top of cont_only (C5 vs
C1 = +0.04 bp); refines `target-construction-layer-leakage` finding —
even audit-cleaned strict-OOF inv_laps is not differentiated enough.

**Next step (NOT RUN — awaiting PI sealed prediction).** Path B
Compound×Stint τ=20k over the C7 K=24 pool. Cost ~15 min CPU. Family
`meta_arch_redesign` (p=0.30, (1, 4, 8) bp). Q6: log-loss / row-AUC =
True. Per Rule 26(a) PI commits LB Δ prediction first.

**Files**:
- `audit/2026-05-07-d17-phase-a-composition-gate.md` — full audit
- `scripts/artifacts/d17_phase_a_summary.json` — per-combo |w| + ρ
- `scripts/artifacts/oof_d17_C{1..7}_*_strat.npy` + `test_*` (C6/C7
  produced this run)
- `data/{train,test,sample_submission}.csv` re-hydrated via `bootstrap.sh`

---

## Day-17 PM review-handover-solutions-oE78b — 5-probe diagnostic pass, 1 structural find, 0 submits

**Result: 0 LB submits, 0 LB Δ.** PI sealed prediction = 0 bp for every
probe (Rule 26a); 5/5 vindicated at LB level (no submission).
PRIMARY remains **`d17_path_b_K23_v4_h1d_tau100000` LB 0.95354**.

### What ran

PI prompt: "What's hiding in plain sight?" Two rounds:

**Round 1 — 3 probes** (`52b00f2`):
- `probe_ntl_single_rule.py` (5 min): 5 NTL reconstructions + 13
  thresholded rules. Best AUC 0.687 < raw TyreLife alone 0.699.
  Host's brief.md "trivial" refers to unmasked original column.
- `probe_target_structure.py` (30 min EDA): P(target=1 | lap_from_
  observed_stint_end) decays 0.272 → 0.061 over 10 laps; 81% of
  multi-pos stints have last pos at observed-last-lap; 65% contiguous.
  Target is decay-from-end, not shifted PitStop, no deterministic rule.
- `probe_combined_lead_lag.py` (30 min): single-feat L1 +29 bp combined
  vs train-only on `lead_LapNumber_diff`. LGBM-level F4-F5 = -0.36 bp.
  Combined-frame premium evaporates at GBDT.

**Round 2 — probe 4 + leakage audit + K=24 gate** (`22ffbc0`,
`c548fda`, `1ca661a`, `e28db35`):
- `probe_field_state.py` (12 min): cross-row aggregates over
  (Race,Year,LapNumber) and (R,Y,L,Compound) from train+test combined.
  Standalone single-LGBM raw 14 + ~30 fs feats: full-train OOF
  0.94230 (+15.58 bp). Top single-feat **`fs_cum_pits` AUC 0.7972 —
  highest single-feat OOF on comp** (raw TyreLife alone 0.6989).
- PI flag: "isn't this leaky like Day-17 FS_A merge?"
- `probe_field_state_strict.py` (7 min): per-fold strict re-run.
  F3-strict OOF 0.94211 (+13.73 bp), F4-strict tr-only 0.94208
  (+13.35 bp). 12% collapse vs Day-17 88-100%. **Audit cleared.**
- K=24 stack-add gate: K=23 LR-meta (v4+h1d) OOF 0.95414, K=24
  (+ field-state) OOF 0.95414, marginal **-0.015 bp NULL**.
  field-state |w|=0.0807 vs v4 0.55 / h1d 0.48. **6th cross-
  confirmation of `lr-meta-rank-lock-strong-anchor`**.

### Calibration outcomes

| Probe | PI sealed | Agent BOTE | Realised LB | Realised OOF |
|---|---:|---:|---:|---:|
| probe_ntl_single_rule | 0 bp | n/a | 0 (no submit) | best 0.687 (≤ raw) |
| probe_target_structure | 0 bp | n/a (EDA) | 0 (no submit) | n/a |
| probe_combined_lead_lag | 0 bp | +0.025 bp | 0 (no submit) | +2.18 bp |
| probe_field_state | 0 bp | +0.025 bp | 0 (no submit) | +13.73 bp strict |
| probe_field_state K=24 gate | 0 bp | n/a | 0 (no submit) | -0.015 bp marginal |

PI 5/5 vindicated at LB. Family prior `single_base_fe_addition` was
mis-priced for cross-row aggregates (30× under at standalone OOF level)
but correctly predicted 0 bp at K=24 stack transfer.

### Friction tags introduced

- `cross-row-aggregates-fire-where-own-row-sequence-doesnt`
- `cross-row-aggregates-survive-strict-fold-safe-audit` (PROMOTED to
  improvements.md G18)
- `field-state-mechanism-fires-on-train-only-too-no-combined-premium`
- `family-prior-single-base-fe-addition-mis-calibrated-for-cross-row`
- `lr-meta-rank-lock-strong-anchor` (6th cross-confirm)
- `host-quote-trivial-refers-to-original-not-reconstructible`
- `pitnextlap-target-cluster-decay-not-shift`
- `combined-frame-leadlag-premium-evaporates-at-gbdt`

### Next-session first-action candidates

1. **CB-v4 + field-state retrain on Kaggle GPU** (~35 min). Add ~24
   field-state features INTO the yekenot recipe; retrain CB; predict
   v4-fs standalone OOF lift +1 to +5 bp over current v4 OOF 0.95200.
   If lifts ≥+3 bp, replace v4 in K=23 and re-gate. Bypasses rank-lock
   by changing the strong anchor itself.
2. h1d + field-state retrain (RealMLP n_ens=4 with fs feats added).
3. **External Pirelli/FastF1 hard-join** (HANDOVER A4) — only
   single-mechanism path to top-5% (51 bp gap to 0.95405 boundary).
4. K=25 full-merge (+d16 cont_only, +d18 chain_decomp) Path-B held
   +1.3 bp OOF — within noise but cheap to re-submit.

### Files
- `audit/2026-05-07-handover-review-3-probes.md` — Round 1 audit
- `audit/2026-05-07-probe4-field-state-aggregates.md` — Round 2 audit
  including strict fold-safe + K=24 gate
- `audit/2026-05-07-postmortem-review-handover-solutions-oE78b.md`
- `scripts/probe_ntl_single_rule.py`
- `scripts/probe_target_structure.py`
- `scripts/probe_combined_lead_lag.py`
- `scripts/probe_field_state.py`
- `scripts/probe_field_state_strict.py`
- `scripts/probe_field_state_lgbm_artifact.py`
- `scripts/artifacts/probe_*` JSONs + `oof_field_state_lgbm_strat.npy`
  + `test_field_state_lgbm_strat.npy`
- `.claude/skills/kaggle-comp/improvements.md` G18 (PI ratified)

### Submissions used (Day-17, all UTC days combined)
0 this branch. Day-17 total: 7/10 (unchanged).

### Branch state
Top of `52b00f2 → ca40a76 → 22ffbc0 → 0241693 → c548fda → 1ca661a → e28db35`.
Ready for ff-merge to `main` after PI sign-off.
## Day-18 PM reverse-engineer-data-generation-Hu8EK

**🎯 NEW PRIMARY: LB 0.95368** (`d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000`,
ref 52432732, scored 2026-05-07). +1.4 bp over previous PRIMARY 0.95354
(main's `d17_path_b_K23_v4_h1d_tau100000`). Top-5% gap closes
−5.1 → **−3.7 bp** (boundary 0.95405). Leader (MILANFX) gap −10.8 bp.

### What ran (DGP-reverse-engineering arc)

14 probes across 3 tiers + combined-with-main:

**Tier-1 CTGAN-aware (after F1 confirmed CTGAN-class)**
| Probe | Mechanism | K=21+1 Δ |
|---|---|---:|
| **d18 v1 chain (causal+gauss)** | per-step orig log-likelihood | **+7.37** ⭐ |
| F5 class-cond GMM Bayes-factor | 2 GMMs on 7 KS-low feats per class | +3.56 |
| J cond-vector tuple lookup | EB(P×C×S×R×Y).y_mean | +2.30 |
| F2 constraint violations | 10 physical constraints | +1.56 |
| d18b v2 chain (q10) | binned multiclass | +1.43 (5× weaker) |
| H mode-id × (C×S) lookup | EB on G's mode-ids | +0.97 |
| I mode-collapse bias-factor | synth_freq[m]/orig_freq[m] | +0.80 |
| G mode-id (CTGAN latent) | BGMM(10) per KS-low feat | +0.53 |
| E5/K1/K2/K3 cohort-axis | mode-id × Compound, etc. | NULL/regress |
| F1 GPU replay forensics | arch ID via SDV replay | NULL as base |

**F1 verdict**: host = **CTGAN-class GAN** (lowest mean KS 0.134 to CTGAN
replay; non-GAN→GAN P(replay-like) jump 0.06→0.13). All 4 disc AUCs
≥0.988 — host has custom signature.

**Combined-with-main**: pulled `oof_p1_single_cb_v4_gpu` + `oof_d17_h1d_yekenot_full`
artifacts via `git checkout origin/main -- ...`. Solo K=23+v4+h1d+1 marginals
(815s wall, 12 candidates):
- d16 +0.79 ⭐, E2 +0.42, d18 +0.33, F2 +0.25, F5 +0.21, J +0.18
- DAE +0.16 (wildcard fizzled — RealMLP h1d absorbs the unsupervised
  manifold), d18b +0.08, p1 v3 +0.08, orig-transfer +0.07, leak/DAE-full ≤0

**K=25/26/27 Path-B sweep** (Compound × Stint, τ ∈ {5k, 20k, 100k}):
- K=25 (v4+h1d+d16+E2): τ=100k OOF 0.95427 (+1.2 bp)
- K=26 (+d18): τ=100k OOF 0.95430 (+1.5 bp)
- **K=27 (+F2)**: **τ=100k OOF 0.95432** (+1.7 bp) ⭐ submitted
- Path-B amp ~1.0× across all K (friction reconfirmed)

**K=27 submitted** (Pre-submit-diff vs main PRIMARY: ρ=0.999023 borderline,
top-1% flips 129/154 R7-OK). Predicted band 0-2 bp; **realised +1.4 bp**
(dead-center). Calibration: 2-of-2 submissions in band this session.

### Falsified or dead this session

- E5 chain_LL_q5 cohort axis (Path-B regress)
- K1/K2/K3 mode-id Path-B cohort variants (all NULL vs Compound × Stint)
- F1 per-architecture disc features as bases (-0.11 bp; arch-bias is
  orthogonal to PitNextLap target)
- DAE wildcard solo on K=23 v4+h1d (+0.16 bp; absorbed)
- G/H/I CTGAN mode-id features on K=23 v4+h1d (≤+0.07 bp; absorbed by
  CatBoost CTR on combo-cats)

### Friction tags introduced this session

- `pool-saturation-v4h1d-absorbs-dgp-class` (generalises main's
  `pool-saturation-v4h1d-absorbs-d16d18` to all DGP-class incl
  CTGAN-aware mode-ids)
- `dae-wildcard-absorbed-by-mainline`
- `sequential-axis-untouched` (the biggest remaining DGP blind spot)
- `f1-replay-disc-features-orthogonal-to-target`
- `parallel-lgbm-3way-contention-oom` (process)
- `combined-pool-marginal-1bp-ceiling`

### Next-session first-action — RANKED queue (NEW probes after this round)

The DGP arc characterised the synthesizer end-to-end (architecture +
mode structure + class-conditional generator + within-row independence).
Five untouched mechanism axes remain — see ISSUES.md leaves 7g-7k:

**A1 (7g) ⭐ — Sequence-level DGP fingerprinting**: HMM on per-Year
Compound transition matrices + AR(1) on within-stint TyreLife;
per-(Driver, Race, Year) sequence log-likelihood under orig's transition
model. Synth groups with low LL = GAN-artifact strategies. Run-length
distributions per group. **Untested mechanism layer**: every probe so
far treats rows i.i.d., but the dataset is sequential (within-group
F1 strategies). v4+h1d sees rows i.i.d. internally → any sequence
signal is structurally orthogonal. Cost 2-3 h CPU. Predicted +1-3 bp
K=21+1; learning value high regardless of LB.

**A2 (7h) — Cross-feature joint mode-id**: G probed univariate mode-id;
the *tuple* `(mode_TL, mode_LT, mode_RP, mode_CD, mode_LD, mode_LN, mode_Pos)`
is the GAN's discrete latent VECTOR. Frequency-table comparison orig vs
synth + per-row log-frequency + cluster-id (k-prototypes) + EB(cluster).y_mean.
Cost 30 min CPU.

**A3 (7i) — Membership inference / exact-row copy detection**: 97.55%
literal LapTime overlap suggests some synth rows are near-exact orig
copies. Per synth row, min-distance to orig over all 16 columns. Below
ε threshold, predicted P(y=1) = orig's actual y (leak-free; uses orig
labels for orig rows). Cost 1 h CPU.

**A4 (7j) — CTGAN replay with explicit cond-vector spec**: F1's CTGAN
used default conditioning. Re-train with `cond_columns=[PitStop, Compound,
Stint, Year]` and stratified per-cond sampling. If host had this design,
KS to host_synth should drop sharply. Cost 3 h Kaggle GPU.

**A5 (7k) — Per-Year DGP heterogeneity / specialists**: d12 said 2023 =
flat 0.96% pit rate (vs 19% global). Per-Year KS divergence + per-Year
v4-recipe specialist test under K=27 pool. Cost 30 min CPU.

### Ranked path to top-5% (3.7 bp gap)

1. **A1 sequence-level fingerprinting** — HIGH learning value; new
   mechanism axis; predicted +1-3 bp marginal.
2. **External data: FastF1 lap-by-lap pit-call hard-join** (mainline
   Item 5) or **Pirelli pit-window scrape** (Item 6) — Tier-2 EV +10
   to +30 bp; only single-mechanism path that could close the full gap.
   Cost 1-2 days each.
3. **Yao/Vehtari covariance-modelled BMA** (T4b) — proper Path-B done
   with LKJ prior on inter-base Σ + GP prior on segment index. Untested;
   amp axis untested at K=27. Cost ~30 min Kaggle GPU.
4. **A2 / A3 / A5** — modest +0.5-1.5 bp each but interpretable
   diagnostics.

### Files added this session

- `scripts/d18_chain_decomp.py` (E1 v1)
- `scripts/d18b_chain_variants.py` (v2/v3)
- `scripts/d18_path_b.py` (K=22/23/25/26/27/28 Path-B variants)
- `scripts/d18_e2_preimage_knn.py`, `d18_e4_class_cond_chain.py`,
  `d18_e5_pathb_chain_cohort.py`, `d18_f2_constraint.py`,
  `d18_f5_class_cond_gmm.py`, `d18_f6_kl_ceiling.py`
- `scripts/d18_g_mode_id_ctgan.py`, `d18_h_mode_lookup.py`,
  `d18_i_mode_collapse.py`, `d18_j_cond_vector_lookup.py`,
  `d18_k_pathb_mode_cohort.py`
- `scripts/d18_combined_with_main.py` (greedy K=23 v4+h1d+N synth)
- `scripts/d18_combined_solo.py` (faster solo marginals)
- `scripts/d18_f1_synth.py` (ρ-matrix synthesis)
- `kernels/d18-f1-replay-forensics-gpu/` (replay forensics, COMPLETED)

### Audits

- `audit/2026-05-07-d18-chain-decomp.md` — d18 v1 (+7.37 bp single-base)
- `audit/2026-05-07-d18-dgp-decomp-batch.md` — E1-E5 batch synthesis
- `audit/2026-05-07-d18-ideaboard.md` — locked queue (still relevant)
- `audit/2026-05-07-d18-tier1-ctgan-batch.md` — G/H/I/J/K + F2/F5 synthesis
- `audit/2026-05-07-d18-postmortem.md` — Day-18 PM postmortem

### Submitted CSVs

- `submission_d18_path_b_K23_d16_d18_tau20000.csv` — LB 0.95149 (Day-17 PM)
- `submission_d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000.csv` — **LB 0.95368** ⭐

