# Irrigation-water postmortem extract — 2026-05-04

Source repo: https://github.com/chris0leite-ui/Kaggle-irrigation-water
Postmortem dir: `writeup/postmortem/` (01..07.md). All 7 files fetched
via raw.githubusercontent.com. Citations use file refs of the form
`PM-0X §<section>` plus URLs.

## 0. Repo access notes

- `https://api.github.com/.../contents/writeup/postmortem` returned
  HTTP 403 (rate-limited); HTML tree view + raw URLs worked.
- Lower-cased URL `kaggle-irrigation-water` also resolves; both
  redirect to the same repo. Capital-K used throughout.

---

## 1. Top mechanisms ranked by LB lift (>5 bp)

From `PM-03 What worked` and `PM-02 Timeline` calibration ladder
(https://raw.githubusercontent.com/chris0leite-ui/Kaggle-irrigation-water/main/writeup/postmortem/03-what-worked.md ,
.../02-timeline.md):

| Rank | Mechanism | LB lift (bp) | Detail | Caveat |
|---|---|---:|---|---|
| 1 | DGP reverse-engineering (closed-form rule on 6 features) | ~+200 over raw, baked into baseline (LGBM 0.97097→0.97271) | Brute-forced thresholds; `dry, norain, hot, windy, nomulch` + Kc weight; `Low ≤3 / Med 4–6 / High ≥7`. PM-03 §1. | Hand FE on top of DGP regressed Δ=−0.00052; trees rediscover interactions. |
| 2 | `recipe_full_te` (TE + multi-seed bagging + High-class specialist) | +84 (0.97468→0.97581→0.97939 cumulative steps) | Target-encoding on cat-rich subset, 5 seeds × 5 folds, routed High-specialist, prior-reweight + log-bias. PM-03 §2. | Workhorse; everything later stacks on this. |
| 3 | 14-bank natural-cal meta + sklearn RF meta-stacker (`v1 RF natural standalone`) | +35 (0.98094→0.98129) | 14 calibrated prob vectors (LGBM/XGB/CatBoost/RF × reweighting) chosen by *error orthogonality* (pairwise Jaccard <0.85), fed to RF. PM-03 §3. | Negative OOF→LB gap (−0.00066) — structural to override family, not exploitable lift. |
| 4 | Selective-override stack (Idea 4b triple-consensus) | +56 over 4-stack (0.98094→0.98150); +21 over `v1 RF` (0.98129→0.98150) | 108 flips (105 H→M, 2 L→M, 1 M→L). Rule: flip iff `bagged_v1' ≠ B AND raw==tier1b AND 14-bank-majority agrees`. PM-03 §4. | Public-private overfit risk — see §7 below. |
| 5 | 3-way log-blend → greedy + XGB(non-rule) → 4-stack `tier1b_greedy_meta` | +20 / +12 / +43 cumulative (0.97097→0.97296→0.97352→0.98094 — note jumps include items 1–2 above) | Stacking era, PM-02 Phase 4. OOF 0.98084→LB 0.98094 (gap −10bp). | Stacking-inflation ceiling first appeared here (multiple metas OOF 0.98030 → LB 0.97995). |
| 6 | 4-gate leakage filter + minimal-input-meta sanity check | Defensive (prevented further losses on top of the +45bp already burned) | G1 standalone OOF clears anchor; G2 blend lift; G3 net rare-class flip ratio ≥0.5; G4 direction asymmetry. PM-03 §5. | Procedural, not a model. |

Honourable mentions (PM-03): GroupKFold leak-check, `lb_status.py`,
atomic submission builders.

---

## 2. Plateau breaks (top 5)

`PM-04 §5` lists 10 plateaus the agent declared structural:
0.97097, 0.97296, 0.97352, 0.97468, 0.97581, 0.97939, 0.97998,
0.98005, 0.98008, 0.98094. Quote: *"Every one of these was broken
by a mechanism the agent had previously labelled 'skip on principled
grounds'"*
(https://raw.githubusercontent.com/chris0leite-ui/Kaggle-irrigation-water/main/writeup/postmortem/04-what-failed.md).

Top 5 break moments (PM-02 Phases 3–6):

1. **0.97097 → 0.97296 (+20 bp)** — 3-way log-blend topology change
   (0.45 hybrid + 0.40 routed + 0.15 spec_678). Broken by **blend
   topology**, not a new model.
2. **0.97468 → 0.97581 (+11 bp)** — multi-seed bagging on top of
   `recipe_full_te`. Broken by **variance-reduction / seed bagging**.
3. **0.97581 → 0.97939 (+36 bp)** — full recipe production pipeline
   (TE + High-class router). Broken by **CV-honest FE pipeline**, not
   a new model class.
4. **0.98094 → 0.98129 (+35 bp)** — sklearn RF meta-stacker on the
   14-bank natural-cal bank. Broken by **new model class on a
   curated, error-orthogonal feature bank** (PM-02 Phase 6).
5. **0.98129 → 0.98150 (+21 bp)** — Idea 4b triple-consensus override.
   Broken by **decision-rule mechanism** (not a probabilistic blend);
   decorrelated from saturated stacking bank.

Common pattern: lifts came from *changing the mechanism class*
(topology / bagging / routing / stacker / decision rule), not from
hyperparameter tuning or larger models. NN expedition produced 0 of
these breaks across 18 architectures (PM-04 §1).

---

## 3. Leakage incidents (7 reported, ~0.0045 LB total)

Verbatim table from `PM-04 §2`:

| Date | Incident | LB Δ |
|---|---|---|
| 2026-04-23 | stage-2 pseudo-label (labeler+target same folds) | −0.00009 |
| 2026-04-23 | stacking-inflation ceiling (3+ blends OOF 0.98030 → LB ~0.97995) | flat |
| 2026-04-24 | soft-distillation student memorizes teacher OOF | −0.00148 |
| 2026-04-25 | LR meta v1 / v4 ET+kNN / P3 perturbed | −0.00103 / −0.00102 / −0.00139 |
| 2026-04-26 | DROP_DETERMINISTIC removed boundary-anchor rows | regressed |
| 2026-04-27 | R2 hybrid grid-selected (24-pt grid → OOF inflation) | −0.00046 |
| 2026-04-28 | stacking feature leak (80% gain from circular meta-of-metas) | regressed |

Common detection: each had positive OOF Δ that did not transfer to
LB. Resolution: 4-gate filter + minimal-input-meta-test
(PM-03 §5 — train candidate meta with ONLY `anchor + new` 2 components;
if 2-comp OOF < anchor, the N-comp lift was cross-component
memorization).

**Within-stratum vs cross-group distinction**: the closest analog in
PM-04 is the 04-26 `DROP_DETERMINISTIC` incident — removing
boundary-anchor rows (rule-deterministic rows the host labeled with
prob 1) regressed LB. Reading: dropping rows that are
*within-stratum* (rule-deterministic) hurts because they anchor
calibration on the rule-matching majority. The pseudo-label /
distillation leaks (04-23, 04-24) are *cross-group* leaks where the
labeler and target shared fold structure. **Transfer to s6e5**: the
anchor-A-vs-B verdict should hinge on whether the new anchor leaks
across folds (cross-group, fatal) or merely re-weights within
stratum (within-stratum, recoverable).

---

## 4. R-rule origins (verbatim quotes)

Source: `PM-07 Recommendations`
(https://raw.githubusercontent.com/chris0leite-ui/Kaggle-irrigation-water/main/writeup/postmortem/07-next-comp-recommendations.md).

**R1 — Two-anchor OOF.** Origin: *"OOF→public was tight (5–10bp);
public→private was wide (50–100bp). We trusted public as the ground
truth and optimized into it."* Fix: *"compute OOF under a different
CV scheme (e.g. GroupKFold on a row-id hash, or repeated stratified
with a different seed) and require gates to pass under BOTH OOFs."*

**R2 — Final selection along public-LB axis.** Origin: *"HEDGE was
an orthogonal-mechanism hedge (RF-natural vs override). Both
submissions overfit public LB the same way. The actually-best
private submissions (`idea5`, `W3_MHonly`, `bagginglr`) were rejected
because they regressed −2 to −44bp on public."* Fix: *"PRIMARY = best
public; HEDGE = best OOF that regressed on public by ≤30bp."*

**R5 — Final OOF-best regression probe.** Origin: *"we had
`idea5_anchor_switch` sitting at public 0.98148 (−2bp vs PRIMARY) on
Day 17 with the same 4-gate verdict. We treated it as inferior.
Private result: 0.98058 — our best."* Fix: *"in the final 3-day
window, add a mandatory final probe of the OOF-best candidate that
was rejected for public regression."*

**R7 — Override-mechanism rules.** Origin: *"The override family
produced our best public score and our worst public→private gap.
Override mechanisms are inherently a public-LB overfitting risk
because they target a small row count (108 flips × 80% public = ~22
public-relevant flips)."* Rules: *"Override candidates require all
four gates PASS on a SECOND OOF scheme (R1). Override-flip count >
200 requires explicit PI sign-off. Below 200 flips, the public-LB
signal is dominated by the public split's row sampling. An override-
family submission cannot be PRIMARY; it can only be HEDGE."*

**R8 — End-of-comp framework metric.** Origin: *"The framework's
stated target is 'top 5% reliably'. This comp: top 5.24%, missed by
11 ranks of 4315."* Fix: *"Track per-comp final rank and percentile
in `improvements.md`. After 3 comps, recompute … If the median is
>5%, demote the target."*

---

## 5. Failed mechanisms (don't repeat)

From `PM-04 What failed`
(https://raw.githubusercontent.com/chris0leite-ui/Kaggle-irrigation-water/main/writeup/postmortem/04-what-failed.md):

- **18 NN architectures** (TabPFN-10k, RealMLP n_ens={1,2,4},
  FT-Transformer, KAN, Mamba, Trompt, TabM, ExcelFormer, narrow MLPs
  at 12 capacity points, score-{6,7,8} specialist heads, training-
  data-routed MLPs). All passed standalone OOF; none passed blend
  gate. Ceiling not capacity (1M-param ≈ 50k-param). Reason given:
  *"DGP is rule-structured. Boosted trees with axis-aligned splits
  align with the rule's thresholds; an MLP learns a smoothed
  approximation that misses the same boundary rows the trees miss."*
  Cost: ~3 days + 1 GPU kernel killed at t+3h34min.
- **Multi-task XGB**: OOF +0.00036 but bank inflated to 149 components,
  flagged as stacking-inflation, never probed.
- **R3 NN-distance override family**: 18th saturation, signal not LB-extractable.
- **Macro-recall surrogate XGB**: first G4-PASS in 25+ saturations,
  but LB null — *"magnitude was below the resolution of the public-LB
  split (80/20 puts a hard floor on probe resolution at ~0.00005)."*
- **L1–L5 loss-function ensembles** (focal, conformal, etc.): 44–47th
  saturations, all regressions.
- **`bagginglr_natural` standalone**: 48th saturation, 0.98106 LB.
- **Soft-distillation student**: −0.00148 LB (memorizes teacher OOF).
- **Hand-coded physics-faithful FE on top of DGP**: Δ = −0.00052
  (PM-02 Phase 2).

Never-properly-tried (PM-04 §6): external LLM judge (rate-limit),
DGP NN-inversion (host arch unknown), public-CSV blending (banned
by ⚠️ rule).

---

## 6. Calibration ladder & gap analysis

From `PM-01` (https://raw.githubusercontent.com/chris0leite-ui/Kaggle-irrigation-water/main/writeup/postmortem/01-overview.md):

| Mechanism | OOF | LB | Gap |
|---|---:|---:|---:|
| recipe_full_te | 0.97967 | 0.97939 | +0.00028 |
| 3-way multi-seed | 0.98029 | 0.98005 | +0.00024 |
| LB-best 3-stack | 0.98061 | 0.98008 | +0.00053 |
| LB-best 4-stack (tier1b_greedy_meta) | 0.98084 | 0.98094 | −0.00010 |
| v1 RF natural standalone | 0.98063 | 0.98129 | −0.00066 |
| 2-OTHER raw+tier1b k=2 unanimous (B) | 0.98088 | 0.98140 | −0.00052 |
| Idea 4b triple-consensus (PRIMARY) | ~0.98088 | 0.98150 | −0.00062 |

Gap blew up at: **3-stack** (gap +53bp) — diagnosed as cross-component
memorization on a saturated bank (PM-02 Phase 4). Gap stayed tight
at: `recipe_full_te` (+28bp, near first-submission gap of +125bp,
within fold-std).

Quote from PM-01: *"Negative-gap entries (LB above OOF) are
override-family. They come from a different mechanism (14-bank
majority + selective flips) than the stacking metas. The negative
gap is structural to the override decision rule, not a margin we
could spend on stacks."* This warning should have been louder —
private results (PM-06) showed those negative gaps were public-split
selection bias.

---

## 7. Final selection (PRIMARY vs HEDGE) and private reveal

Source: `PM-06 Final results` (summarized fetch — verbatim tables
were not returned by WebFetch; numbers below are reproduced from
that summary). URL:
https://raw.githubusercontent.com/chris0leite-ui/Kaggle-irrigation-water/main/writeup/postmortem/06-final-results.md

- **PRIMARY**: `idea4b_selective_override` — public 0.98150 / private
  **0.98051** (regression of 99 bp public→private).
- **HEDGE chosen**: `sklearn_rf_meta_natural` — public 0.98129 /
  private 0.98047.

**Five own-submissions that beat PRIMARY on private** (per CLAUDE.md
quote and PM-06):

| Submission | Public | Private | Note |
|---|---:|---:|---|
| `idea5_anchor_switch` | 0.98148 | **0.98058** | rejected for −2bp public regress |
| `W3_MHonly` | 0.98127 | 0.98057 | rejected for −23bp public regress |
| `bagginglr_natural_standalone` | 0.98106 | 0.98055 | 48th saturation |
| `4b_plus_w5_strict90` | 0.98143 | 0.98052 | rejected for −7bp public regress |
| `idea4b` (PRIMARY) | 0.98150 | 0.98051 | chosen |
| `sklearn_rf_meta_natural` (HEDGE) | 0.98129 | 0.98047 | chosen as orthogonal-mechanism hedge |

Final rank: **226 / 4315 → top 5.24%** (missed top-5% by 11 ranks).
Quote: *"the override mechanism family broke the 0.98094 4-stack
ceiling on public LB (+0.00056) but this lift proved to be largely
selection bias on the public split. The private ceiling remained at
approximately 0.98050."*

---

## 8. Surprises / things not explicitly asked

- **Public→private spread compression ~3×**: OOF→public gaps were
  5–10bp; public→private gap was 50–100bp. The single most important
  number in the postmortem and the direct cause of R1/R2/R5/R7.
  (PM-06 summary, PM-07 R1.)
- **Override decision rules created NEGATIVE OOF→LB gaps** (LB above
  OOF by 50–66bp). PM-01 flagged this as "structural to the decision
  rule, not exploitable margin" — and it turned out to be public-LB
  *overfit* in disguise. Treat negative gaps as a red flag, not a
  feature.
- **Submission-budget burn**: 84 of 100 slots used (8.4/day). PM-04
  documents a `until ... grep -q "successfully submitted"` retry-loop
  that burned 4 slots on case-mismatched success marker. R4 mandates
  daily slot audit; <7/day for 2 days triggers a Research-loop probe.
- **Saturation count**: PM-02 logs "48 independent saturation
  confirmations at LB 0.98150" — most clustered within ±0.00005
  public, ALL within ±0.0001 private. R3: stop chasing public
  micro-lifts when daily spread < 2× public-private OOF gap.
- **CLAUDE.md crossed 1MB**, triggered Anthropic API idle timeouts,
  forced subagents to load 100k+ tokens. PM-05 §8. Hence the ≤50k
  cap and modular-file rule.
- **Stop-early bias was the dominant coordination failure** (PM-05
  §2). The agent argued ceiling at 10 distinct plateaus, every one
  broken by a mechanism the agent had labeled "skip on principled
  grounds". Direct genesis of the NEVER-GIVE-UP / saturation-is-
  bounded rule.
- **Top-5 winners' writeups were never read** (R6) — +37bp gap to
  leader (0.98219 vs 0.98150 public) remained unexplained.

---

## Pointers for the next session

- s6e5-relevant: R1 two-anchor OOF is highest priority — F1 pit-stops
  comp baseline LB 0.94113, top-5% 0.95345; gap = 122 bp. This is the
  same magnitude as the irrigation public→private spread, suggesting
  s6e5 may also have wide private gap.
- Override-mechanism rules (R7) likely irrelevant unless we end up
  with a similarly small flip-count decision rule on F1; deprioritize.
- Heuristic-first (CLAUDE.md rule 6) should pair with DGP-archaeology
  attempt if pit-stops show synthetic-rule structure (PM-03 §1).
