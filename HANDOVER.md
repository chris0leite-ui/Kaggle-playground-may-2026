# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 8 (2026-05-08)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-15
2. `audit/2026-05-08-strategic-menu-wider-steps.md` — ranked menu (BUT
   note: T1.5/T1.3/T1.2 falsified today; see #3)
3. `audit/2026-05-08-d8-cpu-probes-falsified.md` — today's 3 negative
   results + menu re-rank
4. `audit/2026-05-08-data-probe-results.md` — load-bearing probes
   (P1 falsifies sequence; P10 confirms tight calibration)
5. `audit/2026-05-07-d6-f1-2-LB-result.md` — F1.2 PRIMARY landed +2.1bp
6. `scripts/pre_submit_diff.py` — MANDATORY before submit. Tighten
   ρ threshold to 0.9995.

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 9 morning)

- **Day 9, 0/10 used today.** Day-7 unused. Day-8 0/10 used (pure
  research + falsification day; no submissions).
- **PRIMARY** = `d6_k18_multi_rule` LB **0.95026** (M5q + 4
  rule_residuals). Strat OOF 0.95065. Gap −3.9bp.
- **Headroom to top-5%** (0.95345): **31.9bp**.
- **Headroom to leader** (0.95435): 41bp.
- **22 days remaining** (deadline 2026-05-31). 9 slots/day.

## ⚠️ ACTIVE THREAD: GPU bookkeeping

- **`realmlp-bag-gpu` kernel CANCELLED** (per PI on parallel branch)
  — d5 partial-pseudo gap-widening makes pseudo-rebuild EV negative.
  Reflection in this thread confirmed: hold the kernel.
- **Originally-planned RealMLP rank-average (seeds 42+123+456)** —
  status unknown; if seed-42 + bag completed earlier and just the
  pseudo-rebuild was cancelled, the rank-average may still be a
  free +1-3bp move. Verify on parallel branch's HANDOVER.
- **GPU window is currently held** per PI ("we will wait for GPU
  availability"). When GPU returns, FIRST PRIORITY is T1.1 TabM
  smoke (see §"Day-9 first-action plan").

## Day-8 falsified moves (DO NOT retry without new evidence)

- **T1.5 Deotte L2 stacking (LGBM-aug, Ridge-aug, L3 blend)** —
  ZERO OOF lift over LR-meta on K=18. The 18-base pool is at
  information ceiling for any meta family. ρ=0.985 rank-shuffle on
  L3 but no OOF gain. **Meta-only changes are dead.**
- **T1.3 Q12 mandatory-2-compound rule_residual** — standalone
  OOF 0.94518 / ρ=0.926 (legitimately diverse), but minimal-meta
  gate FAILED by 0.06bp. K=19 stack collapsed to ρ=0.99994. Q12
  signal applies to only 4.26% of rows (`must_change=1`); LR meta
  routes around it. **Rule-residual single-rule is dead unless
  coverage > 50% AND the rule's pairwise structure is genuinely
  outside the GBDT split set.**
- **T1.2 Poisson laps-until-next-pit reformulation** — standalone
  AUC 0.57 (data 79.77% censored due to within-stint grouping).
  Deeper reason: `scripts/b_laps_until_pit.py` ALREADY implements
  laps-until-pit. T1.2 was redundant with existing M5q pool.

## Updated CPU queue (post-Day-8 falsifications)

| # | Move | Cost | EV (bp) | Slot? |
|---|---|---|---:|---|
| C1 | **External-data Q3 (SC probability per track)** | 2-3h CPU + 1h scrape | 2-5 | ROC-PASS gate; new info not in pool |
| C2 | **External-data Q1 (Pirelli pit-windows)** | 6-8h scrape + 2h CPU | 3-12 | as 4-6 rule_residual bases (F1.2 pattern) |
| C3 | **EmbMLP on CPU** (PyTorch nn.Embedding for Driver) | 4-6h CPU (8-core) | 1-3 | distinct from RealMLP-TD cat handling |
| C4 | **Hierarchical Bayesian stacking (Yao 2021)** | 2-3h CPU (PyMC+JAX) | 1-6 | different META STRUCTURE (not just family) |
| C5 | **T2.1+T2.2 multi-rule extension** (next_compound + prev_compound × laps_into_stint, F1.2-style) | 30 min CPU | 0-3 | AT RISK of Q12-mode failure but cheap |
| C6 | RaceLength + Year=2023 mask audit | <1h | 0-2 | small but free |
| C7 | Adversarial-validation instance weights | 30 min | 0-2 | small but free |

## Updated GPU queue (when GPU returns)

| # | Move | Cost | EV (bp) | Notes |
|---|---|---|---:|---|
| G1 | **T1.1 TabM 1-fold smoke** (Rule 2!) | 1-2h T4 | gate test | NOT YET FALSIFIED; sole unbroken Tier-1 |
| G2 | **T1.1 TabM 5-fold + bag (3 seeds)** | 6-10h T4×2 | 2-8 | proceed only if smoke gate passes |
| G3 | **T1.4 Hazard-rate NN** (K=20 hazard buckets, nnet-survival loss) | 4-6h T4 | 1-7 | structurally different problem |
| G4 | **T2.6 SCARF pretrain on aadigupta1601 unlabeled** | 6-10h T4 | 1-6 | different unlabeled corpus avoids d5 amp |
| G5 | **TabPFN-v2 / TabICL-v2 ICL ensemble** | 6-12h T4 | 1-5 | regime mismatch; held |

## Day-9 first-action plan (when PI returns)

### If GPU is available:

1. **Push TabM 1-fold smoke kernel** to Kaggle T4. Use `pip install
   tabm` in the kernel; build with the same Strat fold-0 split as
   M5q (`SEED=42, N_FOLDS=5, fold 0`). Target wall <1.5h.
2. **Smoke gate**: fold-0 val AUC ≥ 0.945 (RealMLP fold-0 was 0.947).
   - PASS: schedule 5-fold + 3-seed bag overnight; expect Day-10 stack.
   - FAIL: don't proceed to 5-fold; document and try T1.4 hazard-NN.
3. **In parallel on CPU**: C5 (T2.1+T2.2 multi-rule extension) —
   cheap 30 min probe to test if rule_residual mechanism extends to
   higher-coverage signals. Build alongside TabM kernel push.

### If GPU is NOT available:

1. **C1 external-data Q3 SC probability** (highest CPU EV;
   non-redundant signal). Build SC-probability lookup from public
   F1 stats (axiorablogs.com, Williams Racing posts, oddschecker —
   sources cited in F1-domain audit). Implement as rule_residual
   base: lookup `(Race, sc_prob_decile, Stint, lap_quintile) →
   pit_rate`; HGBC residual.
2. **C2 external-data Q1 pit-windows** (highest absolute EV but
   highest engineering cost). Scrape Pirelli pit-window predictions
   from F1.com strategy-guide articles for 26 races × 4 years.
   Build 4-6 rule_residual bases (in_window, dist_to_window_center,
   etc.) following F1.2 multi-rule template.
3. **C5 T2.1+T2.2 multi-rule extension** in parallel — cheap probe.
4. **C4 Bayesian hierarchical stacking** — different meta STRUCTURE.

## Critical operating rules (reaffirmed Day-8)

1. **Pre-submit-diff before EVERY submit**, ρ < 0.9995.
2. **Mechanism-class-only**: today confirmed 3rd time. Pool tweaks
   AND meta-only changes AND single-rule residuals on raw features
   AND duplicate reformulations are all dead.
3. **Predicted-gap gate**: pred-gap <−7bp needs PI sign-off.
4. **Minimal-input-meta sanity check**: today's Q12 confirms this
   gate is sharp — Q12 failed by 0.06bp at 2-comp meta and the
   K=19 stack collapse confirmed the gate was right.
5. **Strat-only Day-3+** (R1; U3 confirmed i.i.d.).
6. **Track gap direction** — Day-7 K=18 narrowed gap −5.2 →−3.9bp.
   Real positive transfer.
7. **NEW** (Day-8) — **The strategic menu's EV estimates over-credited
   "wider steps" that turn out to be redundant or rank-locked.**
   T1.5 (meta-only): 0bp. T1.3 (single-rule rule_residual): 0bp.
   T1.2 (multi-formulation): redundant with existing pool. Recalibrate
   menu EV: when in doubt, the rank-lock will hold; only mechanism
   classes that ESCAPE GBDT-on-binary-AUC produce real lift.

## Calibration ladder snapshot (Day 9 morning)

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | gap −5.2bp |
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | **gap −3.9bp** |
| d8_l2_l3_blend | 0.95065 | n/a | TIE_EXPECTED ρ=0.985 (held) |
| d8_q12_v_b standalone | 0.94518 | n/a | min-meta FAIL by 0.06bp |
| d8_k19_q12 | 0.95065 | n/a | TIE ρ=0.99994 (rank-lock) |
| d8_poisson_lapsuntil | 0.56951 | n/a | redundant; std AUC ~random |
| d8_k19_poisson | 0.95064 | n/a | TIE ρ=0.99982 |

## Held submissions (do NOT blindly submit)

- (carry-forward from Day-7) `m5x_yetirank.csv`, `m5z_yetirank_nb.csv`,
  `m5_meta_lgbm_medium.csv`, `m5_meta_hgbc.csv`, `d5_meta_k15_*.csv`,
  `m5_k15a/b/c.csv`, `d5_partial_pseudo_m5q.csv`,
  `d6_aux_meta_with_aux.csv`, `d6_2base_v[1-4]_*.csv`,
  `d6_k15_rule_residual.csv`, `d6_k16_two_diverse.csv`
- (Day-8 new holds) `d8_l3_blend.csv`, `d8_k19_q12.csv`, 
  `d8_k19_poisson.csv` — all TIE-class (ρ ≥ 0.985); none merit
  a slot.

## Pointers

- `audit/2026-05-08-strategic-menu-wider-steps.md` — menu (refer
  with Day-8 falsification overlay)
- `audit/2026-05-08-d8-cpu-probes-falsified.md` — today's audit
- `audit/2026-05-08-data-probe-results.md` — P1-P10 probes (load-
  bearing for sequence, kNN, calibration constraints)
- `audit/2026-05-07-d6-f1-2-LB-result.md` — F1.2 K=18 LB win
- `scripts/d6_multi_rule.py` — F1.2 builder (template for C1, C2,
  C5 rule_residual extensions)
- `scripts/d8_l2_stacking.py`, `scripts/d8_q12_forced_pit.py`,
  `scripts/d8_poisson_lapsuntil.py` — falsified probes
- `scripts/probes_d8/run_probes.py` — P1-P10 probe code
- `scripts/pre_submit_diff.py` — MANDATORY (ρ < 0.9995)
