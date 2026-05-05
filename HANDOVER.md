# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 8 (2026-05-08)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16 (Rule 16 NEW Day-8: 5-question pre-flight)
2. `audit/2026-05-08-d8-cpu-probes-falsified.md` — Day-8 T1.5/T1.3/T1.2 falsifications
3. `audit/2026-05-08-d7-realmlp-partial-bag-null.md` — Day-7 RealMLP-bag NULL salvage
4. `audit/2026-05-08-strategic-menu-wider-steps.md` — full ranked menu (read with falsification overlay below)
5. `audit/2026-05-08-data-probe-results.md` — load-bearing probes (P1 falsifies sequence; P10 confirms tight calibration)
6. `audit/2026-05-07-d6-f1-2-LB-result.md` — F1.2 PRIMARY landed +2.1bp
7. `scripts/pre_submit_diff.py` — MANDATORY before submit. ρ threshold 0.9995.

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 9 morning)

- **Day 9, 0/10 used today.** Day-7 used 0/10 (bag salvage NULL, no submit). Day-8 used 0/10 (CPU falsifications, no submit).
- **PRIMARY** = `d6_k18_multi_rule` LB **0.95026** (M5q + 4 rule_residuals). Strat OOF 0.95065. Gap −3.9bp.
- **Headroom to top-5%** (0.95345): **31.9bp**.
- **Headroom to leader** (0.95435): 41bp.
- **22 days remaining** (deadline 2026-05-31). 9 slots/day.

## Day-7 RealMLP bag thread — CLOSED, NULL salvage

`kernels/realmlp-bag-gpu/` was cancelled mid-fold-3 of seed 123 after
parallel-branch probes (P10 in `audit/2026-05-08-data-probe-results.md`)
downgraded bag EV to Tier-3. Salvage in `scripts/d7_realmlp_partial_bag.py`
tested two paths:

| Path | K=18 OOF | Δ d6_k18 | ρ vs d6_k18 |
|---|---:|---:|---:|
| B (bagged TEST + seed-42 OOF) | 0.95065 | −0.02bp | 0.99955 |
| C (hybrid OOF + bagged TEST) | 0.95066 | +0.08bp | 0.99964 |

Both above the 0.9995 tightened tie threshold. **Confirmed Tier-3
classification**: variance-reduction on 1 of 18 bases caps stack lift
at ≤0.1bp. Both submissions HELD — see
`audit/2026-05-08-d7-realmlp-partial-bag-null.md`. **Do not retry RealMLP
bagging.**

## Day-8 CPU probes — three falsifications (load-bearing)

| Probe | Std OOF | ρ vs PRIMARY | min-meta Δ | K=N stack | Verdict |
|---|---:|---:|---:|---|---|
| T1.5 Deotte L2 (LGBM-aug) | 0.95057 | 0.99517 | n/a | OOF -0.76bp | FAIL |
| T1.5 Deotte L3 blend | 0.95065 | 0.98535 | n/a | OOF +0.02bp ρ=0.985 | TIE_EXPECTED |
| T1.3 Q12 forced-pit (V_B) | 0.94518 | 0.92570 | -0.06bp | K=19 ρ=0.99994 | FAIL |
| T1.2 Poisson laps-until-pit | 0.56951 | 0.36998 | -0.08bp | K=19 ρ=0.99982 | FAIL (redundant) |

Three failure modes:
- **T1.5 meta-only**: 18-base pool at info ceiling regardless of meta family. d3-endgame already said this.
- **T1.3 single-rule rule_residual**: low coverage (4.26%) + GBDT-class residual on raw features → LR meta routes around.
- **T1.2 reformulation**: redundant — `a_horizon` + `b_lapsuntilpit` are ALREADY in M5q pool from Day-2.

Friction codified: `tag: menu-overcrediting-redundant-mechanism` →
**CLAUDE.md Rule 16**: 5-question pre-flight check before committing
compute on any new candidate.

## Updated priors (from data probes P1-P10)

- **P1 falsifies sequence models.** Test groups average 2.25 laps;
  only 9.7% have ≥5 consecutive laps. Big LSTM/Transformer bounded.
  What survives: 1-step lookup features (next_compound, prev_compound,
  laps_into_stint).
- **P2 falsifies retrieval.** kNN distances too large → TabR /
  Hopular / TabPFN-context bounded.
- **P5**: 68% of test rows have computable `next_compound` — large
  unused signal.
- **P10**: pool extracts what's extractable from the 14 raw features.
  No residual cohort with |bias|≥2pp. **Lift requires NEW SIGNALS or
  NEW MODEL CLASS, not better extraction.**
- **P6**: StratifiedKFold has 80% within-group leakage. OOF is
  optimistic by ~5bp (matches our gap). Use Strat as LB proxy per R1
  but add GroupKFold(Race, Driver, Year, Stint) as diagnostic.
- **C6**: compute is NOT the binding constraint.

## Updated CPU queue (post-Day-8 falsifications)

| # | Move | Cost | EV (bp) | Notes |
|---|---|---|---:|---|
| C1 | **External-data Q3 (SC probability per track)** | 2-3h CPU + 1h scrape | 2-5 | new info NOT in pool; race-marginal preserved under CTGAN |
| C2 | **External-data Q1 (Pirelli pit-windows)** | 6-8h scrape + 2h CPU | 3-12 | as 4-6 rule_residual bases (F1.2 pattern); highest absolute EV |
| C3 | **EmbMLP on CPU** (PyTorch nn.Embedding for Driver) | 4-6h CPU (8-core) | 1-3 | distinct from RealMLP-TD cat handling |
| C4 | **Hierarchical Bayesian stacking (Yao 2021)** | 2-3h CPU (PyMC+JAX) | 1-6 | different META STRUCTURE (not just family) |
| C5 | T2.1+T2.2 multi-rule extension (next_compound + prev_compound × laps_into_stint, F1.2-style) | 30 min CPU | 0-3 | AT RISK of Q12-mode failure but cheap |
| C6 | RaceLength + Year=2023 mask audit | <1h | 0-2 | small but free |
| C7 | Adversarial-validation instance weights | 30 min | 0-2 | small but free |

## GPU queue (when GPU returns)

| # | Move | Cost | EV (bp) | Notes |
|---|---|---|---:|---|
| G1 | **T1.1 TabM 1-fold smoke** (Rule 2!) | 1-2h T4 | gate test | NOT YET FALSIFIED; sole unbroken Tier-1 |
| G2 | **T1.1 TabM 5-fold + bag (3 seeds)** | 6-10h T4×2 | 2-8 | proceed only if smoke gate passes |
| G3 | **T1.4 Hazard-rate NN** (K=20 hazard buckets, nnet-survival loss) | 4-6h T4 | 1-7 | structurally different problem |
| G4 | **T2.6 SCARF pretrain on aadigupta1601 unlabeled** | 6-10h T4 | 1-6 | different unlabeled corpus avoids d5 amp |
| G5 | TabPFN-v2 / TabICL-v2 ICL ensemble | 6-12h T4 | 1-5 | regime mismatch; held |

## Day-9 first-action plan

### If GPU is available:

1. **Push TabM 1-fold smoke kernel** to Kaggle T4. Use `pip install tabm`;
   build with the same Strat fold-0 split as M5q (`SEED=42, N_FOLDS=5,
   fold 0`). Target wall <1.5h.
2. **Smoke gate**: fold-0 val AUC ≥ 0.945 (RealMLP fold-0 was 0.947).
   - PASS: schedule 5-fold + 3-seed bag overnight; expect Day-10 stack.
   - FAIL: don't proceed to 5-fold; document and try T1.4 hazard-NN.
3. **In parallel on CPU**: C5 multi-rule extension (cheap 30 min) +
   C1 SC-probability scrape.

### If GPU is NOT available:

1. **C1 external-data Q3 SC probability** (highest CPU EV; non-redundant
   signal). Build SC-probability lookup from public F1 stats
   (axiorablogs.com, Williams Racing posts, oddschecker — sources cited
   in F1-domain audit). Implement as rule_residual base: lookup
   `(Race, sc_prob_decile, Stint, lap_quintile) → pit_rate`; HGBC residual.
2. **C2 external-data Q1 pit-windows** (highest absolute EV but highest
   eng cost). Scrape Pirelli pit-window predictions from F1.com strategy-
   guide articles for 26 races × 4 years. Build 4-6 rule_residual bases
   (in_window, dist_to_window_center, etc.) following F1.2 multi-rule
   template.
3. **C5 T2.1+T2.2 multi-rule extension** in parallel — cheap probe.
4. **C4 Bayesian hierarchical stacking** — different meta STRUCTURE.

## Critical operating rules (reaffirmed Day-8)

1. **Pre-submit-diff before EVERY submit**, ρ < 0.9995.
2. **Mechanism-class-only**: today confirmed Nth time. Pool tweaks AND
   meta-only changes AND single-rule residuals on raw features AND
   duplicate reformulations AND single-base bag rebuilds are all dead.
3. **Predicted-gap gate**: pred-gap <−7bp needs PI sign-off.
4. **Minimal-input-meta sanity check**: today's Q12 confirms this gate
   is sharp — Q12 failed by 0.06bp at 2-comp meta and the K=19 stack
   collapse confirmed the gate was right.
5. **Strat-only Day-3+** (R1; U3 confirmed i.i.d.).
6. **Track gap direction** — Day-7 K=18 narrowed gap −5.2 →−3.9bp.
   Real positive transfer.
7. **NEW** (Day-8) **Rule 16 pre-flight (5 questions)**. Apply BEFORE
   committing compute: ladder cross-check + mechanism-vulnerability
   classification + predicted standalone OOF + predicted ρ vs PRIMARY
   + closest gate-PASS/FAIL precedent. EV midpoint × 0.3 if rank-lock-
   vulnerable.

## Falsified / dead — do NOT retry

- **Big sequence models** — P1.
- **kNN / retrieval / TabR / Hopular** — P2.
- **TabPFN-2.5 ICL ensemble** — same regime issue as P2; held.
- **RealMLP bagging** — Day-7 partial-bag NULL salvage.
- **Broad pseudo-labeling** — Day-5 partial-pseudo K=14.
- **F5 aux-feature GBDT-meta** — Day-6.
- **Move B 2-base [M5q, recursive]** — Day-6.
- **Per-Race / per-Stint isotonic** — Day-3 in-CV regress.
- **Reintroduce `Normalized_TyreLife`** — host-removed.
- **T1.5 Deotte L2 stacking** — Day-8 (meta-only changes 0bp).
- **T1.3 Q12 single-rule rule_residual** — Day-8 (low-coverage rule_residual collapses).
- **T1.2 Poisson laps-until-next-pit** — Day-8 (redundant with `a_horizon`/`b_lapsuntilpit`).

## Calibration ladder snapshot (Day 9 morning)

| Mechanism | Strat OOF | LB | Notes |
|---|---:|---:|---|
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | gap −5.2bp |
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | **gap −3.9bp** |
| d7_realmlp_bag_partB | 0.95065 | n/a | TIE ρ=0.99955 (held) |
| d7_realmlp_bag_partC | 0.95066 | n/a | TIE ρ=0.99964 (held) |
| d8_l2_l3_blend | 0.95065 | n/a | TIE ρ=0.985 (held) |
| d8_q12_v_b standalone | 0.94518 | n/a | min-meta FAIL by 0.06bp |
| d8_k19_q12 | 0.95065 | n/a | TIE ρ=0.99994 (rank-lock) |
| d8_poisson_lapsuntil | 0.56951 | n/a | redundant; std AUC ~random |
| d8_k19_poisson | 0.95064 | n/a | TIE ρ=0.99982 |

## Held submissions (do NOT blindly submit)

- **Day-7 NULL salvage**: `d7_realmlp_bag_partB.csv`, `d7_realmlp_bag_partC.csv`
- **Day-8 new holds**: `d8_l3_blend.csv`, `d8_k19_q12.csv`, `d8_k19_poisson.csv`
- **Carry-forward TIE/NULL**: `m5x_yetirank`, `m5z_yetirank_nb`,
  `m5_meta_lgbm_*`, `m5_meta_hgbc`, `d5_meta_k15_*`, `m5_k15a/b/c`
- **Burned**: `d5_partial_pseudo_m5q` (−4.2bp)
- **Day-6 falsified**: `d6_aux_meta_with_aux`, `d6_2base_v[1-4]_*`,
  superseded `d6_k15_rule_residual` / `d6_k16_two_diverse`

## Pointers

- `audit/2026-05-08-d8-cpu-probes-falsified.md` — today's CPU audit
- `audit/2026-05-08-d7-realmlp-partial-bag-null.md` — Day-7 bag NULL
- `audit/2026-05-08-strategic-menu-wider-steps.md` — full menu (refer with falsification overlay)
- `audit/2026-05-08-data-probe-results.md` — P1-P10 probes (load-bearing for sequence, kNN, calibration)
- `audit/2026-05-07-d6-f1-2-LB-result.md` — F1.2 K=18 LB win
- `audit/2026-05-07-d6-f1-2-multi-rule.md` — K=18 build template
- `scripts/d6_multi_rule.py` — F1.2 builder (template for C1, C2, C5 extensions)
- `scripts/d6_rule_residual.py` — F1.1 builder
- `scripts/d7_realmlp_partial_bag.py` — bag salvage (closed)
- `scripts/d8_l2_stacking.py`, `scripts/d8_q12_forced_pit.py`, `scripts/d8_poisson_lapsuntil.py` — falsified probes
- `scripts/probes_d8/run_probes.py` — P1-P10 probe code
- `scripts/pre_submit_diff.py` — MANDATORY (ρ < 0.9995)
