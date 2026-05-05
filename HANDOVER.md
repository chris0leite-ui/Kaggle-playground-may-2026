# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 12 (2026-05-12)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16 (Q7 still PROPOSED; Day-11
   critique reaffirms the Q6 binding constraint)
2. `audit/2026-05-11-d11-strategy-critique.md` — load-bearing pivot doc
3. `audit/2026-05-11-d11-tabm-v3-extended-training-dead.md` — TabM closure
4. `audit/2026-05-09-d9-hazard-nn-LB-postmortem.md` — leak mechanism
5. `audit/2026-05-08-data-probe-results.md` — P1-P10 priors
6. `scripts/pre_submit_diff.py` — MANDATORY before submit (ρ < 0.999)

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 12 morning)

- **PRIMARY UNCHANGED** = `d6_k18_multi_rule` LB **0.95026**.
- **Headroom to top-5%** (0.95345): **31.9bp**. To leader: 41bp.
- **15 days remaining** (deadline 2026-05-31). 9 slots/day after reset.
- **Submits used Day-11**: 0/10. **Total comp: 14 / 270 budget.**
- **Null streak**: 11 in 5 days since Day-6 K=18 landed.

## Day-11 close: TabM v3 dead + strategy pivot

TabM v3 extended training (n_epochs=200, patience=50, lr=3e-4) returned
fold-0 AUC **0.93926**, **−11.3bp worse than v2** (0.94039). Training
log: best val at epoch 11, 189 epochs of monotone slow drift after.
Patience never tripped (drift inside tolerance). Path A confirmed:
**TabM-D is dead at any plausible training schedule on this DGP.**

Strategy critique (Rule 14) wrote
`audit/2026-05-11-d11-strategy-critique.md`. One root cause behind
all 11 nulls: hill-climbing on the **base axis**. LR-meta projects
new bases onto K=18 span; complement is empirically near-empty
(P10). Top-5% requires either C2 Pirelli upper-tail OR external
breakthrough; honest median outcome is **top-15% to top-25%**.

## Day-12 first-action plan

### Path C — F2 multi-rule rebuild with Q6 enforcement (cheap CPU)

Per critique §5: 2-3h CPU. Predicted-NULL by C5/C1 analogy but cheap
to falsify and tightens the dead-list category.

**Build** (template: `scripts/d6_multi_rule.py`):
- Generate ~10 candidate rule_residuals over 3-way keys not yet in
  pool (e.g. `(Driver, Stint, lap_quintile)`, `(Race, Year, Compound)`,
  `(Stint, prev_compound, laps_into_stint_decile)`)
- Score each candidate's **ρ vs K=18 OOF** (not just M5q) — the Q6
  filter — keep only those at ρ ≤ 0.997
- Build K=N+m stack with survivors; gate via min-meta + ρ vs K=18 ≤ 0.999

**Gate**: if zero candidates survive Q6 → confirms dead-list category,
move straight to Path B. If ≥1 survives → predicted +0.5–2bp.

### Path B — C2 Pirelli pit-windows (highest-EV; new INFORMATION)

Critique §5 highest priority. **Do NOT defer past Day-13.**
- 6-8h scrape: F1.com strategy guides + Pirelli press kits, 26 races
  × 4 years for `(Race, Year, Compound) → optimal_pit_window` lookup
- 2h CPU: build 4-6 rule_residual bases following F1.2 template
- Apply Q6 filter on each before K=N add
- EV +3-8bp median; this is the **only** un-explored move that
  brings genuinely new info to the pool

### Path D — Calibration submits on held TIEs (1-2 day overlap)

Spend 3-4 of the unused submit budget on held TIE candidates (m5x,
m5z, d6_k15, d8_k19_q12, d9_k19_sc_prob, d9_k20_neighbor). Goal:
recalibrate `pred_lb` at sub-0.99 ρ where current uncertainty is
~30bp (caused the hazard-NN burn). Single-shot per Rule 1; PI sign-
off per submit.

### Background — G4 SCARF on aadigupta1601 (Day-13 overnight)

Kick off when Path C returns. 6-10h T4. Different inductive bias
(contrastive pretraining); only un-falsified GPU candidate post-
TabM closure.

## Falsified / dead — do NOT retry

(Day-11 additions in **bold**)
- Big sequence models — P1
- kNN / retrieval / TabR / Hopular — P2
- TabPFN-2.5 ICL ensemble — P2 regime
- RealMLP bagging — Day-7
- Broad pseudo-labeling — Day-5
- F5 aux-feature GBDT-meta — Day-6
- Move B 2-base [M5q, recursive] — Day-6
- Per-Race / per-Stint isotonic — Day-3
- Reintroduce `Normalized_TyreLife` — host-removed
- T1.5 Deotte L2 stacking — Day-8
- T1.3 Q12 single-rule rule_residual — Day-8
- T1.2 Poisson laps-until-pit — Day-8
- TabM-D smoke (default config) — Day-9
- C5 prev/next compound multi-rule — Day-9 (K=20 TIE)
- C1 SC-prob curated lookup — Day-9 (K=19 TIE)
- T1.4 Hazard-rate NN — Day-9 LB −31.5bp / Day-10 leak-free 0.92013
- **TabM-D extended training (200 epochs, lr=3e-4)** — Day-11 v3
- **Single-rule rule_residuals on raw features (whole class)** — 3× confirmed
- **In-pool hypothesis-class NN variants (whole class)** — 4× confirmed

## Held submissions (do NOT submit)

- BURNED: `d5_partial_pseudo_m5q.csv` (LB 0.94963), `d9_k19_hazard_nn_stack.csv` (LB 0.94711)
- Day-7-10 TIE/DEAD: `d7_realmlp_bag_part[BC].csv`, `d8_{l3_blend,k19_q12,k19_poisson}.csv`,
  `d9_{hazard_nn,k20_neighbor,k19_sc_prob}.csv`, `d10_hazard_nn_leakfree.csv`

## Calibration ladder snapshot (Day 12 morning)

| Mechanism | Strat OOF | LB | Gap | Notes |
|---|---:|---:|---:|---|
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | **−3.9bp** | UNCHANGED |
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | −5.2bp | parent |
| TabM v2 fold-0 | 0.94039 | n/a | n/a | gate FAIL |
| **TabM v3 fold-0** (extended) | **0.93926** | n/a | n/a | **gate FAIL −11bp from v2** |
| d10 hazard leak-free | 0.92013 | n/a | n/a | DEAD |

## Critical operating rules (re-emphasised Day-11)

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **Predicted-LB calibration is fragile in low-ρ regime.** At ρ < 0.99,
   downgrade pred-LB by additional 30bp until recalibrated.
3. **Rule 16 Q6 (ρ vs FULL K=18, not just M5q)** is the binding gate,
   as Day-11 critique formalised. Apply pre-flight to every new base.
4. **Rule 16 Q7 (target-construction leak test)**: ×0.1 EV unless
   GroupKF OOF.
5. **Strat Day-3+** (R1) BUT GroupKFold load-bearing for any base
   whose target depends on within-group future labels.
6. **Mechanism-class-only**: confirmed 11×. **Stop running NN arch
   variants and single-rule rule_residuals on raw features.** Both
   classes saturated.
7. **Submit budget vastly underused** (14/270). Spend 3-4 calibration
   probes Day-12-14 on held TIEs to recalibrate pred_lb.
8. **Information > bases.** Day-12 priorities: Path C falsification
   then Path B (C2 Pirelli) — the ONLY remaining move with novel info.

## Pointers

- `audit/2026-05-11-d11-strategy-critique.md` — load-bearing pivot
- `audit/2026-05-11-d11-tabm-v3-extended-training-dead.md` — TabM close
- `audit/2026-05-10-d10-hazard-nn-leakfree-confirmed-dead.md` — leak diag
- `audit/2026-05-08-{data-probe-results,strategic-menu-wider-steps}.md`
- `scripts/{d6_multi_rule,pre_submit_diff}.py` — F2 template + MANDATORY gate
