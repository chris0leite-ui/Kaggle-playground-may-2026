# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 10 (2026-05-11)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16 (Rule 16 Q7 PROPOSED below; needs PI go to land in CLAUDE.md)
2. `audit/2026-05-09-d9-hazard-nn-LB-postmortem.md` — load-bearing failure mode discovery
3. `audit/2026-05-10-d10-hazard-nn-leakfree-confirmed-dead.md` — diagnostic close-out
4. `audit/2026-05-09-d9-three-nulls-tabm-c5-c1.md` — Day-9 CPU/GPU sweep
5. `audit/2026-05-08-data-probe-results.md` — load-bearing P1-P10 (P6 explains the leak mechanism)
6. `scripts/pre_submit_diff.py` — MANDATORY before submit (ρ < 0.999)

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 10 morning)

- **PRIMARY UNCHANGED** = `d6_k18_multi_rule` LB **0.95026** (M5q + 4 rule_residuals).
- **Headroom to top-5%** (0.95345): **31.9bp**. To leader (0.95435): 41bp.
- **20 days remaining** (deadline 2026-05-31). 9 slots/day after submit reset.
- **Submits used today**: 0/10. **Total comp: 15.**

## Day-9 result — load-bearing failure mode discovery

Submitted `submission_d9_k19_hazard_nn_stack.csv` (M5q + 4 d6 rules +
hazard_nn_bag, K=19). Strat OOF 0.95446 (+38.1bp vs K=18). Predicted LB
0.95367. **Actual LB 0.94711. Gap −73.5bp** (vs calibrated −5bp).

**Root cause**: hazard target's `bfill` propagates val-row PitNextLap
backward through every earlier row in the same stint group. Per P6,
80% of consecutive-lap pairs cross fold boundaries → 80% of tr rows'
hazard buckets directly encode val labels. Hazard NN's K=20 buckets
per row carry 20× the per-row leak signal vs scalar reformulations.

**Day-10 leak-free re-run** (stint-drop within Strat fold): leak-free
seed-42 OOF **0.92013** vs leaky 0.94319. **Leak magnitude: 230bp.**
Hazard NN architecture itself adds **zero usable signal** beyond
the binary pool. **Hazard NN is DEAD; do not retry.**

## Pool audit finding

`b_lapsuntilpit` (in K=18) and `a_horizon` (in K=18) use the SAME
bfill mechanism. b_lapsuntilpit groups by `(Race, Driver)` — broader
than hazard's `(Race, Driver, Year, Stint)`. Both carry within-group
label leakage but at much smaller magnitude (their L1 weights 0.57 /
0.67 in the K=18 LR-meta — diluted by 14 leak-free binary
classifiers). Hazard NN entered K=19 with **L1=9.52**, dominating the
meta and exposing its 230bp leak.

K=18's calibrated −3.9bp gap is robust because all bases share
similar leak signatures and L1 weights are anchored by the binary
classifiers. **Don't disturb the K=18 pool composition** unless we
have a clean replacement.

## Proposed Rule 16 Q7 (needs PI go for CLAUDE.md)

> **Q7 — Target-construction leak test.** If a candidate base's
> target/feature construction depends on FUTURE same-group labels
> (e.g., `bfill`, `next_pit_lap`, horizon-shifted target with group
> propagation), apply ×0.1 EV downgrade UNLESS its OOF was computed
> under GroupKFold by the relevant group. Apply PRE-FLIGHT, not
> post-submit. Catches the failure mode that pre_submit_diff,
> min-meta, and Q6 all missed (all are sensitive to predictions, not
> targets).

## Day-10 first-action plan

### Path A — TabM v3 with extended training (top-priority)

Day-9 TabM smoke v2 stopped at val cross-entropy −0.4248 at epoch 5,
then 20 epochs of oscillation. Possibly converged to a local basin;
possibly under-trained at the margin.

**Build**: copy `kernels/tabm-smoke-gpu/` → `kernels/tabm-smoke-v3-gpu/`
with explicit longer training:
- introspect `pytabkit.TabM_D_Classifier.__init__` for `n_epochs` /
  `patience` / `lr` knobs; pass max-epochs ≥200, patience ≥50
- if pytabkit doesn't expose those, fall back to standalone `tabm`
  package (simpler API)
- ~10 min on T4

Same gates as smoke v2: fold-0 AUC ≥ 0.945 PROMOTE; else HOLD.

**No leak risk** (binary PitNextLap target). If it clears the gate,
this is the cleanest next-base candidate. EV +1-7bp at K=19.

### Path B — G4 SCARF/VIME pretrain on aadigupta1601 (heavy)

6-10h T4. Different unlabeled corpus avoids d5 partial-pseudo amp.
Different inductive bias (contrastive pretraining). No leak risk.

Defer until Path A resolves.

### Path C — F2 multi-rule rebuild with Q6 enforcement (cheap CPU)

After today's confirmation that K=18 pool is at the LR-meta info
ceiling (8 nulls), the only way to add a base via rule_residual is
to find a rule whose ρ-orthogonality is preserved AGAINST EVERY
existing pool member, not just M5q. Cheap experiment but predicted
NULL by analogy with C5/C1.

## Falsified / dead — do NOT retry

(keep this list growing)
- **Big sequence models** — P1
- **kNN / retrieval / TabR / Hopular** — P2
- **TabPFN-2.5 ICL ensemble** — P2 regime issue
- **RealMLP bagging** — Day-7
- **Broad pseudo-labeling** — Day-5
- **F5 aux-feature GBDT-meta** — Day-6
- **Move B 2-base [M5q, recursive]** — Day-6
- **Per-Race / per-Stint isotonic** — Day-3
- **Reintroduce `Normalized_TyreLife`** — host-removed
- **T1.5 Deotte L2 stacking** — Day-8
- **T1.3 Q12 single-rule rule_residual** — Day-8
- **T1.2 Poisson laps-until-pit** — Day-8
- **TabM-D smoke (default config)** — Day-9
- **C5 prev/next compound multi-rule** — Day-9 (K=20 TIE)
- **C1 SC-prob curated lookup** — Day-9 (K=19 TIE)
- **NEW: T1.4 Hazard-rate NN** — Day-9 LB −31.5bp / Day-10 leak-free OOF 0.92013

## Held submissions (do NOT submit)

- `submission_d9_k19_hazard_nn_stack.csv` — burned at LB 0.94711
- `submission_d9_hazard_nn.csv` (single-base hazard) — predict LB ~0.926
- `submission_d10_hazard_nn_leakfree.csv` — predicted LB ~0.916
- Day-9 `submission_d9_k20_neighbor.csv`, `submission_d9_k19_sc_prob.csv` — TIE-locked
- Day-7 `d7_realmlp_bag_partB.csv`, `d7_realmlp_bag_partC.csv` — TIE-locked
- Day-8 `d8_l3_blend.csv`, `d8_k19_q12.csv`, `d8_k19_poisson.csv`
- Day-5 `d5_partial_pseudo_m5q.csv` — burned at LB 0.94963

## Calibration ladder snapshot (Day 10 morning)

| Mechanism | Strat OOF | LB | Gap | Notes |
|---|---:|---:|---:|---|
| **d6_k18_multi_rule (PRIMARY)** | **0.95065** | **0.95026** | **−3.9bp** | UNCHANGED |
| m5q (M5h + RealMLP, K=14) | 0.95057 | 0.95005 | −5.2bp | parent |
| d9_k19_hazard_nn (LEAKY) | 0.95446 | **0.94711** | **−73.5bp** | burned |
| d9_k19_sc_prob | 0.95065 | n/a | n/a | TIE held |
| d9_k20_neighbor | 0.95065 | n/a | n/a | TIE held |
| Day-10 hazard leak-free | 0.92013 | n/a | n/a | DEAD |

## Critical operating rules (re-emphasised)

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **Predicted-LB calibration is fragile in low-ρ regime.** The
   `pred_lb` function's offset table was calibrated on ρ ≥ 0.99
   submits; at ρ ≈ 0.96 the offset extrapolation was 20× too small.
   **At ρ < 0.99, downgrade pred-LB by an additional 30bp** until we
   have more sub-0.99 submits to recalibrate.
3. **Rule 16 Q7 (PROPOSED): target-construction leak test.** Apply
   ×0.1 EV downgrade pre-flight unless GroupKF OOF available.
4. **Strat-only Day-3+** (R1) BUT: GroupKFold is now load-bearing
   for any base whose target depends on within-group future labels.
5. **Mechanism-class-only**: confirmed Nth time. Pool tweaks AND
   meta-only changes AND single-rule residuals on raw features AND
   duplicate reformulations AND single-base bag rebuilds AND
   leak-saturated NN bases are ALL dead.
6. **NEVER-GIVE-UP / saturation-is-bounded**. Today's failure is a
   data point about WHICH new bases work, not that no new bases
   work. TabM (binary target) is unfalsified.

## Pointers

- `audit/2026-05-09-d9-hazard-nn-LB-postmortem.md`
- `audit/2026-05-10-d10-hazard-nn-leakfree-confirmed-dead.md`
- `audit/2026-05-09-d9-three-nulls-tabm-c5-c1.md`
- `audit/2026-05-09-d9-c5-neighbor-rules-marginal.md`
- `audit/2026-05-08-data-probe-results.md` (P6 = leak mechanism)
- `audit/2026-05-08-strategic-menu-wider-steps.md` (apply Q7 overlay)
- `kernels/hazard-nn-leakfree-gpu/hazard_nn_leakfree.py` (stint-drop pattern)
- `kernels/tabm-smoke-gpu/tabm_smoke_gpu.py` (Day-9 v2; copy for Path A)
- `scripts/pre_submit_diff.py` (MANDATORY)
