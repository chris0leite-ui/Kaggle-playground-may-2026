# Day-11 Strategy Critique (Rule 14) — 2026-05-11

> Trigger: 11 nulls in 5 days since K=18 PRIMARY landed Day-6. Latest
> (today): TabM v3 extended training fold-0 AUC 0.93926, **worse**
> than v2's 0.94039. PI: "do a strategy critique."
>
> This is not a brainstorm. It is an honest audit of whether the
> current trajectory can land top-5% in the remaining 16 days, and
> what to do about the answer.

## 1. Null ledger (since Day-6)

| Day | Move | Class | Result | Why null |
|---:|---|---|---|---|
| 7 | RealMLP-bag (B,C) | bag of in-pool base | TIE ρ=0.9996 | variance reduction caps at 0.1bp |
| 8 | T1.5 Deotte L2/L3 | meta-only | TIE ρ=0.985 | pool at info ceiling for any meta |
| 8 | T1.3 Q12 single-rule | rule_residual on raw | min-meta FAIL | LR routes around low-coverage rule |
| 8 | T1.2 Poisson lapsuntil | reformulation | redundant | duplicate of `a_horizon`/`b_lapsuntilpit` |
| 9 | TabM v2 default | NN arch | gate FAIL 0.94039 | same class as RealMLP |
| 9 | C5 prev/next compound | rule_residual | K=20 TIE ρ=0.9999 | redundant with `rule_driver_compound` |
| 9 | C1 SC-prob lookup | rule_residual external | K=19 TIE ρ=0.9999 | per-Race already in pool |
| 9 | Hazard NN K=19 (LEAKY) | NN reformulation | LB **−73.5bp** burn | bfill leaked val labels |
| 10 | Hazard NN leak-free | NN reformulation | OOF 0.92013 | architecture adds zero signal |
| 11 | TabM v3 extended | NN arch (longer train) | gate FAIL 0.93926 | converges in 10 epochs, degrades after |

Summary: **0 LB lift, 1 burn (−6.3bp net territory), 5 days of compute.**

## 2. The structural pattern

Every null falls into one of four classes, ALL of which the K=18
LR-meta absorbs:

1. **In-pool hypothesis class** (TabM, RealMLP-bag, GBDT-meta variants):
   ρ vs M5q ≥ 0.97 → ρ vs K=18 ≥ 0.999.
2. **Single-rule rule_residuals on raw features** (Q12, C5, C1):
   ρ vs M5q can be 0.89, but ρ vs K=18 collapses to 0.9999 because
   LR-meta projects onto the FULL pool span — Q6 binding constraint.
3. **Reformulations that duplicate in-pool features** (T1.2 Poisson):
   redundant before training.
4. **Target-construction with within-group propagation** (hazard
   bfill, partial-pseudo on test): leaks under Strat → catastrophic
   gap divergence.

**One root cause**: we have been adding bases to a pool whose
hypothesis class is fixed. LR-meta projects new bases onto pool
span. New base earns slot only if signal lives in the **complement
of pool span** — and P10 (anti-corr probe) found NO residual cohort
with |bias|≥2pp. The complement is empirically near-empty for any
*feature-based* base.

## 3. EV math, honest

Headroom to top-5% (0.95345): **31.9bp**. 16 days left. Realistic
sum-of-medians for remaining un-falsified moves:

| Move | Median EV | Class |
|---|---:|---|
| F2 multi-rule rebuild (Q6-enforced) | 0–2bp | predicted-NULL by C5/C1 analogy |
| G4 SCARF/VIME on aadigupta1601 | 1–4bp | different inductive bias (pretraining) |
| Bayesian hierarchical stacking (T2.5) | 1–3bp | different META structure |
| **C2 Pirelli pit-windows** | **3–8bp** | NEW INFORMATION, not new base |
| Adversarial validation weights | 0–2bp | reweighting only |
| TabM HPO sweep | 0–2bp | same class; downgraded |
| **Sum of medians** | **~12bp** | |
| Realistic transfer (50%) | **~6-8bp** | additivity is fictional |

**Expected ceiling at full execution: 0.95080–0.95100** (top-15%
to top-25%). **24bp short of top-5%.**

Top-5% requires either (a) a tail-event from C2 / G4 hitting the
upper end of its distribution, AND (b) the leader pack not moving
further. Neither is under our control.

## 4. What's actually wrong with the strategy

We have been hill-climbing on the **base axis**. The pool can only
extract what's in its inputs. New bases re-extract the same
information through new arch — that's why ρ collapses to 0.999.
The right axis is **information**.

- All 14 features are pool-saturated (P10).
- C5/C1 added derived features (next_compound, sc_prob): TIE.
- F1.2 multi-rule worked **because the rule keys encoded
  3-way interactions the GBDTs hadn't learned** — that was the
  last lift (+2.1bp).
- The remaining structural moves are: (i) genuinely external info
  (C2 Pirelli, race-control logs); (ii) different META structure
  (Bayesian hierarchical) so the pool gets re-projected; (iii)
  pretraining on a different unlabeled corpus (G4 SCARF).

## 5. Recommended pivot (3-day window)

**Stop running NN arch variants and single-rule rule_residuals.**
Both classes are saturated. Replace the queue with:

### Day-11 (today, remaining): F2 multi-rule rebuild w/ Q6
2-3h CPU. Cheap falsification of "Q6-enforced ρ vs K=18 yields a
survivor". Predicted NULL but the negative result tightens
calibration on what mechanism survives the projection.

### Day-12–13: C2 Pirelli pit-windows scrape + build
6-8h scrape + 2h CPU + K=N stack rebuild. Per-(Race, Year, Compound)
pit-window scalars + 4-6 rule_residual bases following F1.2 template.
This is the highest-EV move on the menu and the only one that brings
**genuinely new information** to the pool. Tail-case +8bp; median
+5bp.

### Day-14: G4 SCARF on aadigupta1601 (kicked off Day-13 evening)
6-10h T4 overnight. Different unlabeled corpus avoids d5 amp.
Unblocked by today's TabM closure (only un-falsified GPU candidate).

### Day-15: Bayesian hierarchical stacking probe
2-3h CPU. Different META structure shrinks per-segment (Compound,
Stint, Year×Compound) weights. Per-Race isotonic over-fit; Bayesian
shrinks toward global → noisy segments collapse, signal-rich pulled
local.

## 6. Submit budget reality

We have used 14/270 submits over 11 days (5%). Budget allows ~16
submits/day for remaining 16 days. **We are leaving calibration
data on the table.** Held TIE submissions (m5x, m5z, m5_meta_lgbm,
d6_k15_rule, d8_k19_q12, d9_k19_sc_prob, d9_k20_neighbor) are 7
calibration-spend candidates that would refine pred_lb at sub-0.99 ρ
(currently 30bp uncertainty band per Day-9 postmortem).

Recommend: spend 3-4 calibration submits over the next 5 days on
held submissions to recalibrate pred_lb. The hazard-NN burn was
partly caused by the pred_lb extrapolation being 20× too small at
ρ ≈ 0.96; this is a structural risk for ALL remaining low-ρ submits.

## 7. Honest realistic finish

Median 0.95080–0.95100 (top-15–25%); P75 0.95130 (top-12%); tail-case
C2+G4 upper-tail 0.95200 (top-8%); top-5% (0.95345) requires C2 +8bp
**and** G4 +6bp **and** Bayesian +3bp **and** leader stagnation, P≈5%.

PRIMARY at 0.95026 is already a strong finish given the rank-lock.
Right play: **pursue C2 hard, run G4 background, accept top-15% modal**
rather than burn compute on in-class NN/GBDT variants.

## 8. Falsifications added to dead-list

- **TabM-D extended training (200 epochs, lr 3e-4)** — Day-11.
- All 4 null classes from §2 are now confirmed dead-list categories,
  not single-experiment falsifications.

PRIMARY unchanged: `d6_k18_multi_rule` LB **0.95026**, gap −3.9bp.
0/10 submits used today. 256/270 total budget remaining.
