# Day-11 Strategy Critique (Rule 14) — 2026-05-11

> Trigger: 12 submits since K=18 PRIMARY landed Day-6, of which 11 NULL
> and 1 PASS (d9c FM at +3bp). Latest (Day-11): TabM v3 extended-
> training fold-0 AUC 0.93926, **worse** than v2 (0.94039). PI: "do a
> strategy critique."
>
> This critique was first drafted assuming all 12 were nulls. Revised
> after picking up parallel-branch d9c FM PASS on origin/main.

## 1. Submit ledger (since Day-6)

| Day | Move | Class | Result | Why null/PASS |
|---:|---|---|---|---|
| 7 | RealMLP-bag (B,C) | bag of in-pool base | TIE ρ=0.9996 | variance reduction caps |
| 8 | T1.5 Deotte L2/L3 | meta-only | TIE ρ=0.985 | pool at meta info ceiling |
| 8 | T1.3 Q12 single-rule | rule_residual on raw | min-meta FAIL | LR routes around |
| 8 | T1.2 Poisson lapsuntil | reformulation | redundant | duplicates a_horizon |
| 9 | TabM v2 default | NN arch | gate FAIL 0.94039 | same class as RealMLP |
| 9 | C5 prev/next compound | rule_residual | K=20 TIE ρ=0.9999 | rule already absorbed |
| 9 | C1 SC-prob lookup | rule_residual external | K=19 TIE ρ=0.9999 | per-Race in pool |
| 9 | Hazard NN (LEAKY) | NN reformulation | LB **−73.5bp** | bfill leak |
| 9 | d9 10-cohort math/heuristic | rule_residual | all FAIL min-meta | 5th P10 confirmation |
| 9 | d9b R14 K=20 swap+L4 | sparse-LR rule | LB 0.95025 TIE | quantization-bounded |
| 9 | **d9c K=20 swap + FM** | **new model class (FM)** | **LB +3.0bp PASS** | **low-rank pairwise** |
| 10 | Hazard NN leak-free | NN reformulation | OOF 0.92013 | architecture zero signal |
| 10 | d9d FM hparam sweep + bag | FM tuning | all stack TIE ρ≥0.9999 | flat hparam |
| 11 | TabM v3 extended | NN arch (longer train) | gate FAIL 0.93926 | architecture is bound |

**Net since Day-6: +3bp LB lift, 1 burn (−6.3bp territory), 13 nulls.**
**The single PASS reframes the structural picture.**

## 2. The structural pattern (revised)

The original critique said "K=18 LR-meta absorbs everything; all moves
are TIE." That was wrong about d9c FM. The corrected pattern:

**Bases inside the pool's hypothesis class TIE; structurally different
model classes can lift.** Eleven nulls fall into 4 classes the K=18
LR-meta absorbs:

1. **In-pool hypothesis class** (TabM, RealMLP-bag, GBDT-meta).
2. **Single-rule rule_residuals on raw features** (Q12, C5, C1, d9 ×10).
3. **Reformulations duplicating in-pool features** (T1.2 Poisson).
4. **Target-construction with within-group propagation** (hazard).

But d9c FM **broke the pattern at +3bp** because its low-rank pairwise
embedding manifold is structurally outside what GBDTs/MLPs/sparse-LR
extract. The pool's hypothesis-class span has an **un-explored
direction**: bilinear-low-rank interaction surfaces.

## 3. EV math (revised)

Headroom to top-5% (0.95345): **31.6bp** (post-FM). 15 days left.

| Move | Median EV | Class |
|---|---:|---|
| **FFM (field-aware FM)** | **+0.5–2bp** | extends FM model class |
| **Multi-FM partition diversity** | +0.2–1bp | extends FM model class |
| **C2 Pirelli pit-windows** | **+3–8bp** | NEW INFORMATION |
| G4 SCARF/VIME on aadigupta | +1–4bp | different inductive bias |
| Bayesian hierarchical stacking | +1–3bp | different META structure |
| F2 multi-rule rebuild Q6 | 0–2bp | predicted NULL (P10 5×) |
| Adversarial validation weights | 0–2bp | reweighting |
| **Sum of medians** | **~13bp** | |
| Realistic transfer (50%) | **~7-9bp** | |

**Expected ceiling at full execution: 0.95100–0.95120** (top-12% to
top-18%). Top-5% still requires tail-case stacking (~5% probability).

The FM PASS shifted the ceiling expectation up ~2bp by demonstrating
model-class diversity is a live lever. FFM and multi-FM partitions
inherit that lever directly.

## 4. What's right vs wrong with the strategy

**Right axes (confirmed by FM)**: structurally different model
classes (FM, FFM, hierarchical Bayesian) AND new information (C2
Pirelli) both can break rank-lock.

**Wrong axes (still confirmed dead)**: in-pool hypothesis-class NN
variants (TabM ×2, RealMLP-bag); single-rule rule_residuals on raw
features (10× confirmation); same-mechanism rearrangement (R14 ladder).

**Reframed pivot**: "more bases" was wrong only when the new base
shared the existing hypothesis class. **Add bases that occupy
*different* model-class slots** — FM filled the bilinear-low-rank
slot; FFM extends it; SCARF/Bayesian add yet more orthogonal slots.

## 5. Recommended sequence (3-day window)

### Day-12 (today): FFM + multi-FM diversity
Build FFM (field-aware) following `scripts/d9c_fm.py` template;
~5 min CPU. Build 2 partition-FMs with disjoint field subsets.
Predicted +0.5–2bp K=N swap. **Highest-EV/$ move on the board.**

### Day-12–13: C2 Pirelli pit-windows scrape + build
6-8h scrape + 2h CPU + K=N rebuild following F1.2 template + Q6
filter. Highest-absolute-EV move; orthogonal to FM gains. Tail +8bp.

### Day-13–14: G4 SCARF on aadigupta1601 (overnight T4)
6-10h. Different unlabeled corpus avoids d5 partial-pseudo amp.

### Day-14: Bayesian hierarchical stacking
2-3h CPU. Different META structure; shrinks per-segment weights.

### Day-15+: F2 Q6 (cheap kill), calibration submits

## 6. Submit budget reality

16/270 used. 254 remaining over 15 days = ~17/day allowance.
Spend 3-4 calibration probes Day-12-14 on held TIEs (m5x, m5z,
d6_k15, d8_k19_q12, d9_k19_sc_prob, d9_k20_neighbor, d9d_*) to
recalibrate `pred_lb` at sub-0.99 ρ where current uncertainty is
~30bp. Especially urgent because d9c's +3bp upside (5.7× pred)
demonstrates pred_lb is **also** under-extrapolating in the
positive direction near 0.999 ρ.

## 7. Honest realistic finish

| Outcome | LB | Percentile |
|---|---:|---|
| Median (FFM + C2 land at median) | 0.95100–0.95120 | top-12% to top-18% |
| P75 (FFM + C2 + G4 at median) | 0.95150 | top-10% |
| Tail (C2 upper-tail + Bayesian + FFM) | 0.95220 | top-7% |
| Top-5% (0.95345) | requires 4 medians + tail | P ≈ 8–10% |

PRIMARY at 0.95029 is a stronger position than yesterday. The FM
breakthrough shifts the realistic finish ~5bp up. Right play: **run
FFM and C2 in parallel as the two highest-EV moves; accept top-12%
modal**, attack the tail-case with G4 + Bayesian.

## 8. Falsifications added to dead-list

- **TabM-D extended training (200 epochs, lr 3e-4)** — Day-11.
- **d9d FM hparam sweep + 3-seed bag** (in-class FM tuning) — Day-10.
- All 4 null classes from §2 confirmed as dead-list categories.

PRIMARY: `d9c_K20_swap_FM` LB **0.95029**, gap −2.6bp. 16/270 submits
used. 0/10 used Day-11.
