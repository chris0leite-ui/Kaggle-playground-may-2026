# Day-6 Critic-Loop Audit (Rule 14) — 2026-05-07

> Trigger: 2 consecutive negative gap drifts. d4 slot-2
> (m5_meta_lgbm_shallow) gap −4.7bp (Δ +0.5bp); d5 slot-1 (partial-
> pseudo K=14) gap −12.0bp (Δ −6.8bp). Audit before any new family.

## 1. All-submit gap ledger

| # | Submit | OOF | LB | Gap (bp) | Notes |
|---|---|---:|---:|---:|---|
| 1 | baseline_two_anchor | 0.94075 | 0.94113 | +3.8 | LB-proxy ✓ |
| 2 | e3_hgbc | 0.94876 | 0.94870 | −0.6 | best single GBDT |
| 3 | m5b | 0.94926 | 0.94891 | −3.5 | first stack |
| 4 | m5d | 0.95023 | 0.94963 | −6.0 | β-clones over-corr |
| 5 | m5h_l1pruned | 0.95043 | 0.94991 | −5.2 | pool-prune fix |
| 6 | m5h2 | 0.95044 | 0.94991 | −5.3 | tied (ρ≥0.9997) |
| 7 | m5j | 0.95044 | 0.94991 | −5.3 | tied |
| 8 | m5p (K=6) | 0.94839 | 0.94754 | −10.6 | over-prune |
| 9 | m5n_3b (K=4) | 0.94808 | 0.94700 | −10.8 | over-prune |
| 10 | **m5q (PRIMARY)** | 0.95057 | 0.95005 | −5.2 | M5h+RealMLP |
| 11 | m5_meta_lgbm_shallow | 0.95048 | 0.95001 | −4.7 | meta-switch |
| 12 | d5_partial_pseudo K=14 | 0.95082 | 0.94963 | **−12.0** | pseudo over-amp |

Stack-saturation regime (rows 5–7, 10–11): gap = **−5.0 ± 0.4bp**
across 5 independent variants — well below the 80/20 split-resolution
floor (~0.5bp). **Not stochastic.**

## 2. Verdict — SYSTEMATIC, NOT COLLAPSE

The −5bp gap is a structural property of LR-meta-on-GBDT-heavy-pool.
Three excursions all have verified mechanism causes:

- **m5d −6.0bp**: β-HGBC clones (ρ≈0.99 with E3) inflated OOF.
  Fixed by L1-prune (m5h restored gap to −5.2bp). [`pool-redundancy-
  gap-widen`]
- **m5p / m5n_3b −10.6/−10.8bp**: pruning to K=4–6 exposed meta to
  single-base error. Pool size = signal redundancy, not just
  diversity. [`minimal-orth-basis-falsified`]
- **d5 partial-pseudo −12.0bp**: 6/14 bases rebuilt on pseudo from
  M5q's confidence; meta L1 reshuffled +67–130% toward pseudo bases,
  −45 to −85% on original LB-correct anchors. OOF rewards
  consistency-with-M5q; LB penalizes it. [d5 slot-1 audit]

The two trigger drifts (rows 11–12) are **mechanism-explained**: row
11 is bounded meta-switch cost (~50% OOF→LB transfer per
`rho-0.995-not-tie-meta-switch-bounded`); row 12 is broad-pseudo over-
amp. **Neither implies the M5q LB ceiling has shifted.**

Rule 14 cleared under: any new candidate with predicted gap <−7bp
(= M5q −1.8bp) requires explicit PI sign-off before slotting (§4).

## 3. Five untried mechanism families with citations (Rule 7)

### F1. Multi-formulation reframe + 2-level stacking
Deotte's April-2025 PS winner: 4 problem formulations at L1 (direct,
ratio, residual, missing-imputed); L2 = GBDT/NN with `confidence`
(std across L1) + `consensus` (mean) as aux features; L3 = weighted
avg.[1][2] Our pool is 1-formulation × 14 bases. Adding (a) hazard-
rate (P(pit | survived to lap k)), (b) residual-from-Compound×TyreLife
rule, (c) sequence-imputed-feature backbone diversifies at the ROOT,
breaking LR-meta rank-lock by changing what the ranks rank. **~1d.**

### F2. TabM ensemble (ICLR 2025)
Parameter-efficient MLP imitating an N-head ensemble in a single
forward pass; #4 median rank across 68 tabular datasets, beats
single-head MLPs / RealMLP on multiple benchmarks.[3] Different
inductive bias from RealMLP-TD. Available in pytabkit. **~6h Kaggle
T4, 1-fold smoke first per Rule 2.**

### F3. TabICL-v2 / TabPFN-v2 foundation-model member
TabICL-v2 handles ~1M rows; in-context Bayesian inference is the
most-orthogonal inductive bias to GBDT-stack we can add.[4] Train
440k / test 188k is in-range. Foundation-model bases now standard in
top-tier PS solutions (Deotte's stack included TabPFN).[1] **~4–8h
Kaggle T4.**

### F4. Sequence model on (Race, Driver, Stint) groups
Frontiers AI 2025: Bi-LSTM at ROC-AUC **0.988** on F1 pit-stop
prediction with 10-step sequences (vs our 0.95 ceiling on i.i.d.
rows).[5] `test_lead_pitstop_computable_pct = 0.974` — 97.4% of
test rows have a same-(Race, Driver) successor in test; we can build
true in-test sequences. Day-2 strategy critique flagged this as the
single largest unmined mechanism class. **1–2d build.**

### F5. Hill-climb / GBDT-meta + auxiliary disagreement features
Replace LR-meta with: (a) per-row `std(L1)` and `mean(L1)` features
+ GBDT meta, OR (b) GPU-cuML hill-climb over the existing 14-base
pool (Deotte L2 trick).[2] LR can only fit a linear function of base
outputs; std-of-bases is a rank-lock-breaker because disagreement is
an LB-relevant signal we currently discard. **30min CPU — cheapest
test of mechanism-class change.**

### Side-quest: regularized pseudo for Path B retry
Tschalzev 2023 replaces confidence-only gating with likelihood-
density gating + sample-weighted training.[6] Directly addresses
d5 over-amp. **1–2h CPU.** Holds until F1/F5 land.

## 4. Re-ranked move list

| # | Move | Cost | EV (bp) | Slot? |
|---|---|---|---:|---|
| A | F5: aux-feature GBDT-meta + std/mean of L1 | 30min CPU | 1–4 | yes if ρ<0.999 |
| B | 2-base [M5q, recursive_HGBC] LB | 5min | 1–3 | yes if ρ<0.999 |
| C | F1: hazard-rate L1 reformulation | 1d | 3–8 | follow-up |
| D | F2: TabM 1-fold smoke → 5-fold | 6h GPU | 2–6 | overnight |
| E | F4: sequence-FE LGBM probe (no NN yet) | 2h CPU | 1–4 | smoke first |
| F | Multi-seed RealMLP bag (HANDOVER A.1) | 6h GPU | 1–3 | overnight |
| G | Tschalzev pseudo retry | 2h CPU | 0–4 | held |
| H | F3: TabICL-v2 base | 8h GPU | 1–5 | held |

Sequence: A (today) → B (Day-7 slot-1) → D + F overnight Kaggle →
E + C during the day. F1 hazard reformulation is the highest-EV
medium-term move and should be the Day-7/8 anchor build.

## 5. New decision rules (codify before next slot)

- **Predicted-gap gate**: candidates with predicted gap <−7bp need
  explicit PI sign-off. Predicted gap = M5q gap + (pool-divergence
  penalty for pseudo-aug, pool-shrink, or meta-switch).
- **Minimal-input-meta sanity check** (irrigation R5): for every
  base-add, train 2-component meta on `M5q + new` only; if 2-comp
  OOF < M5q OOF, K-comp lift was cross-component memorization.
  Reject.
- **Mechanism-class-only**: LR-meta pool-tweaks are dead (3×
  rank-lock). New slots must change meta family, L1 formulation,
  OR add structurally orthogonal model class.

## 6. Verdict

Critic-loop **CLEARED**. Gap drift is systematic and bounded; both
triggers are mechanism-explained, not regime-shift. Resume probing
under §5 rules. Highest-EV next move: F5 tonight, F1 as Day-7
anchor build.

---

[1] Deotte, NVIDIA Developer Blog, "Winning April 2025 PS with cuML
stacking", 2025. https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/
[2] NVIDIA, "Kaggle Grandmasters Playbook: 7 Battle-Tested Tabular
Techniques", 2025. https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
[3] Gorishniy et al., "TabM: Advancing Tabular DL With Parameter-
Efficient Ensembling", ICLR 2025, arxiv 2410.24210.
https://github.com/yandex-research/tabm
[4] PriorLabs/TabPFN + TabICL-v2. https://github.com/PriorLabs/TabPFN
[5] Frontiers AI, "Data-driven pit stop decision support for F1
using deep learning", 2025. https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full
[6] Tschalzev et al., "Revisiting Self-Training with Regularized
Pseudo-Labeling for Tabular Data", arxiv 2302.14013, 2023.
