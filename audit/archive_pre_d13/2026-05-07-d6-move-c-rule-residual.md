# Day-6 Move C: F1 rule-residual L1 base — mechanism real, magnitude marginal

> Critic-loop §3 F1 reformulation (b): closed-form rule + GBDT
> residual. Goal: produce an L1 base with mistake structure
> orthogonal to every existing PitNextLap-direct base.

## Standalone result

| Quantity | Value | Notes |
|---|---:|---|
| Rule-only OOF (Compound × TyreLife-decile) | 0.74710 | sanity: weak rule alone |
| **Rule + residual GBDT OOF** | **0.94593** | Δ e3 −28.3bp, Δ M5q −46.4bp |
| Per-fold std | 0.00088 | tight; well-converged |
| 5-fold wall | 122s | comfortable |
| **ρ vs M5q test (standalone)** | **0.92887** | **most diverse base since YetiRank (0.666)** |

Standalone is intentionally weak — the residual GBDT is *forced* to
spend capacity on the rule-corrected signal, not the main effect.
This is the design choice that makes the rank ordering structurally
different.

## Pool-add results

| Stack | Strat OOF | Δ M5q | ρ vs M5q test | rule_residual L1 |
|---|---:|---:|---:|---:|
| M5q anchor (K=14) | 0.95057 | 0 | — | — |
| K=15: M5q + rule_residual | **0.95062** | +0.51bp | 0.99971 | **1.485 (TOP)** |
| K=16: M5q + recursive + rule_residual | 0.95063 | +0.56bp | 0.99970 | 1.422 (TOP) |
| 2-comp [M5q, rule_residual] (minimal-meta) | 0.95061 | +0.38bp | — | — |

**Minimal-input-meta sanity check PASSES** (audit §5.2): 2-comp OOF
0.95061 > M5q 0.95057. The K=15 lift is real, NOT cross-component
memorization. **First non-tie meta-level signal in 5 days.**

L1 weights (K=15 meta): rule_residual 1.485 dominates the pool
(2× the next base, e5_optuna_lgbm at 0.722). The meta clearly wants
to use the rule_residual signal. K=16 only adds +0.05bp over K=15 —
recursive and rule_residual capture overlapping residual structure.

## The catch: ρ ≥ 0.9997

| Stack | ρ vs M5q test | Predicted LB | Pred Δ M5q LB |
|---|---:|---:|---:|
| K=15 + rule_residual | 0.99971 | 0.95010 | +0.5bp |
| K=16 + both | 0.99970 | 0.95011 | +0.6bp |

Both at the Kaggle 5-decimal quantization floor (5e-5 = 1 quantum).
50/50 between TIE and +1-quantum lift. **Not slot-worthy on its
own.**

## Strategic read

The rule-residual mechanism is **real but quantum-bounded** at this
scale. Three strong positive signals despite the marginal LB
forecast:

1. **First minimal-meta PASS in 5 days** (audit §5.2 since codified).
   Recursive failed minimal-meta (-0.2bp); rule_residual passes
   (+0.4bp). The mechanism transfers signal through the meta.
2. **Lowest ρ since RealMLP standalone** (0.929 vs 0.972). RealMLP
   gave +14bp LB lift via a 10× OOF→LB amplification on a low-ρ
   base. Rule_residual is even more diverse and showed minimal-meta
   lift — same path.
3. **L1 dominance** (1.485, 2× the next base). The LR meta is not
   rank-locking around rule_residual; it's USING it as the main
   weight while keeping M5q's logit as the rank backbone.

But the K=15 OOF lift is +0.51bp — too small to break Kaggle's
quantization. To convert to a real LB lift, the rule-residual
mechanism needs to be **strengthened**, not just slotted.

## Day-7 strengthening candidates (F1.2–F1.4)

1. **F1.2 — multi-rule ensemble**. Build 3 more rule_residual bases
   with different rule lookups: Compound × Stint, Driver-class
   × Compound, Year × Race. Each ~0.946 standalone, each different
   ρ vs M5q. Pool-add all 4 → K=17 or K=18. Cost: 3× ~120s CPU.
2. **F1.3 — classifier residual** (instead of regressor). Train HGBC
   on `target` with `sample_weight = 1 / max(rule_proba, 1-rule_proba)`
   (downweight rule-confident rows). Different mistake structure
   from regressor-on-residual. Cost: ~120s.
3. **F1.4 — rule_proba as meta-input**. Append `rule_proba`,
   `m5q_minus_rule_proba` as 2 extra meta-features. Different from
   adding rule_residual as a base — gives the meta a calibration
   signal. Cost: 30s.

**Slot decision**: hold rule_residual K=15 as a marginal slot
candidate. If F1.2/F1.3/F1.4 lift the K-stack to OOF ≥ 0.95068
(M5q + 1.1bp), ρ < 0.999, slot the strengthened version.

## Day-6 wrap (all moves)

| Move | Result | Status |
|---|---|---|
| F5 aux-feature GBDT-meta | +0.12bp over no-aux LGBM, −0.78bp vs M5q | **FALSIFIED** |
| B 2-base [M5q, recursive] | tie regime (V1 ρ=0.99996) or OOF regression (V2-V4) | **FALSIFIED** |
| C F1.1 rule-residual | minimal-meta PASS, K=15 +0.51bp, ρ=0.99971 | **REAL but marginal** |
| F multi-seed RealMLP bag | kernel `realmlp-bag-gpu` v1 pushed | **running on Kaggle** |

Three meta-level falsifications + one real-but-quantum-bounded
mechanism + one parallel kernel running. Day 6 has converged on
**base-pool diversification (F1.x) + variance reduction (Multi-seed
RealMLP)** as the only EV-positive directions.

## Held submissions (do not push)

- `submission_d6_aux_meta_with_aux.csv` — F5 falsified
- `submission_d6_2base_v[1-4]_*.csv` — Move B falsified
- `submission_d6_k15_rule_residual.csv` — marginal candidate
- `submission_d6_k16_two_diverse.csv` — marginal candidate

## Pointers

- `scripts/d6_rule_residual.py` — F1.1 build
- `scripts/d6_k16_two_diverse.py` — K=16 follow-up
- `scripts/artifacts/d6_rule_residual_results.json`
- `scripts/artifacts/d6_k16_two_diverse_results.json`
- `kernels/realmlp-bag-gpu/` — Move F kernel (running on Kaggle)
