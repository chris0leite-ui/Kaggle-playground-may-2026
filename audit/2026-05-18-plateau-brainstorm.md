# 2026-05-18 — Plateau-break brainstorm (larger step size)

Triggered by: PI request after 2026-05-18 Tier-A null. The skill
(`problem-solving.md`) prescribes **re-enter step 1** at plateau —
restate the problem from scratch rather than iterate on the
existing solution. The skill (`personas.md`) prescribes **persona
rotation** when the agent claims structural ceiling.

Did both. Two fresh personas (10 Wild Options + Junior ML Engineer,
both Opus, no prior-conversation memory). Quick EDA verifying
junior's leakage flags. Synthesis below.

## Step-1 re-entry: what's the actual problem?

The team has been solving: **"maximise OOF AUC on the K=11 + LR-
meta + Path-B stack."** That's a step-5 (Analyse) framing, not a
step-1 (Define) framing.

Restated step-1: **"on a CTGAN-class synthetic dataset where the
generator has decoupled PitStop[L+1] from PitNextLap[L] by ~19%,
what is the maximum-AUC classifier — and what evidence would tell
us we've reached it?"** Three implications:

1. The 19% decoupling is informative about the synth — but we have
   not characterised it formally. Is it i.i.d. noise on the target?
   A latent state (race aggressiveness, driver style) that explains
   the decoupling? Conditional on what?
2. Our LR-meta is trained on log-loss; the metric is AUC. We've
   never tested an **AUC-direct loss at the meta layer**.
3. We are optimising public LB; the prize is private LB. The two-
   final-submission rule means our final-window posture should be
   variance reduction across folds, not AUC chasing on public-LB.

## EDA on junior agent's leakage flags

Junior agent flagged 5 things; I verified the load-bearing ones:

| Flag | Verdict |
|---|---|
| `P(PitNextLap=1 \| PitStop=1) = 0.2478` (vs base 0.199) | Synth has INVERTED real-F1 sign (real F1: <0.05). Self-leak feature, but already a feature in pool. |
| `tyrelife_reset_next` as feature | 90.7% observable, +9pp lift but only 0.169 → 0.259 conditional rate. Synth introduces noise in TyreLife trajectory too. |
| Strict L+1 lookup in test rows | 41.7% have L+1 observable (12.4% in test + 29.3% in train). Team's prior measurement of 12.4% missed the cross-frame 29.3%. |
| PitStop_next as direct predictor | Standalone AUC 0.5363 on 41.5% coverage. SYNTH HAS BROKEN THE LAP-ADJACENCY — "observable lead-feature is DEAD" verdict confirmed even on the wider coverage set. |
| 96.3% (D,R,Y) groups overlap train↔test | Confirmed. Combined-frame lead/lag was already tested (-0.36 bp at GBDT base level). |

**Net of EDA: no hidden goldmine.** The synth-broken-pit-adjacency
is genuine, not an artefact of how the team measured. Confirms
2026-05-14 "Bayes-optimal ceiling on row features."

## Filter applied to persona output

Both personas returned strong ideas. Filtering against
mechanism-ledger / hypothesis-board / `kernels/`:

| Idea (source) | Filter | Verdict |
|---|---|---|
| **Cox PH hazard regression** (#3 wild) | hazard-nn killed by Rule 24 leakage (kernels/hazard-nn-bag-gpu); leak-free version -206 bp. | Likely-dead-on-arrival. Skip. |
| **TyreLife/stint_length ratio** (junior #1) | Subsumed by Heilmeier residual (EXP-A3-4) pending in menu. | Re-cast as A3-4, not new. |
| **PitStop_lag1 lag features** (junior #2) | Inter-stint FE (EXP-3) NULL at K=10+1. | Closed, but the junior agent's exact construction (within-(D,R,Y) lag1) is slightly different and untried-as-feature-add. Worth a 10-min retry. |
| **RaceProgress non-linearity TE** (junior #3) | Subsumed by existing Path-B C×S; per-segment shrinkage already captures non-monotonic Race x LapNumber surface. | Closed. |
| **LapTime rolling spike** (junior #4) | Subsumed by m4 relative-state FE. | Closed. |
| **Expected_remaining stint life** (junior #5) | EXP-A2-5 Heilmeier `remaining_pit_stops_proxy` pending in menu. | Re-cast, not new. |

## Genuinely untried larger-step ideas (ranked by EV / novelty)

### Tier S — never-tested mechanisms

**S1. Reciprocal-rank-fusion blend (RRF) over K=11 bases.** Pure
post-process. For each row, `score = Σ 1/(60 + rank_i)` over the
11 bases (parameter k=60 per Anserini default). Different from
team's blend harness (weighted-average). Different from rank-mean
(killed; -32 bp) because RRF is bounded and emphasises top-rank
agreement. **Cost: ~5 min. Predicted: 0 to +0.3 bp.** Risk: low
(pure post-process). Confidence: moderate (RRF is widely cited in
search-engine blending and Kaggle multi-model fusion).

**S2. TabM (multiplicative tabular network).** The team has a
`kernels/tabm-smoke-v3-gpu/tabm_smoke_v3_gpu.py` scaffold but no
artifact files; the formal run was never completed. TabM beats
TabPFN and FT-Transformer on multiple tabular benchmarks (Yandex
2024). **Cost: ~30-60 min Kaggle T4. Predicted: +0.3 to +0.8 bp
standalone (NN-class diversity).** Risk: low (existing kernel
scaffold means setup is done; just needs a full-data run).

**S3. TTA via Gaussian feature jittering on K=11 stack** (10 wild
#10). Per-feature σ = 0.02 × train-std, 16 forward passes per test
row, average logits. Pure post-process at INFERENCE only — the
training is unchanged. **Cost: ~15 min CPU. Predicted: +0.1 to
+0.4 bp** (TTA on tabular is rarely transformative but cheap to
verify). Risk: low.

**S4. AUC-direct meta loss (pairwise rank or `xendcg`).** Replace
LR-meta (log-loss) with a meta optimised directly on AUC. Use
LightGBM's `objective=rank_xendcg` with group=all-rows (which
reduces to cross-entropy with NDCG approximation). Or pairwise
Wilcoxon-Mann-Whitney loss via sklearn `SGDClassifier(loss='hinge')`
with sample pairs. **Cost: ~20 min CPU. Predicted: 0 to +0.3 bp** —
small because the LR-meta is already near optimal under the
3-D logit subspace constraint, but the LOSS-AUC gap is non-zero.

**S5. Conformal-prediction-set width as 11 meta-features** (10
wild #1). For each base, fit split-conformal calibration on a
hold-out fold; the 90%-coverage interval WIDTH per row is an
uncertainty feature. Feed the 11 widths INTO the LR-meta. Different
from base-disagreement std (which was tested as adaptive blend
input, null) because conformal width is per-base-specific, not
cross-base. **Cost: ~25 min CPU. Predicted: 0 to +0.3 bp.**

### Tier A — speculative reframes

**A1. Per-Driver random-slope GLMM** (10 wild #5). Mixed-effects
regression `logit(P) = X·β + (1 + RaceProgress | Driver)` via
statsmodels MixedLM. BLUP shrinkage on the random slope is a
regularised driver-target-encoder with a different functional
form than fixed-effects TE. The team killed per-Driver SPECIALISTS
(d12); GLMM is structurally different (single model, soft
per-driver coefficients). **Cost: ~30 min CPU (MixedLM is slow).
Predicted: 0 to +0.4 bp.** Risk: moderate.

**A2. Hand-coded stint-cap multiplier as PRIMARY post-process**
(10 wild #2). If `Compound==SOFT & TyreLife≥22`, multiply final P
by 1.20 (clipped). Same for MEDIUM≥32, HARD≥42. Test 3-5 cap pairs
on OOF; submit the best as a HEDGE-eligible candidate. **Cost: ~5
min.** Pure heuristic; cannot regress at OOF (calibration ladder
preserves rank), but might LB-tie or marginally lift. Risk: low.

**A3. TabDDPM row imputation + 22nd base** (10 wild #4). Train a
tabular diffusion model on train; generate 200k synthetic rows;
train a 22nd LightGBM base on augmented set; blend at rank 0.05.
Distinct from the existing CTGAN-replay characterisation (d18
forensics) because TabDDPM is generative not discriminative.
**Cost: ~2-3 hr Kaggle T4. Predicted: +0.3 to +0.8 bp** if the
diffusion captures patterns CTGAN smooths over.

**A4. Per-row LightGBM-meta on K=4 with sample weight =
log(1 + base-disagreement-std)**. Tested as ADAPTIVE BLEND (null)
but never as a SAMPLE WEIGHT on a fresh GBDT meta. The hard-row
hypothesis: the GBDT meta might fit the hard rows correctly if
they're weighted up at training. **Cost: ~20 min CPU. Predicted:
0 to +0.3 bp.** Risk: moderate (LGBM-meta on K=4 already regressed
-1.30 bp without weighting; the weighting might or might not
rescue).

### Tier B — heavier / data-side

**B1. Active selection on test predictions with public-LB feedback.**
228 submission slots remain. Submit 5-10 "diagnostic" submissions
in the next 3 days that perturb the PRIMARY in known directions
(e.g., shift +ε to Compound=SOFT rows, etc.); use the public-LB
delta to identify WHICH ROW GROUPS the PRIMARY is wrong on.
**Cost: 5-10 submission slots over 3 days.** Strategic — uses the
LB itself as a measurement instrument. Risk: high (consumes the
slot budget we'd otherwise save for the final-window hedge).

**B2. GraphSAGE on Driver-Race-Compound tripartite graph** (10
wild #9). 2-layer GraphSAGE, 32-d node embeddings. Each lap is a
node; edges are (Driver-Race, Driver-Compound, Race-Compound).
Embeddings as features for a 22nd CatBoost base. **Cost: ~2-3 hr
Kaggle T4. Predicted: +0.2 to +0.6 bp.** Truly different mechanism
class than anything in mechanism-ledger.

### Tier C — final-window reserves

**C1. (carry-over from research) OpenF1 per-Race scalar join.**
~45 min. Not yet tried with 26-Race-level join key.

**C2. (carry-over from research) Swap-noise DAE on combined
train+test.** Porto Seguro 1st-place mechanism. ~2-3 hr Kaggle T4.

**EXP-9 (carry-over) gap-aware sequence transformer.** ~4-6 hr.

## What the skill says about all this

`problem-solving.md`: re-entered step 1 (this artifact).
`personas.md`: rotated 2 personas (this artifact).
`guardrails.md` Rule 4: "never declare structural ceiling without
a fresh Research loop." Did that 2026-05-18 morning.
`guardrails.md` Rule 23: ≥1 slot per 3-day cycle for free-form FE.
The next 3 days should include 1 slot for hand-coded mechanisms
(A2 stint-cap multiplier or similar).

## Recommended execution order

If PI authorises, in cost order (cheapest first; each gate at
K=4+1 or PRIMARY-direct):

1. **S1 Reciprocal-rank-fusion** (~5 min, pure post-process).
2. **A2 Hand-coded stint-cap multiplier** (~5 min, hand-coded).
3. **S3 TTA noise jittering** (~15 min, pure post-process).
4. **S4 AUC-direct meta loss** (~20 min CPU).
5. **S5 Conformal-prediction-set width as meta-feature** (~25 min).
6. **S2 TabM full 5-fold** (~30-60 min Kaggle T4).
7. **C1 OpenF1 per-Race join** (~45 min, carry-over).
8. **A1 Per-Driver random-slope GLMM** (~30 min CPU).

Total ~3-4 hr CPU + 1 GPU slot for the first 8 picks. Submission
slots needed: 1-3 across the batch (only the picks that clear gate).

## Confidence note

The K=4 LR-meta 3-D logit subspace absorbed today's Tier-A picks
(both gate WEAK at +0.3 bp). The same absorption likely applies
to most of the Tier-S picks above — they probably also gate WEAK.
**The single most likely lift is the loss-class reframe** (S4 AUC
direct + future C2 DAE swap-noise) because those introduce a NEW
LOSS-FUNCTION class, not a NEW FEATURE/BASE within the existing
log-loss family. The team has not yet exhausted loss-function
diversity at the meta layer.
