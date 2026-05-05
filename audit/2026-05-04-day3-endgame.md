# Day-3 endgame — synthesis + Day-4 plan (2026-05-04 23:00 UTC)

## Day-3 final scoreboard

10/10 submissions used. LB results:

| # | Submission | K | Strat OOF | LB | Gap | Notes |
|---:|---|---:|---:|---:|---:|---|
| 6 | M5h | 13 | 0.95043 | **0.94991** | 52bp | Day-3 PRIMARY |
| 7 | M5h2 (drop a_horizon) | 12 | 0.95044 | 0.94991 | 53bp | tied — pool-size hypothesis null |
| 8 | M5j (d3a/d2a swap) | 13 | 0.95044 | 0.94991 | 53bp | tied — TE-key swap LB-neutral |
| 9 | **M5p** (orth K=6) | 6 | 0.94839 | **0.94754** | 85bp | **−237bp REGRESSED** |
| 10 | **M5n_3b** (min-orth K=4) | 4 | 0.94808 | **0.94700** | 108bp | **−291bp REGRESSED** |

## Load-bearing finding: GBDT consensus IS the LB rank

The 10 "redundant" GBDT bases in M5h are NOT decorative. The minimal-
basis stacks (M5n_3b, M5p) regressed on LB by **2-3× the OOF drop**.
OOF→LB gap *widened* from 52bp (M5h) → 85bp (M5p) → 108bp (M5n_3b).

Interpretation:
1. The 13-base GBDT-heavy pool encodes a *consensus rank* via averaging.
2. That consensus rank is what transfers to LB (the test set follows
   the same DGP as train; the consensus generalizes).
3. Removing consensus members exposes the rank to whichever model's
   idiosyncratic errors dominate the smaller pool.
4. Rank-breaking submissions (ρ < 0.999 vs M5h) move LB — but the
   direction is downward unless the new structure is *correctly*
   learning new signal. Today's tests showed it doesn't.

**Hypothesis update**: in-pool tweaks of the GBDT-heavy LR meta are
LB-LOCKED at 0.94991 within Kaggle's 5-decimal quantization. Real lift
requires ADDING strong orthogonal-mechanism bases to the FULL M5h
pool — NOT replacing the GBDT consensus.

## RealMLP completed late Day-3

5-fold StratKFold on Kaggle T4 (after 2 P100 sm_60 failures, fixed
with torch 2.4 force-reinstall). Total wall: 175 min.

  Strat OOF: 0.94582  (Δ baseline +50.7bp; std 0.00084)
  Per-fold:  0.9473, 0.9452, 0.9459, 0.9448, 0.9459

**Diversity scorecard (built tonight; persist for Day-4 selection):**

| Base | Strat OOF | ρ vs M5h | |Δ|@Stint2 | Diversity score |
|---|---:|---:|---:|---:|
| LR-FE | 0.89684 | 0.86916 | 0.1540 | 0.2848 (HIGHEST) |
| EBM | 0.93361 | 0.93081 | 0.0664 | 0.1356 |
| H1 pseudo-LGBM | 0.94265 | 0.96539 | 0.0566 | 0.0912 |
| **RealMLP** | **0.94582** | **0.97192** | 0.0368 | 0.0649 (LOWEST) |

**RealMLP is high-quality but low-diversity** — NN-on-tabular converges
to GBDT-like behavior on this DGP. Strong standalone (+51bp baseline)
but predictions are similar to M5h's consensus.

## RealMLP-augmented stack candidates (pre-built for Day-4)

| Variant | K | Strat OOF | Δ M5h | ρ vs M5h | RealMLP L1 | Gate |
|---|---:|---:|---:|---:|---:|---|
| **M5q** (M5h + RealMLP) | 14 | 0.95057 | +1.4 | 0.99865 | 0.573 | **PASS (just)** |
| M5r (M5h2 + RealMLP) | 13 | 0.95056 | +1.3 | 0.99901 | 0.326 | TIE_EXPECTED |
| M5s (M5n_3b + RealMLP) | 5 | 0.94854 | -18.9 | 0.98740 | 2.399 | PASS |

**M5q is the highest-EV slot-1-Day-4 candidate**. ρ=0.99865 is JUST
under the 0.999 tie threshold. RealMLP gets L1=0.573 in the meta
(6th-highest of 14 bases) — meaningful weight, not consensus-clone.
Expected LB: 0.94991 ± 2bp. Probably ties; small chance of +1-2bp lift.

M5s would test "minimal-basis with strong NN" but per Day-3's M5p
result, the minimal-basis approach is now FALSIFIED for this dataset.
Skip M5s on LB; treat as confirmation if the slot is otherwise unused.

## Day-3 process retrospective

### Wins
- **Pool disagreement diagnostic** identified that the "13-base pool" is
  effectively 3-source-diverse + 10 GBDT-clones. Reframed the entire
  search.
- **Pre-submit diff helper** added; would have saved slot 7-8 if
  caught earlier (M5h2/M5j ρ ≥ 0.99999 → tie was preventable).
- **RealMLP successfully ported to Kaggle T4** despite torch sm_60
  P100 incompat — friction logged, sandbox is reproducible.
- **EBM + H1 + LR-FE all ran to completion overnight** — full
  artifact set ready for Day-4 stacking.

### Losses
- **Rule 2 violation**: skipped 1-fold smoke probe before launching
  RealMLP full 5-fold; cost ~3h GPU compute on a kernel that took 3h
  but might have been killed early if it had been slow.
- **Wasted slots 7-8** on tied M5h2/M5j (calibration probes that
  tied at LB 0.94991 within Kaggle's quantization). Pre-submit-diff
  helper should prevent this Day-4+.
- **Day-3 evening hypothesis (minimal-orth basis is the lever)
  FALSIFIED** by slots 9-10. Cost 2 slots, but we learned the GBDT
  consensus IS load-bearing — high-info negative result.

## Day-4 plan (slots reset at 00:00 UTC, ~60 min away)

### Slot 1 (immediate): **M5q** (M5h + RealMLP, K=14)
- Most-likely-to-lift candidate.
- Expected LB: 0.94991 ± 2bp.
- If LB > 0.94991 → RealMLP earns its slot; build M5q-derivatives.
- If LB = 0.94991 → tie despite ρ < 0.999; the rank-shift wasn't
  enough.
- If LB < 0.94991 → RealMLP somehow hurts (very unlikely with
  L1=0.573).

### Slot 2: **M5q + H1 + EBM (K=16)** if M5q ties or lifts
- Stacks ALL strong-orthogonal bases on M5h. Test maximum diversity
  with consensus preserved.

### Slot 3: **HGBC multi-seed bag** (build new)
- Echo the cb_slow-wide-bag pattern that worked on Day-2.
- E3, f1, f2 are single-seed. Bag 3 seeds; new pool member.
- ~30 min CPU.

### Slot 4: **Pseudo-label H1 RUN-2 with RealMLP-anchored agreement**
- Original H1 used 13-GBDT pool agreement. RealMLP is now available.
- Re-run H1 with the 14-base pool (M5h + RealMLP) for the agreement
  guard. Should give different pseudo-labels (RealMLP may flag
  different rows as confident).
- Retrain a base on the new augmented train.
- ~30 min CPU.

### Slot 5: **Sequence-aware base (LSTM with Driver embedding)**
- Major mechanism-family lever still untested. 887 Driver × 16-dim
  embedding + LapNumber + per-lap features over (Race, Driver) seq.
- Needs Kaggle GPU (per Rule 13).
- 1-fold SMOKE FIRST (per Rule 2 — applying tonight's lesson).
- ~3-4h roundtrip.

### Slots 6-10 (depending on Day-4 morning state)
- Day-3 best LB stays as 0.94991 PRIMARY.
- Hedge candidates: anything that comes within 30bp on public LB.
- R5 final-window probe: best OOF that regressed on public ≤30bp.
- Final lock-in: best of {Day-1 to Day-4 by LB}.

## Pool weaknesses (still unaddressed)

| W# | Weakness | Status | Day-4 plan |
|---|---|---|---|
| W1 | Mechanism family monoculture (13/13 are GBDT) | Partially addressed (RealMLP added) | M5q test; LSTM as next |
| W2 | Same cat handling across bases | Addressed (RealMLP uses embeddings) | RealMLP L1 in M5q tells us if it helped |
| W3 | No sequence-aware base | UNADDRESSED | Slot 5 LSTM |
| W4 | Stint 2 blind spot | UNADDRESSED | RealMLP doesn't help (|Δ|@S2 = 0.0368) |
| W5 | LR meta over-emphasizes consensus | M5q tests this | If M5q lifts, partially addressed |
| W6 | Probability calibration variance ignored | Untested | Day-5+ |

## Hypotheses for Day-4+

### NH8. RealMLP brings GENERAL LB lift, not Stint-2 fix
RealMLP's |Δ|@Stint2 is only 0.0368 — it agrees with the GBDT
consensus on the blind-spot segment. Adding RealMLP shouldn't help
Stint 2 specifically. The lift (if any) will come from generic
prediction-quality improvement, not blind-spot fixing.

### NH9. Stint 2 needs feature engineering, not new model
The pool is uniformly wrong on Stint 2. Adding more models doesn't
help. Build features that differentiate within Stint 2:
  - lap_since_last_pit × tyre_compound (interaction)
  - relative_pace = (LapTime - race_min_LapTime) / race_std_LapTime
  - traffic_density = count of cars within Y position-units (proxy)
  - within-Stint-2 TE (target rate within current Stint conditional
    on lap-into-stint).
A targeted feature set on Stint 2 may give 1 base a real edge there.

### NH10. Distillation: train a small model to MIMIC M5h
Take M5h's OOF predictions as soft labels. Train a single LGBM (or
RealMLP) on (X, M5h_oof) regression. The distilled model has the
M5h signal but is a single forward pass. It won't EXCEED M5h's
quality, but its predictions may be SMOOTHER and reduce overfit.
Add as a stack base.

### NH11. Out-of-fold-target-as-feature
For each train row, the M5h OOF probability is the "model's read
on this row". For test rows, use the M5h test prediction. Add
**M5h_oof_probability** as a feature and retrain a base. The base
can learn "where M5h is wrong" via interactions with raw features.
Risky (recursive, possibly leaky), but a known Kaggle technique.

### NH12. Year-Stint cross-OOF
Train 1 base per (Year, Stint) pair (4 × 8 = 32 mini-models). At
inference, pick the matching mini-model. Massive overfitting risk;
need careful CV. But could fix the Stint-2 blind spot since each
mini-model is dedicated to its segment.

## Critical reminders

1. **Pre-submit-diff before EVERY submit** going forward
   (`scripts/pre_submit_diff.py`).
2. **1-fold smoke before any GPU 5-fold** going forward
   (`tag: rule2-smoke-skip-realmlp-day3` to be added if reocurs).
3. **Strat-only Day-3+** (R1).
4. **Spend full 10/day budget** (R12); don't hold slots out of
   misplaced caution.
5. **HANDOVER.md update needed** before next session.

## Pointers

- `audit/2026-05-04-day3-learnings.md` — daytime learnings (pool
  weaknesses, NH1-NH7).
- `scripts/m5op_orthogonal_stacks.py` — M5o/M5p builders.
- `scripts/m5n_minimal_basis.py` — M5n builder.
- `scripts/m5qrs_realmlp_stacks.py` — M5q/M5r/M5s builders.
- `scripts/diag_pool_disagreement.py` — pool diversity diagnostic.
- `scripts/diag_new_base_diversity.py` — new-base diversity scorecard.
- `scripts/pre_submit_diff.py` — MANDATORY pre-submit gate.
- `scripts/artifacts/*` — full set of OOF/test arrays for all bases
  and stacks.
- `submissions/submission_m5q_realmlp.csv` — Day-4 slot-1 candidate.
