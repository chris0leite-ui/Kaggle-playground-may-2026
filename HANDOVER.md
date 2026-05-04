# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file with the next
> session's brief. Day-loop step 7 also auto-refreshes it at EOD.

---

## Today's session — Day 3 (2026-05-05+)

**Read order on session start** (skip the default; this file is the
synthesis):

1. `CLAUDE.md` — state block + Rules 1-15 (especially R1, R12, R13, R14)
2. `comp-context.md` — settled-once facts (compute, schema, structural
   findings, GPU workflow)
3. `audit/2026-05-04-day-2-wrap.md` — Day-2 close + ranked H-list
4. `audit/2026-05-04-strategy-critique.md` — what we DON'T know yet

After reading, open with a 3-bullet read-back of state + the first
diagnostic you'll run. Don't start running until PI confirms.

## Where we are

- **Day 2 closed**, 5/5 submissions used. `our_lb_best = 0.94963`
  (M5d 12-base LR-meta stack). Headroom to top-5% (0.95345): 38.2bp.
- **Single-model gap is ~0bp** (E3 HGBC standalone OOF 0.94876, LB
  0.94870). **Stack gap widens with redundancy:** M5 −4.4 → M5b −3.5
  → M5d −6.0bp. The LR meta over-weights correlated bases.
- **Strategy critique (2026-05-04) flagged five gaps:** no per-segment
  failure map, no calibration check, no model-disagreement
  localization, no sequence-FE scout, optimistic headroom math.

## Day-3 sequence (DO IN ORDER)

**Step 0 — four diagnostics (~30 min total)** before any H-list work.
These re-rank H1-H5 and give H1 its missing safety guard:

1. Per-Race OOF AUC table on M5d (5 min) → which 3 races drag the mean?
2. Reliability diagram on M5d OOF (5 min) → if miscalibrated, H1 uses
   isotonic-calibrated probs, not raw.
3. Multi-base agreement matrix across all 12 bases (10 min) → 2-tail
   subset (count ∈ {0,1,2} ∪ {10,11,12}) is the safe pseudo-label pool.
4. Sequence-FE scout (10 min): single-LGBM probe on baseline +
   `laps_since_last_pitstop`, `cumulative_pitstops_this_race`,
   `rolling_target_rate(window=5)` over (Race, Driver) groups. If
   Strat-OOF lifts ≥5bp, add to EXPLORE queue at high priority.

Output → `audit/2026-05-05-d3-diagnostics.md`.

**Step 1+ — H-list, re-ranked by diagnostic findings:**

- **H3** (~10 min) pairwise-correlation gate ρ≥0.97 → refit M5e on
  diversity-pruned pool. **Submit slot 1 — D3 PRIMARY.**
- **H1** (~2h) pseudo-labeling — but use the multi-base agreement
  guard from diagnostic #3, NOT raw M5d confidence. Add a regression
  test: does pseudo-labeling LIFT or REGRESS the *single-base* OOF?
  **Submit slot 2.**
- **2-way TE** (~1h) Driver×Race / Driver×Compound / Race×Lap-bin
  with α=80, inner 5-fold per outer fold. This is the missed Day-1
  research-loop lever (analyticaobscura Source 1 #2). **Submit slot 3.**
- **Kaggle GPU port — RealMLP / EmbMLP** (~2-3h, first GPU
  experiment). Per Rule 13. yekenot's 56-vote public notebook for
  *this exact comp* uses RealMLP. Document the
  notebook-roundtrip pipeline on first use. **Submit slot 4.**
- **R2 hedge** (~30 min) best OOF that regressed ≤30bp on LB.
  **Submit slot 5.**

Backlog if any of the above null:
- H4 HGBC multi-seed bagging (proper variance reduction)
- H5 hill-climb / Ridge meta drop-in
- H2 reformulations: stint-stratified, residual-from-baseline,
  driver-recent-pit-history

## Workflow rules in force (recap)

- **R1** Submissions are single-shot + PI-approved. Never loop.
- **R12** Spend the full 5/5 daily budget. Calibration probes are
  load-bearing data, not just rank.
- **R13** Kaggle GPU IS available — port heavy NN / deep CatBoost
  5-fold / any 5-fold > 1h local-CPU projection to Kaggle.
- **R14** Strategy-critic-loop fires automatically at EOD, on gap
  drift ≥2bp on consecutive submits, before adding a new mechanism
  family, mid-comp, or at plateau (before Research-loop).

## Anti-patterns (from Day-2 process errors)

- **Don't expand a stack pool when OOF→LB gap is widening.** That's
  the meta over-fitting OOF noise. Prune (H3) or swap meta (Ridge /
  hill-climb) instead. Use the diagnostic #3 agreement matrix to
  identify which bases to drop.
- **Don't pseudo-label from the over-fit stacker.** Use multi-base
  agreement, not single-stacker confidence.
- **Don't declare a mechanism "not cost-justified" on local CPU
  alone.** Re-evaluate on Kaggle GPU per R13 first (RealMLP 5-fold
  and ζ deep CatBoost 5-fold both deserve a re-run).

## Open questions for PI before submitting

- Slot 1 (M5e pruned) is high-confidence; OK to push without further
  PI gating beyond R1 single-shot?
- Pseudo-label aggressiveness — what's the test fraction we're
  comfortable using? Default plan: agreement-guarded ~10-15% of test.
- Kaggle GPU notebook scheduling: PI to confirm Kaggle account /
  notebook ownership for the first artifact-roundtrip run.
