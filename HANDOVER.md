# HANDOVER

> Next-session prompt for playground-series-s6e5. PI says **"handover"**
> → agent reads this file and proceeds per its instructions. PI says
> **"prepare handover"** → agent rewrites this file.

---

## Today's session — Day 14 (2026-05-14)

**Read order on session start** (skip the default; this is the synthesis):

1. `CLAUDE.md` — state block + Rules 1-16
2. `audit/2026-05-13-d13-path-b-hier-meta.md` — Path B mechanism (load-bearing)
3. `audit/2026-05-13-d13d-path-b-gkf-probe.md` — GKF amplification confirms private-robust
4. `audit/2026-05-12-d12-master-synthesis.md` — Day-12 leakage-robust thesis
5. `scripts/pre_submit_diff.py` — MANDATORY before submit

Open with a 3-bullet read-back of state + first action.

## Where we are (Day 14 evening)

- **PRIMARY** = `d13e_compound_stint_tau20000` LB **0.95049** (+8bp Day-13 PM).
- **HEDGE** = `d13_path_b_stint_tau100000` LB 0.95041 (R5 candidate).
- **Gap to top-5%** (0.95345): **29.6bp**. 13 days remaining.
- **Submits used**: 24/270 total (Day-13 used 6/9).

## Day-14 session results (this session)

### Move A — TabPFN fine-tune: DEAD

Both v2.5 and v2.6 exhausted:

- **v2.5 @ 150k rows** (kernel v10): fold-0 AUC **0.94446** — identical to 50k-row result
  (0.94439). Training loss was flat from epoch 1; fine-tuning not learning competition
  signal. Wall 6829s (vs 5134s at 50k — more time, zero gain).
- **v2.6 @ 150k rows** (kernel v2): OOM at epoch 1 (model weights ≈15.37GB alone, P100
  has 16GB). Same OOM at 351k rows. v2.6 too large for P100 regardless of row count.
- **Verdict**: TabPFN fine-tuning ceiling is ~0.944 at this competition (-64bp vs PRIMARY).
  ρ=0.960 (diverse) but gap too large for pool contribution at current PRIMARY level.
  **Dead-list both v2.5 and v2.6 fine-tuning.**

### Move D — FM new inputs (F1-F4): DEAD

`scripts/d13_move_f_features.py` built 4 new fields (F1_PitWindow_q5, F2_HazardDecay_q5,
F3_CompoundPress_q5, F4_RaceStage) — all strong standalone signals (F2/F3 monotone
Δrate=0.22-0.25, F4 mid_b=0.38 vs opening=0.06).

`scripts/d13_move_f_fm_aug16.py` 16-field FM (12 d9h + 4 Move-F):
- Standalone OOF: 0.92741 (**+20.1bp** vs d9h_aug12 0.92540)
- ρ vs PRIMARY: **0.919** (genuinely diverse)
- Min-meta: **-0.07bp FAIL**

Confirms "FM-field-augmentation saturated at 12 fields" thesis (d14 H1 aug13 was -0.13bp;
aug16 is -0.07bp). New input types provide standalone lift but zero pool-level increment.

## Remaining live moves (Day 15)

### Move B — Pseudo-label cascade at K=21+hier-meta level (~3-4h CPU)
EV +5-10bp. Use d13e PRIMARY preds as pseudo-labels, confidence-filter top 30%,
retrain 5 fastest bases on (train+pseudo), re-stack K=21 with hier-meta.
Risk: d5 partial-pseudo widened the gap on m5q (−4.2bp LB vs +2.5bp OOF).

Mitigation: use d13e (stronger PRIMARY) + hier-meta layer. Start conservative:
pseudo at top-confidence threshold only.

### Move C — DeepFM-lite (~3-4h CPU)
FM pairwise + 2-layer MLP head. New model class beyond d9f/d9h.
EV +3-8bp standalone, +1-3bp stacked. Risk: overfit (d9e FFM precedent).
Precedent: add dropout, batch-norm, limit depth to 2.

### Research loop trigger (Rule 7)
**If Day-15 yields no ≥+5bp structural move, fire the research loop:**
- Pause submits
- Web-search top-5 finishers' writeups from comparable playground tabular comps
- Identify untried mechanism families
- 40bp gap likely requires structural insight (target reformulation / unique FE /
  external data) not yet found

### Move E — submit held Compound×Stint variants (only if B/C/research miss)
`d13e_compound_stint_tau{100000}` OOF +0.82bp, ρ=0.9996 vs τ=20k (TIE band).

## Falsified / dead — do NOT retry

All prior dead-list entries from previous HANDOVER remain. Additional dead:

- **TabPFN v2.5 fine-tune** — AUC ceiling 0.9444 at any row count; flat training loss
- **TabPFN v2.6 fine-tune** — OOM on P100 (model weights >16GB); no fix possible
- **FM-field-augmentation** (Move D / d14 H1 aug13 / aug16): saturated at 12 fields;
  new input types add standalone OOF but -0.07 to -0.13bp min-meta NULL
- All prior Day-13 and Day-14 dead entries (see prior HANDOVER section for full list)

## Critical operating rules (unchanged)

1. **Pre-submit-diff before EVERY submit**, ρ < 0.999.
2. **NEW Day-13: ρ/G3/R7 heuristics DO NOT apply to new mechanism families.**
3. **Rule 16 Q6** (ρ vs FULL pool, not just PRIMARY): binding gate.
4. **GroupKF as secondary gate** (Day-12 finding).
5. **Strat-only Day-3+ (R1)** for primary OOF.
6. **Submit budget** 24/270; 13 days × 9 = ~115 remaining. **40bp gap means structural
   moves only**; tuning probes deferred to R5.
7. **Model-class diversification > tuning**.

## Pointers

- `audit/2026-05-13-d13-{path-b-hier-meta,d13d-path-b-gkf-probe}.md` — load-bearing
- `scripts/d13_move_f_features.py` + `scripts/d13_move_f_fm_aug16.py` — Move D result
- `scripts/artifacts/d13_move_f_fm_aug16_results.json` — Move D gate result
- `scripts/artifacts/d12_tabpfn_finetune_150k_results.json` — TabPFN 150k dead verdict
- `kernels/d12-tabpfn-finetune-gpu/` (v2.5) + `kernels/d13-tabpfn-v26-strat/` (v2.6) — archived
