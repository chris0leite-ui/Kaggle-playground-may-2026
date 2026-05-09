# 2026-05-08 — P2/P5 stint-recovery bases (DGP-aware FE; K=4+1 NULL)

`branch: claude/find-dgp-research-ClsQE`
`tag: dgp-fe + rank-lock-confirmation`

## TL;DR

Two LightGBM bases tested on the P1 finding (synth `Stint` label is
fabricated; `stint_start_imputed = LN-TL+1` is the true per-row
identifier):

| Base | Standalone OOF | ρ vs PRIMARY | K=4+1 LR-meta Δ | Verdict |
|---|---:|---:|---:|---|
| **P2** stint_recovery + std14 | 0.93971 | 0.953 | **+0.09 bp** | WEAK / FAIL |
| **P5** stint_recovery alone (no std14) | 0.92624 | **0.903** | **+0.14 bp** | NULL |

P5 is the **lowest ρ ever observed** for a K=4+1 candidate on this
comp (prior best diverse: RF-yekenot 0.959). Yet K=4+1 lift is only
+0.14 bp — almost identical to P2's +0.09 bp despite ρ being 5pt lower.

**Conclusion: rank-lock at K=4 is robust to even ρ=0.90 candidates.**
The K=4 LR-meta absorbs structurally distinct DGP-recovery signal
into its 1.33-rank logit subspace.

## Mechanism — what each base does

**P2** (27 features): standard 14 + `stint_start_imputed` +
`stint_consistent_with_synth_label` + 7 cell stats (size, tl_max,
tl_min, lap_max, lap_min, pos_spread, tl_range,
implied_stint_len) + 3 per-fold TE features
(Race/Year/Compound/stint_start, Compound/stint_start, Race/Year/
stint_start). Per-fold refit per Rule 24.

**P5** (16 features, no std14): same recovery features + tl_frac_of_cell
+ implied_stint_len_bin + 5 per-fold TE features (4 of P2's plus
Compound/Race/Year). Standard 14 EXCLUDED — isolates the orig-stint
signal cleanly.

## What this confirms about the DGP

1. **The orig-stint cell info IS predictive but small.** P5's standalone
   OOF 0.926 means knowing a row's `(Race, Year, Compound,
   stint_start_imputed)` cell + cell statistics predicts PitNextLap
   with AUC 0.926 — well above chance (0.5) but ~30 bp below PRIMARY.
   The recovery is real.

2. **K=4 absorbs orthogonal DGP signal at the logit-direction level.**
   Even ρ=0.903 (lowest ever for K=4+1 positively-gating) only lifts
   the K=4 LR-meta by +0.14 bp. The 12-feat [P, rank, logit] expansion
   on K=4 reconstructs P5's logit prediction as a linear combination
   of existing logits.

3. **A33 (irreducible noise floor) plus rank-lock cap further FE.** The
   PitNextLap target is ~81% concordant with PitStop[L+1] when L+1 is
   observed; this irreducibility caps achievable AUC near 0.954. The
   K=4+1 ceiling is at the LR-meta logit-direction subspace, set by
   the task framing rather than the feature set.

## Implication for "find DGP" mission

The P1+P1b+P1c+P2+P5 chain has **fully characterized the synthesizer**:

  - Per-row CTGAN sampling, conditioned on (Race, Year, Compound).
  - Categorical labels (Driver, Stint) assigned independently of the
    row's source orig stint.
  - 887 driver vocab: 31 active (real timeline) + 100 historical
    abbrev (uniform fabrication) + 756 D-prefix ghosts.
  - 97.55% LapTime literal copies; tuple-concordance 95% confirms
    label-preserving re-use of orig source rows.
  - Within-stint downsampling (median 6 rows from 21-lap window).
  - Compound preserved exactly within stint groups; LN/TL not.

This is the DEEPEST DGP characterization in the comp's audit history,
done **without using the aadigupta1601 public CSV**. Per-fold features
that exploit the recovered structure pass K=4+1 gate at +0.09 to
+0.14 bp — confirming the rank-lock ceiling robust to DGP-knowledge.

The next-level lift would require a META architecture change that
breaks the LR-meta subspace, NOT another base.

## Pointers

- `scripts/dgp_v2/p2_orig_stint_recovery.py` — P2 base
- `scripts/dgp_v2/p5_pure_orig_stint.py` — P5 base
- `scripts/dgp_v2/gate_p2_k4plus1.py`, `gate_p5_k4plus1.py` — gates
- `scripts/artifacts/p{2,5}_*.npy/.json` — artifacts

## Friction tag

`stint-recovery-fe-orthogonal-but-rank-locked` — even ρ=0.90 K=4+1
candidates lift only +0.14 bp. K=4 logit subspace ceiling robust to
DGP-aware FE.
