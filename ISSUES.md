# ISSUES.md — live problem decomposition

> BPS step-2 (disaggregate) + step-3 (prioritise) output. Refreshed
> when the strategy-critic-loop fires (plateau, saturation, kickoff,
> 50% checkpoint). Cap ≤150 lines.

**Problem (BPS step 1).** Close the −29.6bp gap to top-5% (leader
LB 0.95345 vs ours 0.95049) in 13 remaining days, given that
FM-class augmentation, Path-B hier-meta, and the GBDT pool are
all saturated within current configurations (Day-13/14 alt-axis
4-of-4 NULL; d14 Path B cohort sweep on Year/Year×Stint/Race
also NULL).

**Claim convention.** Pick an unclaimed `open` leaf. Edit its
`[owner: ...]` field to your branch slug (part after `claude/`).
Commit + push. One open leaf per branch at a time. Update status
at wrap-up via `WRAPUP.md` section A.

Status values: `open`, `wip`, `done`, `null` (falsified), `parked`.

---

## 1. New model class (existing classes saturated within pool)

- **1a.** TabPFN-2.6 fine-tuned (Kaggle GPU). v2.6 OOM fix landed
  (cap 150k rows); resume fold-0 probe. EV: +5–15bp std-alone,
  +1–3bp stack median, tail +3–9bp. Live "10bp shot" per Day-12
  master synthesis. `[owner: unclaimed | status: open]`
- **1b.** EmbMLP CPU baseline. Embedding-MLP for the leakage-robust
  population. Cheap, untried. Day-15 reframed: Jahrer-style swap-noise
  DAE + LGBM-on-latent (Porto Seguro precedent) is the strict upgrade —
  unsupervised, orthogonal to the supervised pool.
  `[owner: read-handover-LgbQ4 | status: wip]`
- **1c.** DeepFM-lite (FM + 2-layer MLP head). Extends d9c FM with
  non-linear interactions. Cheap CPU.
  `[owner: unclaimed | status: open]`
- **1d.** Regularised FFM re-attempt. d9e FFM strictly worse, but
  with stronger regularisation (dropout, L2) may flip the tie.
  `[owner: unclaimed | status: parked]`

## 2. Meta-layer innovations beyond Path-B hier-meta

- **2a.** Non-Gaussian shrinkage prior on hier-meta. Path B is
  Gaussian-τ; try Beta-Binomial or Student-t shrinkage.
  `[owner: unclaimed | status: open]`
- **2b.** Calibration-aware stacker for private-LB row-iid structure
  (test is i.i.d. per U3). Re-fit calibration on holdout.
  `[owner: unclaimed | status: open]`
- **2c.** GroupKF-meta as R5 HEDGE. d12 K=20 GKF clean; ρ vs
  Strat-meta 0.9856; private-LB-likely-real per d10/d13d probes.
  `[owner: unclaimed | status: open]`

## 3. Target reformulation upstream of K=21 pool

- **3a.** Within-stint relative-progress as TARGET (not feature).
  d13 G1 was a feature-add NULL; reframe as target variable.
  `[owner: unclaimed | status: open]`

## 4. External data

- **4a.** Pirelli pit-window scrape. Tier-2 highest absolute EV per
  Day-8 research. Compound × track × season metadata.
  `[owner: unclaimed | status: open]`

## 5. Final-3-day-window strategy (R5 HEDGE)

- **5a.** d13e Compound×Stint τ=20k vs τ=100k HEDGE. Flip ratio
  55/98 < 200; HEDGE-eligible per R7. Decide at start of final
  window.
  `[owner: unclaimed | status: open]`

## 6. Pool composition surgery

- **6a.** Replace 3 most leakage-eating GBDTs with FM-class bases.
  cb_slow-wide-bag (-17 GKF rank), e5_optuna_lgbm (-13 rank).
  Risk: public-LB row-iid leak-eaters carry signal (d13c T2/T3).
  `[owner: unclaimed | status: open]`

---

## Falsified or dead (do not re-claim)

- Time-to-event LGBM variants (d12 T1.2 4-of-4 dead).
- Year-segmented specialist + AV-reweight (d12 falsified; AV-AUC 0.502).
- LambdaRank meta (d12 -86bp).
- AUC-pairwise XGB base (d12 -451bp fold-0).
- 6-shape FM partition variants (Day-13 PM saturated across shapes).
- Drop-GBDT leak-eaters (d13c T2/T3 falsified -2.5/-2.6bp Strat).
- Single-base FE additions in LGBM/FM class (d13/d14 alt-axis G1/G2'/G3/H1
  4-of-4 NULL).
- Path B cohort sweep on Year, Year×Stint, Race axes (d14 9 variants NULL).
- Hazard NN (d9 -315bp; main-branch leakfree confirmed dead OOF 0.92013).
- 2-level stacking via meta-as-base (2026-05-06: K=22 + d12_lr_meta
  +1.348 bp OOF, but K=22 Path B Compound×Stint τ=100k SUBMITTED
  LB 0.95045, regressed -4 bp; Path B amp does NOT fire on
  meta-derivative additions per friction tag
  `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`).

---

## Re-decomposition trigger

Rewrite this file when ANY of:
- All open leaves resolved (`done` / `null`), OR
- Plateau ≥3 days at unchanged PRIMARY LB, OR
- Saturation count ≥5 at same LB, OR
- 50% comp checkpoint, OR
- PI says "redecompose" / "new issue tree".

The agent that owns the rewrite archives this file to
`audit/archive-YYYY-MM-DD-issues-pre-redecomp.md` before replacing.
