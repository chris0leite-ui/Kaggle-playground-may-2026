# Notebooks rescan — 2026-05-19 (Rule 22, post-CLI-auth-fix)

Last scan: 2026-05-09 (10 days stale; HANDOVER notes the gap was the
kaggle CLI auth block, fixed earlier today by switching to
`KAGGLE_API_TOKEN="$KaggleAPIToke" kaggle ...`).

Scope: top-30 by `scoreDescending` and top-30 by `voteCount`. Pulled
14 distinct kernels and inspected source.

## Public LB top 5 — by score

| Rank | Kernel | LB | Category | One-line summary |
|------|--------|------|----------|------------------|
| 1 | nawfeelrahman1124444/s6e5-0-95452 | 0.95452 | (a) trivial | Reads a private dataset CSV (`pss6ep5448/submission-...csv`) and writes it as `submission.csv`. No model code. Output of a separate private submission. |
| 2 | raunakdey07/f1-pit-stops-blender-0-95450 | 0.95450 | (a) blender | Loads N public + private submissions from a structured `public/` + `ours/` folder. Greedy weighted blend across (rank-blend, asymmetric-max, prob-mean) candidates, picks the one with best diagnostic. Pure ensemble of *other* notebooks. |
| 3 | azzamradman/knock-the-blender-with-one-line | ~0.95450 | (a) one-liner | Literal one line copying raunakdey07's submission.csv to its own output. No mechanism. |
| 4 | kalyankkr/f1-pitstop-ensemble-blender-0-95450 | 0.95450 | (a) blender | Same family as raunakdey07; finds best submission.csv in `/kaggle/input`, blends candidates with rank/amax/prob strategies. No model code. |
| 5 | safar1/lb-score-0-95449 | 0.95449 | (a) blender | 50/50 average of `mikhailnaumov/f1-pit-stops-ensemble/submission.csv` (the 0.95438 model output) and `nina2025/f1-pit-stops-17/0.95439.csv` (private dataset, source unknown). |

The whole top of the public LB is a daisy-chain blender stack
sourced from **one underlying public model** (`mikhailnaumov`) plus a
private nina2025 dataset (LB 0.95439, source-of-truth not in any
public kernel).

Other inspected: `flexonafft/f1-submission-blender-0-9545`,
`nina2025/ps-s6e5-hb11`, `nina2025/predicting-f1-pit-stops-blend`,
`leonchani/02-nn`, `sidcodegg/tabpfn-is-all-you-need`,
`pilkwang/s6e5-driver-s-high-driver-feature-eng`,
`anthonytherrien/predicting-f1-pit-stops-nn-residual-network`,
`arunklenin/ps6e5-f1-pit-stops-prediction-fe-ensemble`.

## Leader-mechanism call-out (≥ 0.95476)

**No public kernel reaches 0.95476.** The current public LB top is
0.95452 (a private-submission echo) and 0.95450 (blenders). The
leader's 0.95476 LB does not appear to be reachable from any public
public-domain mechanism — it is presumably a stronger private
ensemble or a private feature set.

## Categorisation of inspected non-trivial source kernels

| Kernel | LB | Category | Mechanism vs our ledger |
|--------|----|----------|--------------------------|
| **mikhailnaumov/f1-pit-stops-ensemble** | 0.95438 | (b) novel-class | **TabM (Tabular Mixture) via pytabkit** + 4×RealMLP variants + 4×XGB + LGBM + CatBoost. 5-fold StratifiedKF; aadigupta1601 orig data concatenated to every train fold; sklearn TargetEncoder per cat col added as `*_te` features; KBinsDiscretizer of RaceProgress (200 bins) and LapTime (7 bins); count-encoded num→cat; final blend by per-model OOF AUC. **TabM_D_Classifier with `arch_type='tabm-mini'`, `tabm_k=8`, piece-wise-linear num embeddings, 119 bins, d_block=512, 3 blocks — NOT on our ledger.** RealMLP we have but TabM is structurally distinct (mixture of K parameter-efficient heads, not a single MLP). |
| sidcodegg/tabpfn-is-all-you-need | (no LB reported) | (a) known | TabPFN-client (hosted prior-fitted-transformer). Already tested in our ledger (Day-14 TabPFN v2.5/v2.6, AUC ceiling 0.944). |
| pilkwang/s6e5-driver-s-high-driver-feature-eng | sub-PRIMARY | (a) known | Within-(Year,Race,Driver) shift/lag features (`stack_pos_prev1/3`, `stack_lt_prev1/2`, `stack_lt_delta1`, `stack_lt_roll3_mean`) + within-(Race,Lap) field aggregates (`stack_race_lap_meantyrelife`, `stack_tyrelife_vs_field_mean`). Identical to our Day-17 PM "Field-state cross-row aggregates" (24-base stack-add −0.015 bp null) and the C2 candidate in `2026-05-18-notebooks.md` (delta_laptime). |
| arunklenin/ps6e5-f1-pit-stops-prediction-fe-ensemble | sub-PRIMARY | (a) known | aadigupta concat + LapTime/LTDelta/Cumdeg/Pos lag-1/2/3 grouped by (Year,Race,Driver,Stint) + TyreLife danger flags (HARD>15, MEDIUM>15, Stint2×HARD>15, etc.) + `laps_remaining_race`, `RP_remaining`. The danger-flag set is essentially r4_segment_fe v1/v2 (G2-fail in ledger); the lag features are pilkwang/C2 redux. |
| leonchani/02-nn | (no LB reported) | (a) known | Plain MLP (Linear→128→…). Same family as our Day-15 PM NN-with-embeddings base (K=21+1 −0.025 bp null). |
| anthonytherrien/predicting-f1-pit-stops-nn-residual-network | sub-PRIMARY | (a) known | ResNet-style MLP (residual block: Linear→BN→ReLU→Dropout→Linear→add). Same family as RealMLP and the NN-embedding base — already absorbed at meta. |
| flexonafft/f1-submission-blender-0-9545 / nina2025/ps-s6e5-hb11 / nina2025/predicting-f1-pit-stops-blend | various ≥0.954 | (a) blender | All read CSV outputs from other kernels and blend with weighted rank / asymmetric-max / probability mean. No own model code. |

## Novel mechanisms found (category b)

**One genuinely novel mechanism, structurally distinct from our ledger:**

### N1. **TabM (Tabular Mixture) base via pytabkit**
- Source: `mikhailnaumov/f1-pit-stops-ensemble` (the actual
  underlying model behind the entire 0.95449–0.95452 blender stack).
- What it is: a recent (2024) tabular DL model from the pytabkit
  group, structured as a *mixture of K compact MLP heads sharing
  trunk weights* (`arch_type='tabm-mini'`, `tabm_k=8`). Piece-wise
  linear numerical embeddings (`num_emb_type='pwl'`, 119 bins,
  `d_embedding=32`). Effectively K efficient ensembles inside one
  model — closer to a snapshot-ensemble at parameter level than a
  conventional MLP.
- Ledger check: our NN-class entries are RealMLP, plain MLP, NN-with-
  embeddings, gap-aware transformer v1/v2, GRU, HMM, swap-noise DAE.
  TabM is structurally distinct from all (mixture-of-heads ≠ single
  stack; PWL num embeddings ≠ our PLR/categorical embeddings). Not
  in the ledger.
- Predicted EV: standalone TabM_D probably lands in the 0.944–0.948
  range (RealMLP territory on s6e5; mikhailnaumov's full *blend* of
  ~10 models hits only 0.95438 LB, so any single component is mid-
  pack). At K=13+Path-B as a single-base meta-add, the absorption
  pattern (Day-15 NN-with-embeddings, Day-19 LightGBM-yekenot,
  R4-HMM standalone) predicts ≤ +0.2 bp lift. **Probable null at
  meta but worth one slot — TabM is the only structurally
  unseen mechanism family on the public LB.**
- Cost: ~30–60 min Kaggle T4 GPU. `pip install pytabkit` + 5-fold
  bag at default hyperparams.

## Negative findings (confirm closures)

- **FastF1 / OpenF1 / Ergast / jolpica** — zero hits across all 14
  pulled notebooks. No public kernel attempts a lap-by-lap real-F1
  hard join. Our 1.4% match-rate cap is consistent with the entire
  public scoreboard treating s6e5 as synthetic; the FastF1 angle
  remains unexplored on the LB. **Whether that's because nobody
  tried or because everyone tried and it didn't transfer is not
  decidable from this scan.**
- **Sequence/transformer/LSTM** — only leonchani's plain MLP and
  anthonytherrien's residual MLP surface; no LSTM, no transformer,
  no GRU in any top-LB kernel. Our R5/R6 transformer v1/v2 work
  has no public-LB analogue.
- **TabPFN** — appears once (sidcodegg) as the cloud-client API;
  already-tested family on our side. Closes the TabPFN axis: our
  Day-14 v2.5/v2.6 attempts and sidcodegg's TabPFN-client are the
  same model family at different access points.
- **Cross-row driver-pair / undercut / FastF1-hard-join** — none of
  these appear in the public LB. The Frontiers Bi-LSTM paper's
  `DriverAheadPit`/`DriverBehindPit` mechanism (candidate C1 in
  `2026-05-18-notebooks.md`) remains uncontested in public space.

## Recommendation for the R10 mechanism-expansion swing

**Single highest-EV add from this scan: probe TabM_D_Classifier
(N1) as a K=14 base.** Mechanism-orthogonal-to-everything-on-ledger;
pytabkit-installable; mikhailnaumov's hyperparams give a working
starting point (3 blocks, d_block=512, tabm_k=8, pwl-119-bins).
Predicted outcome: standalone OOF ~0.945, K=14+Path-B Δ ≈ 0 to +0.3
bp at meta (absorption-likely per the rank-lock pattern, but TabM's
mixture-of-heads structure has a non-trivial chance of producing
a residual orthogonal direction the LR-meta can use).

Slot it in the 3 daily R10 slots — alongside the two unexplored
mechanism classes outside the public scoreboard (sequence-class
with bigger transformer pretraining; DriverAheadPit cross-driver
contagion features per C1).

## Kernels pulled this pass

`/tmp/kpull/` contains:
- nawfeelrahman1124444/s6e5-0-95452
- raunakdey07/f1-pit-stops-blender-0-95450
- azzamradman/knock-the-blender-with-one-line
- kalyankkr/f1-pitstop-ensemble-blender-0-95450
- safar1/lb-score-0-95449
- flexonafft/f1-submission-blender-0-9545
- nina2025/ps-s6e5-hb11
- mikhailnaumov/f1-pit-stops-ensemble  ← only one with novel mechanism
- nina2025/predicting-f1-pit-stops-blend
- leonchani/02-nn-predicting-f1-pit-stops
- sidcodegg/tabpfn-is-all-you-need
- pilkwang/s6e5-driver-s-high-driver-feature-eng
- anthonytherrien/predicting-f1-pit-stops-nn-residual-network
- arunklenin/ps6e5-f1-pit-stops-prediction-fe-ensemble

Listings saved at `/tmp/k_score.txt` and `/tmp/k_votes.txt`.
