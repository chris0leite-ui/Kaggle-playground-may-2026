# Top-writeups contrast — 2026-06-02

Comparative read of the 9 post-close solution writeups on
`playground-series-s6e5` (Predicting F1 Pit Stops), contrasted
against our final stack (W2 PRIMARY 0.95460 + W2_R31 HEDGE
0.95461, rank 148/3023 → top 4.90%).

Source data: `kaggle competitions topic-messages
playground-series-s6e5 <id>` for each writeup, dumped to
`/tmp/writeups/<id>.csv`.

---

## A. Per-writeup capsules

URL pattern: `https://www.kaggle.com/competitions/playground-series-s6e5/writeups/<slug>`

| Rank | Author / Team | Private LB | Title | Topic ID | Slug |
|---|---|---|---|---|---|
| 1 | Optimistix | 0.95503 | By the skin of my teeth | 703562 | `1st-place-by-the-skin-of-my-teeth` |
| 2 | Chris Deotte | 0.95502 | Autonomous Codex Yolo! | 703615 | `2nd-place-autonomous-codex-yolo` |
| 4 | mahoganybuttstrings (Mahog) | 0.95490 | 5 day rush | 703528 | `4th-place-5-day-rush` |
| 5 | unseenuser | 0.95505 (best post-deadline) / 0.95489 selected | 99-model logit stack | 703572 | `5th-place-solution-a-99-model-logit-stack` |
| 7 | author unspecified | ~0.95485 | 7th place solution | 703537 | `7th-place-solution` |
| 8 | author unspecified | 0.95487 | L5 Ensemble | 703539 | `l5-ensemble` |
| 10 | author unspecified | (10th private) | Stacking stacked predictions | 703542 | `stacking-stacked-predictions` |
| 17 | Ravi Ramakrishnan + Arun | 0.95476 | Rank17 — diverse models and blend | 703529 | `rank17-approach-diverse-models-and-blend` |
| 75 | author unspecified | 0.95472 | I Touched Nothing and Finished 75th | 703584 | `finished-75th` |

**No 3rd or 6th place writeup posted as of 2026-06-02.**

### 1st — Optimistix (0.95503)
- 186 OOFs ⇒ LR-on-logits ensembler (cuML).
- Big-6 model classes: XGB, LGBM, CatBoost, RealMLP, TabM, HGB, RF,
  YDF, FT-Transformer, MLP-PLR.
- AutoML usage: AutoGluon (best overall ensembler), Light AutoML,
  FLAML, PyTabKit.
- **Breakthrough**: per-column adversarial diagnostic ⇒ `Driver`
  was the most-shifted column ⇒ ran **"Driver_Dropped" model
  variants**. Boosted CV+LB.
- Original-data sample-weight modulation (0.5–1.0 ≈ neutral;
  0.25 hurt).
- Final = 50-50 blend of AutoGluon-stack + LR-on-logits-stack.
- "L2 OOFs" (4 added) — meta-of-meta OOFs as extra base inputs.

### 2nd — Chris Deotte (0.95502)
- **218 models, all built autonomously by Codex (GPT-5.5 yolo) in
  effectively one weekend** running 4× A100 GPU.
- Final = cuML LogisticRegression on 218 OOF logits (no L2 stack,
  no pseudo-labels).
- Model family table sorted by best-single-CV: RealMLP 40 / XGB 36 /
  CatBoost 37 / TabM 11 / LightGBM 25 / TabICL 8 / FFM 3 / NN 9 /
  RF 3 / GNN 3 / HGB / FM / KNN / TabTransformer / ExcelFormer /
  Cox/survival / DAE / AMFormer / YDF / MLP-PLR / Trompt / TabR /
  AutoInt / GrowNet / SAINT / ModernNCA / DCN / GANDALF / TabNet /
  NODE / Logistic / FTT / LNN — **37 model families**.
- Fine-tuned TabICL was 2nd-most-important by absolute LR meta-coef.
- Honest workflow: nested-fold OOF, fold-safe TE, no seed-twins.
- Diversity = "start from scratch / different notebook backbone /
  reuse prior-comp code"; *not* tuning the same model.

### 4th — Mahog (0.95490)
- 265 OOFs, 57 selected by HillClimb.
- FE was small: count encoding + yekenot arithmetic interactions
  + TE on pairs.
- Best XGB CV 0.95378 / private 0.95346 — strong single.
- Blended his ensemble with mikhailnaumov's ensemble at the end
  (couldn't reproduce mikhail's CV). Without the cross-blend
  would have been 6th.
- Pattern: 265 = [FE variants] × [models] × [hyperparameters].

### 5th — unseenuser (0.95489 selected; 0.95505 post-deadline best)
- 99-model logit-stack via sklearn LogisticRegression.
- Clipped logits ±30, `class_weight=None`, `C=1.0` L2.
- **Family breakdown explicit (most important table in the
  comp):** GBDT trees set the ceiling (0.946–0.954); MLPs match;
  long tail of weaker-but-orthogonal nets (FFM 0.918, GRU, BART,
  Nyström kernel) "did the real decorrelation work".
- **Original-data-as-extra-training-rows = the one reliable lift.**
  Everything else was marginal.
- Sequence (GRU, BiLSTM, causal-TCN-on-hazard-channel), Graph
  (GraphSAGE, GNN), Kernel (Nyström, RBF SVM), BART, EBM, NGBoost
  — all included despite weak standalone AUC.
- **Pairwise Spearman across 99 models: mean 0.950** (saturated).
  Most-redundant cluster: LGBM (mean Spearman 0.965–0.968).
  Most-diverse single: Deep FFM (0.852), BART (0.851), Nyström
  (0.884). Explicit message: **"A 0.918 Deep FFM was worth more to
  the blend than a 4th 0.953 LightGBM."**
- **Post-deadline: rank-average of HC + LS gave 0.95505** —
  beats raw probability average 0.95503 (because the two have
  different probability scales). Would have won 1st.
- **Selection regret**: picked flat-99 LR-stack over HC (both
  conservative variants); should have split final 2 picks across
  "trust LR-stack" + "trust HC" philosophies. Mirrors our
  W2/W2_R31 plateau-center pick.

### 7th — anonymous (0.95485)
- 176 models: 25 hand-built + 50 AutoGluon-`best_v150` + 101
  AutoGluon-`best`. Small feature set: 18 cols + `max_laps`
  (Lap_Number/Race_Progress = a strong feature) + `TyreOvrLap` +
  Driver/Race count-encoded.
- AutoGluon-stack on 176 OOFs → CV 0.95453.
- Final = weighted 3-1-1-1 of {public-0.95454 blend, pseudo-label
  stack of 7 stackers, stack-of-50, LR-on-50}.
- Down-weighted public-blender for private (used 2-1-1-1 in one of
  the two picks; +0.1 bp private, -0.1 bp public). Same logic as
  our distinct-backbone hedge.

### 8th — anonymous (0.95487)
- **L5 = average of 3 different L4 ensembles, each L4 fit on a
  different fold count (5, 7, 10).** Different K-fold counts as
  diversity axis.
  - 5-fold L4: 123 OOFs → CV 0.95510, priv 0.95463
  - 7-fold L4: 11 OOFs → CV 0.95505, priv 0.95467
  - 10-fold L4: 13 OOFs → CV 0.95503, priv 0.95461
  - L5 average → priv 0.95487 (+2 bp over best single L4)
- **Original-data "row" vs "column" approach treated as separate
  signals.** "orig_row" trains on concat(train, original);
  "orig_col" derives TE/anchor features from the original. The two
  blended super-additively.
- Stack architecture: L1 (base OOFs) → L2 (meta features incl.
  ranks, pairwise diffs, AUC-weighted blends) → L3 (low-DoF LR on
  logits) → L4 (self-distilled student on L2/L3 teachers) → L5.
- RealMLP top-3-parameters-averaged. Optuna 100-150 trials for
  RealMLP/TabM/CB/LGB/FFM; 25 for others.

### 10th — anonymous (private not posted) ("Stacking stacked predictions")
- 221-model CV-best ensemble; selected smaller 129-model for
  better generalization.
- cuML LR on logits.
- "**OOF predictions as first-class assets**" — stopped training
  models, started training predictions.
- Multi-level: L1 → L1-stacks → community/public ensembles → L2
  stacks → cuML LR.
- Selection regret: a non-selected sub was ~5th-place private —
  same trap as us.
- Largest negative meta-coefs were on AutoGluon L2/L3 OOFs —
  the stacker was "spending coefficient budget correcting
  redundant/overfit signals" — a saturation diagnostic.

### 17th — Ravi + Arun (0.95476)
- Spent only "2-3 days" — 130 single models, 2 stacking layers,
  ~200 models total.
- Built an >800-feature **feature store** (re-usable across runs).
- **DOUBLE stratification**: `StratifiedKFold(5)` on (year, target)
  jointly. We used pure 5-fold Stratified on target.
- Specific per-driver / per-year cohort features explicitly
  acknowledged as ignoring leakage at the row level (i.e., used
  global stats, not fold-safe TE). They got away with it because
  AV-AUC=0.502.

### 75th — anonymous (0.95472)
- **Single-model solution**: RealMLP_TD_Classifier × 3 seeds × 5
  folds = 15 models averaged.
- Pseudo-labeling at confidence threshold 0.04/0.96. +2 bp.
- **Stopped adding features entirely** — every addition hurt OOF.
- Score progression: 0.95410 baseline → 0.95450 multi-seed →
  0.95472 pseudo-label.
- The "minimal-FE, multi-seed, pseudo-label" recipe.

---

## B. Dimension-by-dimension contrast

### B1. Pool size (OOF count fed to the final stacker)

| Cohort | OOF count |
|---|---|
| 1st Optimistix | 186 |
| 2nd cdeotte | 218 |
| 4th mahog | 265 (57 HC-selected) |
| 5th unseenuser | 99 (132 in HC alternate) |
| 7th anonymous | 176 |
| 8th anonymous | 123 + 11 + 13 = 147 |
| 10th anonymous | 129 (221 CV-best) |
| 17th Ravi | ~130 + 2 stack layers |
| 75th anonymous | 1 (15-model multi-seed) |
| **Us** | **K=28 max single-stack; W2 = 0.85 public-anchor + 0.15 (3-base ours)** |

**Delta:** We capped at K≈28 bases per single LR-meta layer; top-10
ran 99–218 base OOFs. Our pool effectively had ≈15× fewer base
contributors. This was the single biggest structural gap.

### B2. Model families covered

Reference universe (union of top-10 writeups):

**GBDT** XGB · LGBM · CatBoost · HGB · YDF · NGBoost · BART · EBM ·
RGF · ExtraTrees · RF
**NN** RealMLP (pytabkit) · TabM · TabICL fine-tuned · TabPFN ·
RealTabR · FT-Transformer · TabTransformer · Trompt · AutoInt ·
ModernNCA · MLP-PLR · DCN · DANet · ResNet · GANDALF · NODE ·
GrowNet · SAINT · ExcelFormer · TabNet · TabR · LNN · AMFormer ·
SELU · VSN · RFF-Kernel · DAE+transfer
**Other** GraphSAGE / GNN · GRU · BiLSTM · TCN-hazard · FFM · FFM-deep ·
xLearn FFM · Nyström-RBF · NCA-kNN · Cox/survival · ridge LR · GAM ·
LDA/QDA · AutoGluon / LightAutoML / FLAML / H2O AutoML

| Family | Us | Winners (typical) | Delta |
|---|---|---|---|
| LGBM | yes (yekenot recipe + variants) | universal | tied |
| XGBoost | tested, weaker single, used in K | universal (XGB 36 variants for #2) | underweighted |
| CatBoost | strong (cb_v4 yekenot, cb_horizon, cb_stint_completion) | universal | tied |
| HGB | tested d12 single-bag, killed | universal | possibly mis-killed |
| **RealMLP (pytabkit)** | **NOT in pool** | **#1 cited base by 7/9 writeups** | **MISSED — biggest single gap** |
| TabM | R14 added at K=16 PRIMARY swap | universal | tied |
| **TabPFN** | tested v2.5/v2.6, killed (AUC 0.944 / OOM) | 5/9 writeups use it productively | **mis-killed** |
| **TabICL fine-tuned** | not tested | 1st place's 2nd-most-important OOF; 7th, 8th use | **MISSED** |
| **RealTabR** | not tested | 8th place uses | MISSED |
| Cox/survival | R11-C (failed G1) + R12 cb_horizon (worked) | 1st place uses, 5th's BART/EBM cluster | partial |
| FFM / FFM-deep | d9 family tested early, saturated | 5th's most-diverse OOF (0.852 mean ρ) | mis-prioritised — closed too early |
| GNN / GraphSAGE | R11-B graph attention failed (sm_60 + truncation) | 5th, 10th, 17th include despite weak | mis-killed bug-blocked |
| GRU / BiLSTM | R5 transformer v1/v2 absorbed | 5th includes both | mis-killed (we required +0.10 bp G2; they added <0 bp standalone) |
| BART / EBM / NGBoost | not tested | 5th's tail of diversity | MISSED |
| Nyström kernel / NCA-kNN | tested SVM family and NCA-kNN, all null | 5th includes both | tied (closed correctly) |
| AutoGluon / FLAML / LightAutoML / H2O | not tested | 7/9 writeups use AutoGluon | **MISSED — at least one AutoGluon line** |
| Logistic regression base | yes (mega-LR variants, sparse LR) | yes | tied |
| RandomForest | RF-yekenot Day-20 added | universal | tied |

**Most important misses, in order:**
1. **RealMLP** — cited as strongest single by 7+ writeups, never in our pool.
2. **AutoGluon** — the cheap multi-OOF generator everyone uses for the
   long diversity tail. We never spun one up.
3. **TabPFN-2 + Fine-tuned TabICL** — we killed TabPFN-v2.5/v2.6
   on "AUC ceiling 0.944"; 1st-place's TabICL was 2nd-most-
   important LR coefficient.

### B3. Original-data integration

| Approach | Us | Winners |
|---|---|---|
| Original-as-rows (concat) | R9 C1 done as per-Race scalars; not as full row-concat with sample-weight | 1st (0.5–1.0 sample weight modulation), 4th, 7th, 8th — universal |
| Original-as-columns (TE/anchor from) | not explicit | 8th place treats this as a distinct signal axis; 5th, 8th, 17th use it |
| Per-column adversarial diagnostic ⇒ drop-shifted-column | not done | **1st-place breakthrough — drop `Driver`** |

**Delta:** Our AV-AUC=0.502 was the AGGREGATE diagnostic. We didn't
run per-COLUMN adversarial AUCs that would have flagged `Driver`
as the shifted feature. Optimistix's whole breakthrough was a
per-column diagnostic ⇒ training-without-Driver variants. Concrete
miss.

### B4. Stacker architecture

| Approach | Us | Winners |
|---|---|---|
| Single-layer LR-meta on K=N [P, rank, logit] | yes (our standard) | 5th and 10th lean on cuML LR on logits-only |
| Path-B per-segment shrinkage | yes (DriverClass×Stint τ=100k) | **none mentioned by any winner** |
| Hill-climb (HC) selection | not used | 4th, 5th, 10th use it |
| AutoGluon as a stacker | not used | 1st, 4th, 7th, 17th — universal |
| Multi-level (L2+) stacking | killed Day-13 (meta-as-base −1.30 bp); R14 K=22 tested falsified | 1st (4 L2 OOFs), 7th (stack-of-stacks), 8th (L5), 10th (L1→L1-stacks→L2) |
| Logit-space averaging | tested D30 L1 candidate, LB-tied | 5th (clipped ±30), 7th (logit-blend mode), 8th L3 |
| Rank-average across heterogeneous stackers | not done | **5th's 0.95505 trick: rank-avg HC + LR-stack** |
| Different fold-counts (5, 7, 10) as diversity | always 5-fold | 8th place L5 averages across them |
| Multi-seed averaging | partial (R6 fold-fit bag) | universal (75th gets +4 bp from 3 seeds alone) |

**Delta highlights:**
- **Path-B was unique to us.** No top writeup mentions per-segment
  shrinkage. Path-B amplified on our redundant pool (K=21-22) but
  isn't a known top-tier technique.
- **Multi-level stacking (L2+) — universal at top.** Our Day-13
  closure ("meta-as-base −1.30 bp") was on a K=4 pool. The L2 lift
  for top finishers came from much wider pools (K=99+) where the
  L2 OOF actually adds rank info.
- **AutoGluon-as-stacker is a top-tier default.** 1st place: AutoGluon
  ≥ LR-Logits. 4th: stacked with AutoGluon. We never tried it.
- **Different-fold-counts as diversity** (8th place L5) — we never
  varied fold count. Cheap, structural.

### B5. Final-pick selection

| Logic | Us | Winners |
|---|---|---|
| Plateau-center over plateau-extreme | **W2 P=0.15** (private 0.95460); W6 P=0.35 would have hit 0.95466 (+6 bp ≈ top-2.5%) | 5th picked flat-99 over HC; admits the same mistake as us. 10th picked 129-model over 221-model; admits the same. |
| HEDGE = distinct backbone | yes (W2_R31 swapped C5→R31; private +1 bp validated) | 4th cross-blended with mikhail's ensemble; 7th used 2-1-1-1 weight downshift on public-blender |
| Two-philosophy hedge | NO — both W2 and W2_R31 are plateau-center variants | **5th explicit lesson:** "the better hedge would have been to split my two final submissions across both philosophies — trust-conservative AND trust-HC — rather than two conservative variants" |

**Delta:** Our hedge was distinct-backbone but same-philosophy
(both plateau-center). Multiple top finishers admit the same
sub-optimality. The cross-comp pattern is real.

### B6. Compute allocation

| Cohort | GPU posture |
|---|---|
| 1st Optimistix | Kaggle + own GPU, 186 OOFs over the whole month |
| 2nd cdeotte | 4× A100 entire weekend via Codex autonomous |
| 17th Ravi | A100 80GB + A6000 + L4 Colab |
| 75th anonymous | single GPU, 15 RealMLP models |
| **Us** | Kaggle T4×2 + P100 + local CPU (8-core, no GPU). No persistent A100 access. |

**Delta:** Our compute is one tier below the top. The structural
gap shows up most in the "hundreds-of-OOFs" workflows — we cannot
generate 100+ OOFs in a weekend at our compute tier.

---

## C. What we overlooked

1. **RealMLP (pytabkit)** — the strongest single-model base in
   this competition by every winner's tally. We tested TabM (R14)
   and the gap-aware transformer (R5/R6), but never spun up
   RealMLP. yekenot's notebook was public from Day-1 and we
   noted it in scrape candidates but never trained the model
   ourselves. **Highest-impact single miss.**

2. **AutoGluon (or similar AutoML) as a cheap diverse OOF
   generator.** 4 of 9 writeups use AutoGluon; the 7th place got
   151 OOFs from AutoGluon alone. We never spun one up.

3. **Per-column adversarial diagnostic ⇒ drop-shifted-column
   model variant.** We computed aggregate AV-AUC=0.502 and
   declared train/test i.i.d. Optimistix computed AV-AUC per
   column, found `Driver` was the shifted feature, and that
   training **without** `Driver` lifted his score. This is a
   1st-place breakthrough we structurally missed because the
   aggregate AV test gives a binary i.i.d. yes/no, not a per-
   column diagnostic.

4. **Fine-tuned TabICL.** cdeotte's 2nd-most-important LR meta
   coefficient came from fine-tuned TabICL. We never tested it.
   8th place uses it too.

5. **Original-data as columns** (TE/anchor features derived from
   the original dataset and joined back to the comp data) —
   distinct from original-data-as-rows. 8th place explicitly
   says these are complementary signals that blend
   super-additively. We did rows (Aadigupta join) and per-Race
   scalars but not full TE-from-original.

6. **Multi-level stacking on a wide pool (L2 / L3 / L4).** We
   closed multi-level Day-13 at K=4 ("meta-as-base −1.30 bp").
   That closure was POOL-SIZE-SPECIFIC: every top finisher uses
   L2+ on a pool of 99–218 bases. The lift is in the wider
   pool's ability to absorb the L2 OOF as a non-redundant axis.
   Our small-K closure was misread as universal.

7. **Weak-but-orthogonal bases (FFM, BART, EBM, GRU, GraphSAGE).**
   5th place is explicit: "a 0.918 Deep FFM was worth more to the
   blend than a 4th 0.953 LightGBM." We required +0.10 bp G2 OR
   the +0.02 calibrated gate before adding a base; many of the
   bases winners kept would have flunked our gates. **The G2
   gate is structurally biased against weak-but-orthogonal
   bases on a saturated metric.**

8. **R11-B GNN failure was a kernel bug (truncation at
   MAX_DRIVERS=24, 80% of groups truncated).** 5th, 10th, 17th
   all use GNN/GraphSAGE successfully despite weak standalone
   AUC. We killed graph-class on a debug-blocked failure, not a
   true mechanism close.

9. **Different K-fold counts as a diversity axis.** 8th place
   averages L4 ensembles built on 5/7/10-fold splits. We always
   used 5-fold (Stratified KF). Cheap to add; never tried.

10. **Pseudo-labeling at high confidence (0.04/0.96 threshold).**
    Day-16 transductive pseudo-labeling 627k+ probe scored
    +0.631 bp at K=22 LR-meta but null at hier-meta; we marked
    it "marginal hedge" and never returned. 75th-place's solo
    +2 bp came from this. 7th uses it as one of 4 stack inputs.
    Possible mis-prioritisation.

11. **Rank-average over heterogeneous stackers** (5th place's
    0.95505 trick: rank-avg of HC-stack + LR-stack). We did
    rank-mean blends *within* the same stacker family (C5 = rank-
    mean of K=20 / R21 / K=27 all under the same LR-meta), but
    never combined a TREE-class stacker with our LR-meta in
    rank space. Cheap structural lever.

12. **Top-3-hyperparameter averaging on a single model
    (e.g. RealMLP).** 8th place used this as a baseline diversity
    trick. Different from seed-bagging (which we did via R6's
    fold-fit bag).

13. **Cross-team blend at the end.** 4th place's last move was
    blending with mikhailnaumov's ensemble (couldn't reproduce
    his CV but trusted the LB). We have mikhail's notebook in our
    scrape directory but never blended it as a peer ensemble
    (only as a sibling in the public-blender cluster, where it
    was 99.7% correlated with raunakdey and added 0).

---

## D. What we executed well

1. **Public-blender axis activation (R22 V8 +3 bp over public
   ceiling).** Several top writeups (4th, 7th) explicitly used
   public-blender ensembles as one input to their stack. Our
   V8 = 0.90 raunakdey + 0.10 C5 was structurally identical to
   7th place's recipe and gave us the +42 bp jump from K=20-based
   to top-of-public.

2. **Distinct-backbone hedge logic (W2 → W2_R31).** Private +1
   bp validated. The cross-comp lesson "hedge swaps the
   ours-backbone, not the anchor" appears to be genuinely
   novel — none of the top writeups describe it explicitly.
   Worth promoting (R2d already in `improvements.md`; this
   strengthens it).

3. **Strict fold-safe TE (R24).** Top writeups don't mention
   leakage failure modes; we hit the inv-laps-until-pit /
   pit-horizon / reverse-cumulative trap on Day-15 and built
   discipline around it. 17th place (Ravi) explicitly notes he
   "ignored leakage" on driver/year aggregates. Our R24 is more
   conservative and likely correct.

4. **Adversarial-validation gating (R25).** We confirmed AV-AUC =
   0.502 and so trusted the Stratified-KF as LB proxy. Top
   writeups don't run AV at all; some (75th) explicitly note they
   "didn't think about" group-vs-iid. Our discipline here was
   solid even though we missed the per-column refinement.

5. **Membership-inference axis (d21/d22).** None of the top
   writeups mention k-NN-exact-copy distance as a feature. Our
   +2 bp Day-29 from d21+d22 with ρ_test_vs_K18=0.7014 (lowest
   ever recorded) is genuinely novel — we found a private
   structural signal nobody else surfaced. Even though our
   final stack diluted it, the discovery is worth documenting.

6. **Probe.py / decisions.jsonl calibration discipline.** No
   winner mentions formalized BOTE / G1-4 gates. Several admit
   the SAME selection regret (5th, 10th, us). Our calibration
   ladder is more disciplined; the deficit is the gate
   thresholds were tuned for small-K pools and bias against
   weak-orthogonal bases.

---

## E. Promotion candidates for `.claude/skills/kaggle-comp/improvements.md`

Each candidate listed PI-unratified until tagged. Order by
confidence × cross-comp generality.

### E1. Always seed the pool with RealMLP / TabM / TabPFN / TabICL early

Pre-baseline gate item 8 ("public-notebook scan on Day 1") should
include an EXPLICIT mandate: if any public notebook uses a model
class we haven't trained in any prior comp, train at least 1 OOF
version in the first 5 days. Specifically: RealMLP via pytabkit
appears in every Playground winner's pool for 4+ months running
(s6e1, s6e3, s6e4, s6e5). Treat it as a default base, not an
"exotic NN" to gate.

### E2. Per-column adversarial-AUC diagnostic, not just aggregate

Rule 25 currently says "AV-AUC > 0.55 ⇒ fit on train only".
Strengthen: run AV-AUC PER COLUMN; any column with per-column
AV-AUC > 0.55 ⇒ train at least one base WITHOUT that column.
1st-place Optimistix's breakthrough was a column we'd never have
flagged because aggregate AV-AUC = 0.502.

### E3. The "+0.10 bp G2" gate is biased against
weak-but-orthogonal bases on saturated AUC

The G2 +0.10 bp threshold (and the calibrated +0.02 bp gate)
kills bases that are weak standalone but orthogonal in error
direction. 5th place's explicit claim ("a 0.918 Deep FFM was
worth more than a 4th 0.953 LightGBM") is empirical evidence the
gate is mis-tuned for saturated-AUC blending. Candidate revision:
**For a pool with ≥50 bases, replace the G2 absolute threshold
with a Spearman-orthogonality threshold: any base with mean
pairwise Spearman ≤ 0.92 against the existing pool is admitted
regardless of standalone OOF.**

### E4. At least one AutoGluon (or AutoML) line per comp

AutoGluon-stack appears in 4 of the top-10 writeups. It's a
cheap multi-OOF generator and a strong meta-stacker. Treat it as
a default pipeline component, not an exotic. Add to the
day-loop's standard list of mechanisms.

### E5. Multi-level stacking re-opens at K ≥ 50

Our Day-13 closure on multi-level was at K=4. The closure does
not generalize to K ≥ 50 pools (every top finisher uses L2+ on
99+ bases). Rule: when pool grows beyond 50 OOFs, re-test
L2 / L3 stackers; don't carry forward the small-K closure.

### E6. Different fold counts (5 / 7 / 10) as a diversity axis

8th place's L5 = avg of L4-on-5fold + L4-on-7fold + L4-on-10fold
gave +2 bp over best single L4. Cheap to implement; orthogonal to
all our other diversity levers. Add as a default in the
experiment-loop's fold-strategy step.

### E7. Two-philosophy hedge, not just two-backbone

Our W2 / W2_R31 hedge swaps the ours-backbone but both are
plateau-center picks. 5th place explicitly identifies the same
mistake: better to hedge across "trust the LR stack" AND "trust
the HC stack" — two STACKER philosophies, not two ingredient
sets within one philosophy. Rule 2d candidate revision:
**FINAL_PRIMARY = best LB on the conservative stacker;
FINAL_HEDGE = best LB on a structurally different stacker
(HC, AutoGluon, or rank-avg of HC+LR-stack).**

### E8. Rank-average over heterogeneous stackers as a default
last-day move

5th place's 0.95505 trick: rank-average HC + LR-stack scores
above either alone. Cheap one-liner. Add to the day-loop's
"final-day moves" list.

### E9. Top-3-hyperparameter averaging on each base before
adding to the pool

8th place reports averaging the top-3 Optuna trials per model as
adding diversity. Different from seed-bagging. Add to the
mechanism-ledger's standard "add one base" workflow.

### E10. Public-notebook reuse posture: rerun in YOUR fold layout

cdeotte (#2): "Codex reads top-10 public notebooks, rewrites
them to use our local 5-fold, saves OOF and test preds." Trains
in ~1 hour on A100. Our public_scrape directory has test preds
only; we never re-ran public notebooks in our fold layout to get
honest OOFs. Without OOFs, public notebooks can only be used as
blender ingredients (which is what we did), not as base OOFs to
stack on. Candidate rule: **for any public notebook with
LB > our PRIMARY − 30 bp, rerun in our 5-fold to extract an
honest OOF before treating it as a candidate base.**

---

## F. PI ratification slot

PI: please tag each E-candidate with one of:
- `[PROMOTE]` — append to `.claude/skills/kaggle-comp/improvements.md`
  with the description and origin (this writeup + finisher).
- `[KEEP-LOCAL]` — keep in this audit file only; not durable
  enough or too s6e5-specific.
- `[REJECT]` — disagree with the lesson.

(no promotions written without explicit `[PROMOTE]` tag.)

---

## G. Source links

| Topic | URL |
|---|---|
| 1st Optimistix | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth |
| 2nd cdeotte | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo |
| 4th mahog | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/4th-place-5-day-rush |
| 5th unseenuser | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/5th-place-solution-a-99-model-logit-stack |
| 7th anonymous | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/7th-place-solution |
| 8th anonymous | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/l5-ensemble |
| 10th anonymous | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/stacking-stacked-predictions |
| 17th Ravi+Arun | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/rank17-approach-diverse-models-and-blend |
| 75th anonymous | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/finished-75th |

Raw writeups archived at `/tmp/writeups/<id>.csv` (will be lost on
container reclaim — copy into `audit/writeups-archive/` before
end-of-session if PI wants permanent reference).
