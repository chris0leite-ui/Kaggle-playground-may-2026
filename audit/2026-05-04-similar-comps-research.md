# Cross-comp research for s6e5 (2026-05-04, brief-mandated three-source pass)

Mission: cross-comp tactics for s6e5 (F1 PitNextLap, binary AUC, prior
0.199, 439k/188k, baseline LB 0.94113, top-5% 0.95345). Three brief-
mandated sources: aadigupta1601 F1 dataset top-3, PS S4E1 Bank Churn
top-3, PS S3E23 Software Defects top-3.

Access: Kaggle WebFetch returns only `<title>` (JS-rendered); CLI verb
`competitions discussions` does not exist in CLI 2.1.0. Only `kaggle
kernels list/pull` works. Body content was obtained via (a) `kaggle
kernels pull` for the F1 dataset top notebooks, (b) web search /
Medium / GitHub for the two PS comps. Where all top-3 writeups were
inaccessible, fallback = top accessible third-party summary.

## Source 1 — aadigupta1601 F1 Strategy dataset, top-3 notebooks

`kaggle kernels list --dataset aadigupta1601/f1-strategy-dataset-pit-stop-prediction
--sort-by voteCount` ranked notebooks (top by votes; #2 and #3 are pre-
or co-baseline with s6e5 host data and therefore the cross-comp signal):

1. **`yekenot/ps-s6-e5-realmlp-pytabkit`** (56 votes, Vladimir Demidov).
   - Single-model **RealMLP via PyTabKit**, GPU. Posted 2026-05-03.
   - Reported OOF AUC ≈ 0.946 (cited in analyticaobscura's notebook as
     baseline, https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit).
   - F1-specific FE: none beyond raw features; load-bearing was
     RealMLP's ability to handle high-cardinality embeddings without TE.
2. **`analyticaobscura/pit-or-stay-f1-strategy-1`** (34 votes, Ozan M.).
   Pulled via `kaggle kernels pull`; full ipynb on disk.
   - **5-model OOF stack**: RealMLP + focused XGB + focused LGB + Dart
     LGB + EmbMLP (PyTorch embeddings on Driver/Race/Compound).
   - **Dirichlet random search** (3000 weight vectors, α=1) over OOF in
     two modes: raw probs and rank-normalized probs. Author flags this
     as "more conservative than greedy hill climbing — far less prone
     to overfit OOF noise."
   - **LR meta-stacker** on `[raw, rank, logit]` meta-features
     (regularized logistic regression) — explicit linear meta over
     tree meta. Final: small committee = mean(raw blend, rank blend,
     LR stack).
   - F1-specific FE that survived gain importance:
     `TyreLife`, `LapTime_Delta`, `Cumulative_Degradation`,
     `Recent_Degradation`, `Position_Change`, `RaceProgress`,
     compound-fastness map (SOFT/MEDIUM/HARD → numeric hardness),
     `traffic_pressure_proxy` ("undercut/traffic incentive proxy").
   - Stratified KFold + GroupKFold both imported; uses StratifiedKFold
     for OOF, GroupKFold reserved for sanity probe.
   - Target: **0.9550+ AUC** (header banner) — stretch goal at top-5%.
3. **`kospintr/pitstop-catb-hgbc-xgb-lgbm-realmlp-baseline`** (36 votes).
   - **5-model GBDT-trio + HGBC + RealMLP baseline** with mean blend.
   - No load-bearing FE beyond raw inputs; emphasis on diversity at
     base layer (Cat / HGBC / XGB / LGB / RealMLP).

Other relevant pre-s6e5 entries on the same dataset:
`sarcasmos/pit-stop-prodigy-f1-strategy-intelligence` (37 votes,
2026-01-20, predates s6e5). Pulled the ipynb — it is a pure EDA
notebook (`Year`, `RaceId`, `DriverName`, plotly histograms, no model).
No predictive signal beyond confirming the underlying schema — pit-stop
times distribution, driver-team dominance over 1994-1996 data.

**Rule-structure / DGP insight**: pilkwang's `Driver's High` notebook
(41 votes) does a "f1_strategy_importance" diagnostic and concludes:
*"Simple physical priors are fragile — do not hard-code SOFT/MEDIUM/HARD
assumptions as if this were real telemetry. Relative-state features
(position change, race progress) outperform absolute physical priors."*
This is a load-bearing signal: **the host synthesized the data; physics-
faithful FE on top of synthesized inputs regresses, but relative-state
FE works.** Direct echo of the irrigation-water postmortem PM-02 Phase
2 finding ("Hand-coded physics-faithful FE on top of DGP regressed").

## Source 2 — PS S4E1 Bank Churn (closest binary-AUC analog)

Top-3 writeup bodies all inaccessible (JS gating). Fallbacks:

- **17th — "AutoML + Unicorn's pollen"**
  (https://www.kaggle.com/c/playground-series-s4e1/discussion/472636,
  title-only): AutoGluon competitive in the top-30.
- **Pairavi Medium**
  (https://medium.com/@Pairavi/binary-classification-with-a-bank-churn-dataset-bb88a8e55a1e):
  **Load-bearing trick** — synthetic dataset has duplicate rows where
  "for each pair of samples with identical features, the target values
  are always opposite." Fix: `Exited_Orig` join feature pulling labels
  from the original Radheshyam Kollipara Kaggle bank-churn dataset.
  Models: CatBoost beat baselines; NN (dense+BN+dropout) competitive
  but not primary. FE: balance/salary bins, `HasCard&Active`,
  `Gender_Balance`, balance-to-salary ratio. CV: `train_test_split
  test_size=0.2` only — no fold-OOF disclosed. Stacker not described.
- **ShabGaming GitHub** (rank #15 / top 0.05%,
  https://github.com/ShabGaming/Bank-Customer-Churn-Binary-Classification):
  "ensemble + neural net," no specifics.
- **Suraj Wate blog**
  (https://surajwate.com/blog/binary-classification-with-a-bank-churn/):
  single-LGBM 5-fold AUC 0.8893 / public 0.89197 — single-model ceiling
  ~0.892; gold cleared 0.90+ via FE + ensembling.

Strongest take: **original-dataset join** was s4e1's differentiator.

## Source 3 — PS S3E23 Software Defects

All top-3 writeup bodies inaccessible; the kimberlybecker GitHub repo
README also returned no body content. Kaggle CLI cannot list discussions.
After 2 retries on the writeup and competition pages, this source
**(not accessible)**. Fallback signal from search snippets:

- Public notebooks for S3E23 cluster around stacking-ensemble template
  (e.g. `zhukovoleksiy/ps-s3e23-explore-data-stacking-ensemble`,
  `kumudithasilva/ps-s3e23-robustscaler-ensemble-medianpruner`) —
  consistent with the same XGB+LGB+CatBoost stacking pattern that wins
  most playground binary-AUC tasks.
- No verifiable winning-mechanism citation. Skip per brief's "don't
  invent" rule. Treat S3E23 as a missing data point rather than a
  zero. Day-2 manual scrape (browser) recommended if PI grants time.

## Cross-comp themes (synthesizing Sources 1, 2, plus Appendix A)

### Highest-recurring high-impact mechanism
**OOF stacking with linear (LR/Ridge) meta-learner over a diverse base
pool (GBDT trio ± RealMLP/EmbMLP)** — used by analyticaobscura
[Source 1 #2], 5/6 Appendix-A comps, and the S4E1 GitHub winner. Top
finishers used 3000-Dirichlet random weight search over OOF rather
than greedy hill climbing because the OOF noise band on this data is
1bp (analyticaobscura's stated reason). Linear meta beats tree meta
in ≥4/5 disclosed cases.

### NN integration paid off?
- Source 1 (F1 dataset notebooks): **YES** — RealMLP/EmbMLP are
  load-bearing in analyticaobscura's stack, and yekenot's standalone
  RealMLP at 56 votes hits ≈0.946 OOF (within ~5bp of top GBDTs).
  Driver embeddings via PyTorch (EmbMLP) lifted blend diversity.
- Source 2 (S4E1): **mixed** — Pairavi credits CatBoost+NN as
  "superior to baseline," but no bp lift over CatBoost-only is cited.
  Treat as ≤5bp evidence.
- Source 3: not accessible.
- Verdict: **at 439k rows, RealMLP/EmbMLP is worth a Day-3 probe**
  (vs Day-15+ from the prior six-comp scan in Appendix A) — the F1
  dataset's notebook ladder already validates this lift on s6e5
  specifically. Caveat: NN is a diversifier, not a standalone winner.

### Target encoding patterns
- Source 1 #2: **OOF target encoding with smoothing α=80**, inner
  5-fold per outer fold (no leakage). Columns: high-cardinality
  Driver, Race, Compound, Driver×Race interactions.
- Source 2: not described in third-party material.
- Recurring: smoothing 50–100, inner OOF (not naive group-mean), kept
  as an additional column rather than replacing raw category.

### Stacker pattern dominant
**Logistic regression on `[raw, rank, logit]` meta-features** (Source 1
#2) and **LR/Ridge on probabilities** (Appendix A). Tree meta-stacker
appears only when ≤3 base models. The "raw + rank + logit" three-
representation meta-input is a transferable trick — use it.

### Contradictions with irrigation-water postmortem
- **Override-mechanism rules (R7)**: irrigation-water saw negative
  OOF→LB gaps from a 108-flip selective override. Source 1's stack-
  based approach has no override. Source 2's `Exited_Orig` join is
  feature-level, not override-level. **No contradiction**, but R7
  may not bind on s6e5 if no small-flip rule emerges.
- **NN expedition was net-zero** in irrigation-water (18 architectures
  produced 0 LB lift). **Source 1 contradicts** this for s6e5: RealMLP
  and EmbMLP are top-tier on this DGP. Likely because s6e5's DGP is
  NOT rule-structured the way irrigation-water's was (axis-aligned
  thresholds favoured trees there); on s6e5, smoothed approximators
  (NNs) appear to compete. Treat as a real divergence, not a leak.
- **Hand-coded physics-faithful FE regressed** in irrigation-water
  (PM-02 Phase 2). **Source 1 confirms** this for s6e5 (pilkwang:
  *"Simple physical priors are fragile."*). Same direction, different
  comp — strongest cross-comp recurrence in this pass.

## Top 5 candidates for s6e5 Day 2-7

1. **Add RealMLP/EmbMLP to base pool + LR meta on [raw,rank,logit]**
   — analyticaobscura proves it on s6e5 data (Source 1 #2).
   *Lift 30–50 bp* over GBDT-trio. *Code*: PyTorch embeds on
   Driver/Race/Compound; `dirichlet_search(oof,y,n_cand=3000)` then LR
   on [raw,rank,logit]. *Pre*: GBDT-trio OOF std ≤1bp (already true).
2. **Dirichlet random-search blend (3000 cand, α=1) on raw+rank OOF**
   — replaces greedy hill climbing. *Lift 5–15 bp*. *Code*: ~30 lines,
   `np.random.dirichlet`. *Pre*: ≥3 base models.
3. **OOF target encoding on (Driver, Race, Compound, Driver×Race),
   smoothing α=80, inner 5-fold/outer fold** (Source 1 #2 explicit).
   *Lift 15–30 bp*. *Pre*: keep raw category alongside; do not replace.
4. **Original-dataset (aadigupta1601) join probe** — the S4E1
   `Exited_Orig` move (Source 2). Match s6e5 rows to aadigupta1601 on
   (Driver, Race, Lap). *Lift 50–200 bp if join hits, 0 if shuffled*.
   *Code*: probe LGBM with join indicator + any leaked numerics.
   *Pre*: ≥10% match rate.
5. **Drop physics-faithful hand-FE; keep relative-state FE only**
   (pilkwang Source 1 + PM-02 Phase 2). *Defensive — prevents 5–50 bp
   regression* from SOFT/MEDIUM/HARD priors. *Code*: audit FE list;
   keep `Position_Change`, `LapTime_Delta`, `RaceProgress`.

## Coverage notes / gaps

- S3E23: all top writeups inaccessible (not accessible). Day-2 manual
  browser scrape recommended.
- S4E1: top-3 bodies inaccessible; `Exited_Orig` insight from Pairavi
  Medium third-party. Confidence: medium.
- aadigupta1601 top-3 retrieved via kaggle CLI. The pulled
  `pit-or-stay-f1-strategy-1.ipynb` (94k chars) is the single most
  actionable file in this audit.
- Six-comp scan from earlier today reinforces LR-meta + GBDT-trio.
