# Prior-comp research — 2026-05-18

Triggered by: 2026-05-14 plateau (K=11 + LR-meta + Path-B stack at ~0.95386 LB,
row-feature mechanisms exhausted). R7 Research-loop run.

## Analogue competitions reviewed

1. **Porto Seguro Safe Driver Prediction** (binary, normalized-Gini ≈ AUC,
   class prior ~3.6%, large tabular, anonymised numeric features). Most
   structural analogue to S6E5: thin feature space + saturation on GBDTs.
   - URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
   - 1st (Michael Jahrer) writeup mirror: http://kaggler.com/2017/12/01/winners-solution-porto-seguro.html

2. **IEEE-CIS Fraud Detection** (binary, AUC, class prior ~3.5%, 590k rows,
   group structure via card/addr). Best documented plateau-break in Kaggle
   history; quantified per-step lift.
   - URL: https://www.kaggle.com/c/ieee-fraud-detection
   - 1st (FraudSquad / Chris Deotte) NVIDIA writeup: https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
   - Magic-feature notebook: https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600

3. **Otto Group Product Classification** (multi-class log-loss, anonymous
   tabular, 60k rows). Canonical reference for deep stacking when
   row-features are saturated.
   - URL: https://www.kaggle.com/c/otto-group-product-classification-challenge
   - 1st writeup mirror (Titericz/Semenov): https://github.com/ageek/kaggle/blob/master/2015-Kaggle/otto-group-product-classification/winners-writeup-etc/1st-place-winner-solution-gilberto-titericz-stanislav-semenov.txt

Additional recent Playground references cross-checked:
- S5E5 GPU Hill Climbing 1st (Deotte): https://www.kaggle.com/competitions/playground-series-s5e5/writeups/chris-deotte-1st-place-gpu-hill-climbing
- S5E12 Hill Climbing + Ridge 1st: https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl
- Mar-2026 Telecom Churn 1st (NVIDIA blog): 4-level stack, 150 of 850 models — https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/
- NVIDIA Grandmasters Playbook: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/

## Plateau-break mechanisms in each

**Porto Seguro — Denoising Autoencoder (DAE) representation learning.**
"All neural nets were trained on denoising autoencoder hidden activation,
which did a great job in learning a better representation of the numeric
data" (kaggler.com mirror). Swap-noise (15% of features replaced by values
from another row), trained on train+test combined, hidden activations
become the input to 5 supervised NNs; +1 LightGBM on raw. Equal-weight
blend of 6 models. Public 0.2965 → Private 0.2969 normalised-Gini. Jahrer
cited DAE as the headline plateau-breaker. NB: AV-safety check — DAE
ingests test rows, S6E5 AV-AUC ≈ 0.502 means train+test merge is safe.

**IEEE-CIS — UID magic feature + 45 groupby aggregates.**
UID = `card1_addr1 + '_' + floor(day - D1)`, then 45 mean/std aggregates
on transaction amount, timedelta, counts (NVIDIA Deotte blog). Quoted
lifts: 216-feature baseline 0.9363 → with UID aggregates 0.9472 → final
ensemble 0.9459 private. Net +109 bps from one engineered group ID.
Companion mechanism: **time-consistency feature selection** — train on
month-1 only, score on month-6; drop any feature with AUC < 0.5 there.
**Post-processing:** replace each transaction's prediction with the UID's
mean prediction (UID-level aggregation of OOF predictions).

**Otto — 3-layer stack of weak models.**
L1: 33 base models (NN bags of 120 epochs each, XGB 30x bagged, VW, GLM,
SVC/SVR, Ridge, SGD) + 8 engineered features (per-class NN-distance,
TF-IDF NN, t-SNE NN, clustering, row-non-zero count, raw X).
L2: bagged XGB (250x) + NN (600x) + AdaBoost-on-ExtraTrees (250x).
L3: `0.85·[XGB^0.65 · NN^0.35] + 0.15·[ET]`. Key insight: "we learn not
to discard low performance algorithms, since it has enough predictive
power to improve performance in a 2nd level training" (Titericz writeup).

**Recent Playground pattern (S5E5, S5E12, Mar-2026 churn).** GPU hill-
climbing over a large pool (70–500 candidates) with parallel GBDT-meta
and NN-meta, top blended by ridge/weighted-average. The Mar-2026 churn
1st: "tried Hill Climbing, Ridge/Logistic regression, NN, and GBDT
stackers." S5E5 (Deotte) explicitly named "GPU Hill Climbing" as headline.
April-2025 Podcast: L1 75 models (from 500), L2 = GBDT + NN in parallel,
L3 = weighted average (NVIDIA Grandmasters Playbook).

## Mechanisms that map cleanly to S6E5

Ranked by expected lift × cost-efficiency for our row-feature-saturated
state. "Expected lift" = OOF AUC bps; assumes our baseline ~0.954.

1. **Denoising autoencoder embeddings (swap-noise) fed to a NN meta.**
   Cost: ~2-3 GPU-hours T4x2. Expected lift band: **20–60 bps** if the
   thin 14-feature row space hides interactions the GBDT family can't
   reach. AV-AUC 0.502 means we can fit on train+test safely (R25 PASS).
   Distinct mechanism class from anything tree-based — direct replacement
   for "row-feature exhaustion."

2. **UID/group-ID magic-feature search.** S6E5 has obvious group structure
   (driver, constructor, circuit, race-year, stint). The "compound year-
   group" branch is in our ledger; IEEE-CIS shows the canonical form is
   `groupA + '_' + groupB + '_' + floor(timeC - referenceD)` then 30-50
   aggregates. Cost: <1 CPU-hour per UID candidate. Expected lift band:
   **5–40 bps** per discovered UID; expectation is one UID, not many.
   Must be R24 fold-safe.

3. **UID-level post-hoc averaging of OOF predictions.** Replace each row's
   prediction with the group's mean prediction; tested per fold (R33).
   Cost: minutes. Expected lift band: **2–15 bps** if any UID concentrates
   homogeneous targets. Pair with #2.

4. **Hill-climbing on an expanded candidate pool.** Currently K=11; the
   Otto/S5E5/Mar-2026 pattern is K=50-150 from a pool of 500+. Add seed
   replicates, hyperparameter perturbations, GPU GBDTs, TabPFN, KNN,
   Ridge into the L1 pool; Caruana ensemble-selection picks ~50; L2 is
   a GBDT-meta and an NN-meta in parallel, L3 weighted average. Cost:
   1-3 GPU days (P100 for CatBoost-GPU per R30). Expected lift band:
   **10–40 bps** if model diversity is the binding constraint (not
   features).

5. **Iterative pseudo-labelling (soft labels).** Use current K=11 + LR-
   meta to score test, fold soft probabilities back into L1 training,
   repeat 2-3 rounds. NVIDIA Playbook calls this knowledge-distillation.
   Cost: 1 retrain per round. Expected lift band: **5–25 bps** at our
   AUC level; biggest when L1 models are diverse enough to disagree on
   easy test rows. Must be R24-style fold-safe to avoid OOF leakage.

6. **Parallel NN meta-learner alongside the LR meta.** Final blend is
   `α·LR-meta + (1-α)·NN-meta`. NN meta can pick up non-linear OOF
   interactions a linear stacker misses. Cost: <1 GPU-hour. Expected
   lift band: **3–15 bps**. Cheapest sanity check before #1.

## Mechanisms that DON'T map

- **Original-dataset injection** (canonical lift on most recent Playground
  binaries). The CTGAN source for S6E5 is the public F1 dataset, but the
  public dataset already exists in our corpus — confirm against
  comp-context before assuming this is free lift. If it IS already in
  Path-B / kitchen-sink, then dead-on-arrival; otherwise +10-30 bps.
  FLAG for PI.
- **Cross-domain real-data injection** (e.g., live F1 timing scrapes).
  Out of scope for Playground rules unless the dataset is explicitly
  public and permitted; high effort, low confidence.
- **Sequence / hazard / survival framing** (RNN over stint sequences).
  Sounds tempting given pit-stop = censored event, but row-AUC metric
  doesn't reward calibrated hazard estimates; Q6 fails. The 14-feature
  row space won't sustain an LSTM. SKIP unless reframed as feature
  generator that feeds row-AUC GBDT.
- **TabPFN as base.** Useful on small data; 439k rows >> TabPFN's design
  envelope.

## Sources

- Kaggler.com Porto Seguro mirror — http://kaggler.com/2017/12/01/winners-solution-porto-seguro.html
- Chris Deotte / NVIDIA fraud blog (UID, magic, time consistency, post-proc, quoted lifts) — https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
- Otto 1st-place writeup (raw text) — https://github.com/ageek/kaggle/blob/master/2015-Kaggle/otto-group-product-classification/winners-writeup-etc/1st-place-winner-solution-gilberto-titericz-stanislav-semenov.txt
- NVIDIA Grandmasters Playbook — https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- NVIDIA Generative-AI Kaggle win (Mar-2026 churn, 4-level/150-model stack) — https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/
- Deotte XGB-fraud-with-magic notebook — https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600
- Porto Seguro 1st-place forum (header only — body behind JS) — https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
- S5E12 1st (Hill Climbing + Ridge), header verified — https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl
- S5E5 1st (GPU Hill Climbing, Deotte), header verified — https://www.kaggle.com/competitions/playground-series-s5e5/writeups/chris-deotte-1st-place-gpu-hill-climbing
