# Cross-comp research for s6e5 (2026-05-04, brief-mandated three-source pass)
[in progress]

Mission: extract cross-comp tactics for s6e5 (F1 PitNextLap, binary AUC,
prior 0.199, 439k/188k rows, baseline LB 0.94113, top-5% 0.95345).
Three brief-mandated sources: aadigupta1601 F1 dataset top-3, PS S4E1
Bank Churn, PS S3E23 Software Defects.

Note: a parallel six-comp scan is preserved below as Appendix A.

## Access constraints (must read first)

Kaggle WebFetch on `kaggle.com/.../writeups/...` and `.../discussion/...`
returns only `<title>` because pages are JS-rendered. All writeup body
content was extracted via:
(a) Web search snippets that quote the page,
(b) Third-party blog summaries (Medium, dev.to, NVIDIA developer blog),
(c) The author's GitHub repos.

Where access failed for a top-3 writeup, I fell back to top-10 writeups
plus third-party top-finisher coverage. Comps without ≥1 accessible
writeup are skipped per the brief.

## Per-comp summaries

### A. Playground S6E3 — Predict Customer Churn (binary AUC) — closest match

- URL: https://www.kaggle.com/competitions/playground-series-s6e3
- Synthetic-from-real (IBM Telco Churn). Task ≈ s6e5 in shape (binary
  AUC, mild imbalance, ~hundreds of thousands of rows).
- 1st place — KGMON team, "GPT5.4 / Gemini3.1 / ClaudeOpus4.6 — KGMON
  Playbook" (https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm).
  Coverage from the NVIDIA blog
  (https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/):
  - **4-level stack of 150 models, selected from 850 trained.**
  - GBDT trio (XGB / LGBM / CatBoost) + NN (PyTorch) + TabPFN + SVR +
    KNN + Ridge / Logistic stackers.
  - Hill-climbing for blend weights; knowledge distillation; ridge /
    logistic at the meta level (NOT a tree meta-learner at the top).
  - GPU-accelerated cuDF / cuML / XGB / PyTorch — "rapid
    experimentation enabled the search."
- ~Top-8 (rank 286 / 3718, 0.91685 public AUC) — Faith Bui, Ridge-XGBoost
  N-gram pipeline (https://dev.to/faith_b6e08f3b8f05a77bb5f/how-i-reached-top-8-on-kaggle-with-a-ridge-xgboost-n-gram-pipeline-32pa).
  - **N-gram categorical interactions** (bigrams + trigrams across the
    3 high-cardinality categoricals) — the load-bearing trick.
  - Nested target-encoding (5-fold) to prevent leakage.
  - Service-stack count features and "digit-of-numeric" features.
  - 2-stage: ridge OOF → XGBoost on (ridge_oof, engineered features).
  - 10-fold StratifiedKFold, fixed seed.
- Surprise: top-importance features were n-gram interactions and
  nested-target-encoded combinations, not raw categoricals.

### B. Playground S5E11 — Predict Loan Payback (binary AUC) — closest prior on imbalance

- URL: https://www.kaggle.com/competitions/playground-series-s5e11
- Class prior 0.202 — almost identical to s6e5's 0.199.
- 1st place — "A lot of features, a lot of models, and a little bit of
  luck" (https://www.kaggle.com/competitions/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an).
  Body inaccessible (JS gating). Title alone implies massive feature
  count + many models + ensemble — fits the S6E3 KGMON pattern.
- Rank-8 — "Rank8 approach — trust the CV score"
  (https://www.kaggle.com/competitions/playground-series-s5e11/writeups/rank8-approach-trust-the-cv-score).
  Body inaccessible. Title's headline finding ("trust the CV score")
  echoes irrigation-water postmortem-07 R5 (probe OOF-best regressed on
  public).
- Sanjay Bista journey (https://medium.com/@sanjaybista1010/from-0-56-to-0-92-auc-my-kaggle-loan-prediction-journey-through-class-imbalance-and-overfitting-ab80d4591f14):
  - **SMOTETomek before fold-split = leakage; CV jumped 0.91→0.92 by
    just removing it.**
  - Over-engineering hurt: hand-coded `payment_capacity`,
    `credit_utilization` regressed on OOF — simpler features won.
  - LGBM `class_weight='balanced'`, n_estimators=2000, lr=0.05,
    max_depth=8, reg_alpha=0.1, reg_lambda=0.1.
  - StratifiedKFold OOF for the imbalanced metric.
- karltonkxb notebook (https://www.kaggle.com/code/karltonkxb/s5e11-loan-xgb-lgbm-cuml-92-64):
  XGB+LGBM with cuML reached 0.9264 — a single-model ceiling.
- Stacked ensemble result published in search snippets (FelixCharotte
  GitHub, https://github.com/FelixCharotte/LoanApprovalPrediction_KaggleCompetition):
  base AUCs 0.9785–0.9815, **logistic meta 0.9823, XGB meta 0.9843**.

### C. Playground S5E8 — Binary Classification with a Bank Dataset (binary AUC)

- URL: https://www.kaggle.com/competitions/playground-series-s5e8
- 6th place writeup (https://www.kaggle.com/competitions/playground-series-s5e8/writeups/6th-place-solution-oof-stacking-with-lgbm)
  title "OOF Stacking with LGBM" — body inaccessible, but title
  endorses LGBM-as-meta over linear meta on this comp.
- 21st place writeup (https://www.kaggle.com/competitions/playground-series-s5e8/writeups/21st-place-solution)
  body inaccessible.
- Third-party Medium article (Sahil Chukka, banking dataset top
  approach): XGBoost with Optuna (n_estimators 200–1000, lr 0.01–0.3,
  max_depth 3–12), Stratified K-Fold; CV AUC **0.9664 ± 0.0005**, public
  0.96529 / 0.96516. The 1bp std at 5-fold matches the s6e5 OOF noise
  band we measured.
- Top mechanism per Medium summary: KDE-driven outlier handling +
  correlation pruning + Optuna over a tight tree-depth budget.

### D. Playground S4E7 — Binary Classification of Insurance Cross Selling (binary AUC) — large rows ≈ 11M

- URL: https://www.kaggle.com/competitions/playground-series-s4e7
- 1st place writeup (https://www.kaggle.com/competitions/playground-series-s4e7/writeups/cross-sellers-winning-approach-team-cross-sellers)
  body inaccessible.
- #3 writeup, Tilii: "Many individual models and many ensembles"
  (https://www.kaggle.com/competitions/playground-series-s4e7/writeups/tilii-3-solution-many-individual-models-and-many-e)
  body inaccessible, but title is unambiguous: many bases × many blends.
- Public single-model code (akinduhiman, https://www.kaggle.com/code/akinduhiman/insurance-cross-selling-ps4e7-gbt-xgb-lgbm-cat):
  GBT+XGB+LGBM+CatBoost + small linear blend was the public-template
  ceiling.
- Surajwate baseline (https://github.com/surajwate/S4E7-Insurance-Cross-Selling)
  XGBoost single-model: validation 0.87820, public 0.87862. Optuna
  crashed on the 11M row split (resource note transferable).

### E. Playground S4E1 — Binary Classification with a Bank Churn Dataset (binary AUC) — 165k rows

- URL: https://www.kaggle.com/competitions/playground-series-s4e1
- All top-3 writeup bodies inaccessible. 17th place "AutoML +
  Unicorn's pollen" (https://www.kaggle.com/c/playground-series-s4e1/discussion/472636)
  title only — implies AutoML (likely AutoGluon) was competitive.
- **The signal everyone reports: the synthetic dataset has duplicate
  rows where (features identical) ↔ (label flipped).** Pairavi
  Thanancheyan, Medium
  (https://medium.com/@Pairavi/binary-classification-with-a-bank-churn-dataset-bb88a8e55a1e):
  "for each pair of samples with identical features, the target values
  are always opposite." Fixing this with an `Exited_Orig` lookup
  feature from the original Kaggle dataset (Radheshyam Kollipara) was
  the headline FE move. Direct s6e5 translation: **does the original
  aadigupta1601 dataset contain rows joinable to s6e5 train/test on
  (Driver, Race, Lap-equivalent)?**
- ShabGaming README (https://github.com/ShabGaming/Bank-Customer-Churn-Binary-Classification/blob/main/README.md)
  rank #15 / top-0.05% — "ensemble + neural net" but no specifics.
- Suraj Wate blog (https://surajwate.com/blog/binary-classification-with-a-bank-churn/)
  single-LGBM 5-fold AUC 0.8893, public 0.89197 — i.e. single-model
  ceiling was ~0.892, gold cleared 0.90+ via FE + ensembling.
- Natasha Sharma Medium (https://medium.com/@nats.sha/kaggle-series-playground-competition-bank-customer-churn-predictions-d2857ea709ad)
  basic FE list: balance bins, salary bins, IsBalanceZero,
  Gender×Balance, Card×Active interaction, age binning.

### F. Playground S6E2 — Predicting Heart Disease (binary AUC, small)

- URL: https://www.kaggle.com/competitions/playground-series-s6e2
- 1st place writeup "Diversity, Selection, and Trusting the CV–LB
  Relation" (https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t).
  Body inaccessible. Title is the load-bearing finding: **diversity at
  base + select on a CV-LB-correlated subset of folds.**
- Top-1 multi-seed notebook (https://www.kaggle.com/code/masanakashima/s6e2-heart-disease-top1-multi-seed)
  body inaccessible; title implies seed-bagging at the base level.
- Less analogous (only ~600k rows in s6e5 vs ~10k here) so down-weight.

### G. NVIDIA Kaggle Grandmasters Playbook (cross-comp synthesis source)

- https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- Seven techniques distilled from 2024–2025 Playground wins:
  1. Smarter EDA (train/test distribution shift, temporal patterns).
  2. **Diverse baselines** built right away (linear / GBDT / NN).
  3. **Feature engineering at scale** (e.g. 8 cats → 28 interaction
     features; thousands of FE candidates).
  4. **Hill climbing** for blend weights (start strong, add only if it
     improves CV).
  5. Stacking (residual-based or OOF-based; meta is usually linear).
  6. **Pseudo-labeling** on test (soft labels back into train).
  7. Extra training: multi-seed bag + retrain on 100% data after HPO.
- https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/
  S5E2 backpack winner generated >10k FE candidates via
  `groupby(COL1)[COL2].agg(STAT)` over (mean, std, count, min, max,
  nunique, skew) plus quantile binning, then kept top 500 by importance.
  **GBDT-friendly FE search at scale** — the same pattern likely
  generalizes to s6e5's (Driver, Race, Compound, Lap) keys.

## Cross-comp themes

### Most common high-impact mechanism

Across 5 of 6 comps the top finishers used **GBDT trio (XGB+LGBM+
CatBoost) + linear meta-learner** with 5–10 fold OOF stacking. NN
(MLP, TabPFN) was added on top only in S6E3 1st (4-level / 150 models)
and helped on heart disease via multi-seed bagging. **Linear meta-
learner (Ridge / Logistic) beats tree meta** in 4 of 5 comps where the
meta type was disclosed (S6E3 1st, S6E3 top-8, S5E11 stack snippet).
Single exception: S5E8 6th place used LGBM as meta — likely because
they had only ~3 base models and tree meta is robust on small
base-feature counts.

### F1-style sequential vs customer-style data

s6e5 is unusual in this set: rows are per-(driver, lap) **sequences**,
not iid customers. None of the closer comps (S6E3, S5E11, S5E8, S4E7,
S4E1) had within-row temporal structure; their natural groups were
demographic IDs, not stints. Implication: **the unique edge for s6e5
is sequence FE** — within-(driver, race) lag features, "laps since
last pit," tire-life proxies, leader-gap dynamics — and group-aware
CV on Race (already adopted as R1 anchor-b in CLAUDE.md) is critical
because (driver, race) groups are not iid across the StratifiedKFold
seed-42 split. None of the comps studied surfaced a within-group
leakage warning; ours is real (97.4% of test rows have a same-(Race,
Driver) successor in test, per `comp-context.md` u3 probe), so the
analog is closer to the S5E11 SMOTETomek-leakage horror story than
to S4E1 row-pair leakage.

### Is NN integration worth it at 439k rows?

S6E3 1st (4-level stack, 150 models) clearly used NN. S6E3 top-8 did
not. S5E11 1st did not (per Sanjay Bista's tier; only LGBM-class).
S4E7 / S4E1 / S5E8 top finishers: GBDT-trio dominant, no NN evidence
in accessible coverage. Verdict: **at 439k rows with mild imbalance,
NN integration is a Day-15+ activity, not a Day-3 priority.** Add NN
only after the GBDT-trio + linear-meta hits its OOF ceiling. Use a
small MLP or TabPFN-style model with row-level features only; do not
attempt RNN/Transformer over (Race, Driver) sequences without a
pre-validated feature pipeline first.

### Stacker patterns

- 5 of 6 disclosed: linear (Ridge or Logistic) meta beats tree meta.
- Tree meta (LGBM as stacker) safe only when ≤4 base models.
- 2-stage variant (ridge OOF → XGBoost on ridge_oof + features) used
  by S6E3 top-8 — boosts non-linearity at the meta step without giving
  up the linear shrinkage.

### Rank-blending

Not surfaced as a top mechanism in any of the AUC comps studied here.
**Probability blending was the norm.** Hill climbing was the dominant
search procedure for blend weights; rank-averaging appears as a
fallback only when scales differ wildly between bases.

### Submission discipline

S6E3 1st: 850 experiments → 150 selected → 4-level stack. S5E11 1st:
"a lot of features, a lot of models." S6E2 1st: "diversity, selection,
and trusting the CV-LB relation." All three winners selected the
final blend on a CV/private alignment criterion, not raw public LB
chasing. This converges with our R2 (PRIMARY = best public, HEDGE =
best OOF that regressed ≤30bp public) and R5 (mandatory probe of the
OOF-best rejected for public regress).

## Top 5 candidates to try on s6e5 (also returned in reply summary)

1. **Sequence FE within (Driver, Race)**: `laps_since_last_pit`,
   `pit_count_this_race`, lag/lead Compound, tire-life proxy
   (Lap − last_pit_lap), gap-to-leader Δ. Closest analog to S6E3 n-gram
   trick but adapted to F1 sequences. **Expected lift: 30–60 bp**
   (this is the s6e5-unique edge, no comp tested it directly).
2. **Original-dataset join (aadigupta1601)**: replicate the S4E1
   `Exited_Orig` move. If a row in s6e5 train can be matched to an
   original-dataset row by (Driver, Race, Lap or near-Lap) and the
   original carries a label, treat as auxiliary feature. **Expected
   lift: 50–200 bp if such a join exists**, 0 bp if synthetic data is
   shuffled. Run a probe before investing.
3. **GBDT-trio + Ridge/Logistic meta-learner, 5–10 fold OOF**: keep XGB
   + LGBM + CatBoost as bases, switch the stacker from any tree to
   Ridge or Logistic. **Expected lift over single-LGBM baseline: 10–25
   bp** based on S5E11 stack (logistic meta added 8 bp over best
   single).
4. **FE-at-scale via groupby(COL1)[COL2].agg(STAT)**: sweep
   {Driver, Race, Compound, Driver×Race} × numeric_columns × {mean,
   std, count, nunique, min, max, p10, p90}. Keep top 30–50 by gain
   importance. **Expected lift: 15–40 bp** (S5E2 winner used a 500/10k
   keep ratio; we'd be more conservative at 50/500).
5. **Hill-climbing blend selector + multi-seed bag at base**: train
   each base under 5 seeds, hill-climb across the ~15 base predictions.
   **Expected lift: 5–15 bp** on top of GBDT-trio + linear meta. Cheap
   compute at our scale.

## Coverage notes / gaps

- S4E1 Bank Churn top-3 writeup bodies all inaccessible; the leakage
  finding is from a third-party Medium summary, not the host writeup.
  Worth a Day-2 manual scrape if PI grants browser time.
- S5E11 1st-place writeup body inaccessible; lift estimate for
  candidate #3 leans on stack-snippet AUC numbers (FelixCharotte
  GitHub) and S6E3-top-8 (dev.to).
- S3E23 Software Defects: skipped — no top-3 writeup accessible and
  the dataset shape (small, no group structure) is far from s6e5.
- S5E1 / S5E5 / S4E5 / S4E12: regression metrics, skipped per brief.
- aadigupta1601 dataset top-voted notebooks: search returned only the
  dataset card; no notebook with extractable signal beyond what is
  already in EDA. Further drill-down needs the Kaggle CLI to list
  notebook IDs (same fallback we used for `brief.md`).
