# Notebooks research — 2026-05-18

Triggered by: 2026-05-14 plateau (5 mechanism classes NULL).
Method: WebFetch + WebSearch on playground-series-s6e5 notebook ecosystem.

## Sources attempted

- https://www.kaggle.com/competitions/playground-series-s6e5/code — WebFetch returned only the page `<title>`. JS-rendered SPA; vote counts / cards / authors not present in scraped HTML.
- https://www.kaggle.com/competitions/playground-series-s6e5/code?sortBy=voteCount — same JS-render block; title only.
- https://www.kaggle.com/competitions/playground-series-s6e5/discussion — same JS-render block; title only. (Search redirect noted "Checking your browser - reCAPTCHA".)
- https://www.kaggle.com/competitions/playground-series-s6e5/overview — title only; metric and host-baseline mention not extractable.
- https://www.kaggle.com/code/devraai/f1-pit-stop-analysis-prediction (+ /notebook) — JS-rendered; title only, no body content. CANNOT VERIFY model / score.
- https://tracyrenee61.medium.com/predict-on-the-probability-a-formula-1-driver-will-pit-on-the-next-lap-e6b8adb4a45f — READ successfully. Direct hit on s6e5 (May 2026, Crystal X / Tracy Renee author).
- WebSearch site:kaggle.com — confirms /code and /discussion URLs exist for s6e5; no individual notebook titles/votes returned (reCAPTCHA-gated).
- WebSearch for "s6e5 LGBM/CatBoost/TabPFN/AutoGluon/Optuna/GroupKFold/stacking/blend" — zero direct s6e5 notebook hits across all variants. All matches are from prior playground seasons (s5e8, s6e3, etc.).

Net access status: **Kaggle competition pages are fully reCAPTCHA-gated to WebFetch.** The only public discourse artifact we can verify for s6e5 is the Tracy Renee Medium post.

## Notebooks identified

### 1. Tracy Renee / Crystal X — "Predict on the probability a Formula 1 driver will pit on the next lap"
- URL: https://tracyrenee61.medium.com/predict-on-the-probability-a-formula-1-driver-will-pit-on-the-next-lap-e6b8adb4a45f
- Author: Tracy Renee (Medium handle tracyrenee61; bylined "Crystal X")
- Date: May 2026 (Medium tag)
- Model: **CatBoost Classifier** (quote: "I defined the model as being a catboost classifier")
- CV: **None.** Uses sklearn `train_test_split` only — no KFold, no group-aware splits.
- FE: minimal. Only one-hot-encodes object-dtype columns. No engineered features named.
- LB score: author reports **94%** ("I scored 94%, which is only 1 point less than the highest score"). Author appears to be reporting accuracy/percentile, not row-AUC. Cannot map cleanly to our 0.95386 ROC-AUC. Treat as anecdotal.
- Post-processing: none (raw `predict_proba` written to submission).
- Leakage warning: none discussed; transductive risk not assessed.
- Notes: hobbyist-tier walkthrough, single-model, no CV. Below baseline-quality for our standards. No mechanism we haven't tried.

### Host baseline / starter kit
- **Not verifiable.** Kaggle overview page is JS-gated. No third-party citation of an official host starter notebook found. Playground Series convention is no official baseline notebook from Kaggle staff (community-driven). Assume none unless surfaced by a future scrape.

### Top-5 by vote count
- **Cannot enumerate.** /code listing is JS-rendered; WebFetch returns only the page title. Vote counts, author handles, and notebook titles are not in the scraped HTML. WebSearch returned zero direct s6e5 notebook URLs (all hits were prior seasons or unrelated F1 datasets).

## Synthesis: what's in public discourse

- **Public-notebook signal is effectively dark for us right now.** The only verifiable s6e5 artifact is one hobbyist Medium post on CatBoost + train_test_split. There's no public stacking / blending / TabPFN / AutoGluon writeup we can read.
- **Gap vs our PRIMARY:** Our K=11 + K=9 rank-blend stack with LR-meta + Path-B per-segment shrinkage is vastly ahead of any verifiable public approach. No public reference shows GroupKFold or any group-aware CV; the one readable artifact uses none at all. Public discourse is not where the leader's +9.0 bp lift is coming from.
- **Leakage warning chains:** zero public discussion verifiable. Our R24 fold-safe groupby and R25 AV-AUC (0.502) checks remain the gating discipline — public notebooks aren't a source of new red flags.
- **No host baseline confirmed.** Pre-baseline-gate item 8 (R22 public-notebook scan) outcome for plateau-day-2: **null harvest.** This is a known JS/reCAPTCHA block, not a research gap on our side.

## Untried mechanism candidates from this scan

- **None high-confidence.** The Crystal X article surfaces nothing we haven't tried at higher fidelity. CatBoost single-model with no CV is strictly dominated by our K=11 stack (which includes CatBoost as a base learner).
- **Speculative (no evidence base):** if a future scrape surfaces a high-vote TabPFN or AutoGluon notebook, those would be candidate axes; both are absent from current visible discourse.
- **Operational follow-up (not a mechanism):** consider Kaggle-API-authenticated notebook listing (`kaggle kernels list -c playground-series-s6e5 --sort-by voteCount`) for the next plateau scan — bypasses the reCAPTCHA gate. Cost: ~2 min CLI. Predicted lift: 0 bp directly; enables real R22 scan that today is dark.
