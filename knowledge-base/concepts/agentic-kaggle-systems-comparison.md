# 2026-05-06 — Agentic Kaggle research synthesis

> Branch `claude/research-agentic-kaggle-W6IAP`. Comparative survey
> of agentic ML systems (AIDE, MLE-STAR, RD-Agent, AI Scientist v2,
> Agent K, NVIDIA-GM workflows) + MLE-Bench failure-mode taxonomy
> + 2025-2026 HITL ergonomics literature. Mapped against our s6e5
> system. ≤150-line cap.

## TL;DR

Our system is **stronger than published academic agents on
calibration discipline** (OOF→LB amplification priors per family,
4-gate filter, 5-question pre-flight, submission-as-probe) but
**weaker on three dimensions** the field has converged on: (a)
web-grounded seeding (MLE-STAR), (b) BFTS / formal backtracking
(AI-Scientist-v2), (c) persistent hypothesis graph with cross-branch
sampling (RD-Agent). The PI loop is in better shape than any
benchmark agent's — keep it; the harness needs three small additions.

## What others do — at a glance

| System | Search | Calibration | Sub. discipline | MLE-bench medal % |
|---|---|---|---|---|
| AIDE (Weco) | greedy hill-climb tree | none | offline | 16.9% (o1-pre) |
| MLE-STAR (Google '25) | outer ablation + inner refine + web-retrieval seed + leakage-checker agent | none (web seed substitutes) | offline | **64%** |
| RD-Agent (MS) | persistent R/D hypothesis graph w/ topical+score sampling | val/LB target | offline | top of MLE-bench |
| AI Scientist v2 (Sakana) | BFTS (parallel workers + backtrack) | none | offline | n/a (paper-grade) |
| Agent K (Huawei/UCL) | nested memory + Bayesian HP opt | none | offline (rerun) | claimed; contested |
| Deotte / NVIDIA GMs | linear pipeline → 4-level stack | save-everything | live, daily-cap | 1st on s6e3 |
| **Ours (s6e5)** | linear hypothesis board + parallel branches | **per-family priors + 4-gate + R7-flip + min-meta** | **PI-approved single-shot** | live, 28bp from top-5% |

Sources: arXiv 2502.13138, 2506.15692, 2505.14738, 2504.08066,
2411.03562, 2410.07095, 2503.13657, 2510.10472. NVIDIA dev blog
(Deotte / Aché / GM-Playbook). Full URLs in subagent transcripts.

## Where we lead the field

1. **OOF→LB amplification priors per mechanism family.** Empirically
   measured (FM-class 5.7-6.7×, Path-B 6.7-11.6×, base-add 1.4×).
   No published agent has this; FML-bench's "broader-is-better"
   finding (arXiv 2510.10472) implicitly endorses tracking it.
2. **Pre-flight 5-question check (Rule 16).** None of the agents
   score candidates against a family register before compute. AIDE
   / AI-Scientist will re-run a saturated family because they
   don't track it.
3. **Submission as Bayesian update (Rule 12).** Every agent system
   above runs offline. Live PI-approved single-shot submits are the
   single biggest signal-quality lever we have.
4. **Friction log + tags as first-class memory.** Maps to RD-Agent's
   knowledge-base but is hand-curated and shorter — better fit for
   one comp.

## Where we lag — three small harness additions worth shipping

1. **Web-retrieval seed for new mechanism families.** MLE-STAR's +47bp
   over AIDE comes mostly from this. We do this on plateau (Rule 7);
   make it cheap to do at family-kickoff too. ~30 min: a
   `scripts/research_seed.py` that takes a family name and returns 5
   prior-comp citations + canonical recipes.
2. **Formal backtracking for parallel branches.** Day-15's 4-branch
   probe was hand-orchestrated. AI-Scientist-v2's BFTS (parallel
   workers + LLM-judged node selection) generalises this. ~1 day:
   wrap `scripts/probe.py` with a small BFTS loop reading from
   ISSUES.md leaves.
3. **Persistent scored hypothesis graph.** Our `mechanism_families_
   explored` is a flat list. RD-Agent's topical-similarity × score
   sampling kernel is a better structure for plateau-breaking.
   ~2 hours: convert to JSON `{family: {score_delta, ρ_band, audit_ptr}}`.

Plus one defensive add (from MAST FM-7 metric-misalignment, our d12
LambdaRank -86bp confirms it):
4. **"Metric matches objective" as Q6 in 5-question check.** Free.

## What NOT to do — anti-patterns from MLE-bench / MAST / Sakana

Mapped to our existing defenses; gaps flagged.

| # | Anti-pattern | Citation | Our defense | Gap |
|---|---|---|---|---|
| F1 | Public-LB chasing / shake-up | MLE-bench paper | R1 two-anchor, R2 HEDGE/PRIMARY | track OOF→LB realised gap per family |
| F2 | Hallucinated leakage in TE/aggregations | AIDE 2502.13138 | d10/d13d GKF probes, 4-gate | static check on any path touching target |
| F3 | Cargo-cult FE | AIDE Fig 6 | Rule 16; FAMILY_PRIORS; 5 NULLs closed FE | none |
| F4 | Optuna spam / tuning theatre | AutoGPT loops; AIDE complexity↑ | Rule 6 heuristics-first; BOTE | none |
| F5 | Saturation blindness | AutoGPT #1994; AIDE | Rule 7 research; Rule 14 critic | hard-fire research at plateau≥2 |
| F6 | Premature ensembling on redundant bases | NVIDIA GM Playbook | ρ-test + min-meta probe | none |
| F7 | Metric misalignment | MAST FM-2.6 (13.2%) | falsified by d12 | add to 5-Q pre-flight |
| F8 | Silent compute blowup | LangChain $47k loop | Rule 2 caps, Rule 13 | none |
| F9 | Hallucinated numbers in writeup | Sakana 42% fail; 100+ fake citations NeurIPS '25 | `probe.py gate` uniform report | every bp must trace to artifact path |
| F10 | Premature termination / format errors | MAST FM-3 (23.5%) | smoke + 1-fold probe | none |
| F11 | Spec gaming (surface goal satisfied) | MAST FM-1.1 | predicted-vs-actual in 5-Q | none |
| F12 | Context loss / re-trying falsified | MAST FM-1.4 | mechanism_families_explored | append-only per branch slug |
| F13 | Inter-agent misalignment | MAST FC2 (32.4%) | ISSUES.md claim board | enforce one-open-leaf-per-branch hard |
| F14 | Benchmark gaming (Agent-K critique) | analyticsindiamag | track headroom_to_top5pct | none |

## Tips for the PI (synthesized from Karpathy / Litt / Fowler / Anthropic / Osmani)

1. **Predict before the agent runs.** Add a column to the calibration
   ladder: PI-predicted LB Δ next to agent-predicted next to actual.
   After 10 submits you'll know which of you is better calibrated on
   which family. Single highest-leverage cognitive-ergonomics move.
2. **Read one thing per session the agent didn't summarize for you.**
   Top finisher writeup, one paper, one notebook. Otherwise you
   atrophy (Osmani arXiv 2604.03501; Karpathy '25 year-in-review).
3. **One probe per week hand-run, no agent.** Litt's "code like a
   surgeon" — keeps your reflexes sharp. Suggested: a min-meta
   gate computation by hand.
4. **Track your override rate.** If you've approved 28/28 submits
   without ever overriding, you're a stamp not a PI. Mitigation:
   write your own EV before reading the agent's; if they disagree,
   investigate.
5. **Bound the agent's autonomy explicitly.** Rule 19's "PI corollary"
   (no calendar/today/tomorrow framing from agents) is the right
   shape — extend to: agents do not propose final submissions, do not
   re-decompose ISSUES.md without a Rule-14 trigger, do not edit
   `comp-context.md` after Day 1.
6. **Resist real-time stream-watching.** Queue 3-5 probes, batch-review
   audit notes in one sitting. Watching is dopamine, not signal.
7. **CLAUDE.md compression target ≤300 lines.** We're at 50k tokens.
   Compress `mechanism_families_explored` to one-line entries with
   audit-note pointers; archive ladder rows older than 7 days.

## Concrete process changes ranked by EV/cost

1. (free) Add Q6 "metric matches objective?" to 5-Q pre-flight.
2. (free) Add "PI-predicted LB Δ" column to calibration ladder.
3. (~30 min) `scripts/research_seed.py` — web-retrieval at family kickoff.
4. (~2 h) Convert `mechanism_families_explored` to scored JSON graph.
5. (~1 day) BFTS wrapper around `probe.py` for ISSUES.md leaves.
6. (~ongoing) Compress CLAUDE.md to ≤300 lines, archive ladder.

## Pointers

- Existing-systems detail: subagent transcript a4c625600c4e90188.
- Failure-mode taxonomy: subagent transcript a18d579ab69ff4dd4.
- HITL ergonomics: subagent transcript a2bd670910b783a3e.
- Key reads (do this week):
  - Litt — Code like a surgeon: https://www.geoffreylitt.com/2025/10/24/code-like-a-surgeon
  - Fowler — Harness engineering: https://martinfowler.com/articles/harness-engineering.html
  - Willison — Designing agentic loops: https://simonwillison.net/2025/Sep/30/designing-agentic-loops/
  - MLE-STAR paper: https://arxiv.org/abs/2506.15692
  - MAST taxonomy: https://arxiv.org/html/2503.13657v3
  - NVIDIA GM Playbook: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
