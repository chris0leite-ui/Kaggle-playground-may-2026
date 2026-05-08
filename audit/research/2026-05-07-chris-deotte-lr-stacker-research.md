# Chris Deotte 2nd-place LR-stacker — 7-step research synthesis (2026-05-07)

Branch `claude/ensemble-logistic-regression-research-MbLKu`. PI asked us
to study s6e4 winners' writeups (in particular Chris Deotte 2nd "Claude
Code and Codex — GPU LogReg") and plan probes. Research-only; no
compute spent.

## Sources

- 2nd `2nd-claude-codex-logreg.json` — Chris Deotte; private 0.98151;
  125-base bank, GPU PyTorch multinomial LR + class weights + L2
  (coef-only) + forward selection ~20 of 125; final = Claude-port +
  Codex-port of his March s6e3 solution.
- 1st `1st-one-vs-rest.json` — Kirill; 61 OOFs, two-stage binary,
  meta = `LogisticRegressionCV` on logits.
- 4th `4th-more-ensemblers-than-models.json` — Optimistix; 11 L1 +
  4 L2 ensemblers (HC, **LR-with-logits**, DiffEvol, Top-K); explicitly
  imported "LR-with-logits" from Chris's s6e3.
- 12th `12th-stacked-ordered-te.json` — verbatim recipe: "**Three
  axes must all be true at once**: logits not probs, `class_weight=
  'balanced'`, `multinomial` solver"; raw probs → LB 0.97900; `C` ∈
  {0.01, 0.1, 1.0} barely matters.
- Our s6e4 postmortem (`audit/2026-05-04-irrigation-water-postmortem.md`):
  226/4315 = top 5.24%, 11 ranks short of top-5%.

Convergent across 4 winners: **logits-input multinomial LR with balanced
class weights + forward selection over 60-150 diverse OOFs.**

## 1. Define

**L1**: can the Chris recipe lift our PRIMARY (LB 0.95345 = AT top-5%)
toward 0.95435 leader (−9 bp gap), without re-introducing
target/fold leakage (Rules 24/25)?

**Constraints**: PI sign-off per submit, 35/270 used, CPU-local + Kaggle
GPU, 24 days left, 150-line cap, R26 sealed-prediction protocol.

**Boundary in**: LR-meta architecture + bank composition.
**Boundary out**: multi-class target reformulation (binary AUC), full
base-model retraining (382 OOFs already exist).

## 2. Disaggregate (MECE)

```
Chris-style LR-stacker over bank-of-OOFs
├── A. Meta-input transform     [our state: P+rank+logit OK ✓]
│   A1 logits-only          A2 P+logit
├── B. LR fitting recipe        [our state: lbfgs C=1, no class_weight]
│   B1 class_weight='balanced'  B2 C-sweep  B3 saga+L1
├── C. Base subset selection    [our state: hand-curated K=21 → K=24]
│   C1 forward greedy CV (GAP)  C2 backward  C3 L1 path (done; null)
├── D. Bank size / generation   [our state: K=24]
│   D1 port public kernels      D2 multi-arch FE-fixed  D3 dual-agent
└── E. Multi-stacker top blend  [4th-place pattern]
    E1 {LR,HC,RankAvg,DiffEvol} E2 LR-meta ⊕ Path-B-hier
```

A=input, B=fit, C=subset, D=bank, E=top. MECE ✓.

## 3. Prioritise (impact × effort)

| | Easy ≤30 min CPU | Hard >1 h or GPU |
|---|---|---|
| **High impact** | **C1 fwd-sel on K=24+** ; **B1 class_weight** ; **A1 logits-only** | **D1 port 1-2 more public kernels** (yekenot precedent +19.6 bp) |
| **Low impact** | A2 / B2 (C barely matters) | D3 dual-agent ; E1 multi-stacker |

DO NOW: C1 + B1 + A1 (this branch). PLAN FOR: D1 + E2. PRUNE: A2, B2,
B3 (already null), D3, E1.

## 4. Workplan

| # | Probe | Cost | Hyp | Gate | EV bp | Family |
|---|---|---:|---|---|---:|---|
| **P1** | Forward selection K=24 LR-meta | 5 min | 24 bases include redundant signal; FS drops ≥3, +0.1-0.5 bp | OOF Δ≥+0.3 AND ρ vs PRIMARY ≤0.997 | (+0.3, +1, +3) | `lr_meta_subset_selection` (NEW variant) |
| **P2** | A1 logits-only meta-input on K=22 | 2 min | logits-only cleaner if rank+P add noise post-saturation | Δ within ±0.1 → null; Δ≥+0.2 → submit | (−0.5, 0, +0.5) | `lr_meta_input_transform` |
| **P3** | B1 `class_weight='balanced'` on K=24 | 2 min | AUC rank-based → ≈0; verify not negative | within ±0.05 bp = confirmed null | (0, 0, +0.1) | `lr_meta_fit_recipe` |
| **P4** | Composite of P1-best + B1 + A1 if helped | 5 min | additive-tier check | Rule 18 leaf claim | depends | composite |
| P5 (later) | Port `nina2025` / Mikhail Naumov / 3+ public kernels → 8-12 fresh OOFs → re-run FS | 4-8 h CPU + 1-2 h GPU/kernel | bank expansion = proven s6e4 path | per-kernel min-meta gate | (+2, +8, +20) | `bank_expansion_public_kernel_port` (NEW family) |
| P6 (later) | E2 LR-meta(K=24+) ⊕ Path-B-hier top-level | 1 h | aggregation-family diversity | top-OOF ≥ best component −0.05 AND ρ neither >0.99 | (+0.5, +2, +5) | `multi_stacker_top_blend` |

Owner: this branch P1-P4 only (no submits — Rule 1 + R26).

## 5. Analyse — what artifacts already say

`scripts/probe_min_meta.py` inspection:

- ✓ `_expand()` returns `[P, rank, logit]` → we already feed logits.
  Untested: pure-logit ablation (P2/A1).
- ✗ `LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")` — no
  `class_weight`. P3 = 1-line.
- ✗ Subset selection never run. d4 M5g/M5h2 + d4 M5h L1coef
  attempted **continuous coef-shrinkage** (NULL/TIE). C1 is **discrete
  inclusion** — different mechanism.
- 382 OOF artifacts available; many ineligible per Rule 24.

Live pool: K=24 = K=21 + d16_orig_continuous_only + p1_single_cb_v3_gpu
+ d17_h1d_yekenot_full.

## 6. Synthesise

Chris-recipe load-bearing element for s6e5 is **C1 forward selection on
a saturated bank**, not the fit recipe (we already share logits and the
4-axis recipe is mostly no-op for binary AUC). The cross-confirmed s6e4
axes reduce to **logits ✓ + class_weight ✗** — one 1-line probe with
predicted ≈0 bp.

Big asymmetry: **bank-size**. Chris had 125; we have 24. Our 6×
`lr-meta-rank-lock-strong-anchor` cross-confirmations cap single-base-add
lift at ≈0 bp on K=22+. **Forward selection cannot create signal; only
redistribute weight.** Real upside lever is **D1 bank expansion via
public-kernel porting** (yekenot precedent +19.6 bp). 4th-place built on
@yekenot, @cdeotte, @aerdem4, @siukeitin, @utaazu, @yunsuxiaozi —
we ported only yekenot.

Friction candidates if P1-P3 NULL:
- `forward-selection-cant-create-signal-only-redistribute`
- `class-weight-rank-no-op-on-binary-auc-meta`

## 7. Communicate

**Governing thought**: run P1-P3 this session (≤15 min CPU) to close
the LR-recipe gap; defer P5 (high-EV bank expansion) to next session
with PI sign-off and an ISSUES leaf.

**Order**:
1. Claim ISSUES leaf `lr_meta_subset_selection` (open if absent).
2. R26(a) sealed PI prediction for P1 LB Δ BEFORE agent reveals BOTE.
   Log via `--pi-predicted-lb-bp` to `audit/decisions.jsonl`.
3. Implement P1 in `scripts/probe_lr_fwd_select.py` (greedy CV; adapt
   `_meta_oof()` from `probe_min_meta.py`).
4. Run P1 → P2 → P3; 1-paragraph result-followup audit per probe.
5. If any PASS, defer LB submit to PI.
6. Friction-tag NULLs; promote `mechanism_families_explored` only if a
   true new family emerged (P5 = new; P1-P3 = variants).

**Devil's-advocate (R26(c))**: 6× `lr-meta-rank-lock-strong-anchor`
predict ≈0 bp on every K=22+ meta variant; FS is one more such variant.
Base rate ~17% per Rule 19 family priors; upside bounded by current
K=24 OOF 0.95385. BUT cost ≤15 min total; even 5% × +1 bp midpoint is
+EV, and the rule-out closes axis-C of the tree.

PI: please commit a sealed P1 LB Δ prediction before agent reveals BOTE.
