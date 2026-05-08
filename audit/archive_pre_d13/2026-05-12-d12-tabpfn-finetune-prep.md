# Day-12 — TabPFN-2.5 fine-tune Kaggle GPU kernel prep (Option 2)

PI directive: revisit Day-8 Tier-4 dismissal of TabPFN/Mitra. Dismissal
applied to **ICL-inference-only** (sub-sampling 440k → 10k loses statistical
power). Fine-tuning the foundation-model weights is structurally different —
pre-train inductive bias + comp-specific gradient updates. Untried.

## 1. Research — TabPFN-2.5 vs Mitra fine-tune readiness

| Dimension | TabPFN-2.5 (PriorLabs) | Mitra (Amazon/AutoGluon) |
|---|---|---|
| Released | 2025-11-06 | 2025 (AutoGluon 1.4) |
| HF | `Prior-Labs/tabpfn_2_5` | `autogluon/mitra-classifier` |
| Pip | `pip install tabpfn` (==7.1.1) | `pip install autogluon.tabular[mitra]` |
| FT API | **`tabpfn.finetuning.FinetunedTabPFNClassifier`** built into 7.x — sklearn `.fit()` | `predictor.fit(... 'fine_tune': True, 'fine_tune_steps': 10)` via AutoGluon |
| FT knobs surfaced | epochs, LR, weight_decay, val_split, patience, n_estimators_* | `fine_tune_steps` only; LR/patience hidden |
| Pre-train cap | 50k samples / 2k features (`ignore_pretraining_limits=True` extends) | not documented |
| License gate | One-time accept on ux.priorlabs.ai → `TABPFN_TOKEN` env | Apache-2.0; none |
| Params | TabPFN-2.5 backbone | 12-layer Transformer, 72M |

**Decision: TabPFN-2.5 = primary path.** Direct `FinetunedTabPFNClassifier`
exposes every knob the PI brief calls out (LR=3e-5, epochs ≤10, patience=3)
without AutoGluon-wrapper indirection. Mitra is the documented backup;
not built proactively (would lose LR/patience knobs through `time_limit`).

## 2. Local CPU smoke — BLOCKED on license accept

`scripts/d12_tabpfn_smoke_cpu.py` reached `.fit()` then crashed with
`TabPFNLicenseError`: model download requires interactive license accept
+ `TABPFN_TOKEN`. We don't have a token in this sandbox → subsample
zero-shot AUC = **null**. `oof_d12_tabpfn_smoke10k.npy` not produced.

This is **not** a TabPFN runtime failure. Package installs cleanly
(tabpfn 7.1.1 + torch 2.11.0+cu130), imports, reaches model-download.
On Kaggle a `TABPFN_TOKEN` Kaggle Secret (or one-time interactive accept
in a notebook cell) unblocks immediately.

Result: `scripts/artifacts/d12_tabpfn_smoke_cpu_results.json`
(status: `BLOCKED_LICENSE_ACCEPT`).

## 3. Kaggle GPU kernel — ready

**Path**: `kernels/d12-tabpfn-finetune-gpu/`
- `kernel-metadata.json` — `enable_gpu: true`, `GpuT4x2`, comp-source pinned.
- `d12_tabpfn_finetune.py` — script kernel (realmlp-gpu pattern).

Pinned per PI brief: `learning_rate=3e-5, epochs=10, patience=3,
val_split=0.1, n_est_finetune=2, n_est_validation=2, n_est_final=8,
n_inference_subsample=50_000, seed=42, StratifiedKFold(5, shuffle=True)`.
Cat features (Driver/Compound/Race) factorised on union of
train+val+test per fold; `ignore_pretraining_limits=True`.

Failure-mode handling:
1. nvidia-smi boot probe + cuda-availability check (RuntimeError if absent).
2. License token: `TABPFN_TOKEN` env → `kaggle_secrets.UserSecretsClient`
   → loud-but-non-fatal warning if neither (so PI sees the issue without
   burning a quota slot).
3. Per-fold partial-save of OOF + test arrays so a late-fold crash
   doesn't lose work.

## 4. GPU runtime estimate (T4×2)

| Stage | Cost |
|---|---:|
| Per-fold fine-tune (10 epochs, ~351k rows) | 30-50 min |
| Per-fold OOF inference (~88k rows, 8 estimators) | 5-10 min |
| Per-fold test inference (188k rows, 8 estimators) | 15-30 min |
| **Per-fold total** | **50-90 min** |
| **5-fold wall** | **5-7h** (under Kaggle 9h cap) |

If P100 fallback (realmlp_gpu issue): TabPFN's torch>=2.5 requirement is
already sm_60-compatible (torch 2.11 wheels). No force-reinstall needed.

## 5. EV estimate

**Standalone OOF: +5 to +15bp vs M5q's 0.95057 (target 0.951-0.952).**

- TabPFN-2.5 + FT is documented frontier on tabular AUC (TabArena-lite
  beats AutoGluon ensembles).
- Real-TabPFN-2.5 shows clear step-up over zero-shot per Tech Report.
- Only NN base in pool is RealMLP (OOF 0.94995, ρ=0.972 vs M5q). Foundation
  model + FT is a *different* inductive class than RealMLP-TD's MLP+embedding.

**ρ vs PRIMARY (d9f K=21 OOF 0.95073): expected 0.94–0.97.** Lower if
attention attends over different feature interactions; upper if FT collapses
toward consensus (RealMLP precedent).

5-question pre-flight (Rule 16):
1. Mechanism in explored families? **No** — no foundation-model FT. PASS.
2. Falls in rank-lock-vulnerable bucket? **No** — new model class, new
   paradigm. PASS.
3. Predicted standalone OOF: **0.951-0.952** (RealMLP 0.94995 + 5-15bp
   foundation-model headroom; Real-TabPFN gains ~3-7% rel AUC).
4. Predicted ρ: **0.94-0.97** (RealMLP at 0.972 = upper anchor).
5. Closest gate-PASS precedent: **RealMLP→M5q (+1.4bp OOF, +14bp LB,
   10× amplification, ρ=0.972).** Bet: similar profile at lower ρ →
   potentially larger amplification. PASS.

LB-stack EV after K=22 swap-in:
- Optimistic (ρ=0.94, +1.5bp OOF) → **+3-9bp LB** (d9f-style 6.25× amp).
- Median (ρ=0.965, +0.7bp OOF) → **+1-3bp LB**.
- Pessimistic (ρ=0.985, +0.2bp OOF) → TIE_EXPECTED, ≤+0.5bp LB.

## 6. Submission readiness

User pulls (post-run):
- `oof_d12_tabpfn_finetune_strat.npy` (n_train, 2)
- `test_d12_tabpfn_finetune_strat.npy` (n_test, 2)
- `submission_d12_tabpfn_finetune.csv` (optional standalone LB probe)
- `d12_tabpfn_finetune_results.json` (fold AUCs, walls)

**Workflow** (NOT executed here — Rule 1):
1. PI sets `TABPFN_TOKEN` Kaggle Secret.
2. `kaggle kernels push -p kernels/d12-tabpfn-finetune-gpu/`.
3. On completion: `kaggle kernels output chrisleitescha/d12-tabpfn-finetune-strat -p data/external/d12/`.
4. Local: min-meta gate vs PRIMARY (d9f K=21). Standalone target
   0.951-0.952; ρ target 0.94-0.97.
5. If both PASS, build K=22 swap-in (drop weakest L1-coef base) + K=22
   add probe; choose by min-meta lift.
6. Single-shot LB submit with PI sign-off (Rule 1).

## 7. Files produced

- `scripts/d12_tabpfn_smoke_cpu.py` — local CPU smoke (blocked).
- `scripts/artifacts/d12_tabpfn_smoke_cpu_results.json` — blocker doc.
- `kernels/d12-tabpfn-finetune-gpu/kernel-metadata.json`.
- `kernels/d12-tabpfn-finetune-gpu/d12_tabpfn_finetune.py`.
- `audit/2026-05-12-d12-tabpfn-finetune-prep.md` — this note.

No CLAUDE.md/HANDOVER.md mutations. No git commit. No Kaggle push.

End — 130 lines.
