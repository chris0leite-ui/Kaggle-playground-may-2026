# Archive: HANDOVER.md Day-8 PM section (research-feature-engineering-7oCmj)

Moved here on 2026-05-09 during Day-9 PM `decode-data-process-5uLq3`
wrap to keep HANDOVER.md ≤ 150 lines per WRAPUP.md step 5.

---

## Day-8 PM research-feature-engineering-7oCmj

EXP-NEW Phase 1-5b FE/meta campaign (ISSUES leaf 11) closed `null`.
PRIMARY unchanged @ LB 0.95351; 0 of 270 submission slots used.

| Probe | OOF Δ vs PRIMARY 0.95403 | Verdict |
|---|---:|---|
| A3-7 UID smoothing dry-run | −124 bp | leakage FAIL |
| 6 of 7 Phase-1 smoke picks | null/regress | FAIL |
| A2-2 mandatory_compound_rule smoke | +9.3 bp | smoke-only |
| A2-2 single-LGBM 5-fold | +1.4 bp standalone | partial absorb |
| A2-2 K=4+1 plain LR-meta | +0.302 bp; TIE_EXPECTED | < +0.5 PASS |
| A2-2 K=4+1 Path-B C×S τ=100k | +0.26 bp; ρ 0.999893 | WEAK |
| A2-8 LightGBM stack-meta on K=4 | −1.30 bp | FAIL |

**Rule 7 research scan** (Frontiers AI 2025 Bi-LSTM, Optimum Racing
IJRASET 2025): 4 of 5 mechanisms duplicate A3-1 RankSortedGaps
(already null). Only genuinely untried lever: Bi-LSTM/GRU sequence
base on 10-lap windows, ~30-60 min Kaggle T4, deferred.

**Next-session first actions.** (1) Bi-LSTM/GRU sequence base on
K=4 — cheapest untried lever, +0.5 bp gate. (2) R5 hedge prep
(~30 min CPU). (3) Acceptance posture — −12.5 bp gap to leader
inside sample-noise band; private-LB ceiling may be near.
Artifacts: `scripts/probe_a2_2_pathb_K4.py`,
`scripts/probe_a2_8_stack_meta.py`, postmortem
`audit/2026-05-08-postmortem-research-feature-engineering-7oCmj.md`.
