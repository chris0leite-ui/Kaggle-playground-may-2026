# Postmortem — 2026-06-02 research-improvements-jjI84 (private reveal)

Competition closed 2026-05-31 23:59 UTC; private scores revealed.
This session: private-score read + closing postmortem only. No
compute, no submissions.

## What went wrong

Decision-quality lens. Given priors at decision-time:

- **W2 PRIMARY pick (P=0.15)** — defensible. Public LB rose
  monotonically through P=0.15 then sat flat across P=0.15/0.20/0.25
  at 0.95457, and dipped to 0.95456 at P=0.30 and P=0.35. The public
  plateau provided NO signal that private would continue lifting to
  0.95466 at P=0.35. Conservative center-of-plateau pick was rational.
  Good decision, sub-optimal outcome by ~5–6 bp. Top-5% achieved
  either way.

- **W2_R31 HEDGE pick** — VALIDATED on private. Hedge logic targeted
  the C5-specific-private-bias failure mode by swapping the ours-backbone
  (K=20+R21+K=27 → K=18+R30+R7.2). On public, both PRIMARY and HEDGE
  TIED at 0.95457; on private, HEDGE beat PRIMARY by +1 bp (0.95461 vs
  0.95460). Kaggle scored on best-of-selected → official = 0.95461.
  Good decision, good outcome.

- **L1 logit was the lone wrong-direction private surprise.** L1
  (logit-space V8 with 0.9A+0.1P then expit) tied W2 on public at
  0.95457 but came in at 0.95456 private — 4 bp below W2 on private
  despite the public TIE. The only TIE-band candidate that broke
  *down* on private. Calibration data point: non-linear rank-curve
  manipulations at the public TIE band can move either direction on
  private and add no expected lift. Consistent with Day-30 friction
  `curvature-blends-no-lift-on-AUC-plateau` already in mechanism-ledger.

- **Day-31 V0 re-fire (slot 10)** — already flagged in yesterday's
  postmortem as wasted slot. Private confirmed −9 bp from PRIMARY
  (0.95452 vs 0.95461). No new info from the private reveal; flag stands.

**PI overrides this session:** None this session (read-only).
**Rule-bypass failures this session:** None applicable (no compute).
**Rule-gap failures (candidate, PI declined promote):**
- `public-plateau-hides-monotonic-private-trend` — when ≥3
  same-family candidates share the public-LB max AND form a monotonic
  trend in a single hyperparameter, the public TIE band may be
  structurally bandwidth-limited and not reflect private ordering.
  Candidate rule: among public-TIE candidates on a monotone curve,
  prefer the EXTREME of the trend direction for PRIMARY, not the
  conservative middle — unless public has turned down at both extrema.
  PI: declined promotion ("nothing to add or to promote").

## Frictions logged this session

Cross-link to `audit/friction.md` 2026-06-02 block (appended this
session): `private-reveal-W-curve-monotonic`, `L1-logit-private-down`.

## Promotion candidates (PI ratified: NO promote)

- `public-plateau-hides-monotonic-private-trend` — drafted above,
  PI declined. Remains documented here for cross-comp reference;
  NOT propagated to `.claude/skills/kaggle-comp/improvements.md`.

## PI additions (from step 4)

None. PI: "Nothing to add or to promote."

## Final standing (R8d)

- **Private rank: 148 / 3023 → top 4.90%** (cleared the top-5% target)
- **PRIMARY W2 private:** 0.95460 (rank-equivalent ~165, est)
- **HEDGE W2_R31 private:** 0.95461 (Kaggle official → rank 148)
- **Best of all submitted (not selected):** W6 (P=0.35) and E3 PMEAN
  q=3 both at 0.95466 — would have placed ~top-2.5%
- **Public-vs-private headroom forfeited:** ~5–6 bp by picking
  plateau-center over plateau-extreme

LB ladder (selected + key reference):

| Submit | Recipe | Public | Private | Δ vs W2-priv |
|---|---|---|---|---|
| V0 | raw raunakdey | 0.95453 | 0.95452 | −8 |
| V8 | 0.90 raun + 0.10 C5 | 0.95456 | 0.95458 | −2 |
| **W2 (PRIMARY)** | 0.85 raun + 0.15 C5 | 0.95457 | 0.95460 | — |
| **W2_R31 (HEDGE)** | 0.85 raun + 0.15 R31 | 0.95457 | 0.95461 | +1 |
| W4 (0.25 C5) | 0.75 raun + 0.25 C5 | 0.95457 | 0.95464 | +4 |
| W4_R31 | 0.75 raun + 0.25 R31 | 0.95457 | 0.95464 | +4 |
| W5 (0.30 C5) | 0.70 raun + 0.30 C5 | 0.95457 | 0.95465 | +5 |
| **W6 (0.35 C5)** | 0.65 raun + 0.35 C5 | 0.95456 | **0.95466** | **+6** |
| E3 PMEAN q=3 | 0.90 raun + 0.10 C5 (cubic) | 0.95456 | 0.95466 | +6 |
| L1 logit | 0.90 raun + 0.10 C5 (logit-space) | 0.95457 | 0.95456 | −4 |

## Framework version at session-end

- Commit SHA: <set at commit time>
- Branch: `claude/research-improvements-jjI84`
- Active rules: R0–R36 + R1d–R8d defaults (per CLAUDE.md `## Rules`)
- Loaded skills this session: kaggle-comp, postmortem
- R8d executed: final percentile logged to
  `.claude/skills/kaggle-comp/improvements.md` end-of-comp section.
