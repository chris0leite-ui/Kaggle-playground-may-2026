# Grilling round 4 — drilling F1.1 / F1.3 deeper, opening F1.4

PI's [round-3 answers](./2026-05-06-grilling-round-3.md) confirmed BOTE
is applied (no specific incident), the pre-BOTE-style waste case is
actually post-BOTE (TabPFN), and CLAUDE.md is agent-authored. Three
sharper follow-ups, plus one parked idea.

---

## F1.1.1 — TabPFN: did BOTE actually run?

PI's TabPFN example shows PI override mattering. Need to know what the
*automated* mechanism did before the override:

- Did `python scripts/probe.py bote tabpfn …` get called at all?
- If yes, what did it return — PURSUE, DEFER, or SKIP?
- If PURSUE: BOTE said go and was wrong; FAMILY_PRIORS need attention.
- If never called: Rule 19(a) was bypassed — also a problem.

Sources to check: `audit/2026-05-12-d12-tabpfn-finetune-prep.md`,
`kernels/d12-tabpfn-finetune-gpu/`, recent commit history.

**Why I'm asking.** The TabPFN incident is currently the strongest
evidence we have on whether the focus-setting rules work autonomously.
The detail of what BOTE *said* (vs. what PI *did*) determines whether
the rules need fixing or are simply being skipped.

> **PI answer (2026-05-06).** "BOTE was not established yet." → therefore
> not blameworthy that BOTE didn't run.
>
> **Verification (Claude, 2026-05-06).** Confirmed via git log:
> TabPFN fold-0 smoke completed at 08:11 UTC (`c54a4a4`); BOTE/probe.py
> created at 09:57 UTC (`1e96c59`); Rule 19 added to CLAUDE.md at
> 10:26 UTC (`e48820d`, merged from agent branch
> `claude/ml-handover-alignment-xvUN0`).
>
> But: **Rule 2 was active** during TabPFN. Kernel iterated v7
> (`N_FOLDS=1`), v8 (`SMOKE_FOLD0_ONLY`), v9 ("Rule 2 compliance").
> So TabPFN was killed by **Rule 2 + PI override**, not by missing
> BOTE. F1.2's framing implicitly credited BOTE; correction filed
> as [flag F-1](../flags/2026-05-06.md#f-1--tabpfn-hypothesis-board-credits-framework-mechanism-was-pi-override--rule-2).
> Authorship finding ([F-2](../flags/2026-05-06.md#f-2--standing-watch-rule-19bote-itself-was-agent-authored)):
> Rule 19/BOTE itself was agent-authored.

---

## F1.4 — At-scale verification strategy

PI explicitly admitted: "I will not be able to oversee all of them. And
if I scale this up it will be more extreme."

Three plausible verification strategies:

- **Random sampling.** PI spot-checks a random N% of agent decisions.
  Sample size scales with confidence requirement, not absolute count.
- **Triggered review.** PI is alerted on null-streaks, regressions,
  resource overruns, gate-fail clusters, or BOTE-prediction-vs-actual
  divergence — the rules pull PI in only when something looks off.
- **Aggregate metrics.** PI watches dashboards: BOTE prediction
  calibration, gate hit-rate, OOF→LB amplification distribution,
  override frequency. Per-decision auditing replaced by population stats.

(Not mutually exclusive — could mix.)

**Why I'm asking.** Determines what tooling this KB should host (or
point to). Also determines what counts as an "incident" worth filing —
is one bad BOTE verdict an incident, or only a pattern of bad ones?

> **PI answer (2026-05-06).** **Aggregate metrics — calibration of
> BOTE estimates / predictions / decisions.** PI explicitly accepts
> this will be slow: trust must be earned over a track record.
>
> **Claude pushback.** "Calibration" is the right shape but
> underspecified. Two stages exist (BOTE→OOF, OOF→LB) and the latter
> is currently massively miscalibrated — see [synthesis](../thoughts/2026-05-06-bote-fresh-and-calibration-as-trust.md).
> Plus: PI's plan covers PURSUE-precision but not SKIP-recall.
> Drilled in [F1.6](./2026-05-06-grilling-round-5.md#f16) and
> [F1.7](./2026-05-06-grilling-round-5.md#f17).

---

## F1.5 — Bipartite CLAUDE.md, concretely

PI said they'll keep a section that is theirs. On a scan of the
current `CLAUDE.md`, candidate ownership map:

| Section                                | Likely author    |
|----------------------------------------|------------------|
| Top-level rules 1-19                   | mixed (some PI directional, agent encoded) |
| Defaults from prior-comp postmortem (R1, R5, R7, R8)  | likely agent (sourced from prior comp)|
| `Current state` yaml                   | agent (live state)        |
| Calibration ladder                     | agent (live data)         |
| Hypothesis board                       | agent                     |
| `Pointers`                             | agent                     |

Questions:

- Which numbered rules feel **yours** (PI judgment, you'd defend each)?
- Which feel agent-drafted but you tolerate?
- Are any rules you'd actually **delete** because you don't endorse
  them, or that are obsolete?
- Are there rules that are *missing* — judgment you exercise but the
  framework doesn't capture?

Not asking for a refactor. Asking which parts you actually own.

**Why I'm asking.** Once we know this, the KB can hold the genuinely
PI-owned parts (a "PI manifesto" section) while CLAUDE.md remains the
agent's operational manual. Bipartite by design, not by accident.

> PI answer:
> _to fill_

---

## Parked

**Heuristics library.** PI suggested collecting useful BOTE worked
examples / heuristics for agents to leverage. Adjacent to existing
`FAMILY_PRIORS` in `scripts/probe.py`. Not opened today; will become a
concept-entry candidate when PI returns to it.

**Standing duty inaugurated.** Claude flags points warranting careful
PI review. First flag: TabPFN hypothesis-board entry credits the
framework for an outcome that required PI override.
See [synthesis](../thoughts/2026-05-06-bote-applied-and-scale-trust.md#first-flag-kicking-off-the-standing-duty).
