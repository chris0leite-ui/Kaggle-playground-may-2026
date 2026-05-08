"""Audit-writer for M3 CatBoost two-anchor run. Split out to keep the
main script under the 150-line cap (CLAUDE.md rule 9)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path


def write_audit(R: dict) -> None:
    P = R["P"]; BS, BG = R["BASE_S"], R["BASE_G"]
    cap = P["iterations"] - 1
    es_a = sum(1 for b in R["bi_a"] if b < cap)
    es_b = sum(1 for b in R["bi_b"] if b < cap)
    def gv(auc, base):
        return ("PASS" if auc >= base - 5e-4 else
                "SOFT" if auc >= base - 1e-3 else "FAIL")
    g1a, g1b = gv(R["auc_a"], BS), gv(R["auc_b"], BG)
    fi_md = "\n".join(f"| {n} | {v:.4f} |" for n, v in (R["fi0"] or [])[:10])
    body = f"""# M3 CatBoost — two-anchor (shrunk) ({dt.date.today()})

## Shrunk-config rationale

Prior probe (depth=8/iters=2000) projected 96.4 min — see `audit/2026-05-04-m3-catboost-PROBE-FAIL.md`. Shrunk to depth=6 / iters=800 / lr=0.08 / od_wait=50 to fit under 60-min cap while retaining capacity for the +92bp single-fold signal.

Params: {P}

## Two-anchor results

| anchor | OOF AUC | fold_std | per-fold | Δ vs baseline |
|---|---:|---:|---|---:|
| A — StratKFold | **{R['auc_a']:.5f}** | {R['sd_a']:.5f} | {[round(x,5) for x in R['fs_a']]} | {R['d_a']:+.1f}bp |
| B — GroupKF(Race) | **{R['auc_b']:.5f}** | {R['sd_b']:.5f} | {[round(x,5) for x in R['fs_b']]} | {R['d_b']:+.1f}bp |

## G1 verdict (gate ≥ baseline OOF − 5bp; soft −10bp)

- Strat: base {BS:.5f} → OOF {R['auc_a']:.5f} → {g1a}
- GroupKF: base {BG:.5f} → OOF {R['auc_b']:.5f} → {g1b}

## Wall-time table

| stage | wall (s) |
|---|---:|
| smoke (prior) | ~30 |
| re-probe (1-fold full) | 171.5 |
| Strat 5-fold | {sum(R['w_a']):.0f} |
| GroupKF 5-fold | {sum(R['w_b']):.0f} |
| total two-anchor | {R['total']:.0f} ({R['total']/60:.1f} min) |

## Top-10 feature importances (Strat fold-0, PredictionValuesChange)

| feature | importance |
|---|---:|
{fi_md}

## Early-stopping fired?

- Strat best_iters: {R['bi_a']} → {es_a}/5 stopped early
- GroupKF best_iters: {R['bi_b']} → {es_b}/5 stopped early
- best_iter == {cap} = hit iter cap (no ES). If most folds hit cap, AUC is a floor; more iters could lift further.
"""
    out = Path(f"audit/{dt.date.today().isoformat()}-m3-catboost.md")
    out.write_text(body); print(f"\n→ {out}")
