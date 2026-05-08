"""C2 — Pirelli pit-window scrape [DEPRECATED 2026-05-12 — synthetic-DGP incompatibility].

DEPRECATED before execution. See `audit/2026-05-12-d12-c2-pirelli-prep.md`
for the deprecation rationale: Year=2023 is mode-collapsed in the CTGAN
synth, the 5-way co-occurrence Pirelli windows depend on is exactly what
CTGAN broke, and the pool already absorbs the synth's empirical analog
via existing Compound × TyreLife / Race × Compound × Stint rules.

Skeleton preserved for reference / repurposing if a future comp uses
real F1 telemetry. DO NOT EXECUTE on the s6e5 synthetic data.

Original docstring follows.
---

Scrapes F1.com pre-race strategy guides + Pirelli Motorsport press
kits for pre-race pit-window predictions, building the lookup CSV
that feeds `scripts/d12_c2_pirelli_build.py`.

Output: `data/external/pirelli_windows.csv` with columns
  race, year, compound, n_stops_strategy,
  window_start_lap, window_end_lap, window_center_lap, window_width,
  source_url, scrape_confidence

Per Day-12 prep doc:
  - 24 races × 4 years = 96 race-events to process
  - Estimated 6-8h wall (with manual review per year)
  - Year-by-year with explicit checkpoint between years (resumable)

Sources tried in order per (race, year):
  1. F1.com strategy guide article (most reliable 2022-2025)
  2. Pirelli press kit PDF (highly structured, archive may be patchy)
  3. The Race / Autosport pre-race (cross-check fallback)

Manual review:
  - After each year's scrape, write JSON cache + open in editor for
    a 5-10 min sanity check before moving to next year.
  - Flag rows with confidence='low' for follow-up.

This is a SKELETON; the parser bodies are stubs. Real implementation
needs:
  - HTML parser robust to F1.com 2024 layout change
  - PDF text extraction (pdfplumber) for Pirelli kits
  - Tire-name canonicalization (HARD/MEDIUM/SOFT/INTERMEDIATE/WET; Pirelli C0-C5 → comp letters)
  - Lap-range regex tuned to "lap 22-28" / "laps 22 to 28" / "around lap 25" variants
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Imports kept lazy to avoid import cost at skeleton-test time
# import requests  # Real run
# import pdfplumber  # For Pirelli PDF kits
# from bs4 import BeautifulSoup  # For F1.com HTML

OUT = Path("data/external/pirelli_windows.csv")
CACHE = Path("data/external/pirelli_cache")
CACHE.mkdir(parents=True, exist_ok=True)

YEARS = [2022, 2023, 2024, 2025]
# 24 GP names (mapped to F1.com slugs) — match the Race column levels
RACES = [
    "Bahrain", "Saudi-Arabian", "Australian", "Japanese", "Chinese",
    "Miami", "Emilia-Romagna", "Monaco", "Canadian", "Spanish",
    "Austrian", "British", "Hungarian", "Belgian", "Dutch", "Italian",
    "Azerbaijan", "Singapore", "United-States", "Mexico-City",
    "Sao-Paulo", "Las-Vegas", "Qatar", "Abu-Dhabi",
]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


@dataclass
class WindowRecord:
    race: str
    year: int
    compound: str
    n_stops_strategy: int
    window_start_lap: int
    window_end_lap: int
    window_center_lap: float  # may be non-integer if averaged across sources
    window_width: int
    source_url: str
    scrape_confidence: str  # high / med / low


# ---------------------------------------------------------------------------
# Source 1 — F1.com pre-race strategy guides
# ---------------------------------------------------------------------------

F1COM_STRATEGY_URL = (
    "https://www.formula1.com/en/latest/article/"
    # Real URL pattern needs slug discovery via search; not enumerable
    # by template alone. See `find_f1com_strategy_url` stub below.
)


def find_f1com_strategy_url(race: str, year: int) -> Optional[str]:
    """Search F1.com for the pre-race strategy guide article URL.

    F1.com slugs are not deterministic; need to search by article title
    pattern '<race> grand prix strategy' or '<race> gp strategy guide'.
    """
    # SKELETON: real impl uses requests + DDG/Bing/F1.com search
    # query = f"{race} GP {year} strategy guide site:formula1.com"
    raise NotImplementedError("source_1: F1.com slug discovery")


def parse_f1com_strategy(html: str) -> list[WindowRecord]:
    """Extract pit-window predictions from F1.com strategy article HTML."""
    # SKELETON: regex over the article body for patterns like
    #   r"(?:lap|laps?)\s*(\d+)\s*(?:to|[-–])\s*(\d+).{0,80}(SOFT|MEDIUM|HARD)"
    # Combine with a sentence-level NLU pass for n_stops_strategy
    # ("one-stop", "two-stop") and compound transitions.
    raise NotImplementedError("source_1: F1.com HTML parser")


# ---------------------------------------------------------------------------
# Source 2 — Pirelli Motorsport press-kit PDFs
# ---------------------------------------------------------------------------

PIRELLI_PRESSKIT_URL = "https://press.pirelli.com/motorsport/"


def find_pirelli_presskit_url(race: str, year: int) -> Optional[str]:
    """Pirelli press kits live at /motorsport/<race-slug>-<year>/ or similar."""
    # SKELETON: search Pirelli archive
    raise NotImplementedError("source_2: Pirelli URL discovery")


def parse_pirelli_pdf(pdf_bytes: bytes) -> list[WindowRecord]:
    """Extract 'Strategy Predictions' table from Pirelli press kit PDF."""
    # SKELETON: pdfplumber extract_tables()
    # Each kit has a "Strategy Predictions" page with a table listing
    # 1-stop / 2-stop / 3-stop variants and lap windows per compound.
    raise NotImplementedError("source_2: Pirelli PDF parser")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def scrape_one_event(race: str, year: int) -> list[WindowRecord]:
    """Try sources in order; merge agreeing results; flag confidence."""
    records = []
    cache_path = CACHE / f"{race}_{year}.json"
    if cache_path.exists():
        return [WindowRecord(**r) for r in json.loads(cache_path.read_text())]
    # Try F1.com first
    # url = find_f1com_strategy_url(race, year)
    # if url: html = requests.get(url, timeout=30).text
    #         records.extend(parse_f1com_strategy(html))
    # Fallback to Pirelli
    # url2 = find_pirelli_presskit_url(race, year)
    # if url2: pdf = requests.get(url2, timeout=60).content
    #          records.extend(parse_pirelli_pdf(pdf))
    cache_path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    return records


def main(year_filter: Optional[int] = None):
    all_records: list[WindowRecord] = []
    for year in YEARS:
        if year_filter and year != year_filter:
            continue
        print(f"=== YEAR {year} ===")
        for race in RACES:
            t0 = time.time()
            try:
                recs = scrape_one_event(race, year)
                all_records.extend(recs)
                print(f"  {race:<20s} {len(recs)} window(s)  {time.time()-t0:.1f}s")
            except NotImplementedError as e:
                print(f"  {race:<20s} SKELETON: {e}")
            except Exception as e:
                print(f"  {race:<20s} ERROR: {e}")
        print(f"--- Year {year} done; review cache before next year ---")
        input("Press Enter to continue (or Ctrl-C to stop and resume later)...")
    # Write final CSV
    if all_records:
        import pandas as pd
        df = pd.DataFrame([asdict(r) for r in all_records])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, index=False)
        print(f"\nWrote {len(df)} records to {OUT}")


if __name__ == "__main__":
    print("WARNING: this is a SKELETON. Parser bodies are NotImplementedError stubs.")
    print("DO NOT EXECUTE without PI sign-off. Estimated 6-8h wall when implemented.")
    print("To dry-run skeleton: python scripts/d12_c2_pirelli_scrape.py --year 2024")
    if "--year" in sys.argv:
        y = int(sys.argv[sys.argv.index("--year") + 1])
        main(year_filter=y)
    else:
        main()
