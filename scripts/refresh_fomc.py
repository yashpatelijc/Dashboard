"""Refresh FOMC meeting schedule from federalreserve.gov.

Best-effort scraper. The Fed publishes the calendar on:
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

The page lists meetings per year with format like "January 28-29".
We extract the second day (decision/statement date) and write to
config/fomc_meetings.yaml.

Usage:
  python scripts/refresh_fomc.py
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date

import requests
import yaml

URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
OUT = os.path.join(os.path.dirname(__file__), "..", "config", "fomc_meetings.yaml")


_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_meetings_from_html(html: str) -> list[dict]:
    """Extract meeting dates from the Fed calendars page.

    The page structure varies; we use a forgiving regex pattern.
    Pattern: "<Month> [day]/[day]-[day]" producing year context from headings.
    """
    out = []
    # Find each "fomc-meeting-row" or year section
    year_blocks = re.split(r'class="panel panel-padded panel-default"', html)
    current_year = None
    # Crude approach: find year headings + nearby month/day mentions
    for block in year_blocks:
        ymatch = re.search(r"\b(20\d{2})\s+FOMC\s+Meetings", block)
        if ymatch:
            current_year = int(ymatch.group(1))
        if current_year is None:
            continue
        # Find month/day pairs
        for mblock in re.finditer(
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d{1,2})[/\-]?(\d{1,2})?",
            block, re.IGNORECASE,
        ):
            month = _MONTH_NAMES[mblock.group(1).lower()]
            day1 = int(mblock.group(2))
            day2 = int(mblock.group(3)) if mblock.group(3) else day1
            decision_day = max(day1, day2)
            try:
                d = date(current_year, month, decision_day)
                out.append({"decision_date": d.isoformat(),
                            "press_conf": True,
                            "sep": False})
            except ValueError:
                continue
    # Dedup
    seen = set()
    dedup = []
    for r in out:
        if r["decision_date"] in seen:
            continue
        seen.add(r["decision_date"])
        dedup.append(r)
    return sorted(dedup, key=lambda x: x["decision_date"])


def main():
    print(f"Fetching {URL} ...")
    try:
        r = requests.get(URL, timeout=30,
                          headers={"User-Agent": "STIRS-Dashboard/1.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"FAILED to fetch: {e}", file=sys.stderr)
        sys.exit(1)

    meetings = parse_meetings_from_html(r.text)
    if not meetings:
        print("No meetings parsed — page format may have changed. Keeping existing YAML.",
              file=sys.stderr)
        sys.exit(2)

    out_data = {
        "last_updated": date.today().isoformat(),
        "source": URL,
        "meetings": meetings,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_data, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {len(meetings)} meetings to {OUT}")


if __name__ == "__main__":
    main()
