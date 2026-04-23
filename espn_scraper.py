"""
Scrape cricket fixtures from ESPN cricket scores page.
Date windows are computed in **US Eastern** (``America/New_York``) to match ESPN's
day boundaries; use ``include_today=True``-style behavior by requesting day offset 0.
"""
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

ESPN_CRICKET_SCORES_BASE = "https://www.espn.com/cricket/scores"

# Normalized date format for comparison (e.g. "Feb 25")
CURRENT_DATE_FMT = "%b %d"
# URL query param format (e.g. 20260226)
DATE_PARAM_FMT = "%Y%m%d"

# Use US Eastern so "today" matches United States perspective
DEFAULT_TIMEZONE = ZoneInfo("America/New_York")


def _target_date(timezone: Optional[ZoneInfo] = None, days_ahead: int = 1) -> datetime:
    """Target date as datetime: today + days_ahead in the given timezone."""
    tz = timezone or DEFAULT_TIMEZONE
    return datetime.now(tz) + timedelta(days=days_ahead)


def _target_date_label(timezone: Optional[ZoneInfo] = None, days_ahead: int = 1) -> str:
    """Date label for filtering/display (e.g. 'Feb 26')."""
    return _target_date(timezone, days_ahead).strftime(CURRENT_DATE_FMT).replace(" 0", " ")


def _scores_url_for_date(when: datetime) -> str:
    """Build ESPN cricket scores URL with date=YYYYMMDD."""
    date_str = when.strftime(DATE_PARAM_FMT)
    return f"{ESPN_CRICKET_SCORES_BASE}?date={date_str}"


def _parse_date_from_text(text: str) -> Optional[str]:
    """
    Extract a comparable date string from match text (e.g. 'at Colombo, Feb 25 2026' -> 'Feb 25').
    Returns None if no date found.
    """
    # e.g. "46th Match, (N) at Colombo, Feb 25 2026" or "Feb 24-28 2026"
    m = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})(?:\s+\d{4}|\s*-\s*\d)?", text)
    if m:
        return f"{m.group(1)} {m.group(2)}".replace(" 0", " ")
    return None


def _is_primary_game_link(href: str, link_text: str) -> bool:
    """True if this is a main game link (not Summary, Scorecard, Report, Series Home)."""
    if "/game/" not in href or "/cricket/series/" not in href:
        return False
    lower = link_text.lower().strip()
    if len(lower) < 10:
        return False
    skip = ("summary", "scorecard", "report", "series home", "live", "schedule")
    return not any(s in lower for s in skip)


def _teams_from_slug(slug: str) -> List[str]:
    """Parse 'sri-lanka-vs-new-zealand-46th-match-...' -> ['Sri Lanka', 'New Zealand']."""
    if not slug:
        return []
    parts = [p for p in slug.split("-") if p]
    if "-vs-" not in slug and "vs" not in [p.lower() for p in parts]:
        return []
    team1_parts: List[str] = []
    team2_parts: List[str] = []
    stop_words = {"match", "1st", "2nd", "3rd", "group", "super", "eights", "final", "icc", "odi", "t20i", "test", "st", "at"}
    def is_ordinal_or_number(s: str) -> bool:
        return s.isdigit() or (len(s) >= 2 and s[-2:] in ("th", "st", "nd", "rd") and s[:-2].isdigit()) or s.lower() in stop_words
    i = 0
    while i < len(parts) and parts[i].lower() != "vs":
        team1_parts.append(parts[i])
        i += 1
    if i < len(parts):
        i += 1  # skip "vs"
    while i < len(parts):
        if is_ordinal_or_number(parts[i]) or parts[i].lower() in stop_words:
            break
        team2_parts.append(parts[i])
        i += 1
    out: List[str] = []
    if team1_parts:
        out.append(" ".join(w.capitalize() for w in team1_parts))
    if team2_parts:
        out.append(" ".join(w.capitalize() for w in team2_parts))
    return out[:2]


def _venue_date_from_text(text: str) -> tuple:
    """Extract (venue, date_str) from match text containing 'at Venue, Mon DD YYYY'. """
    venue, date_str = "", ""
    # e.g. "46th Match, Super Eights, Group 2, (N) at Colombo, Feb 25 2026"
    at_match = re.search(r"\bat\s+([^,]+),\s*([A-Za-z]+\s+\d{1,2}\s*\d{0,4})", text, re.I)
    if at_match:
        venue = at_match.group(1).strip()
        date_str = re.sub(r"\s+", " ", at_match.group(2).strip())
    return venue, date_str


# --- T20 classification --------------------------------------------------- #

# Long-format markers - if any match, the fixture is NOT T20.
_T20_EXCLUDE_RE = re.compile(
    r"\b(ODI|TEST|ONE\s*DAY|FIRST[-\s]CLASS|4\s*DAY|5\s*DAY)\b",
    re.IGNORECASE,
)

# T20-indicative patterns: explicit T20/T20I or a known T20 franchise league.
_T20_INCLUDE_RE = re.compile(
    r"("
    r"\bT20I?S?\b"
    r"|INDIAN\s+PREMIER\s+LEAGUE"
    r"|PAKISTAN\s+SUPER\s+LEAGUE"
    r"|BIG\s+BASH"
    r"|CARIBBEAN\s+PREMIER"
    r"|\bSA20\b"
    r"|\bILT20\b"
    r"|THE\s+HUNDRED"
    r"|MAJOR\s+LEAGUE\s+CRICKET"
    r"|WOMEN'?S\s+PREMIER\s+LEAGUE"
    r"|\bWPL\b"
    r"|LANKA\s+PREMIER\s+LEAGUE"
    r"|BANGLADESH\s+PREMIER\s+LEAGUE"
    r"|SUPER\s+SMASH"
    r")",
    re.IGNORECASE,
)


def _is_t20(name: str) -> bool:
    """Classify a fixture by its ESPN-derived name. Returns True only for T20 matches."""
    if not name:
        return False
    if _T20_EXCLUDE_RE.search(name):
        return False
    return bool(_T20_INCLUDE_RE.search(name))


# --- Scraping ------------------------------------------------------------- #


def _scrape_fixtures_for_day(days_ahead: int, timeout: int = 25) -> List[Dict[str, Any]]:
    """Scrape all fixtures for a single day offset from "now" in US Eastern.

    ``days_ahead=0`` is **today**; ``1`` is tomorrow, etc.
    """
    target = _target_date(days_ahead=days_ahead)
    current_date = target.strftime(CURRENT_DATE_FMT).replace(" 0", " ")
    date_iso = target.strftime("%Y-%m-%d")
    url = _scores_url_for_date(target)
    matches: List[Dict[str, Any]] = []
    try:
        resp = requests.get(
            url,
            timeout=min(timeout, 20),
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return []

    seen_names: set = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://www.espn.com" + href
        link_text = re.sub(r"\s+", " ", (a.get_text() or "").strip())
        if not _is_primary_game_link(href, link_text):
            continue
        # Combined pattern captures both the series id and the game id at once
        ids_match = re.search(
            r"/cricket/series/(\d+)/game/(\d+)/([^/?#]+)", href
        )
        if ids_match:
            series_id = ids_match.group(1)
            game_id = ids_match.group(2)
            raw_slug = ids_match.group(3)
        else:
            series_id = ""
            game_id = ""
            slug_match = re.search(r"/game/\d+/([^/?#]+)", href)
            raw_slug = slug_match.group(1) if slug_match else ""
        parsed_date = _parse_date_from_text(link_text)
        if parsed_date is None:
            parsed_date = _parse_date_from_text((a.parent.get_text() if a.parent else "") or "")
        if parsed_date is None or parsed_date != current_date:
            continue
        name = (raw_slug.replace("-", " ").title())[:200] if raw_slug else link_text[:200]
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        venue, _ = _venue_date_from_text(link_text)
        teams = _teams_from_slug(raw_slug)
        matches.append(
            {
                "name": name,
                "venue": venue,
                "date": current_date,
                "date_iso": date_iso,
                "teams": teams,
                "espn_series_id": series_id,
                "espn_game_id": game_id,
            }
        )
    # Stable sort by name so the same series groups together (e.g. Bahrain vs Bhutan then Thailand vs Japan)
    matches.sort(key=lambda m: (m.get("name") or "").lower())
    return matches


def scrape_today_fixtures(timeout: int = 25) -> List[Dict[str, Any]]:
    """Scrape fixtures for **today** in US Eastern. Single-day helper."""
    return _scrape_fixtures_for_day(days_ahead=0, timeout=timeout)


def scrape_upcoming_fixtures(
    n_days: int = 3,
    timeout: int = 25,
    t20_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Scrape fixtures for the next ``n_days`` **calendar** days in US Eastern, **including
    today** (``days_ahead`` 0 through ``n_days - 1``).

    When ``t20_only`` is True (default), non-T20 matches (ODIs, Tests, etc.) are dropped.
    Deduplicates by ``(date_iso, name)`` and sorts by date then name so the dropdown
    groups same-day fixtures together.
    """
    combined: List[Dict[str, Any]] = []
    seen: set = set()
    n = max(1, int(n_days))
    for days_ahead in range(0, n):
        for m in _scrape_fixtures_for_day(days_ahead=days_ahead, timeout=timeout):
            key = (m.get("date_iso", ""), m.get("name", ""))
            if key in seen:
                continue
            if t20_only and not _is_t20(m.get("name", "")):
                continue
            seen.add(key)
            combined.append(m)

    combined.sort(
        key=lambda m: (m.get("date_iso") or "", (m.get("name") or "").lower())
    )
    return combined
