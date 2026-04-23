"""
Requests-only roster fetcher for ESPN Cricinfo match summaries.

Given an ESPN series id + game id (both exposed by `espn_scraper.py` as
``espn_series_id`` / ``espn_game_id``), fetch the public
``site.api.espn.com`` summary JSON and extract either the playing XI
(``rosters[*].roster``) when it is populated — typical once a match is live
or completed — or the full squad (``squads[*].athletes``) for upcoming
fixtures. This module intentionally avoids any browser / Playwright
dependency: only ``requests``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/cricket/"
    "{series_id}/summary?event={game_id}"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    )
}


class RosterFetchError(RuntimeError):
    """Raised when the ESPN summary payload is unreachable or malformed."""


def fetch_match_summary(series_id: str, game_id: str, timeout: int = 15) -> Dict[str, Any]:
    """GET the ESPN summary JSON for a given series/game id pair."""
    if not series_id or not game_id:
        raise RosterFetchError("series_id and game_id are required")
    url = SUMMARY_URL.format(series_id=series_id, game_id=game_id)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    except requests.RequestException as exc:
        raise RosterFetchError(f"Network error contacting ESPN: {exc}") from exc
    if resp.status_code != 200:
        raise RosterFetchError(f"ESPN returned HTTP {resp.status_code} for {url}")
    try:
        return resp.json()
    except ValueError as exc:
        raise RosterFetchError(f"ESPN response was not JSON: {exc}") from exc


def _extract_from_roster(roster_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a single ``rosters[i].roster`` list (populated for live/completed games)."""
    out: List[Dict[str, Any]] = []
    for entry in roster_entry.get("roster") or []:
        athlete = entry.get("athlete") or {}
        name = athlete.get("displayName") or athlete.get("fullName") or athlete.get("name")
        if not name:
            continue
        position = ((athlete.get("position") or {}).get("name")) or ""
        out.append(
            {
                "name": name,
                "position": position,
                "starter": bool(entry.get("starter")),
                "captain": bool(entry.get("captain") or False),
            }
        )
    return out


def _extract_from_squad(squad_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a ``squads[i].athletes`` list (typical for upcoming fixtures)."""
    out: List[Dict[str, Any]] = []
    for athlete in squad_entry.get("athletes") or []:
        name = athlete.get("displayName") or athlete.get("fullName")
        if not name:
            continue
        position = ((athlete.get("position") or {}).get("name")) or ""
        out.append(
            {
                "name": name,
                "position": position,
                # Squad entries do not flag XI membership, so we expose the
                # captain / keeper flags ESPN does give us as context hints.
                "starter": False,
                "captain": bool(athlete.get("captain") or False),
                "keeper": bool(athlete.get("keeper") or False),
            }
        )
    return out


def extract_squads_or_xi(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized ``{team_label: {source, players}}`` dict.

    ``source`` is either ``"xi"`` (starting eleven from a live/completed match)
    or ``"squad"`` (the full squad list for upcoming fixtures). If neither is
    available, the value is ``{"source": "none", "players": []}``.
    """
    result: Dict[str, Any] = {"teams": [], "confidence": "none"}

    rosters = summary.get("rosters") or []
    # Prefer the playing XI when ESPN has populated it.
    if any((r.get("roster") or []) for r in rosters):
        for r in rosters:
            team = (r.get("team") or {})
            players = _extract_from_roster(r)
            result["teams"].append(
                {
                    "team_name": team.get("displayName") or team.get("name") or "",
                    "team_abbrev": team.get("abbreviation") or "",
                    "home_away": r.get("homeAway") or "",
                    "source": "xi",
                    "players": players,
                }
            )
        result["confidence"] = "xi"
        return result

    # Fall back to broader squads when the XI is not yet posted.
    squads = summary.get("squads") or []
    if any((s.get("athletes") or []) for s in squads):
        for s in squads:
            team = (s.get("team") or {})
            players = _extract_from_squad(s)
            result["teams"].append(
                {
                    "team_name": team.get("displayName") or team.get("name") or "",
                    "team_abbrev": team.get("abbreviation") or "",
                    "home_away": "",
                    "source": "squad",
                    "players": players,
                }
            )
        result["confidence"] = "squad"
        return result

    return result


def get_rosters(series_id: str, game_id: str) -> Optional[Dict[str, Any]]:
    """High-level convenience: fetch summary and return parsed rosters or None."""
    try:
        summary = fetch_match_summary(series_id, game_id)
    except RosterFetchError:
        return None
    parsed = extract_squads_or_xi(summary)
    if not parsed["teams"]:
        return None
    return parsed
