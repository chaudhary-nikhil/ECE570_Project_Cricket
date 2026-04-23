"""
Open-Meteo geocoding + forecast for the selected fixture's venue.

This module is display-only: it is *not* fed into the prediction model.
No API key required. Fails gracefully (returns None) on any network or
parsing error.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO code -> (emoji, human label). Groups per
# https://open-meteo.com/en/docs (WMO Weather interpretation codes).
WMO: Dict[int, tuple] = {
    0: ("\u2600\ufe0f", "Clear"),
    1: ("\U0001F324\ufe0f", "Mostly clear"),
    2: ("\u26C5", "Partly cloudy"),
    3: ("\u2601\ufe0f", "Overcast"),
    45: ("\U0001F32B\ufe0f", "Fog"),
    48: ("\U0001F32B\ufe0f", "Rime fog"),
    51: ("\U0001F326\ufe0f", "Light drizzle"),
    53: ("\U0001F326\ufe0f", "Drizzle"),
    55: ("\U0001F327\ufe0f", "Heavy drizzle"),
    56: ("\U0001F327\ufe0f", "Freezing drizzle"),
    57: ("\U0001F327\ufe0f", "Freezing drizzle"),
    61: ("\U0001F326\ufe0f", "Light rain"),
    63: ("\U0001F327\ufe0f", "Rain"),
    65: ("\U0001F327\ufe0f", "Heavy rain"),
    66: ("\U0001F327\ufe0f", "Freezing rain"),
    67: ("\U0001F327\ufe0f", "Freezing rain"),
    71: ("\U0001F328\ufe0f", "Light snow"),
    73: ("\U0001F328\ufe0f", "Snow"),
    75: ("\u2744\ufe0f", "Heavy snow"),
    77: ("\u2744\ufe0f", "Snow grains"),
    80: ("\U0001F326\ufe0f", "Rain showers"),
    81: ("\U0001F327\ufe0f", "Showers"),
    82: ("\u26C8\ufe0f", "Violent showers"),
    95: ("\u26C8\ufe0f", "Thunderstorm"),
    96: ("\u26C8\ufe0f", "Thunderstorm with hail"),
    99: ("\u26C8\ufe0f", "Thunderstorm with hail"),
}


def _wmo(code: Optional[Any]) -> tuple:
    try:
        return WMO[int(code)]
    except (TypeError, ValueError, KeyError):
        return ("\U0001F321\ufe0f", "Unknown")


def _venue_variants(venue: str) -> List[str]:
    """Ordered list of progressively simpler search terms derived from the venue string."""
    v = (venue or "").strip()
    if not v:
        return []

    variants: List[str] = [v]

    # "Stadium Name, City" -> also try "City"
    if "," in v:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        for p in reversed(parts):
            if p and p not in variants:
                variants.append(p)

    # Last-resort: individual meaningful tokens (prefer proper-noun-like words)
    stop = {"stadium", "ground", "park", "oval", "cricket", "international", "the", "of", "and"}
    for tok in v.replace(",", " ").split():
        tl = tok.strip()
        if len(tl) > 3 and tl.lower() not in stop and tl.isalpha() and tl not in variants:
            variants.append(tl)

    return variants


def _geocode(name: str, timeout: int = 10) -> Optional[Dict[str, float]]:
    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:  # noqa: BLE001
        return None

    results = data.get("results") or []
    if not results:
        return None

    top = results[0]
    try:
        return {
            "lat": float(top["latitude"]),
            "lon": float(top["longitude"]),
            "name": str(top.get("name", name)),
            "country": str(top.get("country", "")),
        }
    except (KeyError, TypeError, ValueError):
        return None


def get_forecast(venue: str, when: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """Fetch weather for ``venue`` (and optional ``when``). Returns None on failure.

    Returned dict keys: emoji, label, temp_c, wind_kmh, humidity_pct, precip_mm,
    source ('current' or 'forecast'), when (ISO date or None), resolved_name.
    """
    if not venue:
        return None

    geo: Optional[Dict[str, float]] = None
    for variant in _venue_variants(venue):
        geo = _geocode(variant)
        if geo:
            break
    if not geo:
        return None

    params: Dict[str, Any] = {
        "latitude": geo["lat"],
        "longitude": geo["lon"],
        "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,precipitation,weather_code",
        "timezone": "auto",
    }

    iso_date: Optional[str] = None
    if when is not None:
        iso_date = when.isoformat() if isinstance(when, date) else str(when)
        params["start_date"] = iso_date
        params["end_date"] = iso_date
        params["daily"] = "temperature_2m_max,temperature_2m_min,wind_speed_10m_max,precipitation_sum,weather_code"

    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:  # noqa: BLE001
        return None

    resolved_name = f"{geo.get('name', venue)}, {geo.get('country', '')}".strip().strip(",")

    # Prefer daily snapshot when a future/specific date is requested.
    daily = (data.get("daily") or {}) if isinstance(data, dict) else {}
    times = daily.get("time") or []
    if iso_date and times and iso_date in times:
        i = times.index(iso_date)

        def _nth(key: str) -> Optional[float]:
            arr = daily.get(key) or []
            if i < len(arr) and arr[i] is not None:
                try:
                    return float(arr[i])
                except (TypeError, ValueError):
                    return None
            return None

        t_max = _nth("temperature_2m_max")
        t_min = _nth("temperature_2m_min")
        temp: Optional[float]
        if t_max is not None and t_min is not None:
            temp = (t_max + t_min) / 2.0
        else:
            temp = t_max if t_max is not None else t_min

        code_arr = daily.get("weather_code") or []
        code = code_arr[i] if i < len(code_arr) else None
        emoji, label = _wmo(code)

        return {
            "emoji": emoji,
            "label": label,
            "temp_c": temp,
            "wind_kmh": _nth("wind_speed_10m_max"),
            "humidity_pct": None,
            "precip_mm": _nth("precipitation_sum"),
            "source": "forecast",
            "when": iso_date,
            "resolved_name": resolved_name,
        }

    current = (data.get("current") or {}) if isinstance(data, dict) else {}
    if not current:
        return None

    def _cur(key: str) -> Optional[float]:
        val = current.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    emoji, label = _wmo(current.get("weather_code"))
    return {
        "emoji": emoji,
        "label": label,
        "temp_c": _cur("temperature_2m"),
        "wind_kmh": _cur("wind_speed_10m"),
        "humidity_pct": _cur("relative_humidity_2m"),
        "precip_mm": _cur("precipitation"),
        "source": "current",
        "when": None,
        "resolved_name": resolved_name,
    }
