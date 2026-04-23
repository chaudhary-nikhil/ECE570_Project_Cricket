import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

import backend
import espn_roster
import espn_scraper
import weather


DEFAULT_API_KEY = "2cecc518-dfe7-4fa0-baaf-446cd5060795"
BACKEND_PREDICT_URL = "http://127.0.0.1:8000/predict"
TOTALS_CSV_PATH = "data/ipl_match_team_totals.csv"


def chat_url_from_predict_url(predict_url: str) -> str:
    base = predict_url.rstrip("/").rsplit("/predict", 1)[0].rstrip("/")
    return f"{base}/chat"


st.set_page_config(page_title="Cricket Live Score & AI Prediction", layout="wide")


@st.cache_data(ttl=300)
def fetch_upcoming_fixtures_espn(n_days: int = 3) -> List[Dict[str, Any]]:
    """Fetch T20 fixtures for the next ``n_days`` calendar days (including today, US Eastern)."""
    try:
        return espn_scraper.scrape_upcoming_fixtures(n_days=n_days, timeout=25)
    except RuntimeError as exc:
        st.error(str(exc))
        return []
    except Exception as exc:
        st.error(f"ESPN scraper failed: {exc}")
        return []


@st.cache_data(ttl=60)
def fetch_live_matches(api_key: str) -> List[Dict[str, Any]]:
    """Fetch live / recent matches from CricketData.org (fallback)."""
    try:
        data = backend.fetch_live_scores(api_key)
    except RuntimeError as exc:
        st.error(str(exc))
        return []

    if isinstance(data, list):
        matches = data
    elif isinstance(data, dict):
        matches = data.get("data") or data.get("matches") or []
        if not isinstance(matches, list) and isinstance(matches, dict):
            matches = matches.get("matchList") or matches.get("match") or []
    else:
        st.error("Unexpected response format from CricketData API.")
        return []

    if not isinstance(matches, list):
        st.error("Unexpected matches format from CricketData API.")
        return []

    return matches


@st.cache_data
def load_team_options() -> List[str]:
    """
    Load team names from the training CSV (batting team column).

    Falls back to an empty list if the CSV is missing or unreadable;
    in that case, we'll just use the teams from the selected match.
    """
    try:
        df = pd.read_csv("data/ipl_data.csv", usecols=["bat_team"])
    except Exception:  # noqa: BLE001
        return []

    teams = sorted(set(str(x) for x in df["bat_team"].dropna().unique()))
    return teams


@st.cache_data
def load_totals() -> pd.DataFrame:
    """Aggregated IPL team-innings totals used for historical context."""
    try:
        return pd.read_csv(TOTALS_CSV_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# Reasonable T20 pre-match fallback if no context is available at all.
# Derived from typical IPL first-innings totals; used only when every
# historical lookup returns None.
_IPL_PAR_FALLBACK = 170.0


def prematch_baseline(ctx: Dict[str, Optional[float]]) -> float:
    """
    Produce a pre-match baseline total from historical context.

    Priority: head-to-head / team / venue averages are blended when present,
    falling back progressively to any single available signal, and finally to
    a generic IPL par score when nothing matches (e.g. non-IPL fixtures).
    """
    h2h = ctx.get("h2h_avg") if ctx else None
    team = ctx.get("team_avg") if ctx else None
    venue = ctx.get("venue_avg") if ctx else None

    if h2h is not None and team is not None and venue is not None:
        return 0.4 * float(team) + 0.3 * float(venue) + 0.3 * float(h2h)
    if team is not None and venue is not None:
        return 0.5 * float(team) + 0.5 * float(venue)
    if team is not None and h2h is not None:
        return 0.6 * float(team) + 0.4 * float(h2h)
    if team is not None:
        return float(team)
    if venue is not None:
        return float(venue)
    if h2h is not None:
        return float(h2h)
    return _IPL_PAR_FALLBACK


def anchor_pre_match_prediction(
    result: Dict[str, float],
    baseline: float,
    model_weight: float = 0.45,
) -> Dict[str, float]:
    """
    Blend the raw model prediction with a historical baseline for pre-match state.

    The ball-by-ball training data contains essentially no rows at
    ``overs_completed = 0`` with ``current_score = 0``, so the network
    extrapolates into out-of-distribution territory and tends to under-predict.
    A fixed 45 / 55 blend keeps the model's team-vs-team signal while pulling
    the number back toward what the venue/team historically produce.

    The confidence band is preserved in half-width (±) around the anchored value.
    """
    raw = float(result.get("predicted_final_score", 0.0))
    lower = float(result.get("lower_bound", raw - 15.0))
    upper = float(result.get("upper_bound", raw + 15.0))
    half = max(0.0, (upper - lower) / 2.0)

    w = max(0.0, min(1.0, float(model_weight)))
    anchored = w * raw + (1.0 - w) * float(baseline)

    return {
        "predicted_final_score": anchored,
        "lower_bound": anchored - half,
        "upper_bound": anchored + half,
        "_raw_model_score": raw,
        "_baseline_score": float(baseline),
    }


def historical_context(
    df: pd.DataFrame,
    venue: str,
    batting_team: str,
    bowling_team: str,
) -> Dict[str, Optional[float]]:
    """Venue / team / head-to-head average innings totals from `ipl_match_team_totals.csv`."""
    if df is None or df.empty:
        return {"venue_avg": None, "team_avg": None, "h2h_avg": None}

    v = (venue or "").strip().lower()
    bat = (batting_team or "").strip().lower()
    bowl = (bowling_team or "").strip().lower()

    venue_col = df["venue"].astype(str).str.lower()
    bat_col = df["batting_team"].astype(str).str.lower()
    bowl_col = df["bowling_team"].astype(str).str.lower()

    v_mask = venue_col.str.contains(v, na=False) if v else pd.Series(False, index=df.index)
    bat_mask = bat_col == bat if bat else pd.Series(False, index=df.index)
    bowl_mask = bowl_col == bowl if bowl else pd.Series(False, index=df.index)

    def _mean(mask: pd.Series) -> Optional[float]:
        sub = df.loc[mask, "total_runs"]
        return float(sub.mean()) if len(sub) else None

    return {
        "venue_avg": _mean(v_mask),
        "team_avg": _mean(bat_mask),
        "h2h_avg": _mean(bat_mask & bowl_mask),
    }


@st.cache_data(ttl=1800)
def get_rosters_cached(series_id: str, game_id: str) -> Optional[Dict[str, Any]]:
    """Cached wrapper around :func:`espn_roster.get_rosters`.

    Cached for 30 minutes because squads churn slowly before the toss and the
    playing XI only posts once per match.
    """
    if not series_id or not game_id:
        return None
    try:
        return espn_roster.get_rosters(series_id, game_id)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_weather_cached(venue: str, when_iso: str) -> Optional[Dict[str, Any]]:
    """Cached wrapper around :func:`weather.get_forecast` keyed on venue + ISO date."""
    when_arg: Optional[date] = None
    if when_iso:
        try:
            when_arg = date.fromisoformat(when_iso)
        except ValueError:
            when_arg = None
    return weather.get_forecast(venue, when_arg)


def parse_score_info(match: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Try to extract current runs, wickets, and overs from typical cricScore fields.

    We look into 'de', 'si', 't1s', 't2s' and try to match patterns like:
        '25/2 (3.1 ov, ...)'  -> runs=25, wickets=2, overs=3.1
    """
    text_candidates = [
        str(match.get("de", "")),
        str(match.get("si", "")),
        str(match.get("t1s", "")),
        str(match.get("t2s", "")),
    ]

    runs: Optional[float] = None
    wickets: Optional[float] = None
    overs: Optional[float] = None

    pattern = re.compile(r"(\d+)\s*/\s*(\d+)\s*\(([\d\.]+)\s*ov")

    for text in text_candidates:
        match_obj = pattern.search(text)
        if match_obj:
            runs = float(match_obj.group(1))
            wickets = float(match_obj.group(2))
            overs = float(match_obj.group(3))
            break

    rr: Optional[float] = None
    if runs is not None and overs and overs > 0:
        rr = runs / overs

    return {
        "runs": runs,
        "wickets": wickets,
        "overs": overs,
        "rr": rr,
    }


def call_backend_predict(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Call the FastAPI /predict endpoint.

    Returns a dict with predicted_final_score, lower_bound, upper_bound, or None on error.
    """
    try:
        resp = requests.post(
            BACKEND_PREDICT_URL,
            json=payload,
            timeout=10,
        )
    except requests.RequestException as exc:  # noqa: BLE001
        st.error(f"Error contacting backend /predict endpoint: {exc}")
        return None

    if resp.status_code != 200:
        try:
            err = resp.json()
        except ValueError:
            err = resp.text
        st.error(f"Backend /predict error ({resp.status_code}): {err}")
        return None

    try:
        data = resp.json()
    except ValueError as exc:
        st.error(f"Failed to parse backend /predict response: {exc}")
        return None

    predicted = data.get("predicted_final_score")
    if predicted is None:
        st.error("Backend /predict response missing 'predicted_final_score'.")
        return None

    try:
        result = {
            "predicted_final_score": float(predicted),
            "lower_bound": float(data.get("lower_bound", predicted - 15)),
            "upper_bound": float(data.get("upper_bound", predicted + 15)),
        }
    except (TypeError, ValueError):
        st.error("Backend /predict returned invalid prediction or bounds.")
        return None

    return result


def main() -> None:
    global BACKEND_PREDICT_URL  # noqa: PLW0603

    st.title("🏏 Cricket Live Score & AI Prediction")

    # Sidebar configuration
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input(
        "CricketData.org API Key (optional)",
        value=DEFAULT_API_KEY,
        type="password",
        help="Only needed if you use CricketData live scores; fixtures use ESPN.",
    )

    backend_url = st.sidebar.text_input(
        "Backend URL",
        value=BACKEND_PREDICT_URL,
        help="Base URL of the FastAPI /predict endpoint.",
    )

    # Update global for this run (simple approach for single-page app)
    BACKEND_PREDICT_URL = backend_url.rstrip("/")  # ensure no trailing slash

    st.sidebar.caption(
        "Fixtures: T20 only, today + next 2 days (3 calendar days, US Eastern) from ESPN "
        "(cached 5 min)."
    )
    st.sidebar.caption(
        "Chat: set **OPENAI_API_KEY** in the environment of the FastAPI process (not in Streamlit)."
    )

    # Fetch upcoming T20 fixtures from ESPN
    st.subheader("Upcoming Fixtures")
    st.caption(
        "T20 matches only (men's and women's, internationals and franchise leagues) for "
        "**today and the next two calendar days** on ESPN's schedule (days are **US "
        "Eastern**); use the dropdown to pick a match."
    )
    matches = fetch_upcoming_fixtures_espn(n_days=3)
    if not matches:
        st.info(
            "No T20 fixtures found for today and the next 2 days, or the ESPN schedule "
            "could not be loaded. Try again later."
        )
        return

    # Build friendly labels (ESPN items have "name"; optional date)
    options = []
    for m in matches:
        name = m.get("name") or "Fixture"
        date_str = m.get("date") or ""
        label = name
        if date_str and date_str not in label:
            label = f"{name} ({date_str})"
        options.append(label)

    select_label = (
        f"Select a match ({len(matches)} T20 fixtures: today + next 2 days, US Eastern)"
    )
    selected_idx = st.selectbox(
        select_label,
        options=list(range(len(matches))),
        format_func=lambda i: options[i],
        index=0,
    )

    selected_match = matches[selected_idx]
    # Teams: from "teams" array, or t1/t2, or parse from "name" (e.g. "Team A v Team B")
    t1 = selected_match.get("t1") or selected_match.get("team1") or "Team 1"
    t2 = selected_match.get("t2") or selected_match.get("team2") or "Team 2"
    teams_arr = selected_match.get("teams") or []
    if isinstance(teams_arr, list) and len(teams_arr) >= 2:
        t1, t2 = teams_arr[0], teams_arr[1]
    elif isinstance(teams_arr, list) and len(teams_arr) == 1:
        t1 = teams_arr[0]
        t2 = "TBD" if t2 == "Team 2" else t2
    else:
        name = selected_match.get("name") or ""
        if " v " in name:
            parts = name.split(" v ", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                t1, t2 = parts[0].strip(), parts[1].strip()
        elif " vs " in name or " vs. " in name:
            sep = " vs " if " vs " in name else " vs. "
            parts = name.split(sep, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                t1, t2 = parts[0].strip(), parts[1].strip()
    # Venue: from match or venueInfo
    default_venue = (
        selected_match.get("venue")
        or selected_match.get("stadium")
        or (selected_match.get("venueInfo") or {}).get("name")
        or (selected_match.get("venueInfo") or {}).get("venue")
        or ""
    )
    if not isinstance(default_venue, str):
        default_venue = str(default_venue).strip() if default_venue else ""
    else:
        default_venue = default_venue.strip()

    # Live Scorecard
    st.subheader("Live Scorecard")
    score_info = parse_score_info(selected_match)

    col1, col2, col3 = st.columns(3)
    with col1:
        if score_info["runs"] is not None and score_info["wickets"] is not None:
            st.metric("Current Score", f"{int(score_info['runs'])}/{int(score_info['wickets'])}")
        else:
            st.metric("Current Score", "N/A")

    with col2:
        if score_info["overs"] is not None:
            st.metric("Overs", f"{score_info['overs']:.1f}")
        else:
            st.metric("Overs", "N/A")

    with col3:
        if score_info["rr"] is not None:
            st.metric("Run Rate", f"{score_info['rr']:.2f}")
        else:
            st.metric("Run Rate", "N/A")

    # Show raw summary text as a fallback
    summary_text = selected_match.get("de") or selected_match.get("si") or ""
    if summary_text:
        st.caption(summary_text)

    # Match Conditions (weather) - display only; not fed into the model.
    st.subheader("🌤️ Match Conditions")
    match_date = (
        selected_match.get("date_iso")
        or (date.today() + timedelta(days=1)).isoformat()
    )
    if default_venue:
        wx = get_weather_cached(default_venue, match_date)
    else:
        wx = None

    if wx is None:
        st.caption(
            f"No weather forecast available for venue '{default_venue or 'Unknown'}'. "
            "(Open-Meteo could not geocode this venue or the network call failed.)"
        )
    else:
        wx_cols = st.columns(4)
        header = f"{wx['emoji']} {wx['label']}"
        wx_cols[0].metric("Conditions", header)
        wx_cols[1].metric(
            "Temp (°C)",
            f"{wx['temp_c']:.1f}" if wx.get("temp_c") is not None else "N/A",
        )
        wx_cols[2].metric(
            "Wind (km/h)",
            f"{wx['wind_kmh']:.0f}" if wx.get("wind_kmh") is not None else "N/A",
        )
        precip_val = wx.get("precip_mm")
        if precip_val is not None:
            wx_cols[3].metric("Precip (mm)", f"{precip_val:.1f}")
        elif wx.get("humidity_pct") is not None:
            wx_cols[3].metric("Humidity (%)", f"{wx['humidity_pct']:.0f}")
        else:
            wx_cols[3].metric("Precip (mm)", "N/A")

        resolved = wx.get("resolved_name") or default_venue
        when_label = wx.get("when") or "current conditions"
        st.caption(
            f"Open-Meteo {wx.get('source', 'forecast')} for **{resolved}** · {when_label}. "
            "Informational only; not used by the predictor."
        )

    # ---- Rosters (ESPN squads/XI) ----
    # When the scraped fixture carries espn_series_id + espn_game_id we can pull
    # the squads (upcoming) or playing XI (live/completed) straight from ESPN's
    # public site.api summary endpoint — no browser automation required.
    st.subheader("🧢 Rosters")
    roster_sig = (
        selected_match.get("espn_series_id"),
        selected_match.get("espn_game_id"),
        selected_idx,
    )
    if st.session_state.get("_roster_sig") != roster_sig:
        st.session_state["_roster_sig"] = roster_sig
        st.session_state.pop("roster_team_a_players", None)
        st.session_state.pop("roster_team_b_players", None)
        st.session_state.pop("roster_team_a_name", None)
        st.session_state.pop("roster_team_b_name", None)
        st.session_state.pop("roster_data", None)
        st.session_state.pop("lineup_batting_text", None)
        st.session_state.pop("lineup_bowling_text", None)
        st.session_state.pop("_populate_lineups_from_roster", None)

    rosters = st.session_state.get("roster_data")
    series_id = selected_match.get("espn_series_id") or ""
    game_id = selected_match.get("espn_game_id") or ""

    if series_id and game_id:
        col_fetch, col_caption = st.columns([1, 4])
        with col_fetch:
            if st.button("Fetch squads", help="Pull squads / playing XI from ESPN"):
                with st.spinner("Fetching squads from ESPN..."):
                    rosters = get_rosters_cached(series_id, game_id)
                    st.session_state["roster_data"] = rosters
                st.session_state["_populate_lineups_from_roster"] = True
                st.rerun()
        with col_caption:
            st.caption(
                f"ESPN series={series_id} · game={game_id}. Squads display when available; "
                "the playing XI shows once a match goes live."
            )
    else:
        st.caption("No ESPN game id on this fixture; use manual lineup entry below to benefit from the roster-aware model.")

    if rosters and rosters.get("teams"):
        confidence = rosters.get("confidence", "squad")
        badge = "Playing XI" if confidence == "xi" else "Squad (full list)"
        st.caption(f"Source: **{badge}**. Roster loaded into lineup text boxes below.")
        # Remember the two team lists for direct prediction-form population
        st.session_state["roster_team_a_players"] = [
            p["name"] for p in (rosters["teams"][0].get("players") or [])
        ]
        st.session_state["roster_team_a_name"] = rosters["teams"][0].get("team_name", "")
        if len(rosters["teams"]) > 1:
            st.session_state["roster_team_b_players"] = [
                p["name"] for p in (rosters["teams"][1].get("players") or [])
            ]
            st.session_state["roster_team_b_name"] = rosters["teams"][1].get("team_name", "")

    # AI Prediction section
    st.subheader("🤖 AI Prediction")
    st.markdown(
        "Provide or adjust the current match stats below. "
        "These will be sent to the backend `/predict` endpoint."
    )

    # Sync venue with selected match so the form field pre-fills when the dropdown changes
    venue_key = "prediction_venue"
    if venue_key not in st.session_state:
        st.session_state[venue_key] = default_venue
    if "venue_match_idx" not in st.session_state or st.session_state["venue_match_idx"] != selected_idx:
        st.session_state["venue_match_idx"] = selected_idx
        st.session_state[venue_key] = default_venue

    # Pre-fill numeric inputs from parsed score, where possible
    default_runs = int(score_info["runs"]) if score_info["runs"] is not None else 0
    default_wkts = int(score_info["wickets"]) if score_info["wickets"] is not None else 0
    default_overs = float(score_info["overs"]) if score_info["overs"] is not None else 0.0

    # Load full team list from encoders (if available) and merge with live teams
    encoder_teams = load_team_options()
    all_team_options = sorted({*encoder_teams, t1, t2})

    # Session-state-backed team preferences so the Swap button can flip them.
    fixture_sig = (selected_idx, t1, t2)
    if st.session_state.get("_teams_fixture_sig") != fixture_sig:
        st.session_state["_teams_fixture_sig"] = fixture_sig
        st.session_state["bat_team_pref"] = t1 if t1 in all_team_options else all_team_options[0]
        st.session_state["bowl_team_pref"] = (
            t2 if (t2 in all_team_options and t2 != t1) else all_team_options[0]
        )

    bat_pref = st.session_state["bat_team_pref"]
    bowl_pref = st.session_state["bowl_team_pref"]

    def _match_roster_for_team(team_name: str) -> List[str]:
        roster_a_name = st.session_state.get("roster_team_a_name", "") or ""
        roster_b_name = st.session_state.get("roster_team_b_name", "") or ""
        roster_a_players = st.session_state.get("roster_team_a_players") or []
        roster_b_players = st.session_state.get("roster_team_b_players") or []
        t = (team_name or "").lower()
        if t and t in roster_a_name.lower():
            return list(roster_a_players)
        if t and t in roster_b_name.lower():
            return list(roster_b_players)
        return []

    # Initialize lineup text boxes once for this fixture/team context.
    if "lineup_batting_text" not in st.session_state:
        st.session_state["lineup_batting_text"] = "\n".join(_match_roster_for_team(bat_pref))
    if "lineup_bowling_text" not in st.session_state:
        st.session_state["lineup_bowling_text"] = "\n".join(_match_roster_for_team(bowl_pref))

    # Explicitly requested by user: clicking "Fetch squads" should populate the
    # prediction form text boxes directly.
    if st.session_state.get("_populate_lineups_from_roster"):
        st.session_state["lineup_batting_text"] = "\n".join(_match_roster_for_team(bat_pref))
        st.session_state["lineup_bowling_text"] = "\n".join(_match_roster_for_team(bowl_pref))
        st.session_state["_populate_lineups_from_roster"] = False

    col_swap_btn, col_swap_caption = st.columns([1, 4])
    with col_swap_btn:
        if st.button("⇄ Swap Teams", help="Flip batting and bowling"):
            st.session_state["bat_team_pref"], st.session_state["bowl_team_pref"] = (
                bowl_pref,
                bat_pref,
            )
            old_bat = st.session_state.get("lineup_batting_text", "")
            old_bowl = st.session_state.get("lineup_bowling_text", "")
            st.session_state["lineup_batting_text"] = old_bowl
            st.session_state["lineup_bowling_text"] = old_bat
            st.rerun()
    with col_swap_caption:
        st.caption(
            f"Currently predicting **{bat_pref}** batting vs **{bowl_pref}** bowling. "
            "Use _Swap Teams_ to flip perspective."
        )

    is_pre_match = (default_runs == 0 and default_wkts == 0 and default_overs == 0.0)

    with st.form("prediction_form"):
        venue = st.text_input(
            "Venue",
            value=st.session_state.get(venue_key, default_venue),
            help="Pre-filled from the selected match; edit if needed.",
        )
        col_bt, col_bl = st.columns(2)
        with col_bt:
            bat_default_idx = (
                all_team_options.index(bat_pref) if bat_pref in all_team_options else 0
            )
            batting_team = st.selectbox(
                "Batting Team",
                options=all_team_options,
                index=bat_default_idx,
            )
        with col_bl:
            bowl_default_idx = (
                all_team_options.index(bowl_pref) if bowl_pref in all_team_options else 0
            )
            bowling_team = st.selectbox(
                "Bowling Team",
                options=all_team_options,
                index=bowl_default_idx,
            )

        col_cs, col_wk, col_ov = st.columns(3)
        with col_cs:
            current_score = st.number_input(
                "Current Score (runs)",
                min_value=0,
                value=default_runs,
            )
        with col_wk:
            wickets_lost = st.number_input(
                "Wickets Lost",
                min_value=0,
                max_value=10,
                value=default_wkts,
            )
        with col_ov:
            overs_completed = st.number_input(
                "Overs Completed",
                min_value=0.0,
                max_value=20.0,
                value=default_overs,
                step=0.1,
            )

        runs_in_last_5_overs = st.number_input(
            "Runs in Last 5 Overs",
            min_value=0,
            value=0,
            help="If unknown, you may leave this as 0 or an estimate.",
        )

        st.markdown("**Lineups (optional, roster-aware prediction)**")
        col_bxi, col_oxi = st.columns(2)
        with col_bxi:
            batting_xi_text = st.text_area(
                f"{batting_team} batting XI / squad",
                key="lineup_batting_text",
                height=150,
                help="One player per line. Leave blank for neutral batting strength.",
            )
        with col_oxi:
            bowling_xi_text = st.text_area(
                f"{bowling_team} bowling XI / squad",
                key="lineup_bowling_text",
                height=150,
                help="One player per line. Leave blank for neutral bowling strength.",
            )

        predict_both = st.checkbox(
            "Predict both innings (pre-match)",
            value=False,
            help=(
                "Run the predictor twice with batting/bowling swapped and show both sides' "
                "first-innings projections. Most meaningful before the first ball."
            ),
        )

        submitted = st.form_submit_button("Get AI Prediction")

    if submitted:
        def _split_names(text: str) -> List[str]:
            return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

        batting_xi_list = _split_names(batting_xi_text)
        bowling_xi_list = _split_names(bowling_xi_text)

        payload = {
            "venue": venue or "Unknown",
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "current_score": float(current_score),
            "wickets_lost": int(wickets_lost),
            "overs_completed": float(overs_completed),
            "runs_in_last_5_overs": float(runs_in_last_5_overs),
        }
        if batting_xi_list:
            payload["batting_xi"] = batting_xi_list
        if bowling_xi_list:
            payload["bowling_xi"] = bowling_xi_list

        st.session_state["last_prediction_payload"] = payload

        totals_df = load_totals()
        ctx = historical_context(totals_df, venue, batting_team, bowling_team)
        venue_avg = ctx.get("venue_avg")
        # Reference "par" for the progress bar; fall back to a sensible T20 default.
        reference_max = max(180.0, float(venue_avg)) if venue_avg else 180.0

        def _render_single(result: Dict[str, float], label: str, current: float) -> None:
            predicted_score = result["predicted_final_score"]
            lower = result["lower_bound"]
            upper = result["upper_bound"]
            half = max(0.0, (upper - lower) / 2.0)

            col_metric, col_progress = st.columns([1, 2])
            with col_metric:
                delta = predicted_score - float(current)
                st.metric(
                    label=label,
                    value=f"{predicted_score:.1f}",
                    delta=f"{delta:+.1f} runs from current",
                )
                st.caption(
                    f"Confidence range: **{lower:.0f} – {upper:.0f}** runs (±{half:.0f})"
                )
            with col_progress:
                normalized = max(0.0, min(predicted_score / reference_max, 1.0))
                st.progress(normalized)
                st.caption(
                    f"Bar normalized to venue par ≈ **{reference_max:.0f}** runs."
                )

        # Pre-match anchor: blend model output with a historical baseline when
        # the inputs are all zero, because the ball-by-ball training data has
        # essentially no rows at overs=0 & score=0 and the model under-predicts.
        def _maybe_anchor(
            result: Optional[Dict[str, float]],
            batting: str,
            bowling: str,
        ) -> Optional[Dict[str, float]]:
            if result is None or not is_pre_match:
                return result
            sub_ctx = historical_context(totals_df, venue, batting, bowling)
            baseline = prematch_baseline(sub_ctx)
            return anchor_pre_match_prediction(result, baseline=baseline)

        def _anchor_caption(result: Optional[Dict[str, float]]) -> None:
            if (
                result is None
                or not is_pre_match
                or "_raw_model_score" not in result
            ):
                return
            raw = float(result["_raw_model_score"])
            base = float(result["_baseline_score"])
            st.caption(
                f"Pre-match anchor applied: blended raw model **{raw:.1f}** with "
                f"historical baseline **{base:.0f}** (45 % / 55 %). Live in-play "
                "predictions are shown as-is from the model."
            )

        if batting_team == bowling_team:
            st.warning("Batting team and bowling team must be different. Pick two different sides.")
        elif predict_both:
            if not is_pre_match:
                st.info(
                    "Both-innings prediction is most meaningful pre-match. Showing the "
                    "projections anyway based on the current match state you entered."
                )

            swap_payload = dict(payload)
            swap_payload["batting_team"] = bowling_team
            swap_payload["bowling_team"] = batting_team
            # When we flip innings, the other team's *squad* becomes the
            # batting lineup and vice versa.
            if bowling_xi_list:
                swap_payload["batting_xi"] = bowling_xi_list
            else:
                swap_payload.pop("batting_xi", None)
            if batting_xi_list:
                swap_payload["bowling_xi"] = batting_xi_list
            else:
                swap_payload.pop("bowling_xi", None)

            with st.spinner("Predicting both innings..."):
                res_a = call_backend_predict(payload)
                res_b = call_backend_predict(swap_payload)

            res_a = _maybe_anchor(res_a, batting_team, bowling_team)
            res_b = _maybe_anchor(res_b, bowling_team, batting_team)

            if res_a is not None and res_b is not None:
                st.markdown("#### Pre-match projection (both innings)")
                col_a, col_b = st.columns(2)
                with col_a:
                    _render_single(
                        res_a,
                        label=f"{batting_team} batting first",
                        current=float(current_score),
                    )
                with col_b:
                    _render_single(
                        res_b,
                        label=f"{bowling_team} batting first",
                        current=0.0,
                    )

                diff = res_a["predicted_final_score"] - res_b["predicted_final_score"]
                if abs(diff) < 1:
                    st.caption("Model sees both sides at roughly par — expect a tight contest.")
                else:
                    favored = batting_team if diff > 0 else bowling_team
                    st.caption(
                        f"Model edge: **{favored}** by ~{abs(diff):.1f} runs on first-innings projection."
                    )
                st.caption(
                    "Both predictions are first-innings projections; the current model "
                    "does not include chase-pressure dynamics."
                )
                _anchor_caption(res_a)
        else:
            result = call_backend_predict(payload)
            result = _maybe_anchor(result, batting_team, bowling_team)
            if result is not None:
                _render_single(
                    result,
                    label="Predicted Final Score",
                    current=float(current_score),
                )
                _anchor_caption(result)

        # Historical Context row - pulled from ipl_match_team_totals.csv
        st.markdown("#### Historical Context")
        hc_cols = st.columns(3)
        hc_cols[0].metric(
            "Venue avg",
            f"{ctx['venue_avg']:.0f}" if ctx.get("venue_avg") is not None else "N/A",
            help="Mean innings total at venues matching this name.",
        )
        hc_cols[1].metric(
            f"{batting_team[:22]} avg",
            f"{ctx['team_avg']:.0f}" if ctx.get("team_avg") is not None else "N/A",
            help=f"Mean innings total when {batting_team} bat.",
        )
        hc_cols[2].metric(
            "Head-to-head avg",
            f"{ctx['h2h_avg']:.0f}" if ctx.get("h2h_avg") is not None else "N/A",
            help=f"Mean innings total of {batting_team} vs {bowling_team}.",
        )
        if totals_df.empty:
            st.caption(
                "Context source `ipl_match_team_totals.csv` not loaded; averages unavailable."
            )
        else:
            st.caption(
                "Averages computed from `ipl_match_team_totals.csv`. 'N/A' means no rows "
                "matched for that venue/team (e.g., for non-IPL fixtures)."
            )

    st.subheader("💬 Research chat")
    st.caption(
        "Ask about IPL history (from **ipl_match_team_totals.csv**) or model predictions. "
        "Uses OpenAI function-calling on the server; ensure the API key is set for uvicorn."
    )
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    c_clear, _ = st.columns([1, 5])
    with c_clear:
        if st.button("Clear chat history"):
            st.session_state.chat_messages = []
            st.rerun()

    chat_endpoint = chat_url_from_predict_url(BACKEND_PREDICT_URL)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about IPL records or predicted scores…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        match_context: Dict[str, Any] = {
            "selected_fixture": {
                "team1": t1,
                "team2": t2,
                "venue": default_venue,
            },
            "scoreboard_parse": {k: v for k, v in score_info.items() if v is not None},
        }
        if "last_prediction_payload" in st.session_state:
            match_context["last_prediction_form"] = st.session_state["last_prediction_payload"]

        try:
            resp = requests.post(
                chat_endpoint,
                json={
                    "messages": st.session_state.chat_messages,
                    "match_context": match_context,
                },
                timeout=120,
            )
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                reply = f"Chat error ({resp.status_code}): {detail}"
            else:
                data = resp.json()
                reply = str(data.get("reply", "")) or "(Empty reply)"
        except requests.RequestException as exc:  # noqa: BLE001
            reply = f"Could not reach chat endpoint at {chat_endpoint}: {exc}"

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()

