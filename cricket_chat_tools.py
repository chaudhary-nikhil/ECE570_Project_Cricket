"""
Pandas-backed tools for the IPL research chatbot (historical match totals CSV).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import pandas as pd

import backend

MATCH_TOTALS_CSV = os.environ.get("IPL_MATCH_TOTALS_CSV", "ipl_match_team_totals.csv")
_MAX_ROWS_PRINT = 40
_MAX_CHARS = 6000


@lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    path = MATCH_TOTALS_CSV
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Match totals CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_runs"] = pd.to_numeric(df["total_runs"], errors="coerce")
    df["wickets_fallen"] = pd.to_numeric(df["wickets_fallen"], errors="coerce")
    return df


def _truncate(s: str) -> str:
    if len(s) > _MAX_CHARS:
        return s[: _MAX_CHARS] + "\n... (truncated)"
    return s


def _team_mask(series: pd.Series, query: str) -> pd.Series:
    q = (query or "").strip()
    if not q:
        return pd.Series(False, index=series.index)
    return series.astype(str).str.contains(q, case=False, na=False)


def team_innings_summary(team_name: str, season: Optional[str] = None) -> str:
    """Aggregate batting innings for teams whose name contains team_name."""
    df = _load_df()
    m = _team_mask(df["batting_team"], team_name)
    if season:
        m &= df["season"].astype(str) == str(season)
    sub = df.loc[m]
    if sub.empty:
        return f"No innings found for batting team matching '{team_name}'" + (
            f" in season {season}." if season else "."
        )
    g = sub["total_runs"]
    out = [
        f"Team filter: '{team_name}'" + (f", season={season}" if season else ""),
        f"Innings count: {len(sub)}",
        f"Mean innings total: {g.mean():.1f} runs",
        f"Median: {g.median():.1f}",
        f"Min–max: {g.min():.0f} – {g.max():.0f}",
    ]
    return _truncate("\n".join(out))


def head_to_head_summary(team_a: str, team_b: str, season: Optional[str] = None) -> str:
    """Matches where both teams had a batting innings; show each innings row."""
    df = _load_df()
    if season:
        df = df[df["season"].astype(str) == str(season)]
    ma = _team_mask(df["batting_team"], team_a)
    mb = _team_mask(df["batting_team"], team_b)
    ids_a = set(df.loc[ma, "match_id"].unique())
    ids_b = set(df.loc[mb, "match_id"].unique())
    common = ids_a & ids_b
    if not common:
        return f"No matches found where both '{team_a}' and '{team_b}' batted."
    sub = df[df["match_id"].isin(common)]
    if sub.empty:
        return f"No head-to-head rows for '{team_a}' vs '{team_b}' in season {season}."
    sub = sub.sort_values(["date", "match_id", "innings"], ascending=[False, True, True])
    lines = [
        f"Head-to-head (batting innings): '{team_a}' vs '{team_b}'"
        + (f", season={season}" if season else ""),
        f"Distinct matches: {sub['match_id'].nunique()}, innings rows: {len(sub)}",
    ]
    for _, r in sub.head(_MAX_ROWS_PRINT).iterrows():
        lines.append(
            f"  match_id={int(r['match_id'])} inn={int(r['innings'])} {r['date'].date() if pd.notna(r['date']) else '?'} "
            f"{r['batting_team']} vs {r['bowling_team']}: {int(r['total_runs'])}/{int(r['wickets_fallen']) if pd.notna(r['wickets_fallen']) else '?'} "
            f"at {r['venue']}"
        )
    if len(sub) > _MAX_ROWS_PRINT:
        lines.append(f"... and {len(sub) - _MAX_ROWS_PRINT} more innings rows.")
    return _truncate("\n".join(lines))


def venue_summary(venue_search: str, season: Optional[str] = None) -> str:
    """Batting innings at venues whose name contains venue_search."""
    df = _load_df()
    m = _team_mask(df["venue"], venue_search)
    if season:
        m &= df["season"].astype(str) == str(season)
    sub = df.loc[m]
    if sub.empty:
        return f"No innings at venues matching '{venue_search}'" + (
            f" in season {season}." if season else "."
        )
    g = (
        sub.groupby("venue", as_index=False)
        .agg(
            innings=("total_runs", "count"),
            mean_runs=("total_runs", "mean"),
            median_runs=("total_runs", "median"),
            max_runs=("total_runs", "max"),
        )
        .sort_values("mean_runs", ascending=False)
    )
    lines = [
        f"Venue filter: '{venue_search}'" + (f", season={season}" if season else ""),
        "Venues with mean innings total (min 5 innings):",
    ]
    top = g[g["innings"] >= 5].head(15)
    for _, row in top.iterrows():
        lines.append(
            f"  {row['venue']}: innings={int(row['innings'])}, mean={row['mean_runs']:.1f}, "
            f"median={row['median_runs']:.1f}, max={row['max_runs']:.0f}"
        )
    if top.empty:
        lines.append("  (No venue had at least 5 innings; showing all with any data.)")
        for _, row in g.head(10).iterrows():
            lines.append(
                f"  {row['venue']}: innings={int(row['innings'])}, mean={row['mean_runs']:.1f}"
            )
    return _truncate("\n".join(lines))


def match_by_id(match_id: int) -> str:
    """All innings rows for a match_id."""
    df = _load_df()
    sub = df[df["match_id"] == int(match_id)].sort_values("innings")
    if sub.empty:
        return f"No rows for match_id={match_id}."
    lines = [f"Match {match_id}:"]
    for _, r in sub.iterrows():
        lines.append(
            f"  Innings {int(r['innings'])}: {r['batting_team']} vs {r['bowling_team']} — "
            f"{int(r['total_runs'])}/{int(r['wickets_fallen']) if pd.notna(r['wickets_fallen']) else '?'} "
            f"on {r['date'].date() if pd.notna(r['date']) else '?'} at {r['venue']} ({r['season']})"
        )
    return _truncate("\n".join(lines))


def recent_team_matches(team_name: str, limit: int = 15) -> str:
    """Most recent batting innings for teams matching team_name."""
    df = _load_df()
    m = _team_mask(df["batting_team"], team_name)
    sub = df.loc[m].sort_values("date", ascending=False).head(max(1, min(limit, 50)))
    if sub.empty:
        return f"No innings for team matching '{team_name}'."
    lines = [f"Recent batting innings (up to {len(sub)}) for '{team_name}':"]
    for _, r in sub.iterrows():
        lines.append(
            f"  {r['date'].date() if pd.notna(r['date']) else '?'} match_id={int(r['match_id'])} "
            f"inn={int(r['innings'])} vs {r['bowling_team']}: {int(r['total_runs'])}/{int(r['wickets_fallen']) if pd.notna(r['wickets_fallen']) else '?'} "
            f"at {r['venue']}"
        )
    return _truncate("\n".join(lines))


def top_innings_totals(limit: int = 15, season: Optional[str] = None) -> str:
    """Highest batting innings totals in the dataset."""
    df = _load_df()
    sub = df
    if season:
        sub = sub[sub["season"].astype(str) == str(season)]
    sub = sub.nlargest(max(1, min(limit, 30)), "total_runs")
    lines = ["Highest innings totals:" + (f" season={season}" if season else "")]
    for _, r in sub.iterrows():
        lines.append(
            f"  {int(r['total_runs'])} — {r['batting_team']} vs {r['bowling_team']}, "
            f"match_id={int(r['match_id'])} inn={int(r['innings'])}, {r['date'].date() if pd.notna(r['date']) else '?'}, {r['venue']}"
        )
    return _truncate("\n".join(lines))


def seasons_list() -> str:
    """Distinct seasons in the CSV."""
    df = _load_df()
    seasons = sorted(df["season"].dropna().astype(str).unique())
    return "Seasons in dataset: " + ", ".join(seasons)


def predict_innings_final_score(
    venue: str,
    batting_team: str,
    bowling_team: str,
    current_score: float,
    wickets_lost: int,
    overs_completed: float,
    runs_in_last_5_overs: float,
) -> str:
    """Call the same Keras model as /predict."""
    try:
        out = backend.predict_final_score(
            venue=venue or "Unknown",
            batting_team=batting_team,
            bowling_team=bowling_team,
            current_score=float(current_score),
            wickets_lost=float(wickets_lost),
            overs_completed=float(overs_completed),
            runs_in_last_5_overs=float(runs_in_last_5_overs),
        )
    except RuntimeError as exc:
        return f"Prediction failed: {exc}"
    return (
        f"Predicted final score (first innings): {out['predicted_score']:.1f} runs "
        f"(approximate range {out['lower_bound']:.0f}–{out['upper_bound']:.0f} runs, ±15 MAE band). "
        f"Inputs: venue={venue}, batting={batting_team}, bowling={bowling_team}, "
        f"current={current_score}/{wickets_lost} in {overs_completed} ov, last 5 ov runs={runs_in_last_5_overs}."
    )


TOOL_FUNCTIONS = {
    "team_innings_summary": team_innings_summary,
    "head_to_head_summary": head_to_head_summary,
    "venue_summary": venue_summary,
    "match_by_id": match_by_id,
    "recent_team_matches": recent_team_matches,
    "top_innings_totals": top_innings_totals,
    "seasons_list": seasons_list,
    "predict_innings_final_score": predict_innings_final_score,
}
