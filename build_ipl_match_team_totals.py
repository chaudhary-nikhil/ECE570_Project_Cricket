"""
Build a per-innings summary CSV from Kaggle IPL ball-by-ball data.

Each row is one team's batting innings: total runs = sum of runs_total on every
ball while that team was batting. A normal T20 has two innings → two rows per
match_id; super overs / extra phases can add more rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/kaggle_ipl_dataset/IPL.csv"),
        help="Path to IPL.csv (ball-by-ball)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ipl_match_team_totals.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df["runs_total"] = pd.to_numeric(df["runs_total"], errors="coerce").fillna(0)

    agg_kw = {
        "date": ("date", "first"),
        "venue": ("venue", "first"),
        "season": ("season", "first"),
        "match_type": ("match_type", "first"),
        "batting_team": ("batting_team", "first"),
        "bowling_team": ("bowling_team", "first"),
        "total_runs": ("runs_total", "sum"),
    }

    out = df.groupby(["match_id", "innings"], as_index=False).agg(**agg_kw)

    if "team_wicket" in df.columns:
        w = df.groupby(["match_id", "innings"], as_index=False).agg(
            wickets_fallen=("team_wicket", "max")
        )
        out = out.merge(w, on=["match_id", "innings"], how="left")

    out = out.sort_values(["match_id", "innings"]).reset_index(drop=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
