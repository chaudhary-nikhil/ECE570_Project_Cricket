"""
Build per-player historical priors from Kaggle IPL ball-by-ball data.

Produces ``player_priors.csv`` keyed by a normalized player name, containing
batting and bowling strength indices that downstream code can average over
a lineup to compute ``batting_xi_strength`` and ``bowling_attack_strength``.

Intended to be regenerated occasionally (it is not part of training). The
artifact is consumed at inference time by the prediction backend and, during
retraining, as a fallback when a ball-by-ball row lacks an explicit XI.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


KAGGLE_IPL_CSV = Path("data/kaggle_ipl_dataset/IPL.csv")
OUTPUT_CSV = Path("data/player_priors.csv")


def normalize_name(name: str) -> str:
    """Case-fold + strip punctuation / suffixes so 'V Kohli' and 'V. Kohli' match.

    Output is a lowercase space-joined token string used purely as a lookup key.
    Display names are kept separately in the CSV so UI strings stay pretty.
    """
    if not isinstance(name, str):
        return ""
    s = name.strip()
    if not s:
        return ""
    # Strip parenthetical suffixes like "(c)" or "(wk)"
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.lower()
    # Collapse any non-alphanumeric run into a single space
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _zscore(values: pd.Series) -> pd.Series:
    """Simple standardization with a small floor on std to avoid div-by-zero."""
    mean = values.mean()
    std = values.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(np.zeros(len(values), dtype="float32"), index=values.index)
    return ((values - mean) / max(std, 1e-6)).astype("float32")


def build_priors(csv_path: Path) -> pd.DataFrame:
    """Compute career-to-date batting + bowling aggregates per normalized name."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} to exist; did you sync the Kaggle dataset?")

    usecols = [
        "match_id",
        "date",
        "batter",
        "bowler",
        "non_striker",
        "runs_batter",
        "balls_faced",
        "valid_ball",
        "runs_bowler",
        "wicket_kind",
        "player_out",
    ]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    # ---- Batting aggregates ----
    bat = df.copy()
    bat["batter_key"] = bat["batter"].map(normalize_name)
    bat = bat[bat["batter_key"] != ""]
    bat["runs_batter"] = pd.to_numeric(bat["runs_batter"], errors="coerce").fillna(0.0)
    bat["balls_faced"] = pd.to_numeric(bat["balls_faced"], errors="coerce").fillna(0.0)
    bat["batter_dismissed"] = (
        bat["player_out"].fillna("").map(normalize_name) == bat["batter_key"]
    ).astype(int)

    bat_stats = bat.groupby("batter_key").agg(
        display_name=("batter", "last"),
        bat_innings=("match_id", "nunique"),
        bat_runs=("runs_batter", "sum"),
        bat_balls=("balls_faced", "sum"),
        bat_dismissals=("batter_dismissed", "sum"),
    )
    bat_stats["bat_sr"] = 100.0 * bat_stats["bat_runs"] / np.maximum(bat_stats["bat_balls"], 1.0)
    bat_stats["bat_avg"] = bat_stats["bat_runs"] / np.maximum(bat_stats["bat_dismissals"], 1.0)

    # ---- Bowling aggregates ----
    bowl = df.copy()
    bowl["bowler_key"] = bowl["bowler"].map(normalize_name)
    bowl = bowl[bowl["bowler_key"] != ""]
    bowl["valid_ball"] = pd.to_numeric(bowl["valid_ball"], errors="coerce").fillna(0.0)
    bowl["runs_bowler"] = pd.to_numeric(bowl["runs_bowler"], errors="coerce").fillna(0.0)
    # Bowler gets credit for wickets except run-outs
    bowl["bowler_wicket"] = (
        bowl["wicket_kind"].fillna("").astype(str).str.strip().str.lower().isin(
            {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}
        )
    ).astype(int)

    bowl_stats = bowl.groupby("bowler_key").agg(
        bowl_innings=("match_id", "nunique"),
        bowl_balls=("valid_ball", "sum"),
        bowl_runs=("runs_bowler", "sum"),
        bowl_wickets=("bowler_wicket", "sum"),
    )
    bowl_stats["bowl_econ"] = (
        6.0 * bowl_stats["bowl_runs"] / np.maximum(bowl_stats["bowl_balls"], 1.0)
    )
    bowl_stats["bowl_sr"] = bowl_stats["bowl_balls"] / np.maximum(bowl_stats["bowl_wickets"], 1.0)

    priors = bat_stats.join(bowl_stats, how="outer")
    priors = priors.fillna(
        {
            "bat_innings": 0,
            "bat_runs": 0,
            "bat_balls": 0,
            "bat_dismissals": 0,
            "bat_sr": 0.0,
            "bat_avg": 0.0,
            "bowl_innings": 0,
            "bowl_balls": 0,
            "bowl_runs": 0,
            "bowl_wickets": 0,
            "bowl_econ": 0.0,
            "bowl_sr": 0.0,
        }
    )

    # ---- Strength indices ----
    # Only trust stats from players with enough volume. Below the floor we set
    # the strength to 0 (league average) so noisy debutants don't swing team
    # aggregates.
    MIN_BAT_BALLS = 60
    MIN_BOWL_BALLS = 60

    eligible_bat = priors["bat_balls"] >= MIN_BAT_BALLS
    bat_score = priors["bat_sr"] * 0.6 + np.log1p(priors["bat_avg"].clip(lower=0)) * 8.0
    priors["bat_strength"] = 0.0
    priors.loc[eligible_bat, "bat_strength"] = _zscore(bat_score[eligible_bat]).astype(float)

    eligible_bowl = priors["bowl_balls"] >= MIN_BOWL_BALLS
    # Lower economy and lower strike rate are both better for a bowler.
    bowl_score = -priors["bowl_econ"] - 0.05 * priors["bowl_sr"]
    priors["bowl_strength"] = 0.0
    priors.loc[eligible_bowl, "bowl_strength"] = _zscore(bowl_score[eligible_bowl]).astype(float)

    priors = priors.reset_index().rename(columns={"index": "player_key"})
    priors = priors[priors["player_key"] != ""]

    numeric_cols = [
        "bat_innings",
        "bat_runs",
        "bat_balls",
        "bat_dismissals",
        "bat_sr",
        "bat_avg",
        "bowl_innings",
        "bowl_balls",
        "bowl_runs",
        "bowl_wickets",
        "bowl_econ",
        "bowl_sr",
        "bat_strength",
        "bowl_strength",
    ]
    for c in numeric_cols:
        priors[c] = priors[c].astype(float).round(4)

    return priors[["player_key", "display_name"] + numeric_cols]


def main() -> int:
    priors = build_priors(KAGGLE_IPL_CSV)
    priors.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(priors)} player priors to {OUTPUT_CSV}")
    preview = priors.sort_values("bat_strength", ascending=False).head(5)
    print("Top 5 by bat_strength:")
    print(preview[["display_name", "bat_runs", "bat_balls", "bat_sr", "bat_strength"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
