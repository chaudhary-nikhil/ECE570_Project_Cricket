"""Shared training/inference constants for the score predictor."""

from __future__ import annotations

# Order must match training and backend numeric vector construction.
NUMERIC_COLS = [
    "current_score",
    "wickets_lost",
    "overs_completed",
    "runs_in_last_5_overs",
    "overs_remaining",
    "run_rate",
    "wickets_remaining",
    # Roster-aware lineup features; 0 == league average, positive == above par.
    "batting_xi_strength",
    "bowling_attack_strength",
]

CATEGORICAL_COLS = ["venue", "batting_team", "bowling_team"]

T20_MAX_OVERS = 20.0
MAX_WICKETS = 10.0

# Saved inside label_encoders.pkl under "meta"
META_KEY_MODEL = "embedding_remaining_v2"

# Path to per-player priors artifact (produced by build_player_priors.py)
PLAYER_PRIORS_PATH = "player_priors.csv"
