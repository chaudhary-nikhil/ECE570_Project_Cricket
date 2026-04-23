"""
OpenAI chat with function calling over cricket CSV tools and score prediction.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from cricket_chat_tools import TOOL_FUNCTIONS

try:
    from openai import APIStatusError, OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]
    APIStatusError = RuntimeError  # type: ignore[misc, assignment]  # unused; run_chat exits early without OpenAI

MAX_TOOL_ROUNDS = 6


class OpenAIQuotaOrRateLimitError(RuntimeError):
    """OpenAI HTTP 429 (billing/quota or rate limit)."""



TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "team_innings_summary",
            "description": (
                "Summary stats for batting innings where the team name contains the given string "
                "(e.g. 'Kolkata' matches Kolkata Knight Riders). Optional IPL season label like 2007/08."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string", "description": "Substring or full IPL franchise name"},
                    "season": {
                        "type": "string",
                        "description": "Optional season label from the dataset, e.g. 2007/08",
                    },
                },
                "required": ["team_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "head_to_head_summary",
            "description": (
                "Historical IPL innings (from ball-by-ball totals) where both teams batted in the same match. "
                "Shows recent innings rows with scores and venues."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team_a": {"type": "string"},
                    "team_b": {"type": "string"},
                    "season": {"type": "string", "description": "Optional season filter"},
                },
                "required": ["team_a", "team_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "venue_summary",
            "description": "Aggregate batting innings at venues whose name contains the search string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "venue_search": {"type": "string"},
                    "season": {"type": "string"},
                },
                "required": ["venue_search"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_by_id",
            "description": "Look up all innings for a numeric match_id in the historical dataset.",
            "parameters": {
                "type": "object",
                "properties": {"match_id": {"type": "integer"}},
                "required": ["match_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_team_matches",
            "description": "Most recent batting innings for teams matching the given name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max rows, default 15, max 50"},
                },
                "required": ["team_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_innings_totals",
            "description": "Highest team innings totals in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many rows to return, default 15"},
                    "season": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "seasons_list",
            "description": "List all season labels present in the historical IPL totals CSV.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_innings_final_score",
            "description": (
                "Predict final first-innings total using the project's trained neural model (same as /predict). "
                "Use when the user asks for a predicted score, forecast, or what total to expect. "
                "Requires current match state: venue, batting and bowling teams, runs, wickets, overs, runs in last 5 overs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "venue": {"type": "string"},
                    "batting_team": {"type": "string"},
                    "bowling_team": {"type": "string"},
                    "current_score": {"type": "number"},
                    "wickets_lost": {"type": "integer"},
                    "overs_completed": {"type": "number"},
                    "runs_in_last_5_overs": {"type": "number"},
                },
                "required": [
                    "venue",
                    "batting_team",
                    "bowling_team",
                    "current_score",
                    "wickets_lost",
                    "overs_completed",
                    "runs_in_last_5_overs",
                ],
            },
        },
    },
]


def _system_prompt(match_context: Optional[Dict[str, Any]]) -> str:
    base = (
        "You are a helpful IPL cricket assistant. You answer using ONLY tool outputs for facts about "
        "historical matches (from ipl_match_team_totals.csv: per-innings team totals). "
        "Do not invent match results or statistics. If tools return no data, say so. "
        "For score predictions, call predict_innings_final_score with explicit parameters. "
        "Explain predictions clearly: they are model estimates with an approximate ±15 run uncertainty band. "
        "Teams and venues should use names consistent with IPL (e.g. Kolkata Knight Riders, M Chinnaswamy Stadium)."
    )
    if match_context:
        ctx = json.dumps(match_context, indent=2, default=str)
        base += (
            "\n\nThe user's app context (selected fixture and/or last submitted prediction form) follows. "
            "Use last_prediction_form values as defaults when calling predict_innings_final_score if the user "
            "asks for a prediction without specifying all numbers.\n"
            f"{ctx}"
        )
    return base


def _call_tool(name: str, arguments: str) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        return f"Invalid JSON arguments for {name}: {exc}"
    try:
        if name == "seasons_list":
            return fn()
        return fn(**args)
    except TypeError as exc:
        return f"Tool {name} argument error: {exc}"
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return f"Tool {name} failed: {exc}"


def run_chat(
    messages: List[Dict[str, str]],
    match_context: Optional[Dict[str, Any]] = None,
) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    api_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(match_context)},
        *messages,
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=api_messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except APIStatusError as exc:
            if getattr(exc, "status_code", None) == 429:
                raise OpenAIQuotaOrRateLimitError(
                    "OpenAI returned HTTP 429 (quota or rate limit). "
                    "If you saw insufficient_quota, this account has no usable paid credits. "
                    "Open https://platform.openai.com/account/billing , add a payment method or buy credits, "
                    "and use an API key from that same organization. Free-tier keys often need billing enabled."
                ) from exc
            raise RuntimeError(f"OpenAI API error ({exc.status_code}): {exc}") from exc
        choice = resp.choices[0]
        msg = choice.message

        if msg.tool_calls:
            api_messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                result = _call_tool(tc.function.name, tc.function.arguments or "{}")
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            continue

        return (msg.content or "").strip() or "(No response text.)"

    return "Sorry: too many tool calls in one turn. Please ask a simpler question."
