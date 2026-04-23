from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import backend


API_KEY = "2cecc518-dfe7-4fa0-baaf-446cd5060795"


app = FastAPI(
    title="Cricket Score Predictor",
    description="FastAPI backend to fetch live cricket scores and predict final scores.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    venue: str = Field(..., description="Match venue name")
    batting_team: str = Field(..., description="Name of the batting team")
    bowling_team: str = Field(..., description="Name of the bowling team")
    current_score: float = Field(..., ge=0, description="Current runs scored")
    wickets_lost: int = Field(..., ge=0, le=10, description="Wickets lost so far")
    overs_completed: float = Field(..., ge=0, le=20, description="Overs completed so far")
    runs_in_last_5_overs: float = Field(..., ge=0, description="Runs scored in the last 5 overs")
    # Optional lineup list form — backend maps names to prior indices. Absent ==
    # use a neutral 0.0 strength (roughly league average).
    batting_xi: Optional[List[str]] = Field(
        None,
        description="Optional batting XI / squad used to compute batting_xi_strength",
    )
    bowling_xi: Optional[List[str]] = Field(
        None,
        description="Optional bowling XI / squad used to compute bowling_attack_strength",
    )
    # Optional pre-computed scalars; take precedence over the lists when present
    batting_xi_strength: Optional[float] = Field(
        None, description="Direct override for batting XI strength (league-centered z-score)"
    )
    bowling_attack_strength: Optional[float] = Field(
        None, description="Direct override for bowling attack strength (league-centered z-score)"
    )


class PredictionResponse(BaseModel):
    predicted_final_score: float
    lower_bound: float
    upper_bound: float


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history ending with latest user turn")
    match_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional UI context: selected_fixture, last_prediction_form, scoreboard_parse",
    )


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Cricket score prediction API is running."}


@app.get("/live-scores")
def get_live_scores() -> Dict[str, Any]:
    """Fetch live match scores from CricketData.org via backend helper."""
    try:
        data = backend.fetch_live_scores(API_KEY)
    except RuntimeError as exc:
        # Map backend errors to HTTP errors
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return data


@app.post("/predict", response_model=PredictionResponse)
def predict_final_score(req: PredictionRequest) -> PredictionResponse:
    """Predict the final score given current match state."""
    bat_strength = req.batting_xi_strength
    bowl_strength = req.bowling_attack_strength
    if bat_strength is None and req.batting_xi:
        bat_strength = backend.aggregate_lineup_strength(req.batting_xi, kind="bat")
    if bowl_strength is None and req.bowling_xi:
        bowl_strength = backend.aggregate_lineup_strength(req.bowling_xi, kind="bowl")

    try:
        result = backend.predict_final_score(
            venue=req.venue,
            batting_team=req.batting_team,
            bowling_team=req.bowling_team,
            current_score=req.current_score,
            wickets_lost=req.wickets_lost,
            overs_completed=req.overs_completed,
            runs_in_last_5_overs=req.runs_in_last_5_overs,
            batting_xi_strength=bat_strength or 0.0,
            bowling_attack_strength=bowl_strength or 0.0,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        predicted_final_score=result["predicted_score"],
        lower_bound=result["lower_bound"],
        upper_bound=result["upper_bound"],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """LLM + tools over ipl_match_team_totals.csv and the score prediction model."""
    try:
        from cricket_chat_service import OpenAIQuotaOrRateLimitError, run_chat
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Chat module unavailable: {exc}",
        ) from exc

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        reply = run_chat(msgs, req.match_context)
    except OpenAIQuotaOrRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(reply=reply)

