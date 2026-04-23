import os
import pickle
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from model_io import MAX_WICKETS, PLAYER_PRIORS_PATH, T20_MAX_OVERS


MODEL_PATH = "cricket_model.h5"
ENCODERS_PATH = "label_encoders.pkl"
# currentMatches returns full match objects including "venue"; cricScore returns minimal score-only data
CURRENT_MATCHES_URL = "https://api.cricapi.com/v1/currentMatches"
CRIC_SCORE_URL = "https://api.cricapi.com/v1/cricScore"

# Margin (runs) for confidence range, based on observed MAE during training.
PREDICTION_MARGIN_MAE = 15

_model: tf.keras.Model | None = None
_encoders: Dict[str, Any] = {}
_encoder_meta: Dict[str, Any] = {}


def _load_model() -> tf.keras.Model:
    """Load and cache the Keras model once."""
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at '{MODEL_PATH}'.")

    try:
        # For inference only; avoid deserializing training losses/metrics.
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    return _model


def _load_encoders() -> Dict[str, Any]:
    """Load and cache label encoders once."""
    global _encoders

    if _encoders:
        return _encoders

    if not os.path.exists(ENCODERS_PATH):
        raise RuntimeError(f"Encoders file not found at '{ENCODERS_PATH}'.")

    try:
        with open(ENCODERS_PATH, "rb") as f:
            loaded = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load encoders: {exc}") from exc

    global _encoder_meta

    if isinstance(loaded, dict) and "encoders" in loaded:
        encs: Dict[str, Any] = loaded["encoders"]
        _encoder_meta = loaded.get("meta", {}) or {}
    else:
        encs = loaded
        _encoder_meta = {}

    # Ensure "Unknown" class exists for robust handling of unseen categories.
    for col, le in encs.items():
        if col not in ("venue", "batting_team", "bowling_team"):
            continue
        classes = list(le.classes_)
        if "Unknown" not in classes:
            classes.append("Unknown")
            le.classes_ = np.array(classes, dtype=object)

    _encoders = encs
    return _encoders


# Common IPL team-name variants / historical rebrands that must resolve to the
# same embedding as the canonical name the encoder was fit on. Keys are the
# incoming variant, values are the target canonical name. Matching is case
# insensitive and whitespace trimmed.
_TEAM_ALIASES: Dict[str, str] = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "rcb": "Royal Challengers Bengaluru",
    "bangalore": "Royal Challengers Bengaluru",
    "bengaluru": "Royal Challengers Bengaluru",
    "csk": "Chennai Super Kings",
    "chennai": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "mumbai": "Mumbai Indians",
    "dc": "Delhi Capitals",
    "delhi daredevils": "Delhi Capitals",
    "delhi": "Delhi Capitals",
    "kkr": "Kolkata Knight Riders",
    "kolkata": "Kolkata Knight Riders",
    "rr": "Rajasthan Royals",
    "rajasthan": "Rajasthan Royals",
    "srh": "Sunrisers Hyderabad",
    "sunrisers": "Sunrisers Hyderabad",
    "hyderabad": "Sunrisers Hyderabad",
    "pbks": "Punjab Kings",
    "kings xi punjab": "Punjab Kings",
    "punjab": "Punjab Kings",
    "gt": "Gujarat Titans",
    "gujarat": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
    "lucknow": "Lucknow Super Giants",
}


def _resolve_against_classes(value: str, classes: list) -> str | None:
    """Return the canonical class that best matches ``value``.

    Tries an exact case-insensitive match first, then a token-subset match
    (e.g. "Chinnaswamy" -> "M Chinnaswamy Stadium"), and finally a substring
    match. Returns None if nothing plausible lines up so the caller can decide
    whether to fall back to "Unknown".
    """
    if not value:
        return None

    norm = value.strip().lower()
    if not norm:
        return None

    class_norms = [str(c).strip().lower() for c in classes]

    # Exact (case insensitive)
    for raw, low in zip(classes, class_norms):
        if low == norm:
            return str(raw)

    value_tokens = {t for t in re.split(r"[^a-z0-9]+", norm) if t}
    if not value_tokens:
        return None

    # Token-subset: every value token appears in the candidate class name
    best = None
    best_overlap = 0
    for raw, low in zip(classes, class_norms):
        cand_tokens = {t for t in re.split(r"[^a-z0-9]+", low) if t}
        if not cand_tokens:
            continue
        if value_tokens.issubset(cand_tokens):
            overlap = len(value_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best = str(raw)
    if best is not None:
        return best

    # Substring containment as a last resort
    for raw, low in zip(classes, class_norms):
        if norm in low or low in norm:
            return str(raw)

    return None


def _normalize_category_value(value: str, col_name: str, classes: list) -> str:
    """Map an incoming team / venue string to a trained encoder class when possible."""
    if value is None:
        return "Unknown"

    raw = str(value).strip()
    if not raw:
        return "Unknown"

    norm_key = raw.lower()

    if col_name in ("batting_team", "bowling_team"):
        aliased = _TEAM_ALIASES.get(norm_key)
        if aliased and aliased in classes:
            return aliased

    matched = _resolve_against_classes(raw, classes)
    if matched is not None:
        return matched

    return raw  # caller will fall back to "Unknown" if still unmatched


def _encode_category(value: str, col_name: str) -> int:
    """Encode a categorical feature using the stored LabelEncoder."""
    encoders = _load_encoders()

    if col_name not in encoders:
        raise RuntimeError(f"Label encoder for '{col_name}' is missing.")

    le = encoders[col_name]
    classes = list(le.classes_)

    # Normalize to a canonical class name (aliases + token/substring match)
    # before falling back to "Unknown", so near-miss variants still reach
    # a trained embedding instead of collapsing to a generic prior.
    val = _normalize_category_value(value, col_name, classes)
    if val not in classes:
        if "Unknown" in classes:
            val = "Unknown"
        else:
            val = classes[0]

    try:
        encoded = int(le.transform([val])[0])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to encode '{col_name}' value '{value}': {exc}") from exc

    return encoded


def _clip_embedding_index(model: tf.keras.Model, embedding_layer_name: str, idx: int) -> int:
    """
    LabelEncoder may gain an extra 'Unknown' class at load time, producing an index equal to
    Embedding.input_dim, which is out of range (valid indices are 0 .. input_dim-1).
    """
    layer = model.get_layer(embedding_layer_name)
    n = int(layer.input_dim)
    return int(np.clip(int(idx), 0, n - 1))


_player_priors_cache: Optional[Dict[str, Dict[str, float]]] = None
_player_alias_cache: Optional[Dict[str, str]] = None


def _normalize_player_name(name: str) -> str:
    """Mirror ``build_player_priors.normalize_name`` without importing at module load."""
    if not isinstance(name, str):
        return ""
    s = re.sub(r"\([^)]*\)", " ", name).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _load_player_priors() -> Dict[str, Dict[str, float]]:
    """Load (and cache) the per-player priors keyed by normalized name."""
    global _player_priors_cache, _player_alias_cache
    if _player_priors_cache is not None:
        return _player_priors_cache
    if not os.path.exists(PLAYER_PRIORS_PATH):
        _player_priors_cache = {}
        _player_alias_cache = {}
        return _player_priors_cache
    try:
        df = pd.read_csv(PLAYER_PRIORS_PATH)
    except Exception:
        _player_priors_cache = {}
        _player_alias_cache = {}
        return _player_priors_cache
    mapping: Dict[str, Dict[str, float]] = {}
    # Alias index: "kohli" -> "v kohli" when it is unambiguous. ESPN rosters
    # frequently ship full first names ("Virat Kohli") while the Kaggle CSV
    # uses initials ("V Kohli"), so a lastname-only fallback recovers most
    # names without hand-curated alias tables.
    last_to_keys: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        key = str(row.get("player_key", "")).strip()
        if not key:
            continue
        mapping[key] = {
            "bat_strength": float(row.get("bat_strength", 0.0) or 0.0),
            "bowl_strength": float(row.get("bowl_strength", 0.0) or 0.0),
        }
        tokens = key.split()
        if tokens:
            last_to_keys.setdefault(tokens[-1], []).append(key)
    alias: Dict[str, str] = {}
    for last, keys in last_to_keys.items():
        if len(keys) == 1:
            alias[last] = keys[0]
    _player_priors_cache = mapping
    _player_alias_cache = alias
    return mapping


def _resolve_player_key(normalized: str) -> Optional[str]:
    """Resolve an input player name to a priors key via exact / initial+last / last-only."""
    priors = _load_player_priors()
    if not normalized:
        return None
    if normalized in priors:
        return normalized
    tokens = normalized.split()
    if not tokens:
        return None
    last = tokens[-1]
    # Initial + last (e.g. "virat kohli" -> "v kohli")
    if len(tokens) >= 2:
        candidate = f"{tokens[0][0]} {last}"
        if candidate in priors:
            return candidate
    # Lastname-only fallback, but only when unique across the dataset.
    alias = _player_alias_cache or {}
    return alias.get(last)


def aggregate_lineup_strength(players: Iterable[str], kind: str) -> float:
    """Return the mean batting or bowling strength across a list of player names.

    Unknown names contribute 0.0 (league average) rather than being dropped, so
    the aggregate stays comparable across rosters with varying name-match rates.
    """
    if kind not in {"bat", "bowl"}:
        raise ValueError("kind must be 'bat' or 'bowl'")
    priors = _load_player_priors()
    field = "bat_strength" if kind == "bat" else "bowl_strength"
    vals: List[float] = []
    for name in players or []:
        normalized = _normalize_player_name(str(name))
        key = _resolve_player_key(normalized)
        if key is None:
            vals.append(0.0)
            continue
        vals.append(float(priors.get(key, {}).get(field, 0.0)))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _build_numeric_matrix(
    current_score: float,
    wickets_lost: float,
    overs_completed: float,
    runs_in_last_5_overs: float,
    batting_xi_strength: float = 0.0,
    bowling_attack_strength: float = 0.0,
) -> np.ndarray:
    """Numeric feature vector; order must match ``model_io.NUMERIC_COLS``."""
    cs = float(current_score)
    wk = float(wickets_lost)
    oc = float(overs_completed)
    rl = float(runs_in_last_5_overs)
    over_rem = max(0.0, float(T20_MAX_OVERS) - oc)
    rr = cs / max(oc, 0.1)
    wk_rem = max(0.0, float(MAX_WICKETS) - wk)
    bat_xi = float(batting_xi_strength)
    bowl_atk = float(bowling_attack_strength)
    vec = np.array(
        [[cs, wk, oc, rl, over_rem, rr, wk_rem, bat_xi, bowl_atk]],
        dtype=np.float32,
    )
    return vec


def _build_legacy_flat_vector(
    venue: str,
    batting_team: str,
    bowling_team: str,
    current_score: float,
    wickets_lost: float,
    overs_completed: float,
    runs_in_last_5_overs: float,
) -> np.ndarray:
    """Single-matrix input for older checkpoints (7 features, label-encoded teams in-vector)."""
    venue_enc = _encode_category(venue, "venue")
    bat_team_enc = _encode_category(batting_team, "batting_team")
    bowl_team_enc = _encode_category(bowling_team, "bowling_team")
    numeric_features = [
        float(current_score),
        float(wickets_lost),
        float(overs_completed),
        float(runs_in_last_5_overs),
    ]
    return np.array(
        [venue_enc, bat_team_enc, bowl_team_enc] + numeric_features,
        dtype=np.float32,
    ).reshape(1, -1)


def predict_final_score(
    venue: str,
    batting_team: str,
    bowling_team: str,
    current_score: float,
    wickets_lost: float,
    overs_completed: float,
    runs_in_last_5_overs: float,
    batting_xi_strength: float = 0.0,
    bowling_attack_strength: float = 0.0,
) -> Dict[str, float]:
    """
    Predict the final score given the current match state.

    Returns a dict with predicted_score and a confidence range (lower_bound,
    upper_bound) using a fixed margin based on training MAE.
    """
    model = _load_model()
    _load_encoders()

    try:
        if len(model.inputs) == 4:
            v_raw = _encode_category(venue, "venue")
            b_raw = _encode_category(batting_team, "batting_team")
            o_raw = _encode_category(bowling_team, "bowling_team")
            v_id = np.array(
                [[_clip_embedding_index(model, "emb_venue", v_raw)]],
                dtype=np.int32,
            )
            b_id = np.array(
                [[_clip_embedding_index(model, "emb_bat", b_raw)]],
                dtype=np.int32,
            )
            o_id = np.array(
                [[_clip_embedding_index(model, "emb_bowl", o_raw)]],
                dtype=np.int32,
            )
            num = _build_numeric_matrix(
                current_score=current_score,
                wickets_lost=wickets_lost,
                overs_completed=overs_completed,
                runs_in_last_5_overs=runs_in_last_5_overs,
                batting_xi_strength=batting_xi_strength,
                bowling_attack_strength=bowling_attack_strength,
            )
            # The saved checkpoint may have been trained on a shorter numeric
            # vector (pre-lineup-strength). Trim extras so old .h5 files keep
            # working while the training script is being re-run.
            expected_num_features = int(model.inputs[3].shape[-1])
            if num.shape[1] > expected_num_features:
                num = num[:, :expected_num_features]
            prediction = model.predict([v_id, b_id, o_id, num], verbose=0)
            remaining = float(prediction[0][0])
            cs = float(current_score)
            val = max(cs, cs + remaining)
        else:
            features = _build_legacy_flat_vector(
                venue=venue,
                batting_team=batting_team,
                bowling_team=bowling_team,
                current_score=current_score,
                wickets_lost=wickets_lost,
                overs_completed=overs_completed,
                runs_in_last_5_overs=runs_in_last_5_overs,
            )
            prediction = model.predict(features, verbose=0)
            val = float(prediction[0][0])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Model prediction failed: {exc}") from exc

    return {
        "predicted_score": val,
        "lower_bound": val - PREDICTION_MARGIN_MAE,
        "upper_bound": val + PREDICTION_MARGIN_MAE,
    }


def get_feature_sensitivity(
    venue: str,
    batting_team: str,
    bowling_team: str,
    current_score: float,
    wickets_lost: float,
    overs_completed: float,
    runs_in_last_5_overs: float,
) -> Dict[str, float]:
    """
    Estimate the impact of each numeric feature on the predicted score.

    For each numeric feature, this computes the base prediction, then
    increases that feature slightly (by +1) while keeping the others
    fixed, and returns the difference.
    """
    base_result = predict_final_score(
        venue=venue,
        batting_team=batting_team,
        bowling_team=bowling_team,
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        runs_in_last_5_overs=runs_in_last_5_overs,
    )
    base = base_result["predicted_score"]

    impacts: Dict[str, float] = {}

    # Define simple +1 perturbations for each numeric feature
    deltas = {
        "current_score": 1.0,
        "wickets_lost": 1.0,
        "overs_completed": 1.0,
        "runs_in_last_5_overs": 1.0,
    }

    for feature_name, delta in deltas.items():
        kwargs = {
            "venue": venue,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "current_score": current_score,
            "wickets_lost": wickets_lost,
            "overs_completed": overs_completed,
            "runs_in_last_5_overs": runs_in_last_5_overs,
        }
        kwargs[feature_name] = kwargs[feature_name] + delta

        new_result = predict_final_score(**kwargs)
        impacts[feature_name] = new_result["predicted_score"] - base

    return impacts


def fetch_live_scores(api_key: str) -> Dict[str, Any]:
    """
    Fetch current matches from CricketData.org (includes venue, teams, etc.).

    Uses currentMatches endpoint so each match has "venue", "teams", "name".
    """
    try:
        # (connect_timeout, read_timeout): give server more time to respond
        resp = requests.get(
            CURRENT_MATCHES_URL,
            params={"apikey": api_key, "offset": 0},
            timeout=(10, 30),
        )
    except requests.RequestException as exc:  # noqa: BLE001
        raise RuntimeError(f"Error while contacting CricketData API: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"CricketData API returned status {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except ValueError as exc:  # JSON decode error
        raise RuntimeError("Failed to parse response from CricketData API.") from exc

    return data

