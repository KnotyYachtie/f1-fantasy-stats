from __future__ import annotations
import httpx
import pandas as pd
from typing import Dict, Any

BASE = "https://api.openf1.org/v1"

def _get(endpoint: str, params: Dict[str, Any]) -> pd.DataFrame:
    url = f"{BASE}/{endpoint}"
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, list):
        data = [data]
    return pd.DataFrame(data)

def meetings(season: int) -> pd.DataFrame:
    """List meetings (events) in a season."""
    return _get("meetings", {"year": season})

def sessions(meeting_key: int) -> pd.DataFrame:
    """List all sessions for a given meeting (weekend)."""
    return _get("sessions", {"meeting_key": meeting_key})

def sessions_by_year(year: int) -> pd.DataFrame:
    """
    Return all sessions for a given season.
    Useful for building the list of Race sessions (for picking by race name).
    """
    return _get("sessions", {"year": year})

def session_result(session_key: int) -> pd.DataFrame:
    """Get results for a given session (Quali, Race, Sprint, etc)."""
    return _get("session_result", {"session_key": session_key})

def starting_grid(session_key: int) -> pd.DataFrame:
    """Get the starting grid for a Race session."""
    return _get("starting_grid", {"session_key": session_key})

def race_control(meeting_key: int) -> pd.DataFrame:
    """Race control messages: flags, SC, VSC, etc."""
    return _get("race_control", {"meeting_key": meeting_key})

def weather(meeting_key: int) -> pd.DataFrame:
    """Weather observations for a meeting."""
    return _get("weather", {"meeting_key": meeting_key})

def drivers(season: int | None = None, meeting_key: int | None = None) -> pd.DataFrame:
    """Driver info for a season or specific meeting."""
    params = {}
    if season is not None:
        params["year"] = season
    if meeting_key is not None:
        params["meeting_key"] = meeting_key
    return _get("drivers", params)
