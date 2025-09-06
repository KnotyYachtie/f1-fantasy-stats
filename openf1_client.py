from __future__ import annotations
import httpx
import pandas as pd
from typing import Dict, Any, Optional, Tuple

BASE = "https://api.openf1.org/v1"

def _get(endpoint: str, params: Dict[str, Any]) -> pd.DataFrame:
def sessions_by_year(year: int) -> pd.DataFrame:
    """
    Return all sessions for a given season.
    Useful for building the list of Race sessions (for picking by race name).
    """
    return _get("sessions", {"year": year})
    url = f"{BASE}/{endpoint}"
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, list):
        data = [data]
    return pd.DataFrame(data)

def meetings(season: int) -> pd.DataFrame:
    # List meetings (events) in a season
    return _get("meetings", {"year": season})

def sessions(meeting_key: int) -> pd.DataFrame:
    return _get("sessions", {"meeting_key": meeting_key})

def results(meeting_key: int, session_name: str = "Race") -> pd.DataFrame:
    # session_name examples: "Race", "Qualifying", "Sprint", "Practice 1", etc.
    sess_df = sessions(meeting_key)
    sess_df = sess_df[sess_df["session_name"] == session_name]
    if sess_df.empty:
        return pd.DataFrame()
    session_key = int(sess_df.iloc[0]["session_key"])
    return _get("results", {"session_key": session_key})

def laps(session_key: int) -> pd.DataFrame:
    return _get("laps", {"session_key": session_key})

def race_control(meeting_key: int) -> pd.DataFrame:
    # flags, SC/VSC, etc across sessions
    return _get("race_control", {"meeting_key": meeting_key})

def weather(meeting_key: int) -> pd.DataFrame:
    return _get("weather", {"meeting_key": meeting_key})

def drivers(season: Optional[int] = None, meeting_key: Optional[int] = None) -> pd.DataFrame:
    params = {}
    if season is not None:
        params["year"] = season
    if meeting_key is not None:
        params["meeting_key"] = meeting_key
    return _get("drivers", params)

def grid(meeting_key: int) -> pd.DataFrame:
    # Starting grid for race
    race_df = sessions(meeting_key)
    race_df = race_df[race_df["session_name"] == "Race"]
    if race_df.empty:
        return pd.DataFrame()
    session_key = int(race_df.iloc[0]["session_key"])
    return _get("grid", {"session_key": session_key})
